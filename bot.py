"""
FineDiet - Telegram-бот, который относит товары из чеков к категориям продуктов
(справочник: Тутельян В. А. «Химический состав и калорийность российских продуктов питания», 2012).

Запуск:
    pip install -r requirements.txt
    python train_model.py --data receipt_items.csv     # один раз: обучить и сохранить model.joblib
    set BOT_TOKEN=123456:ABC...   (Windows)   /   export BOT_TOKEN=...   (Linux/macOS)
    python bot.py
"""
import json
import logging
import os
import tempfile

import joblib

from model import ReceiptClassifier, clean  # noqa: F401  (класс нужен для загрузки model.joblib)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes,
                          MessageHandler, filters)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("finediet")

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
MAX_FILE_MB = 2
LOW_CONFIDENCE = 0.30          # ниже этой вероятности бот честно говорит, что не уверен

model = joblib.load(MODEL_PATH)
log.info("модель загружена: %s", MODEL_PATH)

sent_messages: dict[int, list[int]] = {}   # chat_id -> id сообщений бота за сессию


def greeting(update: Update) -> str:
    u = update.effective_user
    return f"@{u.username}" if u.username else u.first_name


def menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Загрузить чек (JSON)", callback_data="upload_json")],
        [InlineKeyboardButton("Ввести название продукта", callback_data="manual_input")],
        [InlineKeyboardButton("Завершить сессию", callback_data="end_session")],
    ])


def classify(names: list[str]) -> list[tuple[str, str, float]]:
    proba = model.predict_proba(names)
    classes = model.classes_
    return [(n, classes[p.argmax()], float(p.max())) for n, p in zip(names, proba)]


def format_result(name: str, cls: str, p: float) -> str:
    note = "  (не уверен)" if p < LOW_CONFIDENCE else ""
    return f"• {name}\n   → {cls} — {p:.0%}{note}"


async def remember(chat_id: int, message) -> None:
    sent_messages.setdefault(chat_id, []).append(message.message_id)


# ---------- команды и кнопки ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Старт", callback_data="start_action")]])
    await update.message.reply_text(f"Привет, {greeting(update)}! Я FineDiet: определяю категорию продуктов из чеков.",
                                    reply_markup=keyboard)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    action = query.data

    if action == "start_action":
        await query.edit_message_text(f"{greeting(update)}, выбери, что сделать:", reply_markup=menu())
    elif action == "upload_json":
        m = await query.message.reply_text(
            "Пришли JSON-файл чека (формат приложения «Проверка чеков» ФНС: ticket → document → receipt → items).")
        await remember(chat_id, m)
    elif action == "manual_input":
        m = await query.message.reply_text("Напиши название продукта, например: «молоко 2,5% 1л».")
        await remember(chat_id, m)
    elif action == "end_session":
        for mid in sent_messages.pop(chat_id, []):
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=mid)
            except Exception as e:  # сообщение уже удалено или слишком старое
                log.warning("не удалось удалить сообщение %s: %s", mid, e)
        await query.edit_message_text("Сессия завершена. До встречи!")


# ---------- обработка чека в JSON ----------
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    doc = update.message.document
    if doc.file_size and doc.file_size > MAX_FILE_MB * 1024 * 1024:
        m = await update.message.reply_text(f"Файл больше {MAX_FILE_MB} МБ, такой чек я не обработаю.")
        await remember(chat_id, m)
        return
    try:
        tg_file = await doc.get_file()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "receipt.json")          # своё имя файла, а не присланное
            await tg_file.download_to_drive(path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        if isinstance(data, list):                             # иногда выгрузка - список чеков
            data = data[0] if data else {}
        items = data.get("ticket", {}).get("document", {}).get("receipt", {}).get("items", [])
        names = [it.get("name") for it in items if it.get("name")]
        if not names:
            raise ValueError("в файле нет товаров (ожидается ticket → document → receipt → items)")
        lines = [format_result(*r) for r in classify(names)]
        for i in range(0, len(lines), 30):                     # длинный чек - несколькими сообщениями
            m = await update.message.reply_text("\n".join(lines[i:i + 30]))
            await remember(chat_id, m)
    except Exception as e:
        log.exception("ошибка обработки JSON")
        m = await update.message.reply_text(f"Не получилось прочитать чек: {e}")
        await remember(chat_id, m)
    await update.message.reply_text("Что дальше?", reply_markup=menu())


# ---------- ручной ввод ----------
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    name, cls, p = classify([update.message.text])[0]
    proba = model.predict_proba([name])[0]
    top = proba.argsort()[::-1][1:3]
    alt = "\n".join(f"   также возможно: {model.classes_[i]} — {proba[i]:.0%}" for i in top)
    m = await update.message.reply_text(format_result(name, cls, p) + "\n" + alt)
    await remember(chat_id, m)
    await update.message.reply_text("Что дальше?", reply_markup=menu())


def main() -> None:
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise SystemExit("Задай токен бота в переменной окружения BOT_TOKEN (см. README).")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^(start_action|upload_json|manual_input|end_session)$"))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("бот запущен, Ctrl+C для остановки")
    app.run_polling()


if __name__ == "__main__":
    main()
