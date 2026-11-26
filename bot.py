"""
FineDiet — Telegram-бот: чеки → категории продуктов → химсостав → анализ рациона и корзина на неделю.
Агент-нутрициолог (LLM с инструментами) отвечает на вопросы своими словами, но все числа берёт из инструментов.

Запуск (подробно в README):
    pip install -r requirements.txt
    python train_model.py --data receipt_items.csv      # один раз → model.joblib
    положить nutrients.csv (химсостав категорий на 100 г) рядом с ботом
    $env:BOT_TOKEN="..."                                # PowerShell; токен не хранить в коде
    $env:LLM_BASE_URL="http://localhost:11434/v1"       # необязательно: агент через Ollama
    python bot.py
"""
import json
import logging
import os
import tempfile


def load_dotenv(path=".env"):
    """Переменные из .env (если файл есть); уже заданные в терминале важнее."""
    if os.path.exists(path):
        for line in open(path, encoding="utf-8-sig"):
            k, sep, v = line.strip().partition("=")
            if sep and k and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


load_dotenv()

import joblib  # noqa: E402
from telegram import InlineKeyboardButton as B, InlineKeyboardMarkup as KB, InputMediaPhoto, Update  # noqa: E402
from telegram.constants import ChatAction  # noqa: E402
from telegram.ext import (ApplicationBuilder,  # noqa: E402
                           CallbackQueryHandler, CommandHandler, ContextTypes,
                          MessageHandler, filters)

import agent  # noqa: E402
from model import ReceiptClassifier, clean  # noqa: F401,E402  (класс нужен для загрузки model.joblib)
from nutrition import (LABELS, MICROS, NutrientTable, analyze, daily_targets, format_report,  # noqa: E402
                       format_summary, pretty)
from optimizer import build_pool, format_basket, optimize, swaps  # noqa: E402
from classify import Classifier  # noqa: E402
try:
    import charts  # noqa: E402
except ImportError:                                  # matplotlib не установлен — отчёты без картинок
    charts = None
from readers import ReadError, read_receipt  # noqa: E402
  # noqa: E402
from storage import Storage  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)          # иначе в логах будет URL с токеном
log = logging.getLogger("finediet")

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
NUTRIENTS_PATH = os.environ.get("NUTRIENTS_PATH", "nutrients.csv")
DB_PATH = os.environ.get("DB_PATH", "finediet.db")
MAX_FILE_MB = 15
LOW_CONFIDENCE = 0.30

model = joblib.load(MODEL_PATH)
table = NutrientTable(NUTRIENTS_PATH)
store = Storage(DB_PATH)
clf = Classifier(model, table)
fixed = store.reclassify(clf.predict)
if fixed:
    log.info("категории уточнены у %d сохранённых покупок", fixed)
covered = sum(str(c) in table.rows for c in model.classes_)
log.info("модель: %d категорий; химсостав есть для %d; агент: %s", len(model.classes_), covered,
         f"{agent.LLM_MODEL} @ {agent.LLM_BASE_URL} (быстрый путь: {'да' if agent.FAST_PATH else 'нет'}, "
         f"рецепты моделью: {'да' if agent.RECIPES_BY_LLM else 'нет'})" if agent.enabled() else "выключен (нет LLM_BASE_URL)")

PROFILE_STEPS = [
    ("sex", "Пол?", ["м", "ж"]),
    ("age", "Возраст (лет)?", (14, 100)),
    ("height", "Рост (см)?", (120, 230)),
    ("weight", "Вес (кг)?", (35, 250)),
    ("activity", "Активность?", ["низкая", "умеренная", "средняя", "высокая"]),
    ("goal", "Цель?", ["поддерживать вес", "снизить вес", "набрать массу"]),
    ("exclude", "Что не ешь? Напиши через запятую (например: свинина, молоко) или «-», если всё ешь.", None),
]


def menu() -> KB:
    rows = [
        [B("🧾 Загрузить чек", callback_data="m:upload"), B("📊 Мой рацион", callback_data="m:report")],
        [B("🛒 План покупок на неделю", callback_data="m:basket"), B("👤 Профиль", callback_data="m:profile")],
        [B("🍳 Меню на день", callback_data="m:meal"), B("🔎 Определить продукт", callback_data="m:classify")],
    ]
    if agent.enabled():
        rows.append([B("💬 Спросить нутрициолога", callback_data="m:ask")])
    rows.append([B("🗑 Удалить мои данные", callback_data="m:forget")])
    return KB(rows)


def after_receipt_kb() -> KB:
    return KB([[B("Что улучшить?", callback_data="m:improve"), B("🛒 План покупок на неделю", callback_data="m:basket")],
               [B("📊 Все покупки", callback_data="m:report"), B("Меню", callback_data="m:menu")]])


def name_of(update: Update) -> str:
    u = update.effective_user
    return u.first_name or (f"@{u.username}" if u.username else "друг")


async def send_chart(update: Update, png: bytes | None, caption: str = "") -> None:
    """Картинка с диаграммой; если графиков нет или matplotlib не установлен — тихо пропускаем."""
    if png:
        await update.effective_message.reply_photo(photo=png, caption=caption or None)


def safe_chart(fn, *args):
    if charts is None:
        return None
    try:
        return fn(*args)
    except Exception as e:                      # график — украшение, отчёт важнее
        log.warning("диаграмма не построена: %s", e)
        return None


async def say(update: Update, text: str, kb: KB | None = None):
    msg = update.effective_message
    for i in range(0, len(text), 4000):                       # лимит Telegram — 4096 символов
        last = i + 4000 >= len(text)
        await msg.reply_text(text[i:i + 4000], reply_markup=kb if last else None)


async def say_card(update: Update, title: str, text: str, kb: KB | None = None, kicker: str = "") -> None:
    """Содержательный ответ — карточкой в стиле постеров (бумага, те же шрифты). Без matplotlib — обычным текстом.
    Длинный текст режем на несколько карточек по ~70 строк (у Telegram есть предел пропорций фото)."""
    if charts is None:
        await say(update, text, kb)
        return
    lines, chunks, cur = text.strip("\n").split("\n"), [], []
    for ln in lines:
        cur.append(ln)
        if len(cur) >= 70 and not ln.strip():
            chunks.append("\n".join(cur))
            cur = []
    if cur:
        chunks.append("\n".join(cur))
    for k, chunk in enumerate(chunks):
        png = safe_chart(charts.text_card, title if k == 0 else f"{title} · {k + 1}", chunk, kicker)
        last = k == len(chunks) - 1
        if png is None:
            await say(update, chunk, kb if last else None)
        else:
            await update.effective_message.reply_photo(photo=png, reply_markup=kb if last else None)


def no_table_note() -> str:
    return "" if table else ("\n\n⚠️ Таблица химсостава не подключена (nutrients.csv), "
                             "поэтому считаю только траты.")


# ---------- команды ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()
    p = store.profile(update.effective_user.id)
    text = (f"Привет, {name_of(update)}! Я FineDiet 🥦\n\n"
            "Пришли чек — PDF из письма, фото бумажного чека или JSON из «Проверки чеков», и я:\n"
            "• разложу покупки по 1 053 категориям продуктов;\n"
            "• посчитаю калории, БЖУ и клетчатку и на сколько дней их хватит;\n"
            "• покажу, куда уходят деньги и где белок дешевле;\n"
            "• соберу самую дешёвую корзину на неделю из того, что ты обычно покупаешь.")
    if agent.enabled():
        text += "\n\nА ещё можно просто спросить: «чего мне не хватает?» или «собери корзину на 3000 ₽ без молочки»."
    if not p.get("weight"):
        text += "\n\nСоветую начать с профиля: по нему считаются твои нормы."
    text += "\n\nЯ не врач, нормы — ориентир."
    card = safe_chart(charts.welcome_card, name_of(update)) if charts else None
    if card:
        await update.message.reply_photo(photo=card)
        short = "Пришли чек — PDF, фото или JSON. Или выбери, с чего начать:"
        if not p.get("weight"):
            short = "Советую начать с профиля — по нему считаются твои нормы. Или сразу пришли чек:"
        await update.message.reply_text(short, reply_markup=menu())
    else:
        await update.message.reply_text(text, reply_markup=menu())


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("mode", None)
    await say(update, "Что делаем?", menu())


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE, full: bool = False) -> None:
    uid = update.effective_user.id
    p = store.profile(uid)
    r = analyze(store.items(uid), table, daily_targets(p), p.get("exclude", []))
    if full:
        await say_card(update, "Подробный разбор", format_report(r) + no_table_note(), menu(), "ВСЕ ПОКУПКИ ЗА 90 ДНЕЙ")
        return
    slides = safe_chart(charts.report_charts, r) or []
    poster = bool(slides)
    if len(slides) > 1:              # два слайда одним альбомом — листаются свайпом
        await update.effective_message.reply_media_group([InputMediaPhoto(png) for png in slides])
    elif slides:
        await send_chart(update, slides[0])
    kb = KB([[B("📄 Подробнее", callback_data="m:report_full"), B("🛒 План покупок", callback_data="m:basket")],
             [B("Меню", callback_data="m:menu")]])
    await say_card(update, "Главное", (format_summary(r) if poster else format_report(r)) + no_table_note(), kb,
                   "МОЙ РАЦИОН")


async def cmd_meal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Меню на день из купленных продуктов: сразу продукты и КБЖУ, потом рецепты (нутрициолог или шаблон)."""
    import asyncio
    import menu as daymenu
    uid = update.effective_user.id
    p = store.profile(uid)
    seed = context.user_data.get("meal_seed", 0) + 1
    context.user_data["meal_seed"] = seed
    pool = build_pool(store.items(uid), table, p.get("exclude", []))
    m = daymenu.build_day(pool, daily_targets(p), seed=seed)
    kb = KB([[B("🔄 Другое меню", callback_data="m:meal"), B("Меню", callback_data="m:menu")]])
    if not m.get("ok"):
        await say_card(update, "Меню на день", daymenu.format_menu(m), kb, "МЕНЮ ИЗ ТВОИХ ПОКУПОК")
        return
    await say_card(update, "Меню на день", daymenu.format_menu(m).split("\n", 1)[1].replace(
        " Рецепты пришлю следующим сообщением 👩‍🍳", ""), None, "МЕНЮ ИЗ ТВОИХ ПОКУПОК")
    recipes = None
    # рецепты от маленькой локальной модели ненадёжны («отвари авокадо»): с Ollama — шаблоны,
    # с облачной моделью — пишет модель (переопределить: RECIPES_LLM=1/0 в .env)
    if agent.enabled() and agent.RECIPES_BY_LLM:
        wait = await update.effective_message.reply_text("👩‍🍳 Придумываю рецепты…")

        async def typing():
            while True:
                try:
                    await update.effective_chat.send_action(ChatAction.TYPING)
                except Exception:
                    pass
                await asyncio.sleep(4)
        keep = asyncio.create_task(typing())
        try:
            recipes = await agent.write_recipes(daymenu.recipe_prompt(m))
        finally:
            keep.cancel()
            try:
                await wait.delete()
            except Exception:
                pass
    steps = daymenu.parse_steps(recipes, m) if recipes else {}
    if recipes:
        log.info("рецепты от модели разобраны для: %s", ", ".join(steps) or "ничего — шаблон")
    await say_card(update, "Как приготовить", daymenu.render_recipes(m, steps), kb, "РЕЦЕПТЫ")


def basket_kb(mode: str) -> KB:
    other = [B("💸 Самая дешёвая", callback_data="m:basket_cheap")] if mode == "привычная" \
        else [B("🏠 Ближе к привычному", callback_data="m:basket")]
    return KB([other, [B("Меню", callback_data="m:menu")]])


async def cmd_basket(update: Update, context: ContextTypes.DEFAULT_TYPE, mode: str = "привычная") -> None:
    """/basket [дней] [бюджет]"""
    uid = update.effective_user.id
    args = context.args or []
    try:
        days = int(args[0]) if args else 7
        budget = float(args[1]) if len(args) > 1 else None
    except ValueError:
        await say(update, "Формат: /basket 7 5000 — дней и (необязательно) бюджет в рублях.")
        return
    days = min(max(days, 1), 14)
    p = store.profile(uid)
    pool = build_pool(store.items(uid), table, p.get("exclude", []))
    res = optimize(pool, daily_targets(p), days, budget, mode=mode)
    await say_card(update, "План покупок", format_basket(res) + no_table_note(), kicker="ИЗ ТВОИХ ПОКУПОК")
    if res.get("ok"):
        await send_chart(update, safe_chart(charts.basket_chart, res))
    await say(update, "Другой вариант корзины:", basket_kb(mode))


async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["mode"] = "profile"
    context.user_data["step"] = 0
    context.user_data["draft"] = {}
    await ask_profile_step(update, context)


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store.forget(update.effective_user.id)
    context.user_data.clear()
    await say(update, "Готово: профиль и все покупки удалены.", menu())


# ---------- профиль: пошаговый опрос ----------
async def ask_profile_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    step = context.user_data["step"]
    if step >= len(PROFILE_STEPS):
        uid = update.effective_user.id
        p = store.profile(uid)
        p.update(context.user_data.pop("draft"))
        store.save_profile(uid, p)
        context.user_data.pop("mode", None)
        t = daily_targets(p)
        card = safe_chart(charts.profile_card, p, {k: v for k, v in t.items() if k not in MICROS},
                          {LABELS[k]: t[k] for k in MICROS}) if charts else None
        if card:
            await send_chart(update, card)
            await say(update, "Профиль сохранён — это твои ориентиры на день. Теперь пришли чек.", menu())
        else:
            excl = ", ".join(p.get("exclude") or []) or "нет"
            await say(update, f"Профиль сохранён. Твои ориентиры на день:\n"
                              f"{t['kcal']} ккал · белок {t['protein']} г · жиры {t['fat']} г · "
                              f"углеводы {t['carbs']} г · клетчатка {t['fiber']:.0f} г\nИсключения: {excl}\n\n"
                              "Посчитано по формуле Миффлина–Сан Жеора; это ориентир, не назначение врача.", menu())
        return
    field, question, opts = PROFILE_STEPS[step]
    kb = KB([[B(o, callback_data=f"p:{o}") for o in opts]]) if isinstance(opts, list) else None
    await say(update, f"{step + 1}/{len(PROFILE_STEPS)}. {question}", kb)


async def profile_answer(update: Update, context: ContextTypes.DEFAULT_TYPE, value: str) -> None:
    field, _, opts = PROFILE_STEPS[context.user_data["step"]]
    if isinstance(opts, list):
        if value not in opts:
            await say(update, "Выбери вариант кнопкой 👆")
            return
    elif isinstance(opts, tuple):
        try:
            v = float(value.replace(",", "."))
        except ValueError:
            await say(update, "Нужно число, например 165.")
            return
        if not opts[0] <= v <= opts[1]:
            await say(update, f"Похоже на опечатку: жду число от {opts[0]} до {opts[1]}.")
            return
        value = v
    else:
        value = [] if value.strip() in ("-", "—", "нет") else [w.strip().lower() for w in value.split(",") if w.strip()]
        # «молоко» → «молок»: ловит и «молочный», и «молоко»
        value = [w[:-1] if len(w) > 4 and w[-1] in "оаыяий" else w for w in value]
    context.user_data["draft"][field] = value
    context.user_data["step"] += 1
    await ask_profile_step(update, context)


# ---------- кнопки ----------
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    data = q.data
    if data.startswith("p:"):
        if context.user_data.get("mode") == "profile":
            await profile_answer(update, context, data[2:])
        return
    action = data[2:]
    context.user_data.pop("mode", None)
    if action == "menu":
        await say(update, "Что делаем?", menu())
    elif action == "upload":
        await say(update, "Пришли чек в любом виде:\n"
                          "• PDF или ссылку-файл из письма магазина / ОФД;\n"
                          "• фото бумажного чека (ровно, при хорошем свете);\n"
                          "• JSON из приложения «Проверка чеков» ФНС;\n"
                          "• Word, Excel или текст.\n"
                          "Можно присылать несколько чеков подряд.")
    elif action == "report":
        await cmd_report(update, context)
    elif action == "report_full":
        await cmd_report(update, context, full=True)
    elif action in ("basket", "basket_cheap"):
        context.args = []
        await cmd_basket(update, context, "дешёвая" if action == "basket_cheap" else "привычная")
    elif action == "profile":
        await cmd_profile(update, context)
    elif action == "classify":
        context.user_data["mode"] = "classify"
        await say(update, "Напиши название товара как в чеке, например «молоко 2,5% 930мл».")
    elif action == "ask":
        context.user_data["mode"] = "ask"
        await say_card(update, "Привет! Я нутрициолог",
                       "Смотрю на твои чеки и отвечаю по цифрам — без общих советов.\n\n"
                       "Спроси своими словами или нажми на вопрос ниже.",
                       KB([[B(q, callback_data=f"q:{i}")] for i, q in enumerate(SAMPLE_QUESTIONS)]
                          + [[B("Меню", callback_data="m:menu")]]), "НУТРИЦИОЛОГ")
    elif action == "meal":
        await cmd_meal(update, context)
    elif action == "improve":
        await improve(update, context)
    elif action == "forget":
        await say(update, "Удалить профиль и все покупки без возможности восстановить?",
                  KB([[B("Да, удалить", callback_data="m:forget_yes"), B("Отмена", callback_data="m:menu")]]))
    elif action == "forget_yes":
        await cmd_forget(update, context)


async def improve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if agent.enabled():
        await ask_agent(update, context, "Что улучшить в моём последнем чеке? Дай 2–3 конкретных совета "
                                         "с заменами из того, что я покупаю.")
        return
    uid = update.effective_user.id
    p = store.profile(uid)
    r = analyze(store.last_receipt(uid), table, daily_targets(p), p.get("exclude", []))
    text = format_report(r)
    pricey = r["protein_rub_per_100g"]["most_expensive"]
    if pricey:
        s = swaps(pricey[0]["category"], build_pool(store.items(uid), table, store.profile(uid).get("exclude", [])))
        if s.get("ok") and s["alternatives"]:
            text += f"\n\nБелок из «{s['product']}» стоит {s['rub_per_100g_protein']} ₽ за 100 г. Дешевле:\n"
            text += "\n".join(f"  • {a['example']} — {a['rub_per_100g_protein']} ₽" for a in s["alternatives"])
    await say_card(update, "Что улучшить", text + no_table_note(), after_receipt_kb(), "ПОСЛЕДНИЙ ЧЕК")


# ---------- чек: JSON, PDF, фото, Word, Excel, текст ----------
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if doc.file_size and doc.file_size > MAX_FILE_MB * 1024 * 1024:
        await say(update, f"Файл больше {MAX_FILE_MB} МБ, такой чек я не обработаю.")
        return
    is_image = (doc.mime_type or "").startswith("image/")
    await handle_receipt(update, context, await doc.get_file(), doc.file_name or "", is_image)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo = update.message.photo[-1]                          # самое большое разрешение
    await handle_receipt(update, context, await photo.get_file(), "photo.jpg", True)


async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE, tg_file, filename: str,
                         is_image: bool) -> None:
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        data = bytes(await tg_file.download_as_bytearray())
        items, meta = await read_receipt(data, filename, is_photo=is_image)
        if not items:
            raise ReadError("в файле нет позиций чека")
    except ReadError as e:
        await say(update, f"Не получилось прочитать чек: {e}")
        return
    except Exception as e:
        log.exception("чек не прочитан")
        await say(update, f"Не получилось прочитать чек ({type(e).__name__}). Попробуй PDF или JSON.")
        return

    for it, (cat, conf) in zip(items, clf.predict([i.name for i in items])):
        it.category, it.confidence = cat, conf          # чистка названия + правила (корма, «несъедобное»)
    rows = [it.to_dict() for it in items]
    uid = update.effective_user.id
    date = meta.get("date")
    if not store.add_items(uid, rows, date):
        await say(update, "Этот чек уже загружен, второй раз не считаю. Вот он ещё раз:")

    total = sum(it.total for it in items)
    lines = [f"Чек{' от ' + str(date)[:10] if date else ''}: {len(items)} позиций на {total:.0f} ₽"]
    if meta.get("total") and abs(meta["total"] - total) > 1:
        lines[0] += f" (в чеке ИТОГО {meta['total']:.0f} ₽ — проверь, всё ли распознано)"
    lines.append(f"Прочитано: {meta.get('source', '?')}\n")
    for it in items[:25]:
        flag = " ❓" if it.confidence < LOW_CONFIDENCE else ""
        w = f", {it.grams:.0f} г" if it.grams else ""
        lines.append(f"• {it.name[:40]} → {pretty(it.category)}{flag} ({it.total:.0f} ₽{w})")
    if len(items) > 25:
        lines.append(f"… и ещё {len(items) - 25}")
    if any(it.confidence < LOW_CONFIDENCE for it in items):
        lines.append("❓ — не уверен в категории, в расчёт рациона не беру")
    p = store.profile(uid)
    report = analyze(rows, table, daily_targets(p), p.get("exclude", []))
    card = safe_chart(charts.receipt_card, rows, meta, table.groups if table else {}) if charts else None
    if card:
        await send_chart(update, card)
    else:
        await say(update, "\n".join(lines))
    await say_card(update, "Чек расшифрован", format_report(report) + no_table_note(), after_receipt_kb(), "ЧЕК")


# ---------- текст ----------
SAMPLE_QUESTIONS = ["🥦 Чего мне не хватает в рационе?", "🐟 Где взять витамин D?",
                    "🛒 План покупок на 5 дней за 3000 ₽", "🔄 Чем заменить колбасу?"]


async def on_sample_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка с примером вопроса — как будто человек сам его написал."""
    q = update.callback_query
    await q.answer()
    try:
        text = SAMPLE_QUESTIONS[int(q.data.split(":")[1])]
    except (ValueError, IndexError):
        return
    context.user_data["mode"] = "ask"
    await q.message.reply_text(text)
    await ask_agent(update, context, text.split(" ", 1)[1])


async def ask_agent(update: Update, context: ContextTypes.DEFAULT_TYPE, question: str) -> None:
    import asyncio
    tools = agent.Tools(clf, table, store, update.effective_user.id)
    history = context.user_data.setdefault("history", [])
    wait = await update.effective_message.reply_text("🤔 Думаю…")

    async def typing():                      # «печатает…» гаснет через 5 с — обновляем, пока модель думает
        while True:
            try:
                await update.effective_chat.send_action(ChatAction.TYPING)
            except Exception:
                pass
            await asyncio.sleep(4)
    keep = asyncio.create_task(typing())
    try:
        answer, used = await agent.run(question, tools, history)
    except Exception as e:
        keep.cancel()
        log.exception("агент упал")
        why = ("модель не запущена — открой Ollama (или выполни «ollama serve»)" if "Connect" in type(e).__name__
               else "модель отвечает слишком долго" if "Timeout" in type(e).__name__
               else f"модель {agent.LLM_MODEL} не найдена — выполни «ollama pull {agent.LLM_MODEL}»"
               if "404" in str(e) else type(e).__name__)
        await say(update, f"Нутрициолог сейчас недоступен: {why}. Кнопки меню работают и без него.", menu())
        return
    finally:
        keep.cancel()
    try:
        await wait.delete()
    except Exception:
        pass
    log.info("агент: инструменты %s", used)
    await say_card(update, "Нутрициолог", answer, KB([[B("Ещё вопрос", callback_data="m:ask"),
                                                          B("Меню", callback_data="m:menu")]]), "ОТВЕТ")


async def classify_text(update: Update, text: str) -> None:
    best, conf = clf.predict([text])[0]
    alts = [(c, p) for c, p in clf.top(text) if c != best][:2]
    n = table.get(best) if table else None
    kb = KB([[B("Ещё продукт", callback_data="m:classify"), B("Меню", callback_data="m:menu")]])
    card = safe_chart(charts.product_card, text, best, conf, n, alts) if charts else None
    if card:
        await send_chart(update, card)
        await say(update, "Напиши ещё название или вернись в меню.", kb)
        return
    note = "  (не уверен)" if conf < LOW_CONFIDENCE else ""
    out = [f"• {text}\n   → {pretty(best)} — {conf:.0%}{note}"]
    out += [f"   также возможно: {pretty(c)} — {p:.0%}" for c, p in alts]
    if n:
        out.append("\nНа 100 г: " + f"{n['kcal']:.0f} ккал, белок {n.get('protein', 0):.1f} г, "
                                     f"жиры {n.get('fat', 0):.1f} г, углеводы {n.get('carbs', 0):.1f} г")
    await say(update, "\n".join(out), kb)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text.strip()
    mode = context.user_data.get("mode")
    if mode == "profile":
        await profile_answer(update, context, text)
    elif mode == "classify" or not agent.enabled():
        await classify_text(update, text)
    else:
        await ask_agent(update, context, text)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сетевые сбои Telegram (TimedOut, NetworkError) — в лог одной строкой, без простыни трейсбэка."""
    from telegram.error import NetworkError
    if isinstance(context.error, NetworkError):
        log.warning("Telegram: %s — повторится при следующем сообщении", context.error)
    else:
        log.error("ошибка обработки", exc_info=context.error)


async def warm_up(app) -> None:
    """Загружаем модель в память заранее: первый ответ агента не будет ждать загрузку 5 ГБ с диска."""
    if agent.enabled():
        import asyncio
        asyncio.get_running_loop().create_task(agent.warm_up())


def main() -> None:
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise SystemExit("Задай токен бота в переменной окружения BOT_TOKEN (см. README).")
    # пока модель грузит процессор, Telegram может отвечать медленно — даём запросам больше времени
    app = (ApplicationBuilder().token(token).connect_timeout(30).read_timeout(60).write_timeout(60)
           .pool_timeout(30).get_updates_read_timeout(60).post_init(warm_up).build())
    app.add_error_handler(on_error)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CommandHandler("profile", cmd_profile))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("basket", cmd_basket))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^[mp]:"))
    app.add_handler(CallbackQueryHandler(on_sample_question, pattern=r"^q:"))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("бот запущен, Ctrl+C для остановки")
    app.run_polling()


if __name__ == "__main__":
    main()
