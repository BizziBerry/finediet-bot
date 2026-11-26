"""Агент-нутрициолог: LLM сама решает, какие инструменты вызвать, а все числа считают инструменты.

LLM подключается через любой OpenAI-совместимый API:
  - Ollama локально (бесплатно, данные не уходят из компьютера): LLM_BASE_URL=http://localhost:11434/v1, LLM_MODEL=qwen2.5:7b
  - облачный провайдер с OpenAI-совместимым API: свой LLM_BASE_URL, LLM_MODEL и LLM_API_KEY
Если LLM_BASE_URL не задан, бот работает без агента (кнопки и отчёты всё равно работают).
"""
import json
import logging
import os

from nutrition import LABELS, MICROS, SPARSE, analyze, daily_targets, pretty, where_to_get
from optimizer import build_pool, format_basket, optimize, swaps

log = logging.getLogger("finediet.agent")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
# доп. заголовки, если провайдер просит (например anthropic-workspace-id): LLM_HEADERS=имя:значение;имя2:значение2
EXTRA_HEADERS = dict(kv.split(":", 1) for kv in os.environ.get("LLM_HEADERS", "").split(";") if ":" in kv)
LOCAL = any(h in LLM_BASE_URL for h in ("localhost", "127.0.0.1", "0.0.0.0"))   # Ollama на своём компьютере


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "auto").strip().lower()
    return default if v in ("", "auto") else v in ("1", "true", "yes", "да", "on")


# слабая локальная модель → быстрый путь (инструмент выбирают правила) и рецепты по шаблону;
# облачная модель → полный агент сам выбирает инструменты и сам пишет рецепты. Можно переопределить в .env.
FAST_PATH = _flag("LLM_FAST_PATH", LOCAL)
RECIPES_BY_LLM = _flag("RECIPES_LLM", not LOCAL)
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1200"))   # облачные API любят явный предел длины ответа
MAX_STEPS = 6            # предохранитель от бесконечного цикла вызовов
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "300"))   # секунд на один ответ модели (на процессоре бывает долго)
RESULT_CHARS = 3500      # сколько знаков результата инструмента отдаём модели: меньше текста — быстрее ответ
FAST_TIMEOUT = float(os.environ.get("LLM_FAST_TIMEOUT", "90"))  # быстрый путь: не дождались — отвечаем шаблоном
FAST_MAX_TOKENS = 350
NO_LLM = {"build_basket"}   # на эти вопросы отвечаем готовым текстом сразу, без модели
HISTORY_TURNS = 6        # сколько последних реплик диалога помнить

SYSTEM = """Ты — FineDiet, помощник по питанию в Telegram. Отвечай только по-русски, на «ты», коротко и по делу (до 900 знаков), без markdown-таблиц.
Правила:
1. Любые числа (калории, БЖУ, цены, граммы, нормы) бери ТОЛЬКО из результатов инструментов. Не придумывай и не считай в уме.
2. Вопросы о покупках, рационе, корзине, заменах — сначала вызови нужный инструмент.
3. Если данных нет (нет чеков, нет химсостава) — так и скажи и предложи загрузить JSON-чек.
4. Ты не врач. При болезнях, беременности, кормлении грудью, питании детей, расстройствах пищевого поведения, лекарствах — дай общий ответ и посоветуй обратиться к врачу. Не советуй голодание и жёсткие дефициты.
5. Нормы — ориентир (КБЖУ по формуле Миффлина–Сан Жеора, витамины и минералы по МР 2.3.1.0253-21), а не медицинское назначение. Не советуй БАДы и витаминные препараты — только продукты; про добавки — к врачу.
6. Если пользователь сообщает рост, вес, возраст, пол, активность, цель или что исключить — сохрани через update_profile.
7. Не выдумывай продукты, которых нет в инструментах.
8. Витаминов и минералов 17. Цинк, селен, B6, фолаты, B12 и витамин D известны не для всех продуктов: если инструмент
   пишет, что данных мало (sparse_coverage < 0.6), — честно скажи, что оценить нельзя. Витамин D в основном
   вырабатывается на солнце: из еды норму набрать трудно, уровень проверяют анализом крови (это к врачу).
9. «Где взять» нутриент — вызывай nutrient_sources; называй продукты оттуда, отмечай те, что человек ещё не покупает.
10. Химсостав — USDA FoodData Central и справочник Тутельяна; это средние значения по категории, не по конкретной марке."""

# короткая инструкция для быстрого пути: модель только пересказывает готовые данные
FAST_SYSTEM = ("Ты — FineDiet, дружелюбный помощник по питанию. Ответь только по-русски (без английских слов), "
               "обращайся на «ты», 3–5 предложений, тепло и по делу, без таблиц. Используй только числа и продукты из данных. Не советуй БАДы; ты не врач.")

TOOLS = [
    {"type": "function", "function": {
        "name": "get_profile",
        "description": "Профиль и суточные нормы.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "update_profile",
        "description": "Сохранить то, что пользователь сообщил о себе.",
        "parameters": {"type": "object", "properties": {
            "sex": {"type": "string", "enum": ["м", "ж"]},
            "age": {"type": "number"}, "height": {"type": "number", "description": "см"},
            "weight": {"type": "number", "description": "кг"},
            "activity": {"type": "string", "enum": ["низкая", "умеренная", "средняя", "высокая"]},
            "goal": {"type": "string", "enum": ["поддерживать вес", "снизить вес", "набрать массу"]},
            "exclude": {"type": "array", "items": {"type": "string"},
                        "description": "что не ест, напр. «молок», «свинин»"}}}}},
    {"type": "function", "function": {
        "name": "analyze_purchases",
        "description": "Анализ покупок: траты, БЖУ, на сколько дней хватит, каких витаминов мало, куда уходят деньги.",
        "parameters": {"type": "object", "properties": {
            "scope": {"type": "string", "enum": ["last_receipt", "all"],
                      "description": "последний чек или все за 90 дней"}}}}},
    {"type": "function", "function": {
        "name": "build_basket",
        "description": "План покупок на N дней из привычных продуктов по нормам КБЖУ и витаминов.",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "minimum": 1, "maximum": 14},
            "budget_rub": {"type": "number", "description": "бюджет на весь период"},
            "exclude": {"type": "array", "items": {"type": "string"}},
            "protein_g": {"type": "number", "description": "белок в день, если назван"},
            "mode": {"type": "string", "enum": ["привычная", "дешёвая"]}},
            "required": ["days"]}}},
    {"type": "function", "function": {
        "name": "find_swaps",
        "description": "Чем заменить продукт подешевле.",
        "parameters": {"type": "object", "properties": {"product": {"type": "string"}}, "required": ["product"]}}},
    {"type": "function", "function": {
        "name": "nutrient_sources",
        "description": "Продукты-источники витамина или минерала (% нормы в 100 г; new_for_you — ещё не покупает).",
        "parameters": {"type": "object", "properties": {
            "nutrient": {"type": "string", "description": "железо, кальций, витамин D…"}},
            "required": ["nutrient"]}}},
    {"type": "function", "function": {
        "name": "classify_products",
        "description": "Категория и химсостав на 100 г по названиям товаров.",
        "parameters": {"type": "object", "properties": {
            "names": {"type": "array", "items": {"type": "string"}}}, "required": ["names"]}}},
]


ALIASES = {"фолиев": "folate", "ниацин": "vit_pp", "никотин": "vit_pp", "ретинол": "vit_a", "тиамин": "vit_b1",
           "рибофлав": "vit_b2", "аскорбин": "vit_c", "токофер": "vit_e", "кобалам": "vit_b12", "пиридокс": "vit_b6",
           "кальциф": "vit_d", "холекальц": "vit_d"}


def find_micro(name: str) -> str | None:
    """«витамин д», «D», «b12», «железа», «фолиевая кислота» → ключ MICROS."""
    n = str(name).lower().replace("витамин", "").replace("vitamin", "").strip(" .,")
    for a, k in ALIASES.items():
        if n.startswith(a):
            return k
    lat = n.translate(str.maketrans("авсдеб", "abcdeb"))         # кириллица, похожая на латиницу: «с», «д», «в»
    for k, v in MICROS.items():
        label = v[1].split(",")[0].lower()                       # «витамин d», «железо»
        short = label.replace("витамин", "").strip()
        if n and (lat == short or n == label):
            return k
    if len(n) >= 3:
        for k, v in MICROS.items():
            label = v[1].split(",")[0].lower()
            if "витамин" not in label and (label.startswith(n[:4]) or v[2].lower().startswith(n[:4])):
                return k
    return None


def compact_report(r: dict) -> dict:
    """Отчёт для LLM: без длинных списков (иначе важное обрезается по лимиту контекста)."""
    out = {k: v for k, v in r.items() if k not in ("products", "unknown_weight", "unknown_category")}
    prods = r.get("products") or []
    out["top_products_by_kcal"] = [{"name": p["name"], "kcal": p["kcal"]} for p in prods[:6]]
    for k in ("daily_targets", "micro_days_covered", "weight_estimated_share_of_spend", "sparse_coverage"):
        out.pop(k, None)
    out["micro_lowest"] = [{**m, "where_to_get": [w["product"] for w in (m.get("where_to_get") or [])][:3]}
                           for m in (r.get("micro_lowest") or [])]
    out["products_count"] = len(prods)
    cov = r.get("sparse_coverage") or {}
    out["not_enough_data_for"] = [LABELS[m].split(",")[0] for m, v in cov.items() if v < 0.6]
    return out


class Tools:
    """Инструменты агента; у каждого вызова свой пользователь."""

    def __init__(self, model, table, store, user_id: int):
        self.model, self.table, self.store, self.uid = model, table, store, user_id

    def _targets(self, protein_g=None):
        t = daily_targets(self.store.profile(self.uid))
        if protein_g:
            t["protein"] = float(protein_g)
        return t

    def get_profile(self):
        p = self.store.profile(self.uid)
        return {"profile": p or "не заполнен", "daily_targets": daily_targets(p),
                "note": "нормы по умолчанию для взрослого" if not p.get("weight") else "по формуле Миффлина–Сан Жеора"}

    def update_profile(self, **kw):
        p = self.store.profile(self.uid)
        p.update({k: v for k, v in kw.items() if v not in (None, "", [])})
        self.store.save_profile(self.uid, p)
        return {"saved": p, "daily_targets": daily_targets(p)}

    def analyze_purchases(self, scope="all"):
        items = self.store.last_receipt(self.uid) if scope == "last_receipt" else self.store.items(self.uid)
        if not items:
            return {"error": "покупок нет: пользователь ещё не загружал чеки"}
        if not self.table:
            return {"error": "таблица химсостава не подключена: можно анализировать только траты"}
        r = analyze(items, self.table, self._targets(), self.store.profile(self.uid).get("exclude", []))
        self.last_full_report = r                  # для ответа без нейросети (format_summary)
        return compact_report(r)

    def nutrient_sources(self, nutrient):
        key = find_micro(nutrient)
        if not key:
            return {"error": f"не знаю нутриент «{nutrient}»", "known": [LABELS[m].split(",")[0] for m in MICROS]}
        if not self.table:
            return {"error": "таблица химсостава не подключена"}
        t = self._targets()
        bought = {i["category"] for i in self.store.items(self.uid)}
        src = where_to_get(self.table, key, t[key], bought, self.store.profile(self.uid).get("exclude", []), k=6)
        return {"nutrient": LABELS[key], "daily_norm": t[key], "sources": src,
                "note": "значение известно не для всех продуктов (USDA)" if key in SPARSE else ""}

    def build_basket(self, days=7, budget_rub=None, exclude=None, protein_g=None, mode="привычная"):
        excl = list(self.store.profile(self.uid).get("exclude", [])) + list(exclude or [])
        pool = build_pool(self.store.items(self.uid), self.table, excl)
        r = optimize(pool, self._targets(protein_g), int(days), budget_rub, mode=mode)
        if not r.get("ok"):
            return {"ok": False, "reason": r.get("reason")}
        self.last_plan_text = format_basket(r)              # полный текст — для ответа без нейросети
        # модели — готовый короткий текст плана и пара итогов, а не весь расчёт (иначе маленькая модель «думает» минутами)
        return {"ok": True, "plan_text": self.last_plan_text[:1800], "total_rub": r.get("total_rub"),
                "usual_rub_for_same_days": r.get("habit_rub"), "excluded": excl,
                "hint": "перескажи план коротко: итог, 3–5 главных продуктов, что изменилось по сравнению с привычным"}

    def find_swaps(self, product):
        excl = self.store.profile(self.uid).get("exclude", [])
        return swaps(product, build_pool(self.store.items(self.uid), self.table, excl))

    def classify_products(self, names):
        names = [str(n) for n in names][:20]
        out = []
        for n, (cat, conf) in zip(names, self.model.predict(names)):     # self.model — Classifier
            out.append({"name": n, "category": pretty(cat), "confidence": round(conf, 2),
                        "per_100g": self.table.get(cat) if self.table else None})
        return out

    def call(self, name, args):
        if name not in {t["function"]["name"] for t in TOOLS}:
            return {"error": f"нет инструмента {name}"}
        try:
            return getattr(self, name)(**args)
        except TypeError as e:
            return {"error": f"неверные аргументы: {e}"}


def enabled() -> bool:
    return bool(LLM_BASE_URL)


async def _chat(messages, use_tools: bool = True, timeout: float | None = None, max_tokens: int | None = None):
    import httpx    # ставится вместе с python-telegram-bot
    body = {"model": LLM_MODEL, "messages": messages, "temperature": 0.2}
    if use_tools:
        body["tools"] = TOOLS
    body["max_tokens"] = max_tokens or MAX_TOKENS   # ограничение длины ответа = ограничение времени и денег
    async with httpx.AsyncClient(timeout=timeout or LLM_TIMEOUT) as client:
        r = await client.post(f"{LLM_BASE_URL}/chat/completions",
                              headers={"Authorization": f"Bearer {LLM_API_KEY}", **EXTRA_HEADERS}, json=body)
        r.raise_for_status()   # ConnectError → Ollama не запущена (bot.py покажет понятное сообщение)
        return r.json()["choices"][0]["message"]


def fallback(name: str, r: dict, tools=None) -> str:
    """Ответ без LLM: те же данные инструмента, коротким текстом."""
    if not isinstance(r, dict) or r.get("error"):
        return (r or {}).get("error", "Не получилось посчитать.") if isinstance(r, dict) else "Не получилось посчитать."
    if name == "build_basket":
        return getattr(tools, "last_plan_text", None) or r.get("plan_text") or r.get("reason") \
            or "Не получилось собрать план."
    if name == "nutrient_sources":
        lines = [f"Где взять {r['nutrient'].split(',')[0]} (норма {r['daily_norm']:g} {r['nutrient'].split(',')[-1].strip()} в день):"]
        for x in r.get("sources", []):
            lines.append(f"• {x['product']} — {x['per_100g_percent_of_norm']}% нормы в 100 г"
                         + (" ★ ты это пока не покупаешь" if x.get("new_for_you") else ""))
        if not r.get("sources"):
            lines.append("в справочнике мало продуктов с известным значением")
        if r.get("note"):
            lines.append(f"({r['note']})")
        return "\n".join(lines)
    if name == "find_swaps":
        if not r.get("ok"):
            return r.get("reason", "Не нашла, чем заменить.")
        lines = [f"Чем заменить «{r['product']}» ({r.get('criterion', '')}):"]
        for a in r.get("alternatives", []):
            v = a.get("rub_per_100g_protein") or a.get("rub_per_1000_kcal") or a.get("rub_per_100g")
            lines.append(f"• {a['example']}" + (f" — {v:.0f} ₽" if isinstance(v, (int, float)) else ""))
        return "\n".join(lines)
    if name == "analyze_purchases":
        from nutrition import format_summary
        try:
            return format_summary(getattr(tools, "last_full_report", None) or r)
        except Exception:
            low = ", ".join(f"{m['nutrient']} ({m['days']:.0f} дн.)" for m in r.get("micro_lowest", []))
            return f"Потрачено на еду {r.get('spend_rub', 0):.0f} ₽. Меньше всего хватит: {low}."
    return json.dumps(r, ensure_ascii=False, default=str)[:1500]


async def write_recipes(prompt: str) -> str | None:
    """Рецепты к меню: одно обращение к модели без инструментов. None — модель не ответила (будет шаблон)."""
    import time
    t0 = time.monotonic()
    try:
        msg = await _chat([{"role": "system", "content": "Ты — домашний повар FineDiet. Пиши по-русски, коротко и понятно."},
                           {"role": "user", "content": prompt}], use_tools=False, max_tokens=650)
        log.info("рецепты: модель ответила за %.0f с", time.monotonic() - t0)
        return (msg.get("content") or "").strip() or None
    except Exception as e:
        log.warning("рецепты: модель не ответила за %.0f с (%s) — шаблон", time.monotonic() - t0, type(e).__name__)
        return None


async def warm_up() -> None:
    """Только для Ollama: загрузить модель в память заранее. Облачной модели прогрев не нужен (и стоит денег)."""
    if not LOCAL:
        return
    import httpx
    import time
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            r = await client.post(f"{LLM_BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                                  json={"model": LLM_MODEL, "messages": [{"role": "user", "content": "ok"}],
                                        "max_tokens": 1})
            r.raise_for_status()
        log.info("модель %s загружена за %.0f с", LLM_MODEL, time.monotonic() - t0)
    except Exception as e:
        log.warning("модель %s не прогрелась: %s", LLM_MODEL, type(e).__name__)


def route(question: str) -> tuple[str, dict] | None:
    """Быстрый путь для типовых вопросов: инструмент выбираем правилами, модель только формулирует ответ.
    На слабом компьютере это 1 обращение к модели вместо 2–3 и вдвое короче промпт (без описаний инструментов)."""
    import re
    q = question.lower()
    if re.search(r"\bне (?:ем|ешь|пью|употребля)|аллерг|мне \d+\s*(?:лет|год)|\bвешу|\bрост\b|\bмой вес|цель", q):
        return None           # человек сообщает о себе — пусть модель сохранит профиль (update_profile)
    m = re.search(r"(?:где|как|чем|из чего|в ч[её]м|откуда)\b.*?(?:взять|получ|закры|добрать|восполн|есть|много|содерж)?"
                  r".*?\b(витамин\s*[a-zа-яё]\d*|[a-zа-я]\d{1,2}\b|железо|железа|кальци\w*|магни\w*|кали[йя]|фосфор\w*|"
                  r"цинк\w*|селен\w*|фолат\w*|фолиев\w*|ниацин\w*)", q)
    if m and find_micro(m.group(1)):
        return "nutrient_sources", {"nutrient": m.group(1)}
    if re.search(r"план|корзин|список покупок|что купить|закупк", q):
        days = re.search(r"(\d{1,2})\s*(?:дн|день|дня|сут)", q)
        week = re.search(r"недел", q)
        budget = re.search(r"(\d[\d\s]{2,})\s*(?:₽|р\b|руб)", q)
        args = {"days": int(days.group(1)) if days else 7 if week or not days else 7}
        if budget:
            args["budget_rub"] = float(budget.group(1).replace(" ", ""))
        if re.search(r"дешев|дешёв|эконом|подешевле", q):
            args["mode"] = "дешёвая"
        return "build_basket", args
    m = re.search(r"(?:чем|на что)\s+(?:можно\s+)?замен\w*\s+(.+?)[?.!]*$", q)
    if m:
        prod = re.sub(r"\b(подешевле|дешевле|подешев\w*|на что-?нибудь|чем-?нибудь|пожалуйста)\b", "", m.group(1))
        return "find_swaps", {"product": prod.strip(" ,")}
    if re.search(r"не хватает|чего мало|рацион|анализ|сколько (?:я )?потрат|куда (?:ушли|уходят) деньги|"
                 r"как я питаюсь|что с моим питанием|последн\w* чек", q):
        return "analyze_purchases", {"scope": "last_receipt" if re.search(r"последн\w* чек", q) else "all"}
    return None


async def run(question: str, tools: Tools, history: list[dict]) -> tuple[str, list[str]]:
    """Цикл агента. Возвращает ответ и список вызванных инструментов (для логов и подписи)."""
    import time
    routed = route(question) if FAST_PATH else None
    if routed:
        name, args = routed
        result = tools.call(name, args)
        log.info("быстрый путь: %s(%s)", name, json.dumps(args, ensure_ascii=False))
        if name in NO_LLM:                  # готовый список: пересказ моделью только медленнее и рискованнее
            answer = fallback(name, result, tools)
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer[:600]}]
            del history[:-HISTORY_TURNS]
            return answer, [name]
        data = fallback(name, result, tools)[:1200]     # готовый короткий текст вместо JSON: в разы меньше токенов
        messages = [{"role": "system", "content": FAST_SYSTEM}, *history[-2:],
                    {"role": "user", "content": f"{question}\n\nДанные (числа бери только отсюда):\n{data}"}]
        t0 = time.monotonic()
        try:
            msg = await _chat(messages, use_tools=False, timeout=FAST_TIMEOUT, max_tokens=FAST_MAX_TOKENS)
            log.info("шаг 1: модель ответила за %.0f с", time.monotonic() - t0)
            answer = (msg.get("content") or "").strip() or fallback(name, result, tools)
        except Exception as e:                       # модель не успела / не запущена — цифры всё равно покажем
            log.warning("модель не ответила за %.0f с (%s) — отвечаю шаблоном", time.monotonic() - t0,
                        type(e).__name__)
            answer = fallback(name, result, tools)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        del history[:-HISTORY_TURNS]
        return answer, [name]

    messages = [{"role": "system", "content": SYSTEM}, *history[-HISTORY_TURNS:],
                {"role": "user", "content": question}]
    used = []
    for step in range(MAX_STEPS):
        t0 = time.monotonic()
        msg = await _chat(messages)
        log.info("шаг %d: модель ответила за %.0f с", step + 1, time.monotonic() - t0)
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = (msg.get("content") or "").strip() or "Не получилось сформулировать ответ, попробуй переспросить."
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
            del history[:-HISTORY_TURNS]
            return answer, used
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            raw = c["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = {}
            result = tools.call(name, args or {})
            used.append(name)
            log.info("tool %s(%s)", name, raw)
            messages.append({"role": "tool", "tool_call_id": c.get("id", name), "name": name,
                             "content": json.dumps(result, ensure_ascii=False, default=str)[:RESULT_CHARS]})
    return "Запрос получился слишком сложным, попробуй разбить его на части.", used
