"""Меню на день из купленных продуктов: завтрак, обед, ужин.

Продукты, граммы и КБЖУ подбирает код (из того, что человек реально покупает, под его норму калорий).
Рецепты пишет LLM — только из этих продуктов; если модели нет или она не успела — простые рецепты по шаблону.
"""
import random
import re

from nutrition import NOT_ADVISED, pretty
from optimizer import ALCOHOL, SEASONING, TREATS

SKIP_GROUPS = {"Напитки", "Соусы и специи", "Масла и жиры", "Сладости", "Другое"}
NOT_MEAL = re.compile(r"лимон|лайм|имбир|клюкв|облепих|шиповн", re.I)    # приправа, а не еда на тарелке
READY_DISH = re.compile(r"по.корейск|воздушн|экструд|маринов|квашен|консерв|\bпо \d-\d|по-домашн|салат |суп|борщ|щи |солянк|пельмен|вареник|пицц|бургер|шаурм|ролл|суши|"
                        r"котлет|запеканк|омлет|блин|пирож|пирог|сырник|голубц|плов|рагу|паштет", re.I)

# роли: (как узнать продукт, порция по умолчанию, г)
ROLES = {
    "молочное": (lambda p: p["group"] == "Молочное" and re.search(r"творог|йогурт|кефир|ряженк|сыр\b|сырок|творожн",
                                                                   p["category"], re.I), 180),
    "яйца": (lambda p: p["group"] == "Яйца", 110),
    "хлеб": (lambda p: p["group"] == "Хлеб, крупы, макароны" and re.search(r"хлеб|батон|багет|лаваш|чиабат|хлопь|"
                                                                            r"овсян|мюсли|геркулес", p["category"], re.I), 60),
    "гарнир": (lambda p: (p["group"] == "Хлеб, крупы, макароны" and re.search(
        r"рис|греч|макарон|паст|булгур|кускус|киноа|перлов|пшен|лапш|крупа", p["category"], re.I))
        or re.search(r"картоф", p["category"], re.I), 70),
    "мясо": (lambda p: p["group"] in ("Мясо и птица", "Рыба и морепродукты"), 150),
    "белок вечером": (lambda p: p["group"] in ("Рыба и морепродукты", "Мясо и птица", "Яйца")
                      or (p["group"] == "Молочное" and re.search(r"творог", p["category"], re.I)), 150),
    "овощи": (lambda p: p["group"] == "Овощи и грибы" and not re.search(r"картоф", p["category"], re.I), 150),
    "фрукты": (lambda p: p["group"] == "Фрукты и ягоды", 150),
    "перекус": (lambda p: p["group"] in ("Фрукты и ягоды", "Бобовые, орехи, семена")
                or (p["group"] == "Молочное" and re.search(r"йогурт|кефир|творог|ряженк", p["category"], re.I)), 150),
}
MEALS = [("Завтрак", "☀️", 0.25, [["молочное", "яйца"], ["хлеб"], ["фрукты"]]),
         ("Обед", "🍲", 0.40, [["мясо"], ["гарнир"], ["овощи"]]),
         ("Ужин", "🌙", 0.35, [["белок вечером"], ["овощи"], ["овощи", "фрукты"]])]
MAIN = {"молочное", "яйца", "хлеб", "гарнир", "мясо", "белок вечером"}   # их порции подгоняем под калории
MAX_G = {"молочное": 250, "яйца": 165, "хлеб": 100, "гарнир": 100, "мясо": 250, "белок вечером": 220,
         "овощи": 250, "фрукты": 200, "перекус": 250}
EGG = re.compile(r"\bяйц|яйцо", re.I)
DRY = re.compile(r"\bрис\b|греч|макарон|\bпаст[аы]\b|булгур|кускус|киноа|перлов|пшено|пшённ|лапш|крупа|хлопь|геркулес|овсян",
                 re.I)


def short(name: str) -> str:
    """«хлеб зерновой (мука в/с и зерно дроблёное)» → «хлеб зерновой»; «творог полужирный 9,0% жирности» → «творог полужирный 9%»."""
    s = re.sub(r"\s*\([^)]*\)", "", pretty(name))
    s = re.sub(r"(\d+),0%", r"\1%", s)
    s = re.sub(r"\s*жирности|\s+сезонные|\s+1 кат\.?|\s+потрошен\w*|\s+мыт\w+|\s+\d+(?:\.\d+)+\.?$", "", s)
    for a, b in SIMPLE.items():
        s = re.sub(a, b, s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip(" ,.")


SIMPLE = {r"^хлеб пшеничный .*": "хлеб пшеничный", r"^хлеб ржано.*|^хлеб ржаной.*": "хлеб ржаной",
          r"^батон .*": "батон", r"^крупа рисовая": "рис", r"^крупа гречиха ядрица|^крупа гречневая": "гречка", r"^крупа манная": "манка",
          r"^крупа перловая.*": "перловка", r"^крупа просо.*|^крупа пшено.*": "пшено", r"^яйцо целое": "яйца",
          r"^хлопья «геркулес».*": "овсяные хлопья", r"^куриные грудки": "куриная грудка", r"^земляника$": "клубника"}


def _cap(p: dict, role: str) -> int:
    c = p["category"]
    if EGG.search(c):
        return 165                                   # не больше 3 яиц за приём
    if re.search(r"орех|миндал|фундук|кешью|фисташ|арахис|семечк|семена", c, re.I):
        return 40
    if re.search(r"авокад", c, re.I):
        return 100
    if re.search(r"картоф", c, re.I):
        return 300
    if DRY.search(c):
        return 100
    return MAX_G.get(role, 250)


def candidates(pool: list[dict]) -> list[dict]:
    out = []
    for p in pool:
        c = p["category"]
        if (p["group"] in SKIP_GROUPS or TREATS.search(c) or ALCOHOL.search(c) or SEASONING.search(c)
                or NOT_ADVISED.search(c) or READY_DISH.search(c) or NOT_MEAL.search(c) or p.get("kcal", 0) <= 0):
            continue
        out.append(p)
    return out


def _stem(cat: str) -> str:
    w = re.findall(r"[а-яё]{3,}", short(cat).lower())
    return w[0][:5] if w else cat


def _pick(cands, roles, used, rng):
    stems = {_stem(c) for c in used}
    opts = [p for p in cands if p["category"] not in used and _stem(p["category"]) not in stems
            and any(ROLES[r][0](p) for r in roles)]
    if not opts:
        return None, None
    weights = [max(p.get("habit_g", 0), 50) ** 0.5 for p in opts]       # чаще покупаешь — чаще попадает в меню
    p = rng.choices(opts, weights=weights, k=1)[0]
    role = next(r for r in roles if ROLES[r][0](p))
    return p, role


def build_day(pool: list[dict], targets: dict, seed: int = 0) -> dict:
    cands = candidates(pool)
    if len(cands) < 4:
        return {"ok": False, "reason": "мало продуктов в покупках, из которых можно готовить — загрузи ещё чеки"}
    rng = random.Random(seed)
    kcal_day = targets.get("kcal") or 2000
    used, meals = set(), []
    for title, icon, share, slots in MEALS:
        items = []
        for roles in slots:
            p, role = _pick(cands, roles, used, rng)
            if p is None:
                continue
            used.add(p["category"])
            g0 = 200 if re.search(r"картоф", p["category"], re.I) else ROLES[role][1]
            items.append({"p": p, "role": role, "grams": min(g0, _cap(p, role))})
        if not items:
            continue
        # подгоняем порции основных продуктов под долю дневной нормы (с потолками порций)
        goal = kcal_day * share
        fixed = sum(i["p"]["kcal"] * i["grams"] / 100 for i in items if i["role"] not in MAIN)
        main = sum(i["p"]["kcal"] * i["grams"] / 100 for i in items if i["role"] in MAIN)
        f = min(max((goal - fixed) / main, 0.5), 1.8) if main > 0 else 1.0
        for i in items:
            if i["role"] in MAIN:
                i["grams"] = min(max(30, round(i["grams"] * f / 10) * 10), _cap(i["p"], i["role"]))
        tot = {k: sum(i["p"].get(k, 0) * i["grams"] / 100 for i in items) for k in ("kcal", "protein", "fat", "carbs")}
        meals.append({"title": title, "icon": icon, "share": share,
                      "items": [{"name": short(i["p"]["category"]), "example": i["p"].get("example", ""),
                                 "role": i["role"], "grams": i["grams"], "kcal": round(i["p"]["kcal"] * i["grams"] / 100)}
                                for i in items],
                      **{k: round(v) for k, v in tot.items()}})
    # не добрали калорий из-за потолков порций — перекус из фруктов, молочного или орехов
    deficit = kcal_day - sum(m["kcal"] for m in meals)
    if deficit > 150:
        snack = []
        for roles in (["перекус"], ["перекус"]):
            p, role = _pick(cands, roles, used, rng)
            if p is None or deficit <= 80:
                break
            used.add(p["category"])
            g = min(_cap(p, role), max(30, round(deficit / max(p["kcal"], 1) * 100 / 10) * 10))
            snack.append({"name": short(p["category"]), "example": p.get("example", ""), "role": role, "grams": g,
                          "kcal": round(p["kcal"] * g / 100), "p": p})
            deficit -= p["kcal"] * g / 100
        if snack:
            tot = {k: round(sum(x["p"].get(k, 0) * x["grams"] / 100 for x in snack)) for k in ("kcal", "protein", "fat", "carbs")}
            meals.append({"title": "Перекус", "icon": "🍎", "share": 0,
                          "items": [{k: v for k, v in x.items() if k != "p"} for x in snack], **tot})
    total = {k: sum(m[k] for m in meals) for k in ("kcal", "protein", "fat", "carbs")}
    return {"ok": True, "meals": meals, "total": total, "kcal_target": round(kcal_day)}


def _amount(it) -> str:
    if it["role"] == "яйца" or EGG.search(it["name"]):
        return f"{max(1, round(it['grams'] / 55))} шт"
    dry = " (сухой вес)" if DRY.search(it["name"]) else ""
    return f"{it['grams']} г{dry}"


def format_menu(m: dict) -> str:
    if not m.get("ok"):
        return "Не получилось составить меню: " + m.get("reason", "")
    t = m["total"]
    lines = [f"🍽 Меню на день из твоих покупок",
             f"≈ {t['kcal']} ккал из {m['kcal_target']} · белки {t['protein']} г · жиры {t['fat']} г · углеводы {t['carbs']} г"]
    for meal in m["meals"]:
        lines.append(f"\n{meal['icon']} {meal['title']} · {meal['kcal']} ккал · Б {meal['protein']} · Ж {meal['fat']} · "
                     f"У {meal['carbs']}")
        for it in meal["items"]:
            lines.append(f"  • {it['name']} — {_amount(it)}")
    lines.append("\nКБЖУ — по справочникам, это ориентир. Рецепты пришлю следующим сообщением 👩‍🍳")
    return "\n".join(lines)


MAIN_DISH = [(r"солен|солён|копчен|слабосол|малосол", "Слабосолёная рыба на тарелке"),
             (r"лосос|сёмг|семг|форел|горбуш|треск|минтай|хек|скумбр|рыб|кет|нерк|палтус|тунец|судак", "Запечённая рыба"),
             (r"креветк|кальмар|мидии|морепрод", "Морепродукты на сковороде"),
             (r"фарш", "Домашние котлеты"),
             (r"кур|бройлер|цыпл", "Курица на сковороде"), (r"индей", "Индейка на сковороде"),
             (r"говядин|телят|стейк", "Тушёная говядина"), (r"свинин|корейк|шейк|вырезк", "Свинина на сковороде"),
             (r"яйц", "Омлет"), (r"творог", "Творожная тарелка"), (r"йогурт|кефир|ряженк", "Йогурт-боул"),
             (r"сыр", "Бутерброды с сыром")]
MINUTES = [(r"картоф", 40), (r"запеч|рыба", 25), (r"тушён", 45), (r"рис|греч|макарон|паст|булгур|перлов|пшен|крупа", 25),
           (r"котлет|курица|индейка|свинина", 25), (r"омлет", 10)]


def dish(meal: dict) -> tuple[str, int]:
    """Название блюда и время — правилами по составу (одинаково и с моделью, и без неё)."""
    items = meal["items"]
    main, rest = None, []
    for it in items:
        label = next((t for pat, t in MAIN_DISH if re.search(pat, it["name"], re.I)), None)
        if label and main is None:
            main = label
        else:
            rest.append(it["name"])
    title = " · ".join([main or items[0]["name"].capitalize()] + rest[:3])
    text = (title + " " + " ".join(i["name"] for i in items)).lower()
    minutes = max([mins for pat, mins in MINUTES if re.search(pat, text)] or [10])
    return title, minutes


def cooked_meals(m: dict) -> list[dict]:
    return [meal for meal in m["meals"] if meal["title"] != "Перекус"]


def recipe_prompt(m: dict) -> str:
    parts = []
    for meal in cooked_meals(m):
        title, minutes = dish(meal)
        ing = "; ".join(f"{it['name']} {_amount(it)}" for it in meal["items"])
        parts.append(f"### {meal['title']}\nБлюдо: {title} (~{minutes} мин)\nПродукты: {ing}")
    return ("Напиши шаги приготовления для каждого блюда. Правила:\n"
            "- только перечисленные продукты в этих количествах + соль, перец, сухие специи, 1 ложка масла, вода;\n"
            "- нельзя добавлять сахар, лук, чеснок, зелень, соусы и другие продукты;\n"
            "- 3–4 коротких шага, каждый с новой строки и с номером; обращайся на «ты»: нарежь, отвари, подай;\n"
            "- перед шагами каждого блюда повтори его заголовок (### Завтрак и т. д.), больше ничего не пиши.\n\n"
            + "\n\n".join(parts))


def parse_steps(text: str, m: dict) -> dict:
    """Ответ модели → {приём пищи: [шаги]}; что не разобрали — потом заполнит шаблон."""
    out, cur = {}, None
    names = [meal["title"] for meal in cooked_meals(m)]
    for line in (text or "").splitlines():
        st = line.strip().strip("*").strip()
        hit = next((n for n in names if re.match(rf"^[#\s{{(\[]*\W*\s*{n}\b", st, re.I)), None)
        if hit:
            cur = hit
            out.setdefault(cur, [])
            continue
        mm = re.match(r"^(\d+)[.)]\s+(.+)", st)
        if cur and mm:
            step = mm.group(2).strip()
            if not re.search(r"\bсахар|\bлук\b|чеснок|зелен|майонез|кетчуп", step, re.I):
                out[cur].append(step[:1].upper() + step[1:])
    # шаги, где упомянут продукт из ДРУГОГО приёма пищи, — модель перепутала: такое блюдо отдаём шаблону
    stems = {meal["title"]: {w[:4] for it in meal["items"] for w in re.findall(r"[а-яё]{4,}", it["name"].lower())}
             for meal in cooked_meals(m)}
    good = {}
    for k, v in out.items():
        own = stems.get(k, set())
        foreign = set().union(*(st for t, st in stems.items() if t != k)) - own
        text = " ".join(v).lower()
        if len(v) >= 2 and not any(re.search(rf"\b{f}", text) for f in foreign):
            good[k] = v[:4]
    return good


def render_recipes(m: dict, steps: dict | None = None) -> str:
    """Рецепты одним форматом: «☀️ Завтрак: блюдо (~N мин)» + шаги; перекус — без готовки."""
    steps = steps or {}
    lines = []
    for meal in m["meals"]:
        if meal["title"] == "Перекус":
            names = " и ".join(it["name"] for it in meal["items"])
            lines += [f"{meal['icon']} Перекус: {names}", "Без готовки — просто вымой или нарежь.", ""]
            continue
        title, minutes = dish(meal)
        lines.append(f"{meal['icon']} {meal['title']}: {title} (~{minutes} мин)")
        own = steps.get(meal["title"])
        if own:
            lines += [f"{k}. {st}" for k, st in enumerate(own, 1)]
        else:
            for k, it in enumerate(meal["items"], 1):
                how = _how(it)
                lines.append(f"{k}. {how[:1].upper()}{how[1:]}")
        lines.append("")
    return "\n".join(lines).strip()


# ---------- рецепты без LLM ----------
def _how(it) -> str:
    n = it["name"].lower()
    if re.search(r"солен|солён|копчен|слабосол|малосол|вялен", n):
        return f"{it['name']}: нарежь тонкими ломтиками и подай с овощами или на хлебе"
    if re.search(r"кур|бройлер|индей|цыпл|утк", n):
        return f"{it['name']}: нарежь, посоли и поперчи, обжарь на ложке масла по 5–7 мин с каждой стороны до готовности"
    if re.search(r"лосос|сёмг|семг|форел|горбуш|треск|минтай|хек|скумбр|сельд|рыб|кет|нерк|палтус|тунец|судак", n):
        return f"{it['name']}: посоли, поперчи, запеки в фольге при 190 °C 15–20 мин"
    if re.search(r"креветк|кальмар|мидии|морепрод", n):
        return f"{it['name']}: обжарь на сковороде 3–4 мин с чесноком или специями"
    if re.search(r"фарш", n):
        return f"{it['name']}: посоли, сформуй котлетки и обжарь по 5 мин с каждой стороны, затем 5 мин под крышкой"
    if re.search(r"говядин|свинин|баранин|телят|стейк|вырезк|корейк|шейк", n):
        return f"{it['name']}: нарежь кусочками, обжарь на сильном огне 5 мин, затем туши под крышкой 20–30 мин"
    if it["role"] == "яйца" or EGG.search(n):
        return f"Яйца: взбей со щепоткой соли и приготовь омлет на сковороде под крышкой 5–6 мин"
    if re.search(r"картоф", n):
        return f"{it['name']}: нарежь дольками, сбрызни маслом, посоли и запеки при 200 °C 30–35 мин"
    if DRY.search(n) and it["role"] == "гарнир":
        return f"{it['name']}: отвари в подсоленной воде по инструкции на упаковке"
    if it["role"] == "овощи" or re.search(r"авокад", n):
        if re.search(r"огур|томат|помид|перец сладк|перец красн|редис|салат|пекинск|авокад|зелен|руккол", n):
            return f"{it['name']}: нарежь в салат, заправь ложкой масла и лимонным соком"
        return f"{it['name']}: нарежь и потуши 7–10 мин с ложкой масла или запеки при 200 °C 20 мин"
    if it["role"] == "фрукты":
        return f"{it['name']}: нарежь и подай к блюду или на десерт"
    if it["role"] == "хлеб":
        return f"{it['name']}: подсуши на сухой сковороде или в тостере" if not DRY.search(n) else \
            f"{it['name']}: залей горячей водой или молоком на 5 мин"
    if it["role"] in ("молочное", "белок вечером"):
        return f"{it['name']}: выложи в тарелку, можно добавить фрукты или зелень"
    return f"{it['name']}: подай как есть"


def template_recipes(m: dict) -> str:
    return "👩‍🍳 Как приготовить\n" + render_recipes(m) if m.get("ok") else ""
