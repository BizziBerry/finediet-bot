"""Химсостав категорий, суточные нормы и анализ покупок.

Таблица химсостава (nutrients.csv) в репозиторий не кладётся: значения взяты из справочника
Тутельяна (защищён авторским правом). Формат: одна строка на категорию классификатора, значения на 100 г.
"""
import os
import re
from collections import defaultdict

import pandas as pd

# какие названия колонок понимаем (регистр не важен)
COLUMNS = {
    "category": ["категория", "category", "продукт", "название"],
    "kcal": ["ккал", "калорийность", "энергетическая ценность", "kcal", "энергия"],
    "protein": ["белки", "белок", "protein"],
    "fat": ["жиры", "жир", "fat"],
    "carbs": ["углеводы", "carbs", "углеводы общие"],
    "fiber": ["клетчатка", "пищевые волокна", "fiber"],
    "sugar": ["сахар", "моно- и дисахариды", "сахара", "sugar"],
    "sodium": ["натрий", "na", "sodium"],          # мг на 100 г
}
# витамины и минералы: ключ → (колонка в nutrients.csv, подпись, род. падеж, норма м, норма ж)
# нормы для взрослых 18–59 лет: МР 2.3.1.0253-21 «Нормы физиологических потребностей…»
MICROS = {
    "calcium": ("кальций, мг", "кальций, мг", "кальция", 1000, 1000),
    "iron": ("железо, мг", "железо, мг", "железа", 10, 18),
    "magnesium": ("магний, мг", "магний, мг", "магния", 400, 400),
    "potassium": ("калий, мг", "калий, мг", "калия", 2500, 2500),
    "phosphorus": ("фосфор, мг", "фосфор, мг", "фосфора", 800, 800),
    "vit_a": ("витамин a, мкг", "витамин A, мкг", "витамина A", 900, 900),
    "vit_e": ("витамин e, мг", "витамин E, мг", "витамина E", 15, 15),
    "vit_b1": ("витамин b1, мг", "витамин B1, мг", "витамина B1", 1.5, 1.5),
    "vit_b2": ("витамин b2, мг", "витамин B2, мг", "витамина B2", 1.8, 1.8),
    "vit_pp": ("витамин pp, мг", "витамин PP, мг", "витамина PP", 20, 20),
    "vit_c": ("витамин c, мг", "витамин C, мг", "витамина C", 90, 90),
    # есть только в USDA FoodData Central: у части категорий значения нет (это не ноль, а «неизвестно»)
    "zinc": ("цинк, мг", "цинк, мг", "цинка", 12, 12),
    "selenium": ("селен, мкг", "селен, мкг", "селена", 70, 55),
    "vit_b6": ("витамин b6, мг", "витамин B6, мг", "витамина B6", 2.0, 2.0),
    "folate": ("фолаты, мкг", "фолаты, мкг", "фолатов", 400, 400),
    "vit_b12": ("витамин b12, мкг", "витамин B12, мкг", "витамина B12", 3.0, 3.0),
    "vit_d": ("витамин d, мкг", "витамин D, мкг", "витамина D", 15, 15),   # старше 65 лет — 20
}
CORE_MICROS = ["calcium", "iron", "magnesium", "potassium", "phosphorus",
               "vit_a", "vit_e", "vit_b1", "vit_b2", "vit_pp", "vit_c"]   # есть в обоих справочниках
SPARSE = [m for m in MICROS if m not in CORE_MICROS]
MIN_SPARSE_COVERAGE = 0.6   # если значение известно меньше чем для 60% купленного веса — вывод не делаем
for _k, (_col, *_rest) in MICROS.items():
    COLUMNS[_k] = [_col]
NUTRIENTS = ["kcal", "protein", "fat", "carbs", "fiber", "sugar", "sodium", *MICROS]
NON_FOOD = {"несъедобное", "нераспознанные продукты"}          # категория модели для пакетов, бытовой химии и т. п.


def is_non_food(category: str) -> bool:
    c = str(category).strip().lower()
    return c in NON_FOOD or c.startswith("нераспознанн") or c.startswith("корм для")   # «корм для животных» тоже
MIN_CONFIDENCE = 0.30   # позиции с меньшей уверенностью (пакеты, бытовая химия) в расчёт не идут
GENITIVE = {"kcal": "калорий", "protein": "белка", "fat": "жиров", "carbs": "углеводов", "fiber": "клетчатки"}
LABELS = {"kcal": "ккал", "protein": "белок, г", "fat": "жиры, г", "carbs": "углеводы, г",
          "fiber": "клетчатка, г", "sugar": "сахара, г", "sodium": "натрий, мг",
          **{k: v[1] for k, v in MICROS.items()}}
GENITIVE.update({k: v[2] for k, v in MICROS.items()})


def pretty(category: str) -> str:
    """Название категории для людей: без кода справочника и ссылок на рецептуры.
    «батон нарезной из муки в/с 6.1.3.2.2» → «батон нарезной из муки в/с»."""
    s = re.sub(r"\s+\d{1,2}(?:[ .]\d{1,2}){1,4}\.?\s*$", "", str(category))
    s = re.sub(r"\bпо\s+\d+-\d+\b|\(\d{3,4}\)", "", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def load_pack_weights(path: str = os.environ.get("PACK_WEIGHTS_PATH", "pack_weights.csv")) -> tuple[dict, dict]:
    """Типичный вес упаковки по категории (make_pack_weights.py) — для товаров без веса в названии,
    и сколько раз категория встречается в чеках датасета — как мера «это реально покупают»."""
    if not os.path.exists(path):
        return {}, {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    cats = df.iloc[:, 0].astype(str)
    counts = dict(zip(cats, df.iloc[:, 2].astype(int))) if df.shape[1] > 2 else {}
    return dict(zip(cats, df.iloc[:, 1].astype(float))), counts


PACK_WEIGHTS, POPULARITY = load_pack_weights()
# в советах «где взять» не предлагаем сладости, алкоголь и снеки
NOT_ADVISED = re.compile(r"шоколад|конфет|печень[ея]|торт|пирожн|вафл|мармелад|зефир|пастил|халв|сахар|варень|джем|"
                         r"газиров|лимонад|чипс|сухарик|мороженое|сироп|батончик|пиво|вин[оа]|водк|ликёр|ликер|"
                         r"настойк|сидр|майонез|соус|маргарин|кофе|какао|чай|жир |шпик|сало|колбас|сосиск|"
                         r"сардельк|пельмен|фарш колбас|консерв|"
                         r"зелень|кинза|базилик|петрушк|укроп|кориандр|лавров|пряност|приправ|отруб|дрожж|желатин|агар|"
                         r"горчиц|хрен|мак\b|порошок|сухой|сухое|сушен|сушён|концентрат", re.I)   # едят граммами или не едят как есть


def item_weight(item: dict) -> tuple[float | None, bool]:
    """(граммы, оценка ли это). Вес из чека или названия точнее; иначе — типичная упаковка категории."""
    if item.get("grams"):
        return item["grams"], False
    w = PACK_WEIGHTS.get(item.get("category", ""))
    if w and (item.get("quantity") or 0) > 0:
        q = item["quantity"]
        return w * (q if q == int(q) else 1), True
    return None, False


class NutrientTable:
    def __init__(self, path: str | None):
        self.rows: dict[str, dict] = {}
        self.groups: dict[str, str] = {}          # категория → группа продуктов (для диаграмм)
        self.available: list[str] = []
        if path and os.path.exists(path):
            df = pd.read_csv(path, encoding="utf-8-sig") if path.endswith(".csv") else pd.read_excel(path)
            lower = {c.strip().lower(): c for c in df.columns}
            mapping = {}
            for key, names in COLUMNS.items():
                for n in names:
                    if n in lower:
                        mapping[key] = lower[n]
                        break
            if "category" not in mapping or "kcal" not in mapping:
                raise ValueError(f"{path}: нужны колонки «Категория» и «ккал», есть: {list(df.columns)}")
            self.available = [k for k in NUTRIENTS if k in mapping]
            gcol = lower.get("группа")
            if gcol:
                self.groups = dict(zip(df[mapping["category"]].astype(str).str.strip(), df[gcol].fillna("Другое")))
            for _, r in df.iterrows():
                vals = {k: pd.to_numeric(str(r[mapping[k]]).replace(",", "."), errors="coerce")
                        for k in self.available}
                # пусто в основных колонках = 0; в «редких» (цинк, D…) пусто = неизвестно, ключа нет
                self.rows[str(r[mapping["category"]]).strip()] = {
                    k: (0.0 if pd.isna(v) else float(v)) for k, v in vals.items() if not (k in SPARSE and pd.isna(v))}

    def __bool__(self):
        return bool(self.rows)

    def get(self, category: str) -> dict | None:
        return self.rows.get(category)


# ---------- суточные нормы ----------
ACTIVITY = {"низкая": 1.2, "умеренная": 1.375, "средняя": 1.55, "высокая": 1.725}
DEFAULT_TARGETS = {"kcal": 2000.0, "protein": 75.0, "fat": 67.0, "carbs": 250.0, "fiber": 25.0}


def micro_targets(p: dict) -> dict:
    male = p.get("sex") == "м"
    t = {k: float(v[3] if male else v[4]) for k, v in MICROS.items()}
    try:
        if float(p.get("age") or 0) > 65:
            t["vit_d"] = 20.0                    # МР 2.3.1.0253-21: старше 65 лет — 20 мкг
    except (TypeError, ValueError):
        pass
    return t


def daily_targets(p: dict) -> dict:
    """КБЖУ по формуле Миффлина–Сан Жеора, витамины и минералы по МР 2.3.1.0253-21. Ориентир, не назначение."""
    return {**macro_targets(p), **micro_targets(p)}


def macro_targets(p: dict) -> dict:
    try:
        w, h, a = float(p["weight"]), float(p["height"]), float(p["age"])
    except (KeyError, TypeError, ValueError):
        return dict(DEFAULT_TARGETS)
    male = p.get("sex") == "м"
    bmr = 10 * w + 6.25 * h - 5 * a + (5 if male else -161)
    kcal = bmr * ACTIVITY.get(p.get("activity"), 1.375)
    goal = p.get("goal")
    if goal == "снизить вес":
        kcal = max(kcal * 0.85, 1500 if male else 1200)      # без резкого дефицита
    elif goal == "набрать массу":
        kcal *= 1.10
    protein = (1.4 if goal in ("снизить вес", "набрать массу") else 1.1) * w
    fat = 0.30 * kcal / 9
    carbs = max(kcal - protein * 4 - fat * 9, 0) / 4
    return {"kcal": round(kcal), "protein": round(protein), "fat": round(fat),
            "carbs": round(carbs), "fiber": 25.0}


# ---------- где взять нутриент ----------
def where_to_get(table, key: str, target: float, bought: set = frozenset(), exclude=(), k: int = 3) -> list[dict]:
    """Продукты-источники из справочника: много нутриента на 100 ккал, 100 г дают ≥ 10% нормы,
    категория реально покупается (≥ 20 товаров в датасете), без сладостей, алкоголя и исключений профиля."""
    ex = [e.lower() for e in exclude if e]
    cands = []
    for cat, row in table.rows.items():
        val, kc = row.get(key, 0.0), row.get("kcal", 0.0)
        if val <= 0 or kc < 15 or val < 0.10 * target or is_non_food(cat) or NOT_ADVISED.search(cat):
            continue
        if POPULARITY and POPULARITY.get(cat, 0) < 20:
            continue
        if any(e in cat.lower() for e in ex):
            continue
        # и «плотно» (много на 100 ккал), и «ощутимо» (много в 100 г): среднее геометрическое двух долей нормы
        cands.append((((val / target) * (val / kc * 100 / target)) ** 0.5, cat, val))
    cands.sort(reverse=True)
    out, seen = [], set()
    for _, cat, val in cands:
        # разнообразие: «печень говяжья», «паштет из печени», «пирожки с печенью» — это одно и то же
        stems = {w[:5] for w in re.findall(r"[а-яё]{3,}", pretty(cat).lower())} - {"для", "или", "без", "при", "муки"}
        code = re.search(r"(\d{1,2})[ .](\d{1,2})(?:[ .]\d{1,2})*\.?\s*$", cat)   # 1.6.1.8 → группа «1.6» (сыры)
        if code:
            stems.add("код" + code.group(1) + "." + code.group(2))
        if re.search(r"\bсыр|сулугун|брынз|моцарел", cat, re.I):              # сыры без кода и без слова «сыр»
            stems.add("код1.6")
        if stems & seen:
            continue
        seen |= stems
        out.append({"product": pretty(cat), "per_100g_percent_of_norm": round(val / target * 100),
                    "new_for_you": cat not in bought})
        if len(out) == k:
            break
    return out


# ---------- анализ покупок ----------
def analyze(items: list[dict], table: NutrientTable, targets: dict, exclude=()) -> dict:
    """Итоги по покупкам: деньги, БЖУ, «на сколько дней хватит» по каждому нутриенту, куда уходят деньги."""
    non_food = sum(i["total"] for i in items if is_non_food(i["category"]))
    items = [i for i in items if not is_non_food(i["category"])]
    spend = sum(i["total"] for i in items)
    totals = defaultdict(float)
    covered_spend = 0.0
    unknown_weight, unknown_cat = [], []
    estimated = 0.0
    by_cat = defaultdict(lambda: {"spend": 0.0, "grams": 0.0, "kcal": 0.0, "protein": 0.0, "names": set(),
                                  "micro": defaultdict(float)})
    known_g = defaultdict(float)          # для редких нутриентов: сколько граммов покупок со значением
    all_g = 0.0
    for i in items:
        n = table.get(i["category"]) if table and i.get("confidence", 1) >= MIN_CONFIDENCE else None
        c = by_cat[i["category"]]
        c["spend"] += i["total"]
        c["names"].add(i["name"])
        grams, approx = item_weight(i)
        if not grams:
            unknown_weight.append(i["name"])
            continue
        if n is None:
            unknown_cat.append(i["name"])
            continue
        if approx:
            estimated += i["total"]
        k = grams / 100
        covered_spend += i["total"]
        c["grams"] += grams
        all_g += grams
        for key, v in n.items():
            totals[key] += v * k
            if key in SPARSE:
                known_g[key] += grams
        c["kcal"] += n["kcal"] * k
        c["protein"] += n.get("protein", 0) * k
        for m in MICROS:
            if m in n:
                c["micro"][m] += n[m] * k

    # редкие нутриенты: если значение известно для большей части покупок — досчитываем остальное
    # пропорционально (оценка), иначе вывод не делаем
    coverage = {m: (known_g[m] / all_g if all_g else 0.0) for m in SPARSE}
    for m in SPARSE:
        if coverage[m] >= MIN_SPARSE_COVERAGE:
            totals[m] = totals.get(m, 0.0) / coverage[m]
        else:
            totals.pop(m, None)
    days = {key: round(totals[key] / targets[key], 1) for key in targets if key in totals and targets[key]}
    macro_days = {k: v for k, v in days.items() if k not in MICROS}
    micro_days = {k: v for k, v in days.items() if k in MICROS}
    # для трёх самых «коротких» витаминов/минералов — где их больше всего среди покупок
    micro_low = []
    for m in sorted(micro_days, key=micro_days.get)[:3]:
        src = sorted(((cat, c["micro"][m]) for cat, c in by_cat.items() if c["micro"].get(m)), key=lambda t: -t[1])
        micro_low.append({"nutrient": LABELS[m].split(",")[0], "days": micro_days[m],
                          "best_sources_in_purchases": [pretty(cat) for cat, _ in src[:2]],
                          "where_to_get": where_to_get(table, m, targets[m], set(by_cat), exclude)})
    top_spend = sorted(by_cat.items(), key=lambda kv: -kv[1]["spend"])[:5]
    by_group = defaultdict(float)
    for cat, c in by_cat.items():
        by_group[table.groups.get(cat, "Готовая еда и другое") if table else "Другое"] += c["spend"]
    kc = totals.get("kcal", 0)
    pfc = totals.get("protein", 0) * 4 + totals.get("fat", 0) * 9 + totals.get("carbs", 0) * 4
    energy = ({"белки": totals.get("protein", 0) * 4 / pfc, "жиры": totals.get("fat", 0) * 9 / pfc,
               "углеводы": totals.get("carbs", 0) * 4 / pfc} if pfc else {})   # доли от БЖУ: в сумме 100%
    # баланс: насколько хватает нутриента относительно калорий (1.0 — в той же пропорции, что и калорий)
    balance = {LABELS[k].split(",")[0]: round(days[k] / days["kcal"], 2) for k in days
               if k != "kcal" and days.get("kcal")}
    protein_sources = [(cat, c["spend"] / c["protein"] * 100) for cat, c in by_cat.items()
                       if c["protein"] >= 20 and c["grams"] and c["protein"] / c["grams"] * 100 >= 5
                       and not re.search(r"пиво|вин[оа]|водк|ликёр|настойк|сидр", cat, re.I)]
    protein_sources.sort(key=lambda t: t[1])
    protein_sources = [(pretty(c), r) for c, r in protein_sources]
    return {
        "items": len(items),
        "spend_rub": round(spend, 2),
        "non_food_rub": round(non_food, 2),
        "nutrition_known_for_share_of_spend": round(covered_spend / spend, 2) if spend else 0,
        "weight_estimated_share_of_spend": round(estimated / spend, 2) if spend else 0,
        "totals": {k: round(v, 1) for k, v in totals.items()},
        "daily_targets": {k: v for k, v in targets.items() if k not in MICROS},
        "spend_by_group": {g: round(v, 2) for g, v in sorted(by_group.items(), key=lambda kv: -kv[1])},
        "products": [{"name": pretty(cat), "group": (table.groups.get(cat, "Другое") if table else "Другое"),
                      "kcal": round(c["kcal"]), "grams": round(c["grams"]), "spend": round(c["spend"], 2)}
                     for cat, c in sorted(by_cat.items(), key=lambda kv: -kv[1]["kcal"]) if c["kcal"] > 0],
        "energy_split": {k: round(v, 3) for k, v in energy.items()},
        "balance_vs_calories": balance,
        "days_covered": macro_days,                             # на сколько дней хватит при твоей норме
        "micro_days_covered": micro_days,
        "micro_lowest": micro_low,
        "sparse_coverage": {m: round(v, 2) for m, v in coverage.items()},
        "micro_highest": [{"nutrient": LABELS[m].split(",")[0], "days": micro_days[m]}
                          for m in sorted(micro_days, key=micro_days.get, reverse=True)[:3]],
        "rub_per_1000_kcal": round(covered_spend / totals["kcal"] * 1000, 1) if totals.get("kcal") else None,
        "top_spend": [{"category": pretty(cat), "rub": round(c["spend"], 2), "share": round(c["spend"] / spend, 2)}
                      for cat, c in top_spend] if spend else [],
        "protein_rub_per_100g": {"cheapest": [{"category": c, "rub": round(r, 1)} for c, r in protein_sources[:3]],
                                 "most_expensive": [{"category": c, "rub": round(r, 1)}
                                                    for c, r in protein_sources[-3:][::-1]]},
        "weight_unknown": unknown_weight[:10],
        "no_nutrient_data": unknown_cat[:10],
    }


def format_summary(r: dict, period: str = "За 90 дней") -> str:
    """Короткая сводка под постером: главные цифры наверху, 2–3 вывода."""
    if not r["items"]:
        return "Продуктов пока нет: пришли чек (PDF, фото или JSON)."
    t, d = r.get("totals") or {}, r.get("days_covered") or {}
    num = lambda v: f"{v:,.0f}".replace(",", " ")
    lines = [f"{period} · {r['items']} продуктов"]
    if t.get("kcal"):
        lines.append(f"🔥 {num(t['kcal'])} ккал — хватит на {d.get('kcal', 0):.0f} дн.")
    money = f"💰 {num(r['spend_rub'])} ₽ на еду"
    if r.get("rub_per_1000_kcal"):
        money += f" — {r['rub_per_1000_kcal']:.0f} ₽ за 1000 ккал"
    lines.append(money)
    if t.get("protein"):
        lines.append(f"💪 белок {num(t['protein'])} г — на {d.get('protein', 0):.0f} дн.")
    tips = []
    for m in (r.get("micro_lowest") or [])[:2]:
        if m["days"] < 0.8 * d.get("kcal", 0) or not tips:
            short = lambda n: " ".join(n.split(" (")[0].replace(",", "").split()[:2])
            src = ", ".join(short(x["product"]) for x in (m.get("where_to_get") or [])[:3])
            tips.append(f"• мало: {m['nutrient']} ({m['days']:.0f} дн.)" + (f" → {src}" if src else ""))
    split = r.get("energy_split") or {}
    if split.get("жиры", 0) > 0.35:
        tips.append(f"• жиров {split['жиры']:.0%} калорий при норме до 35%")
    if r.get("top_spend"):
        top = r["top_spend"][0]
        tips.append(f"• №1 по деньгам — {top['category']}, {top['share']:.0%}")
    if tips:
        lines += ["", "Главное:"] + tips
    return "\n".join(lines)


def format_report(r: dict) -> str:
    if not r["items"]:
        return "Продуктов пока нет: пришли чек (PDF, фото или JSON)."
    lines = [f"Продуктов: {r['items']}, на еду {r['spend_rub']:.0f} ₽"]
    if r.get("non_food_rub"):
        lines[0] += f" (+ {r['non_food_rub']:.0f} ₽ не еда или не распознано)"
    small = r["totals"] and r["days_covered"].get("kcal", 0) < 0.5
    if small:
        t = r["totals"]
        lines.append(f"\nЧек небольшой: {t.get('kcal', 0):.0f} ккал ({r['days_covered'].get('kcal', 0):.0%} дневной нормы), "
                     f"белок {t.get('protein', 0):.0f} г, жиры {t.get('fat', 0):.0f} г, углеводы {t.get('carbs', 0):.0f} г. "
                     "Выводы о рационе лучше смотреть в «Мой рацион» по всем покупкам.")
    if r["totals"] and not small:
        t, d = r["totals"], r["days_covered"]
        lines.append("\nНа сколько дней хватит при твоей норме:")
        for key in ["kcal", "protein", "fat", "carbs", "fiber"]:
            if key in d:
                lines.append(f"  {LABELS[key]:<13} {t[key]:>8.0f}  → {d[key]:.1f} дн.")
        if d:
            lo = min(d, key=d.get)
            hi = max(d, key=d.get)
            if d[hi] >= 1 and d[hi] > 1.5 * d[lo]:
                lines.append(f"\nПерекос: {GENITIVE[hi]} хватит на {d[hi]:.1f} дн., "
                             f"а {GENITIVE[lo]} только на {d[lo]:.1f} дн.")
        if r.get("micro_lowest"):
            lines.append("\nВитамины и минералы — меньше всего хватит:")
            for m in r["micro_lowest"]:
                src = f" (больше всего в: {', '.join(m['best_sources_in_purchases'])})" \
                    if m["best_sources_in_purchases"] else " (в покупках почти нет)"
                lines.append(f"  {m['nutrient']}: {m['days']:.1f} дн.{src}")
                if m.get("where_to_get"):
                    tips = ", ".join(f"{t['product']}{' ★' if t['new_for_you'] else ''} ({t['per_100g_percent_of_norm']}%)"
                                     for t in m["where_to_get"])
                    lines.append(f"     где взять: {tips}")
            if any(m.get("where_to_get") for m in r["micro_lowest"]):
                lines.append("  (в скобках — % дневной нормы в 100 г; ★ — нет в твоих покупках)")
            if any(m["nutrient"].startswith("витамин D") for m in r["micro_lowest"]):
                lines.append("  Витамин D в основном вырабатывается в коже на солнце — из еды норму набрать трудно; "
                             "проверить уровень можно анализом крови.")
            sparse = [m for m, v in (r.get("sparse_coverage") or {}).items() if v < MIN_SPARSE_COVERAGE]
            if sparse:
                lines.append("  Не хватает данных о составе, чтобы оценить: "
                             + ", ".join(LABELS[m].split(",")[0] for m in sparse))
            plenty = [m for m in r["micro_highest"] if m["days"] >= 1]
            if plenty:
                lines.append("Больше всего хватит: " + ", ".join(f"{m['nutrient']} {m['days']:.0f} дн." for m in plenty))
        if r["rub_per_1000_kcal"]:
            lines.append(f"Цена 1000 ккал: {r['rub_per_1000_kcal']:.0f} ₽")
    if r["totals"]:
        share = r["nutrition_known_for_share_of_spend"]
        if r.get("weight_estimated_share_of_spend", 0) >= 0.05:
            lines.append(f"(≈ для {r['weight_estimated_share_of_spend']:.0%} суммы вес взят как типичная упаковка — "
                         "в названии его не было)")
        if share < 0.9:
            lines.append(f"(химсостав посчитан для {share:.0%} суммы чека: у остальных неизвестен вес или категория)")
    if r["top_spend"]:
        lines.append("\nКуда ушли деньги:")
        lines += [f"  {c['category']}: {c['rub']:.0f} ₽ ({c['share']:.0%})" for c in r["top_spend"]]
    ps = r["protein_rub_per_100g"]
    if ps["cheapest"] and len(ps["cheapest"]) + len(ps["most_expensive"]) > 1:
        lines.append("\n100 г белка обходятся:")
        lines += [f"  дешевле всего: {c['category']} — {c['rub']:.0f} ₽" for c in ps["cheapest"][:1]]
        if ps["most_expensive"] and ps["most_expensive"][0]["rub"] >= 1.5 * ps["cheapest"][0]["rub"]:
            lines += [f"  дороже всего: {c['category']} — {c['rub']:.0f} ₽" for c in ps["most_expensive"][:1]]
    return "\n".join(lines)
