"""«Здоровая корзина за минимальные деньги»: задача о диете (линейное программирование)
на продуктах, которые человек сам покупал, с его реальными ценами из чеков."""
from collections import defaultdict
from statistics import median

import re

import numpy as np
from scipy.optimize import linprog

from nutrition import CORE_MICROS, MICROS, is_non_food, item_weight, pretty

# приправы и зелень едят граммами — не больше 50 г в день, иначе «2 кг лука в неделю»
SEASONING = re.compile(r"\bлук\b|лук репч|чеснок|зелень|петрушк|укроп|кинз|базилик|имбир|хрен|перец остр|"
                       r"перец черн|щавел|шпинат", re.I)
VEG, FRUIT, GRAIN, DAIRY = "Овощи и грибы", "Фрукты и ягоды", "Хлеб, крупы, макароны", "Молочное"
PROTEIN_GROUPS = {"Мясо и птица", "Рыба и морепродукты", "Яйца", "Молочное", "Бобовые, орехи, семена"}
# правила «здоровой тарелки»: (ключ, понятное описание) — в порядке ослабления, если не выполнить
PLATE_RULES = [("dairy", "молочное от 150 г в день"), ("protein", "белковые продукты от 20% калорий"),
               ("grain_min", "крупы и хлеб от 15% калорий"), ("grain_max", "крупы и хлеб не больше 35% калорий"),
               ("veg_fruit", "овощи и фрукты от 400 г в день")]

# разделы списка покупок — как в магазине
SECTIONS = [("БЕЛОК", {"Мясо и птица", "Рыба и морепродукты", "Яйца", "Бобовые, орехи, семена"}),
            ("МОЛОЧНОЕ", {"Молочное"}), ("ОВОЩИ И ФРУКТЫ", {VEG, FRUIT}), ("КРУПЫ И ХЛЕБ", {GRAIN})]


def section_of(group: str) -> str:
    return next((name for name, gs in SECTIONS if group in gs), "ДРУГОЕ")


def by_sections(basket: list) -> list:
    """[(раздел, [позиции])] в порядке разделов."""
    order = [n for n, _ in SECTIONS] + ["ДРУГОЕ"]
    out = {n: [] for n in order}
    for b in basket:
        out[section_of(b.get("group", ""))].append(b)
    return [(n, out[n]) for n in order if out[n]]


# алкоголь в корзину не предлагаем никогда
ALCOHOL = re.compile(r"пиво|вин[оа]|водк|коньяк|ликёр|ликер|настойк|наливк|сидр|джин|текил|виски|ром\b|шампанск", re.I)


# сладости и снеки: вместе не больше 10% калорий (ВОЗ: свободные сахара < 10% энергии)
TREATS = re.compile(r"шоколад|конфет|печень|торт|пирожн|вафл|мармелад|зефир|пастил|халв|сахар|варень|джем|"
                    r"газиров|лимонад|кола|чипс|сухарик|мороженое|сироп|батончик|круассан|пончик", re.I)


def build_pool(items: list[dict], table, exclude: list[str] = ()) -> list[dict]:
    """Кандидаты: категории из чеков, для которых известны вес, цена и химсостав."""
    ex = [e.lower().strip() for e in exclude if e.strip()]
    acc = defaultdict(lambda: {"rub_per_100g": [], "names": [], "grams": 0.0})
    for i in items:
        grams, _ = item_weight(i)
        if (not grams or i["total"] <= 0 or not table or table.get(i["category"]) is None
                or i.get("confidence", 1) < 0.30 or is_non_food(i["category"])
                or ALCOHOL.search(i["category"]) or table.get(i["category"]).get("kcal", 0) <= 0):
            continue
        text = (i["category"] + " " + i["name"]).lower()
        if any(e in text for e in ex):
            continue
        a = acc[i["category"]]
        a["rub_per_100g"].append(i["total"] / grams * 100)
        a["names"].append(i["name"])
        a["grams"] += grams
    pool = []
    for cat, a in acc.items():
        n = table.get(cat)
        pool.append({"category": cat, "example": a["names"][-1], "rub_per_100g": median(a["rub_per_100g"]),
                     "habit_g": a["grams"], "group": table.groups.get(cat, "Другое"), **n})
    return pool


def optimize(pool: list[dict], targets: dict, days: int = 7, budget: float | None = None,
             max_g_per_day: float = 300.0, max_kcal_share: float = 0.20, mode: str = "привычная",
             habit_weight: float = 1.5) -> dict:
    """mode="привычная": минимум (стоимость + штраф за отход от обычного набора покупок);
    mode="дешёвая": просто минимум стоимости. Ограничения по нормам одинаковые."""
    if len(pool) < 3:
        return {"ok": False, "reason": "мало продуктов с известными ценой, весом и химсоставом: "
                                       f"{len(pool)} (нужно хотя бы 3). Загрузи ещё чеки."}
    n = len(pool)
    cost = np.array([p["rub_per_100g"] for p in pool])           # переменные: сотни граммов в день
    col = lambda key: np.array([p.get(key, 0.0) for p in pool])
    kcal, prot, fat = col("kcal"), col("protein"), col("fat")
    # лёгкие продукты (овощи, кисломолочка) можно есть больше
    caps = np.full(len(pool), max_g_per_day / 100)                          # любой продукт — до 300 г в день
    # и ни один продукт не даёт больше 20% калорий дня - иначе LP выбирает 2-3 самых дешёвых
    caps = np.minimum(caps, max_kcal_share * targets["kcal"] / np.maximum(kcal, 1))

    # привычный набор: что и в каких пропорциях человек покупает, масштабировано под его норму калорий
    habit = col("habit_g") / 100
    habit_kcal = (habit * kcal).sum()
    h = habit * (targets["kcal"] / habit_kcal) if habit_kcal > 0 else np.zeros(n)
    use_habit = mode == "привычная" and h.sum() > 0
    if use_habit:
        caps = np.maximum(caps, h)                     # что ешь много — можно и дальше
    # штраф за 100 г отхода от привычного — не меньше цены самого продукта: иначе любимые дорогие продукты
    # (сёмга, клубника) выгоднее выкинуть и «заплатить штраф», чем оставить
    kappa = np.maximum(cost, float(np.median(cost)))

    grp = np.array([p.get("group", "Другое") for p in pool])
    is_veg, is_fruit, is_grain, is_dairy = grp == VEG, grp == FRUIT, grp == GRAIN, grp == DAIRY
    is_prot = np.isin(grp, list(PROTEIN_GROUPS))
    # один овощ — до 200 г, фрукт — до 250 г, приправы — до 50 г в день
    caps = np.where(is_veg, np.minimum(caps, 2.0), caps)
    caps = np.where(is_fruit, np.minimum(caps, 2.5), caps)
    caps = np.where([bool(SEASONING.search(p["category"] + " " + p["example"])) for p in pool], np.minimum(caps, 0.5), caps)
    T = targets["kcal"]
    plate_rows = {}                                       # правило → (строка A, b), только если применимо
    if (caps * (is_veg | is_fruit)).sum() >= 4:
        plate_rows["veg_fruit"] = (-(is_veg | is_fruit).astype(float), -4.0)
    if is_grain.any():
        plate_rows["grain_max"] = (kcal * is_grain, 0.35 * T)
        plate_rows["grain_min"] = (-kcal * is_grain, -0.15 * T)
    if is_prot.any():
        plate_rows["protein"] = (-kcal * is_prot, -0.20 * T)
    if is_dairy.any():
        plate_rows["dairy"] = (-is_dairy.astype(float), -1.5)
    treat = np.array([1.0 if TREATS.search(p["category"]) or p.get("sugar", 0) > 25 else 0.0 for p in pool])
    # в план покупок — только нутриенты, известные для всех продуктов (цинк, D… известны не везде:
    # оптимизатор бы считал «нет данных» нулём и перекашивал корзину)
    micros = [m for m in CORE_MICROS if m in targets and targets[m] and any(p.get(m) for p in pool)]
    k = len(micros)

    def solve(protein_k=1.0, use_fiber=True, fat_min=0.20, micro_penalty=0.0, plate=None):
        """Переменные: x — сотни граммов продукта в день; d — отход от привычного (режим «привычная»);
        u — недобор витамина/минерала в долях нормы (мягкое ограничение со штрафом)."""
        A, b = [], []
        A += [kcal, -kcal]; b += [targets["kcal"] * 1.05, -targets["kcal"] * 0.95]
        A.append(-prot); b.append(-targets["protein"] * protein_k)
        A.append(fat * 9); b.append(targets["kcal"] * 0.35)                   # жиры ≤ 35% калорий
        if fat_min:                                                           # и ≥ 20% (МР 2.3.1.0253-21)
            A.append(-fat * 9); b.append(-targets["kcal"] * fat_min)
        if use_fiber and any(p.get("fiber") for p in pool):
            A.append(-col("fiber")); b.append(-targets.get("fiber", 25))
        if treat.any():
            A.append(kcal * treat); b.append(targets["kcal"] * 0.10)
        for key in (plate if plate is not None else plate_keys):
            a_, b_ = plate_rows[key]
            A.append(a_); b.append(b_)
        if budget:
            A.append(cost); b.append(budget / days)
        nd = n if use_habit else 0
        ku = k if micro_penalty else 0
        rows = [np.concatenate([a, np.zeros(nd + ku)]) for a in A]
        for j, m in enumerate(micros[:ku]):                                   # x·a/норма + u ≥ 1
            r = np.zeros(n + nd + ku)
            r[:n] = -col(m) / targets[m]
            r[n + nd + j] = -1
            rows.append(r); b.append(-1.0)
        if nd:                                                                # d ≥ |x − h|
            eye = np.eye(n)
            for sign in (1, -1):
                for i in range(n):
                    r = np.zeros(n + nd + ku)
                    r[i], r[n + i] = sign, -1
                    rows.append(r); b.append(sign * h[i])
        c = np.concatenate([cost, (habit_weight * kappa)[:nd], np.full(ku, micro_penalty)])
        bounds = list(zip([0] * n, caps)) + [(0, None)] * nd + [(0, 1)] * ku
        return linprog(c, A_ub=np.array(rows), b_ub=np.array(b), bounds=bounds, method="highs")

    relaxed = []
    plate_keys = [k for k, _ in PLATE_RULES if k in plate_rows]
    # «здоровая тарелка»: если с какими-то правилами не решается — ослабляем по одному, в порядке PLATE_RULES
    for key, label in PLATE_RULES:
        if solve(fat_min=0.15).status == 0:
            break
        if key in plate_keys:
            plate_keys.remove(key)
            relaxed.append(f"из твоих продуктов не получилось: {label}")
    fat_min = 0.20
    res = solve()
    if res.status != 0:
        for fm in (0.15, 0.0):
            if solve(fat_min=fm).status == 0:
                fat_min = fm
                relaxed.append("жиров в твоих продуктах мало: " + (f"нижняя граница снижена до {fm:.0%} калорий"
                               if fm else "нижняя граница по жирам снята — добавь в покупки масло, орехи, рыбу"))
                break
    kw = {"fat_min": fat_min}
    res = solve(**kw)
    if res.status != 0 and any(p.get("fiber") for p in pool):
        kw["use_fiber"] = False
        res = solve(**kw)
        relaxed.append("ограничение по клетчатке снято: из твоих продуктов её не набрать")
    if res.status != 0:
        kw["protein_k"] = 0.8
        res = solve(**kw)
        relaxed.append("белок снижен до 80% нормы")
    if res.status != 0:
        why = "в этот бюджет не уложиться" if budget else "из этих продуктов норму не собрать"
        return {"ok": False, "reason": why + ". Попробуй больше дней, другой бюджет или загрузи ещё чеки."}
    if micros:
        # недобор целой нормы витамина «стоит» как дневная корзина: дешёвые способы добрать витамин
        # оптимизатор возьмёт, разорительные (килограмм сёмги ради витамина E) — нет
        base_cost = float(cost @ res.x[:n])
        r2 = solve(micro_penalty=1.0 * max(base_cost, 1.0), **kw)
        if r2.status == 0:
            res = r2

    x = res.x[:n]
    basket, day = [], {k: 0.0 for k in ["kcal", "protein", "fat", "carbs", "fiber", *MICROS]}
    for p, v, hv in sorted(zip(pool, x, h), key=lambda t: -t[1] * t[0]["rub_per_100g"]):
        if v < 0.1:                      # меньше 10 г в день не показываем
            continue
        g = round(v * 100 * days / 50) * 50
        change = "" if not use_habit else ("+" if v > hv * 1.3 + 0.2 else "-" if v < hv * 0.7 - 0.2 else "=")
        basket.append({"category": pretty(p["category"]), "example": p["example"], "grams_total": g,
                       "grams_per_day": round(v * 100), "rub": round(p["rub_per_100g"] * v * days),
                       "vs_habit": change, "group": p.get("group", "Другое")})
        for k in day:
            day[k] += p.get(k, 0.0) * v
    micro_pct = {MICROS[m][1].split(",")[0]: round(day[m] / targets[m] * 100) for m in micros}
    by_group = defaultdict(lambda: [0.0, 0.0])                     # группа → [обычно ₽, в корзине ₽]
    for p, v, hv in zip(pool, x, h):
        g = by_group[p.get("group", "Другое")]
        g[0] += p["rub_per_100g"] * hv * days
        g[1] += p["rub_per_100g"] * v * days
    by_group = {k: (round(a), round(b)) for k, (a, b) in by_group.items() if a >= 1 or b >= 1}
    dropped = [p["example"] for p, v, hv in sorted(zip(pool, x, h), key=lambda t: -t[2])
               if use_habit and hv >= 0.3 and v < 0.1][:4]
    return {"ok": True, "days": days, "total_rub": round(float(cost @ x) * days), "budget": budget,
            "mode": "привычная" if use_habit else "дешёвая",
            "habit_rub": round(float(cost @ h) * days) if h.sum() > 0 else None,
            "dropped_from_habit": dropped,
            "by_group": by_group,
            "per_day": {k: round(day[k]) for k in ["kcal", "protein", "fat", "carbs", "fiber"]},
            "targets": {k: v for k, v in targets.items() if k not in MICROS},
            "micro_percent_of_norm": micro_pct,
            "basket": basket, "relaxed": relaxed, "pool_size": n}


def format_basket(r: dict) -> str:
    if not r["ok"]:
        return "Не получилось собрать корзину: " + r["reason"]
    habit = r.get("mode") == "привычная"
    title = "ближе к твоим привычным покупкам" if habit else "самый дешёвый вариант"
    lines = [f"🛒 План покупок на {r['days']} дн. ({title}): ≈ {r['total_rub']} ₽",
             "Что купить, чтобы неделю есть по норме, — из того, что ты обычно покупаешь."]
    if r["budget"]:
        lines[0] += f" (бюджет {r['budget']:.0f} ₽)"
    if r.get("habit_rub"):
        diff = r["habit_rub"] - r["total_rub"]
        lines.append(f"Твой обычный набор на ту же калорийность ≈ {r['habit_rub']} ₽"
                     + (f" — экономия {diff} ₽" if diff > 0 else f" — дороже на {-diff} ₽ ради норм" if diff < 0 else ""))
    mark = {"+": " ↑", "-": " ↓", "=": "", "": ""}
    for sec, items in by_sections(r["basket"]):
        lines.append(f"\n{sec.capitalize()}:")
        for b in items:
            g = b["grams_total"]
            amount = f"{g / 1000:.1f} кг" if g >= 1000 else f"{g} г"
            if "яйц" in b["category"].lower():
                amount = f"{round(g / 55)} шт"
            lines.append(f"• {b['example']} — {amount}, ≈{b['grams_per_day']} г/день (~{b['rub']} ₽)"
                         f"{mark[b.get('vs_habit', '')]}")
    if habit:
        lines.append("↑ больше, ↓ меньше, чем ты обычно покупаешь")
    if r.get("dropped_from_habit"):
        lines.append("Убрано из привычного: " + ", ".join(r["dropped_from_habit"]))
    d, t = r["per_day"], r["targets"]
    lines.append(f"\nВ день: {d['kcal']} ккал (норма {t['kcal']:.0f}), белок {d['protein']} г "
                 f"(норма {t['protein']:.0f}), жиры {d['fat']} г (норма {t['kcal'] * 0.20 / 9:.0f}–{t['kcal'] * 0.35 / 9:.0f}), "
                 f"клетчатка {d['fiber']} г")
    mp = r.get("micro_percent_of_norm") or {}
    if mp:
        full = [k for k, v in mp.items() if v >= 95]
        low = sorted(((k, v) for k, v in mp.items() if v < 95), key=lambda t: t[1])
        if full:
            lines.append(f"Витамины и минералы на 100% нормы: {len(full)} из {len(mp)}")
        if low:
            lines.append("Не добирают: " + ", ".join(f"{k} {v}%" for k, v in low))
    lines += [f"Примечание: {x}" for x in r["relaxed"]]
    if habit:
        lines.append("Меняется только то, что нужно для норм; витамины и минералы добираются, если это не слишком дорого.")
    else:
        lines.append("Самый дешёвый набор: КБЖУ в норме, витамины и минералы добираются, если это не слишком дорого.")
    lines.append("Правила тарелки: овощи и фрукты от 400 г в день, крупы и хлеб 15–35% калорий, белковые продукты "
                 "от 20%, молочное от 150 г, приправы до 50 г.")
    return "\n".join(lines)


def swaps(name_or_cat: str, pool: list[dict], top: int = 3) -> dict:
    """Чем заменить продукт: дешевле на 100 г белка или на 1000 ккал среди того, что человек покупает."""
    norm = lambda t: str(t).lower().replace("ё", "е")
    q = norm(name_or_cat)
    base = next((p for p in pool if q in norm(p["category"]) or q in norm(p["example"])), None)
    if base is None:      # «сёмгу» → «семга»: сравниваем по основам слов (первые 4 буквы)
        stems = [w[:4] for w in re.findall(r"[а-яa-z]{4,}", q)]
        if stems:
            def hit(p):
                words = re.findall(r"[а-яa-z]{4,}", norm(p["category"]) + " " + norm(p["example"]))
                return all(any(w.startswith(st) for w in words) for st in stems)
            base = max((p for p in pool if hit(p)), key=lambda p: p.get("habit_g", 0), default=None)
    if base is None:
        return {"ok": False, "reason": f"«{name_or_cat}» нет среди покупок с известной ценой и составом"}

    def per_protein(p):
        return p["rub_per_100g"] / p["protein"] * 100 if p.get("protein", 0) >= 5 else None
    out = {"ok": True, "product": base["example"], "category": pretty(base["category"]),
           "rub_per_100g": round(base["rub_per_100g"], 1), "alternatives": []}
    bp = per_protein(base)
    if bp:
        alts = [(p, per_protein(p)) for p in pool if p is not base and per_protein(p)]
        alts = [(p, v) for p, v in alts if v < bp * 0.8]
        alts.sort(key=lambda t: t[1])
        out["rub_per_100g_protein"] = round(bp)
        out["alternatives"] = [{"example": p["example"], "rub_per_100g_protein": round(v),
                                "fat_per_100g": p.get("fat"), "kcal_per_100g": p.get("kcal")}
                               for p, v in alts[:top]]
        out["criterion"] = "цена 100 г белка"
    else:
        alts = [p for p in pool if p is not base and abs(p["kcal"] - base["kcal"]) < 0.3 * max(base["kcal"], 1)
                and p["rub_per_100g"] < base["rub_per_100g"] * 0.8]
        alts.sort(key=lambda p: p["rub_per_100g"])
        out["alternatives"] = [{"example": p["example"], "rub_per_100g": round(p["rub_per_100g"], 1),
                                "kcal_per_100g": p["kcal"]} for p in alts[:top]]
        out["criterion"] = "похожая калорийность, дешевле"
    return out
