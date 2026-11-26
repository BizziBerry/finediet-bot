"""Химсостав категорий чеков по USDA FoodData Central → nutrients_usda.csv.

    python import_usda.py                                  # архивы из C:\\projects\\kaggle\\usda
    python import_usda.py --dir D:\\data\\usda --data receipt_items.csv

Источник: U.S. Department of Agriculture, Agricultural Research Service. FoodData Central. fdc.nal.usda.gov
Данные USDA FDC — общественное достояние (CC0 1.0), поэтому nutrients_usda.csv можно публиковать.

Сопоставление «категория → продукт USDA» лежит в usda_map.csv, колонка usda:
    apples raw with skin          все слова должны быть в названии продукта USDA (по началу слова)
    -fuji                         такого слова в названии быть не должно
    a b|c d                       альтернативы: первая, для которой что-то нашлось
    a b & c d                     смесь продуктов поровну (морепродукты ассорти)
    a b @0.97 & c d @0.03         смесь в заданных долях (кофе с сахаром)
    !запрос                       брать USDA, даже если калории сильно расходятся со справочником
    sr: / ff: / fn:               только SR Legacy / Foundation Foods / FNDDS
    fdc:171705                    конкретный продукт по FDC ID
    =0                            всё по нулям (чай и кофе сухие, вода, эритрит)
    -                             USDA не подходит — остаётся справочник Тутельяна (у тебя локально)
Архивы читаются как есть, распаковывать не нужно.
"""
import argparse
import csv
import glob
import io
import os
import re
import zipfile
from collections import Counter

import pandas as pd

TYPES = {"foundation_food": ("ff", "Foundation Foods"), "sr_legacy_food": ("sr", "SR Legacy"),
         "survey_fndds_food": ("fn", "FNDDS")}
# при равной полноте берём SR Legacy: запросы в usda_map.csv выверены по его названиям;
# Foundation Foods — когда в SR подходящего нет или явно указано «ff:»
TYPE_RANK = {"sr": 0, "ff": 1, "fn": 2}

# ключ → (колонка в таблице бота, [(название нутриента в USDA, единица), …] по приоритету)
NUTR = {
    "kcal": ("ккал", [("Energy", "KCAL"), ("Energy (Atwater General Factors)", "KCAL"),
                      ("Energy (Atwater Specific Factors)", "KCAL")]),
    "protein": ("белки", [("Protein", "G")]),
    "fat": ("жиры", [("Total lipid (fat)", "G"), ("Total fat (NLEA)", "G")]),
    "carbs": ("углеводы", [("Carbohydrate, by difference", "G"), ("Carbohydrate, by summation", "G")]),
    "fiber": ("пищевые волокна", [("Fiber, total dietary", "G")]),
    "sugar": ("сахар", [("Sugars, total including NLEA", "G"), ("Sugars, Total", "G"), ("Total Sugars", "G")]),
    "sodium": ("натрий", [("Sodium, Na", "MG")]),
    "calcium": ("кальций, мг", [("Calcium, Ca", "MG")]),
    "iron": ("железо, мг", [("Iron, Fe", "MG")]),
    "magnesium": ("магний, мг", [("Magnesium, Mg", "MG")]),
    "potassium": ("калий, мг", [("Potassium, K", "MG")]),
    "phosphorus": ("фосфор, мг", [("Phosphorus, P", "MG")]),
    "vit_a": ("витамин a, мкг", [("Vitamin A, RAE", "UG")]),
    "vit_e": ("витамин e, мг", [("Vitamin E (alpha-tocopherol)", "MG")]),
    "vit_b1": ("витамин b1, мг", [("Thiamin", "MG")]),
    "vit_b2": ("витамин b2, мг", [("Riboflavin", "MG")]),
    "vit_pp": ("витамин pp, мг", [("Niacin", "MG")]),
    "vit_c": ("витамин c, мг", [("Vitamin C, total ascorbic acid", "MG")]),
    # новые — в справочнике Тутельяна их нет или там заглушки
    "zinc": ("цинк, мг", [("Zinc, Zn", "MG")]),
    "selenium": ("селен, мкг", [("Selenium, Se", "UG")]),
    "vit_b6": ("витамин b6, мг", [("Vitamin B-6", "MG")]),
    "folate": ("фолаты, мкг", [("Folate, DFE", "UG"), ("Folate, total", "UG")]),
    "vit_b12": ("витамин b12, мкг", [("Vitamin B-12", "UG")]),
    "vit_d": ("витамин d, мкг", [("Vitamin D (D2 + D3)", "UG")]),
}
TO = {("G", "MG"): 1000, ("MG", "G"): 0.001, ("MG", "UG"): 1000, ("UG", "MG"): 0.001, ("G", "UG"): 1e6}
NON_FOOD = re.compile(r"^(несъедобное|нераспознанн|корм для)", re.I)


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def base(n: str) -> str:
    return re.split(r"[\\/]", n)[-1].lower()


def open_zip(path: str) -> zipfile.ZipFile:
    """Архив USDA; если внутри ещё один zip (так бывает у новых выпусков) — открываем его."""
    z = zipfile.ZipFile(path)
    if not any(base(n) == "food.csv" for n in z.namelist()):
        inner = [n for n in z.namelist() if n.lower().endswith(".zip")]
        if inner:
            return zipfile.ZipFile(io.BytesIO(z.read(inner[0])))
    return z


def read_member(z: zipfile.ZipFile, name: str, **kw) -> pd.DataFrame | None:
    m = next((n for n in z.namelist() if base(n) == name), None)
    return pd.read_csv(io.BytesIO(z.read(m)), low_memory=False, **kw) if m else None


JSON_TYPES = {"foundation": "foundation_food", "sr legacy": "sr_legacy_food", "survey (fndds)": "survey_fndds_food"}


def load_json(z: zipfile.ZipFile):
    """Выгрузка FDC в JSON (один файл: {"FoundationFoods": [{fdcId, description, foodNutrients: […]}, …]})."""
    import json
    want = {n for _, alts in NUTR.values() for n, _ in alts}
    name = next(n for n in z.namelist() if n.lower().endswith(".json"))
    data = json.loads(z.read(name))
    items = next((v for v in data.values() if isinstance(v, list)), []) if isinstance(data, dict) else data
    foods, vals, types = [], [], set()
    for it in items:
        t = JSON_TYPES.get(str(it.get("dataType", "")).lower())
        if not t or "fdcId" not in it:
            continue
        types.add(t)
        foods.append({"fdc_id": int(it["fdcId"]), "data_type": t, "description": it.get("description", "")})
        for fn in it.get("foodNutrients", []):
            nu = fn.get("nutrient") or {}
            if nu.get("name") in want and fn.get("amount") is not None:
                vals.append({"fdc_id": int(it["fdcId"]), "name": nu["name"], "unit_name": nu.get("unitName", ""),
                             "amount": fn["amount"]})
    return (pd.DataFrame(foods, columns=["fdc_id", "data_type", "description"]),
            pd.DataFrame(vals, columns=["fdc_id", "name", "unit_name", "amount"]), types)


def load(paths: list[str]):
    foods, values, release = [], [], {}
    want = {n for _, alts in NUTR.values() for n, _ in alts}
    for p in paths:
        with open_zip(p) as z:
            if any(n.lower().endswith(".json") for n in z.namelist()):
                f_j, v_j, t_j = load_json(z)
                if len(f_j):
                    m = re.search(r"(\d{4}-\d{2})", os.path.basename(p))
                    for t in t_j:
                        release[TYPES[t][0]] = m.group(1) if m else ""
                    foods.append(f_j)
                    values.append(v_j)
                    print(f"  {os.path.basename(p)} (JSON): продуктов {len(f_j)}, значений {len(v_j)}")
                    continue
            food = read_member(z, "food.csv", usecols=["fdc_id", "data_type", "description"])
            nut = read_member(z, "nutrient.csv")
            if food is None or nut is None:
                print(f"  {os.path.basename(p)}: нет food.csv / nutrient.csv — пропускаю. Внутри архива:")
                for n in z.namelist():
                    print(f"      {n}")
                continue
            food = food[food["data_type"].isin(TYPES)]
            nut = nut[nut["name"].isin(want)]
            fn = read_member(z, "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"])
            fn = fn[fn["fdc_id"].isin(food["fdc_id"]) & fn["nutrient_id"].isin(nut["id"])]
            fn = fn.merge(nut[["id", "name", "unit_name"]], left_on="nutrient_id", right_on="id")
            m = re.search(r"(\d{4}-\d{2})", os.path.basename(p))
            for t in food["data_type"].unique():
                release[TYPES[t][0]] = m.group(1) if m else ""
            foods.append(food)
            values.append(fn[["fdc_id", "name", "unit_name", "amount"]])
            print(f"  {os.path.basename(p)}: продуктов {len(food)}, значений {len(fn)}")
    foods = pd.concat(foods).drop_duplicates("fdc_id")
    values = pd.concat(values)
    # fdc_id → {(название, единица): значение}
    vals: dict[int, dict] = {}
    for fid, name, unit, amt in values.itertuples(index=False):
        if pd.notna(amt):
            vals.setdefault(int(fid), {})[(name, str(unit).upper())] = float(amt)
    index = []
    for fid, t, desc in foods.itertuples(index=False):
        d = str(desc)
        nutr = per100(vals.get(int(fid), {}))
        index.append({"fdc_id": int(fid), "type": TYPES[t][0], "desc": d, "words": words(d),
                      "brand": is_brand(d), "nutr": nutr,
                      "cov": sum(v is not None for v in nutr.values())})
    return index, release


NOT_BRAND = {"USDA", "NLEA", "RTE", "NFS", "UHT", "USDA'S", "II", "III"}


def is_brand(desc: str) -> bool:
    """В SR Legacy марки пишут капсом: «Snacks, KELLOGG'S…» — такие продукты берём в последнюю очередь."""
    return any(w.strip("'.").upper() not in NOT_BRAND for w in re.findall(r"\b[A-Z][A-Z'&.]{2,}", desc))


def per100(raw: dict) -> dict:
    out = {}
    for key, (_, alts) in NUTR.items():
        v = None
        for name, unit in alts:
            for (n, u), a in raw.items():
                if n == name and (u == unit or (u, unit) in TO):
                    v = a if u == unit else a * TO[(u, unit)]
                    break
            if v is not None:
                break
        out[key] = v
    if out["kcal"] is None and None not in (out["protein"], out["fat"], out["carbs"]):
        out["kcal"] = 4 * out["protein"] + 9 * out["fat"] + 4 * out["carbs"]
    return out


GRAIN = re.compile(r"^(Bread|Rolls|Bagels|Crackers|Cookies|Cake|Croissants|Danish|Doughnuts|Pie|Pancakes|Waffles|"
                   r"English muffins|Muffins|Toaster|Pasta|Noodles|Macaroni|Spaghetti|Rice|Cereals|Wheat flour|Semolina|"
                   r"Cornmeal|Corn flour|Cornbread|Pizza|Croutons|Breadsticks|Pretzels)", re.I)
NOT_ENRICHED = re.compile(r"unenriched|not fortified|unfortified|whole[- ]wheat|whole[- ]grain|\boats\b|buckwheat|"
                          r"bulgur|\bbrown\b|soba|rice noodles|rice flour|homemade", re.I)


def fortification(desc: str) -> str:
    """Обогащение по-американски: в России так не делают, значения этих витаминов/минералов не переносим.
    «хлопья» — сухие завтраки (обогащены почти всем), «зерно» — мука/хлеб/макароны/рис (Fe, B1, B2, PP,
    фолиевая кислота, в хлебе ещё кальций), «» — нет."""
    if re.match(r"Cereals ready-to-eat", desc, re.I) and not NOT_ENRICHED.search(desc):
        return "хлопья"
    if (GRAIN.match(desc) or re.search(r"\benriched\b", desc, re.I)) and not NOT_ENRICHED.search(desc):
        return "зерно"
    if re.search(r"with added (vitamin|ascorbic|calcium|nutrients)|\bfortified\b", desc, re.I) \
            and not re.search(r"not fortified|unfortified", desc, re.I):
        return "витамины"          # молоко с витамином D, нектары с аскорбинкой, маргарин с A и D
    return ""


def has(ws: list[str], t: str) -> bool:
    """Слово запроса есть в названии: точно или во множественном числе (egg → eggs). «with» ≠ «without»."""
    return any(x == t or x == t + "s" or x == t + "es" or (t.endswith("s") and x == t[:-1]) for x in ws)


def search(index, query: str):
    """Один запрос без «|» и «&» → лучший продукт и число подходящих."""
    q = query.strip()
    only = None
    m = re.match(r"^(sr|ff|fn):\s*(.*)$", q)
    if m:
        only, q = m.group(1), m.group(2)
    m = re.fullmatch(r"fdc:(\d+)", q)
    if m:
        hit = [f for f in index if f["fdc_id"] == int(m.group(1))]
        return (hit[0], 1) if hit else (None, 0)
    inc = [w for w in q.lower().split() if not w.startswith("-")]
    exc = [w[1:] for w in q.lower().split() if w.startswith("-") and len(w) > 1]
    inc = [t for w in inc for t in words(w)]
    exc = [t for w in exc for t in words(w)]
    if not inc:
        return None, 0
    cands = []
    for f in index:
        if only and f["type"] != only or f["cov"] < 4:
            continue
        ws = f["words"]
        if all(has(ws, t) for t in inc) and not any(any(x.startswith(t) for x in ws) for t in exc):
            cands.append(f)
    if not cands:
        return None, 0
    top = max(c["cov"] for c in cands)
    cands.sort(key=lambda c: (c["brand"], c["cov"] < 0.8 * top, TYPE_RANK[c["type"]], len(c["words"]), len(c["desc"])))
    return cands[0], len(cands)


def resolve(index, spec: str):
    """Колонка usda целиком → ([(продукт, доля), …], сколько кандидатов) или (None, …).
    «кофе @0.97 & сахар @0.03» — смесь в заданных долях; без «@» доли равные."""
    for alt in spec.lstrip("!").split("|"):
        alt = alt.strip()
        if alt == "-":
            return None, "-"
        picked, n = [], 0
        for part in alt.split("&"):
            m = re.search(r"@\s*([\d.]+)\s*$", part)
            w = float(m.group(1)) if m else None
            f, k = search(index, part[:m.start()] if m else part)
            if f is None:
                picked = None
                break
            picked.append([f, w])
            n += k
        if picked:
            given = sum(w for _, w in picked if w is not None)
            free = [p for p in picked if p[1] is None]
            for p in free:
                p[1] = max(0.0, 1 - given) / len(free)
            total = sum(w for _, w in picked) or 1
            return [(f, w / total) for f, w in picked], n
    return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.get("USDA_DIR", r"C:\projects\kaggle\usda"),
                    help="папка с архивами FoodData_Central_*_csv_*.zip")
    ap.add_argument("--map", default="usda_map.csv")
    ap.add_argument("--data", help="CSV датасета чеков — посчитать покрытие по товарам")
    ap.add_argument("--out", default="nutrients_usda.csv")
    args = ap.parse_args()

    zips = sorted(glob.glob(os.path.join(args.dir, "FoodData_Central_*.zip")))
    if not zips:
        raise SystemExit(f"в {args.dir} нет архивов FoodData_Central_*.zip (укажи папку: --dir ...)")
    print("читаю USDA FoodData Central:")
    index, release = load(zips)

    mp = pd.read_csv(args.map, encoding="utf-8-sig", dtype=str).fillna("")
    out, review, unmatched = [], [], []
    for _, r in mp.iterrows():
        cat, grp, spec = r["Категория"].strip(), r["группа"], r["usda"].strip()
        if not spec or spec == "-" or NON_FOOD.match(cat):
            continue
        row = {"Категория": cat, "группа": grp}
        if spec == "=0":
            row.update({col: 0.0 for col, _ in NUTR.values()})
            row.update({"источник": "0 — не считаем (сухой чай/кофе, вода, подсластитель)", "fdc_id": "", "usda_описание": ""})
            out.append(row)
            continue
        picked, n = resolve(index, spec)
        if picked is None:
            if n != "-":
                unmatched.append({"Категория": cat, "usda": spec})
            continue
        for key, (col, _) in NUTR.items():
            vs = [f["nutr"][key] * w for f, w in picked if f["nutr"][key] is not None]
            row[col] = round(sum(vs), 3) if len(vs) == len(picked) else None
        kinds = sorted({TYPES_LABEL[f["type"]] + (f" {release.get(f['type'], '')}".rstrip()) for f, _ in picked})
        row["источник"] = "USDA FoodData Central, " + ", ".join(kinds) + " (CC0)"
        row["fdc_id"] = " & ".join(str(f["fdc_id"]) for f, _ in picked)
        mix = len(picked) > 1
        row["usda_описание"] = " & ".join((f"{w:.0%} " if mix else "") + f["desc"] for f, w in picked)
        row["принудительно"] = 1 if spec.startswith("!") else ""
        kinds_f = {fortification(f["desc"]) for f, _ in picked} - {""}
        row["обогащение"] = next((k for k in ("хлопья", "зерно", "витамины") if k in kinds_f), "")
        out.append(row)
        review.append({"Категория": cat, "взято из USDA": row["usda_описание"], "fdc_id": row["fdc_id"],
                       "обогащение": row["обогащение"],
                       "ккал": row["ккал"], "белки": row["белки"], "жиры": row["жиры"], "углеводы": row["углеводы"],
                       "кандидатов": n, "запрос": spec})

    cols = ["Категория", "группа"] + [c for c, _ in NUTR.values()] + ["источник", "fdc_id", "usda_описание", "принудительно", "обогащение"]
    pd.DataFrame(out, columns=cols).to_csv(args.out, index=False, encoding="utf-8-sig")
    pd.DataFrame(review).to_csv("usda_review.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(unmatched, columns=["Категория", "usda"]).to_csv("usda_unmatched.csv", index=False, encoding="utf-8-sig")

    food_cats = [c for c in mp["Категория"] if not NON_FOOD.match(c)]
    kept = int((mp["usda"].str.strip() == "-").sum())
    print(f"\n{args.out}: {len(out)} из {len(food_cats)} пищевых категорий"
          f" (из USDA {len(review)}, нули {len(out) - len(review)})")
    print(f"  остаются на справочнике Тутельяна: {kept}  (в usda_map.csv стоит «-»)")
    print(f"  запрос не нашёл продукт: {len(unmatched)} → usda_unmatched.csv")
    for u in unmatched[:10]:
        print(f"    {u['Категория']}  ←  {u['usda']}")
    print("  что взято — usda_review.csv (проверь глазами, особенно где «кандидатов» много)")
    if out:
        df = pd.DataFrame(out)
        print("заполненность: " + ", ".join(f"{c} {df[c].notna().mean():.0%}" for c, _ in NUTR.values()))
    if args.data:
        dd = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_excel(args.data)
        freq = Counter(dd["Категория"].astype(str).str.strip())
        total = sum(v for k, v in freq.items() if not NON_FOOD.match(k))
        usda = sum(freq.get(r["Категория"], 0) for r in review)
        print(f"покрытие по товарам в чеках: USDA {usda / total:.0%}, вместе с нулями "
              f"{sum(freq.get(r['Категория'], 0) for r in out) / total:.0%}")


TYPES_LABEL = {code: label for code, label in TYPES.values()}

if __name__ == "__main__":
    main()
