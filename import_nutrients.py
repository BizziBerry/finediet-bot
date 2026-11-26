"""Переносит химсостав из Excel (справочник по вкладкам) в nutrients.csv для бота.

    python import_nutrients.py "C:\\projects\\kaggle\\purchase\\nutrition.xlsx"

Что делает:
  1. проходит все вкладки; в каждой сам находит заголовок (в том числе в две строки: «Бел» / «гр»)
     и колонки ккал, белки, жиры, углеводы, пищевые волокна, сахара (МДС), натрий;
  2. сопоставляет категории модели со строками Excel:
       по коду справочника («абрикосы 9.1.1.1» ↔ «Абрикосы 9.1.1.1») →
       по точному названию →
       по похожему названию, но только внутри той же группы справочника (первая цифра кода);
  3. пишет nutrients.csv (для бота), nutrients_review.csv (похожие пары — проверить глазами)
     и nutrients_missing.csv (не нашлось).
"""
import argparse
import os
import re
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from model import ReceiptClassifier  # noqa: F401

# подстроки в длинных названиях колонок
NUTR = {
    "ккал": ["ккал", "калорийн", "энергетическая", "kcal"],
    "белки": ["белк", "белок", "protein"],
    "жиры": ["жиры", "жир,", "жир ", "fat"],
    "углеводы": ["углевод", "carb"],
    "пищевые волокна": ["волокн", "клетчат", "fiber"],
    "сахар": ["сахар", "дисахар", "sugar"],
    "натрий": ["натрий", "sodium"],
}
# сокращения справочника Тутельяна: первое слово заголовка («Бел гр», «ЭЦ Ккал», «ПВ гр», «МДС гр», «Na мг»)
ABBR = {"ккал": "эц", "белки": "бел", "жиры": "жир", "углеводы": "угл",
        "пищевые волокна": "пв", "сахар": "мдс", "натрий": "na"}
# витамины и минералы: (как узнать по первому слову заголовка, в каких единицах хранить)
MICRO = {
    "кальций, мг": ({"са"}, "мг"), "железо, мг": ({"fe", "железо"}, "мг"), "магний, мг": ({"mg", "магний"}, "мг"),
    "калий, мг": ({"к", "калий"}, "мг"), "фосфор, мг": ({"p", "фосфор"}, "мг"),
    "витамин a, мкг": ({"рэ"}, "мкг"), "витамин e, мг": ({"тэ"}, "мг"), "витамин b1, мг": ({"b1", "в1"}, "мг"),
    "витамин b2, мг": ({"b2", "в2"}, "мг"), "витамин pp, мг": ({"pp", "рр"}, "мг"), "витамин c, мг": ({"с"}, "мг"),
}
# цинк, селен, йод, фолаты, B12, D в таблице справочника — средние по группе (1–8 значений на лист),
# поэтому по продуктам их не берём
# максимум на 100 г, который реально встречается в продуктах (с запасом: печень, шиповник, ламинария,
# рыбий жир проходят). Всё, что выше, — ошибка ввода или единиц в Excel, такие значения не берём.
PLAUSIBLE_MAX = {
    "кальций, мг": 2500, "железо, мг": 50, "магний, мг": 800, "калий, мг": 4000, "фосфор, мг": 2500,
    "витамин a, мкг": 30000, "витамин e, мг": 80, "витамин b1, мг": 5, "витамин b2, мг": 6,
    "витамин pp, мг": 60, "витамин c, мг": 1500, "ккал": 900, "белки": 100, "жиры": 100, "углеводы": 100,
    "пищевые волокна": 100, "сахар": 100, "натрий": 40000,
}
TO_MG = {"гр": 1000.0, "г": 1000.0, "мг": 1.0, "мкг": 0.001}
_CYR = str.maketrans("acekopxyACEKOPXYH", "асекорхуАСЕКОРХУН")


def cyr(w: str) -> str:
    """Латинские буквы-двойники → кириллица («Ca» и «Са» одинаковы)."""
    return w.translate(_CYR).lower()


def find_micro_cols(df: pd.DataFrame) -> dict:
    """{«кальций, мг»: (колонка, множитель для перевода в нужные единицы)}"""
    found = {}
    for key, (firsts, unit) in MICRO.items():
        targets = {cyr(f) for f in firsts}
        for c in df.columns:
            words = str(c).split("|", 1)[1].replace("nan", "").split()
            if not words or cyr(words[0]) not in targets or (len(words) > 1 and cyr(words[1]).startswith("ф")
                                                              and key.startswith("калий")):
                continue
            src_unit = words[-1].lower() if words[-1].lower() in TO_MG else unit
            if to_num(df[c]).notna().mean() > 0.3:
                found[key] = (c, TO_MG[src_unit] / TO_MG[unit])
                break
    return found


NOT_NUTR = ["кдж", "kj", "насыщ", "кислот", "жирн"]
AUTO_OK, REVIEW = 0.85, 0.50
CODE = re.compile(r"(?<![\d-])(\d{1,2}(?:\.\d{1,2}){1,4})\s*$")
NON_FOOD = {"несъедобное", "нераспознанные продукты"}
# группа продуктов — по листу справочника, с которого взят химсостав (для диаграмм)
SHEET_GROUP = {"1": "Молочное", "2": "Яйца", "3": "Мясо и птица", "4": "Рыба и морепродукты", "5": "Масла и жиры",
               "6": "Хлеб, крупы, макароны", "7": "Бобовые, орехи, семена", "8": "Овощи и грибы",
               "9": "Фрукты и ягоды", "10": "Сладости", "11": "Напитки", "12": "Соусы и специи"}


def group_of(rows) -> str:
    sheets = rows["sheet"].dropna().astype(str) if "sheet" in rows else []
    if len(sheets):
        m = re.match(r"\d+", sheets.mode().iloc[0].strip())
        if m:
            return SHEET_GROUP.get(m.group(0), "Другое")
    return "Другое"


def zero_group(category: str) -> str:
    return "Напитки" if re.search(r"вод[аы]|чай|кофе|цикорий|напит", category, re.I) else "Соусы и специи"


def is_non_food(category: str) -> bool:
    """Не еда или не распознано: химсостав не считаем и в покрытие не включаем."""
    c = norm(category)
    return c in NON_FOOD or c.startswith("нераспознанн") or c.startswith("корм для")


def code_of(s):
    m = CODE.search(str(s).strip())
    return m.group(1) if m else None


def strip_code(s) -> str:
    s = CODE.sub("", str(s).strip())
    return re.sub(r"\bпо\s+\d+-\d+\b", "", s)            # «по 1-310» — ссылка на рецептуру


def norm(s) -> str:
    s = strip_code(s).lower().replace("ё", "е")
    s = re.sub(r"[^\w%.,]+", " ", s)
    s = re.sub(r"\s+%", "%", s)
    return re.sub(r"\s+", " ", s).strip(" .,")


def stem(s) -> str:
    """«яйца куриные» и «яйцо куриное» → «яйц курин»."""
    return " ".join(re.sub(r"[аеиоуыэюяйь]+$", "", w) if len(w) > 4 else w for w in norm(s).split())


def to_num(col: pd.Series) -> pd.Series:
    s = col.astype(str).str.replace(",", ".", regex=False).str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def header_variants(raw: pd.DataFrame, h: int):
    """Заголовок из одной строки h или из двух строк h и h+1 («Бел» + «гр»)."""
    one = [str(x).strip() if pd.notna(x) else "" for x in raw.iloc[h]]
    yield one, h + 1
    if h + 1 < len(raw):
        two = [f"{a} {str(b).strip() if pd.notna(b) else ''}".strip() for a, b in zip(one, raw.iloc[h + 1])]
        yield two, h + 2


def find_nutrient_cols(df: pd.DataFrame) -> dict:
    found = {}
    for key in NUTR:
        for c in df.columns:
            label = str(c).split("|", 1)[1].lower()
            words = label.split()
            first = words[0] if words else ""
            hit = first == ABBR[key] or (len(first) > 3 and any(k in label + " " for k in NUTR[key]))
            if hit and not any(b in label for b in NOT_NUTR) and to_num(df[c]).notna().mean() > 0.3:
                found[key] = c
                break
    return found


def find_name_col(df: pd.DataFrame, codes: set, names: set, skip: set):
    best, best_score = None, 0.0
    for c in df.columns:
        if c in skip:
            continue
        vals = df[c].dropna().astype(str)
        if len(vals) == 0 or to_num(vals).notna().mean() > 0.5:
            continue
        code_share = vals.map(code_of).isin(codes).mean()
        name_share = vals.map(norm).isin(names).mean()
        score = 2 * code_share + name_share + 0.01 * (vals.str.len().mean() > 3)
        if score > best_score:
            best, best_score = c, score
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("excel")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--data", default=None,
                    help="датасет чеков с колонкой «Категория» — тогда пропуски сортируются по частоте в чеках")
    ap.add_argument("--manual", default="nutrients_manual.csv",
                    help="ручные пары: «Категория»,«Взять из» (название или код строки Excel; «-» — не брать)")
    args = ap.parse_args()

    classes = [str(c) for c in joblib.load(args.model).classes_]
    if os.path.exists("category_rules.csv"):                  # категории из правил, которых нет у модели
        extra = pd.read_csv("category_rules.csv", encoding="utf-8-sig", dtype=str).fillna("").iloc[:, 1]
        classes += [c.strip() for c in extra if c.strip() and c.strip() not in classes]
    class_codes = {code_of(c) for c in classes} - {None}
    class_names = {norm(c) for c in classes}
    print(f"категорий в модели: {len(classes)}, из них с кодом справочника: "
          f"{sum(code_of(c) is not None for c in classes)}")

    parts = []
    for sname, raw in pd.read_excel(args.excel, sheet_name=None, header=None).items():
        best = None
        for h in range(min(15, len(raw) - 1)):
            for header, start in header_variants(raw, h):
                df = raw.iloc[start:].copy()
                df.columns = [f"{i}|{x}" for i, x in enumerate(header)]   # индекс — чтобы имена были уникальны
                nut = find_nutrient_cols(df)
                if "ккал" not in nut:
                    continue
                units = sum(str(c).split()[-1].lower() in TO_MG for c in df.columns if str(c).split())
                score = len(nut) + 0.01 * units          # при равенстве — вариант, где видны единицы (мг/мкг)
                if best is None or score > best[0]:
                    best = (score, df, nut)
        if best is None:
            print(f"лист {sname!r}: нет колонки калорий, пропускаю")
            continue
        _, df, nut = best
        name_col = find_name_col(df, class_codes, class_names, set(nut.values()))
        if name_col is None:
            print(f"лист {sname!r}: не нашла колонку с названиями, пропускаю")
            continue
        part = pd.DataFrame({"name": df[name_col].astype(str).str.strip()})
        for k, c in nut.items():
            part[k] = to_num(df[c])
        micro = find_micro_cols(df)
        for k, (c, factor) in micro.items():
            part[k] = to_num(df[c]) * factor
        # колонки, которые в этом листе почти везде совпадают друг с другом, — скопированы по ошибке
        mk = list(micro)
        dup = set()
        for i, a in enumerate(mk):
            for b in mk[i + 1:]:
                both = part[a].notna() & part[b].notna() & (part[a] != 0)
                if both.sum() >= 3 and (part.loc[both, a] == part.loc[both, b]).mean() > 0.5:
                    dup |= {a, b}
        for k in dup:
            part[k] = np.nan
        # в отдельной строке: 3+ витамина с одним и тем же заметным числом (> 5) — копипаст
        if mk:
            vals = part[mk].to_numpy(dtype=float)
            for r in range(len(vals)):
                row = vals[r]
                for v, cnt in Counter(row[(row != 0) & ~np.isnan(row)]).items():
                    if cnt >= 3 and v > 5:
                        row[row == v] = np.nan
            part[mk] = vals
        if dup:
            print(f"   лист {sname!r}: колонки совпадают между собой, не беру: {', '.join(sorted(dup))}")
        part = part[part["ккал"].notna() & (part["name"].str.len() > 1) & (part["name"] != "nan")
                    & ~part["name"].str.contains("удалить|убрать", case=False)]
        m = re.match(r"\d+", str(sname).strip())
        part["group"] = [(code_of(n) or "").split(".")[0] or (m.group(0) if m else "") for n in part["name"]]
        part["sheet"] = str(sname)
        print(f"лист {sname!r}: {len(part)} строк; " + ", ".join(
            f"{k} ← {str(v).split('|', 1)[1]!r}" for k, v in nut.items()) + f"; витаминов и минералов: {len(micro)}")
        parts.append(part)
    if not parts:
        sys.exit("Не нашла ни одного листа с калориями.")

    data = pd.concat(parts, ignore_index=True)
    nut_keys = [k for k in [*NUTR, *MICRO] if k in data.columns]
    removed = {}
    for k in nut_keys:
        bad = data[k] > PLAUSIBLE_MAX.get(k, np.inf)
        if bad.any():
            removed[k] = int(bad.sum())
            data.loc[bad, k] = np.nan
    if removed:
        print("\nнеправдоподобные значения убраны (ошибки в Excel): "
              + ", ".join(f"{k} — {v}" for k, v in removed.items()))
    data["code"] = data["name"].map(code_of)
    data["key"] = data["name"].map(norm)
    by_code = data.dropna(subset=["code"]).groupby("code")
    by_name = data.groupby("key")
    print(f"\nстрок с химсоставом: {len(data)}, из них с кодом: {data['code'].notna().sum()}")

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(
        [stem(c) for c in classes] + data["key"].map(stem).tolist())
    D = vec.transform(data["key"].map(stem))
    head = lambda t: (stem(t).split() or [""])[0][:5]      # главное слово: «креветка варёная» → «кревет»
    data_head = data["key"].map(head).to_numpy()

    manual = {}
    if os.path.exists(args.manual):
        m = pd.read_csv(args.manual, encoding="utf-8-sig", dtype=str).fillna("")
        manual = {norm(a): b.strip() for a, b in zip(m.iloc[:, 0], m.iloc[:, 1]) if a.strip()}
        print(f"ручных пар: {len(manual)} ({args.manual})")

    out, review, missing, how = [], [], [], Counter()
    for c in classes:
        code, key = code_of(c), norm(c)
        src, sim = None, 1.0
        if key in manual:
            target = manual[key]
            if target in ("-", ""):
                how["вручную: не брать"] += 1
                missing.append(c)
                continue
            if target == "0":                                  # вода, чай, жвачка: химсостав по сути нулевой
                how["вручную"] += 1
                out.append({"Категория": c, "группа": zero_group(c), **{k: 0.0 for k in nut_keys}})
                continue
            picked = []
            for part_t in [t.strip() for t in target.split(";") if t.strip()]:   # «код1; код2» — среднее
                tc, tk = code_of(part_t) or (part_t if CODE.fullmatch(part_t) else None), norm(part_t)
                if tc and tc in by_code.groups:
                    r_ = by_code.get_group(tc)
                elif tk in by_name.groups:
                    r_ = by_name.get_group(tk)
                else:
                    s_ = (vec.transform([stem(part_t)]) @ D.T).toarray()[0]
                    r_ = data.iloc[[int(s_.argmax())]]
                    print(f"  вручную: «{part_t}» нет в Excel дословно, взяла «{r_.iloc[0]['name']}»")
                if len(r_) > 1:                                    # у кода несколько строк — по названию
                    sims = (vec.transform([stem(part_t)]) @ vec.transform(r_["key"].map(stem)).T).toarray()[0]
                    r_ = r_.iloc[[int(sims.argmax())]]
                picked.append(r_.iloc[[0]])
            rows = pd.concat(picked)
            how["вручную"] += 1
            out.append({"Категория": c, "группа": group_of(rows), **{k: rows[k].mean() for k in nut_keys}})
            continue
        if code and code in by_code.groups:
            rows, src, method = by_code.get_group(code), None, "код"
            if len(rows) > 1 and rows[nut_keys].nunique().max() > 1:
                # один код у разных продуктов (опечатка в Excel) — берём строку с самым похожим названием
                sims = (vec.transform([stem(c)]) @ vec.transform(rows["key"].map(stem)).T).toarray()[0]
                rows = rows.iloc[[int(sims.argmax())]]
        elif key in by_name.groups:
            rows, method = by_name.get_group(key), "название"
        else:
            # похожее название — только в той же группе справочника
            grp = code.split(".")[0] if code else None
            mask = (data["group"] == grp).to_numpy() if grp else np.ones(len(data), bool)
            mask = mask & (data_head == head(c))           # первое слово должно совпадать
            if not mask.any():
                missing.append(c)
                continue
            s = (vec.transform([stem(c)]) @ D.T).toarray()[0]
            s[~mask] = -1
            j = int(s.argmax())
            sim = float(s[j])
            if sim < REVIEW:
                missing.append(c)
                continue
            rows, method, src = data.iloc[[j]], "похожее", data.iloc[j]["name"]
        how[method] += 1
        out.append({"Категория": c, "группа": group_of(rows), **{k: rows[k].median() for k in nut_keys}})
        if method == "похожее" and sim < AUTO_OK:
            review.append({"Категория модели": c, "Взято из строки Excel": src, "похожесть": round(sim, 2)})

    # USDA FoodData Central (CC0) — основной слой; Тутельян заполняет то, чего в USDA нет.
    # Группу оставляем из справочника, если категория там была: от неё зависят правила тарелки в плане покупок.
    SPARSE_COLS = {"цинк, мг", "селен, мкг", "витамин b6, мг", "фолаты, мкг", "витамин b12, мкг", "витамин d, мкг"}
    if os.path.exists("nutrients_usda.csv"):
        us = pd.read_csv("nutrients_usda.csv", encoding="utf-8-sig")
        extra_cols = [c for c in us.columns if c not in ("Категория", "группа", "источник", "fdc_id", "usda_описание", "принудительно", "обогащение")]
        by_cat = {o["Категория"]: o for o in out}
        known = {norm(c): c for c in classes}
        diff, n_us, n_fort = [], 0, 0
        for _, row in us.iterrows():
            target = known.get(norm(str(row["Категория"])))
            if not target:
                continue
            old = by_cat.get(target)
            forced = str(row.get("принудительно", "")).strip() in ("1", "1.0")
            if old and not forced and pd.notna(old.get("ккал")) and pd.notna(row["ккал"]) and max(old["ккал"], row["ккал"]) > 30:
                ratio = (row["ккал"] + 1) / (old["ккал"] + 1)
                if ratio > 1.5 or ratio < 1 / 1.5:
                    # скорее всего неудачная пара (сухое ↔ готовое, другой продукт) — пока оставляем справочник
                    diff.append({"Категория": target, "ккал Тутельян": round(old["ккал"]), "ккал USDA": round(row["ккал"]),
                                 "USDA": row.get("usda_описание", "")})
                    continue
            new = dict(old) if old else {"Категория": target, "группа": row.get("группа", "Другое")}
            fort = str(row.get("обогащение", "") or "").strip()
            fort = "" if fort == "nan" else fort
            # американское обогащение: эти значения из USDA не берём — у Тутельяна (если есть) необогащённые
            skip = set()
            if fort == "витамины":
                skip = {"витамин a, мкг", "витамин d, мкг", "витамин c, мг", "кальций, мг"}
                n_fort += 1
            elif fort:
                skip = {"кальций, мг", "железо, мг", "витамин b1, мг", "витамин b2, мг", "витамин pp, мг", "фолаты, мкг"}
                if fort == "хлопья":
                    skip |= {"витамин a, мкг", "витамин e, мг", "витамин c, мг", "цинк, мг", "витамин b6, мг",
                             "витамин b12, мкг", "витамин d, мкг"}
                n_fort += 1
            for c in extra_cols:
                if c in skip:
                    if not old or pd.isna(old.get(c)):
                        # справочника нет: редкие нутриенты — «неизвестно»; основные — всё же USDA (лучше, чем 0)
                        new[c] = None if c in SPARSE_COLS else (row[c] if pd.notna(row[c]) else None)
                    continue
                if pd.notna(row[c]):
                    new[c] = row[c]
            new["источник"] = row.get("источник", "USDA FoodData Central")
            new["fdc_id"] = row.get("fdc_id", "")
            by_cat[target] = new
            n_us += 1
        out = list(by_cat.values())
        missing = [m for m in missing if m not in by_cat]
        pd.DataFrame(diff, columns=["Категория", "ккал Тутельян", "ккал USDA", "USDA"]).to_csv(
            "usda_vs_tutelyan.csv", index=False, encoding="utf-8-sig")
        print(f"из nutrients_usda.csv (USDA FoodData Central): {n_us}; калории расходятся в 1,5+ раза"
              f" — оставлен справочник: {len(diff)} → usda_vs_tutelyan.csv")
        print(f"  обогащено по-американски (мука, хлеб, хлопья, молоко с витамином D…) — добавленные витамины и минералы не берём: {n_fort}")
        nut_keys = nut_keys + [c for c in extra_cols if c not in nut_keys]

    # продукты, которых нет в справочнике 2012 года (авокадо…), — из открытых источников, с указанием источника
    if os.path.exists("nutrients_extra.csv"):
        ex = pd.read_csv("nutrients_extra.csv", encoding="utf-8-sig")
        known = {norm(c) for c in classes}
        added = 0
        for _, row in ex.iterrows():
            cat = str(row["Категория"]).strip()
            if norm(cat) not in known:
                continue
            target = next(c for c in classes if norm(c) == norm(cat))
            out = [o for o in out if o["Категория"] != target]
            out.append({"Категория": target, "группа": row.get("группа", "Другое"),
                        **{k: row[k] for k in nut_keys if k in row and pd.notna(row[k])}})
            missing = [m for m in missing if m != target]
            added += 1
        print(f"из nutrients_extra.csv (открытые источники): {added}")
    pd.DataFrame(out).to_csv("nutrients.csv", index=False, encoding="utf-8-sig")
    rv = pd.DataFrame(review, columns=["Категория модели", "Взято из строки Excel", "похожесть"])
    rv.sort_values("похожесть").to_csv("nutrients_review.csv", index=False, encoding="utf-8-sig")
    freq = Counter()
    if args.data:
        dd = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_excel(args.data)
        freq = Counter(dd["Категория"].astype(str).map(norm))
    missing = [c for c in missing if not is_non_food(c)]
    miss = pd.DataFrame({"Категория": missing, "Взять из": "", "товаров в чеках": [freq.get(norm(c), 0) for c in missing]})
    miss = miss.sort_values("товаров в чеках", ascending=False)
    miss.to_csv("nutrients_missing.csv", index=False, encoding="utf-8-sig")

    n = len(classes)
    print(f"\nnutrients.csv: {len(out)} из {n} категорий ({len(out) / n:.0%})")
    print(f"  по коду: {how['код']}, по точному названию: {how['название']}, по похожему: {how['похожее']}"
          + (f", вручную: {how['вручную']}" if how["вручную"] else "")
          + (f", вручную исключено: {how['вручную: не брать']}" if how["вручную: не брать"] else ""))
    print(f"  проверь глазами: {len(review)} → nutrients_review.csv")
    print(f"  не нашлось: {len(missing)} → nutrients_missing.csv")
    for r in sorted(review, key=lambda r: r["похожесть"])[:8]:
        print(f"    {r['похожесть']:.2f}  {r['Категория модели']}  ←  {r['Взято из строки Excel']}")
    if freq:
        total = sum(v for k, v in freq.items() if not is_non_food(k))
        covered = sum(freq.get(norm(r["Категория"]), 0) for r in out)
        print(f"\nпокрытие по товарам в чеках: {covered / total:.0%} (а по категориям {len(out) / n:.0%})")
        print("самые частые категории без химсостава (впиши «Взять из» и сохрани как nutrients_manual.csv):")
        for _, r in miss.head(15).iterrows():
            print(f"    {r['товаров в чеках']:>6}  {r['Категория']}")
    if out:
        cov = pd.DataFrame(out)[nut_keys].notna().mean()
        print("заполненность колонок: " + ", ".join(f"{k} {v:.0%}" for k, v in cov.items()))


if __name__ == "__main__":
    main()
