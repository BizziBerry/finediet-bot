"""Проверка nutrients.csv: максимальные значения каждого витамина/минерала (ищем ошибки единиц).
    python check_nutrients.py
    python check_nutrients.py фасоль хлеб        # показать конкретные категории
"""
import sys

import pandas as pd

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 45)
d = pd.read_csv("nutrients.csv", encoding="utf-8-sig")
micro = [c for c in d.columns if "," in c and c not in ("Категория",)]

if len(sys.argv) > 1:
    mask = d["Категория"].str.lower().str.contains("|".join(a.lower() for a in sys.argv[1:]), regex=True)
    print(d.loc[mask, ["Категория", *micro]].to_string(index=False))
else:
    for c in micro:
        top = d.nlargest(4, c)[["Категория", c]]
        print(f"\n{c}: медиана {d[c].median():.2f}, максимум:")
        for _, r in top.iterrows():
            print(f"    {r[c]:>10.2f}  {r['Категория']}")
