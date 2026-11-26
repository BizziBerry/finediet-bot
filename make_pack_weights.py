"""Типичный вес упаковки по категории — из датасета чеков (колонки «Вес» и «ЕИ»).
Нужен, когда в названии товара нет веса («Пельмени с грудкой цыпленка», «Омлет с томатами»).

    python make_pack_weights.py "C:\\projects\\kaggle\\purchase\\df_cleaned_filtered.xlsx"
"""
import sys

import pandas as pd

TO_G = {"г": 1, "гр": 1, "кг": 1000, "мл": 1, "л": 1000}

path = sys.argv[1] if len(sys.argv) > 1 else "df_cleaned_filtered.xlsx"
df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
unit = df["ЕИ"].astype(str).str.strip().str.lower().str.rstrip(".")
grams = pd.to_numeric(df["Вес"], errors="coerce") * unit.map(TO_G)
d = pd.DataFrame({"Категория": df["Категория"].astype(str), "г": grams}).dropna()
d = d[(d["г"] >= 5) & (d["г"] <= 10000)]
w = d.groupby("Категория")["г"].agg(["median", "count"]).reset_index()
w = w[w["count"] >= 3].rename(columns={"median": "вес упаковки, г", "count": "товаров"})
w.to_csv("pack_weights.csv", index=False, encoding="utf-8-sig")
print(f"pack_weights.csv: {len(w)} категорий (из {df['Категория'].nunique()}), "
      f"по {len(d)} товарам с весом. Примеры:")
print(w.sort_values("товаров", ascending=False).head(10).to_string(index=False))
