"""Выгружает список категорий модели в classes.csv — для сопоставления с USDA FoodData Central.

    python dump_classes.py                      # model.joblib, nutrients.csv из .env/по умолчанию
    python dump_classes.py --data receipt_items.csv   # + сколько покупок в каждой категории

В файл попадают только названия категорий (без данных Тутельяна), его можно прислать в чат.
"""
import argparse
import csv
import os
from collections import Counter

import joblib

from model import ReceiptClassifier, clean  # noqa: F401  (нужно для загрузки model.joblib)
from nutrition import is_non_food

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "model.joblib"))
ap.add_argument("--data", help="CSV датасета чеков: посчитать частоту категорий")
ap.add_argument("--out", default="classes.csv")
args = ap.parse_args()

model = joblib.load(args.model)
classes = [str(c) for c in model.classes_]

freq = Counter()
if args.data:
    import pandas as pd
    df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_excel(args.data)
    col = next((c for c in df.columns if c.lower() in ("category", "категория", "class", "label")), df.columns[-1])
    freq = Counter(df[col].astype(str))

with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "food", "count"])
    for c in sorted(classes, key=lambda c: -freq.get(c, 0)):
        w.writerow([c, 0 if is_non_food(c) else 1, freq.get(c, "")])

print(f"{len(classes)} категорий → {args.out}")
