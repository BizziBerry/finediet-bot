"""Какие категории модели содержат слово и есть ли для них химсостав.
    python find_category.py салат тыкв инжир шиповн семеч
"""
import sys

import joblib
import pandas as pd

from model import ReceiptClassifier  # noqa: F401

classes = [str(c) for c in joblib.load("model.joblib").classes_]
try:
    known = set(pd.read_csv("nutrients.csv", encoding="utf-8-sig").iloc[:, 0].astype(str))
except FileNotFoundError:
    known = set()
for word in sys.argv[1:]:
    hits = [c for c in classes if word.lower() in c.lower()]
    print(f"\n«{word}»: {len(hits)}")
    for c in sorted(hits, key=len):
        print(f"   {'✓' if c in known else '·'} {c}")
