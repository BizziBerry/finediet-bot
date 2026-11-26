"""Создаёт nutrients_template.csv: все категории модели и пустые колонки химсостава (на 100 г).
Заполни из справочника (или таблицы коллег) и сохрани как nutrients.csv рядом с ботом.
Таблица в git не попадает: значения справочника защищены авторским правом."""
import joblib
import pandas as pd

from model import ReceiptClassifier  # noqa: F401

model = joblib.load("model.joblib")
cols = ["Категория", "ккал", "белки", "жиры", "углеводы", "пищевые волокна", "сахар", "натрий"]
df = pd.DataFrame({"Категория": [str(c) for c in model.classes_]}).reindex(columns=cols)
df.to_csv("nutrients_template.csv", index=False, encoding="utf-8-sig")
print(f"nutrients_template.csv: {len(df)} категорий. Обязательны «Категория» и «ккал», "
      "клетчатка/сахар/натрий - по желанию (натрий в мг).")
