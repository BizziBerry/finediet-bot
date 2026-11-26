"""Профиль и покупки пользователей в SQLite (файл finediet.db не попадает в git)."""
import json
import sqlite3
import time

from receipt import PET_CATEGORY, is_pet_food


def _fix(row: dict) -> dict:
    """Правила поверх модели применяются и к уже сохранённым покупкам."""
    if is_pet_food(row["name"]):
        row["category"], row["confidence"] = PET_CATEGORY, 1.0
    return row


class Storage:
    def __init__(self, path: str = "finediet.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (user_id INTEGER PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS purchases (
                user_id INTEGER, ts REAL, receipt_date TEXT, name TEXT, category TEXT, confidence REAL,
                price REAL, quantity REAL, total REAL, grams REAL);
            CREATE INDEX IF NOT EXISTS ix_purchases_user ON purchases(user_id, ts);
            CREATE TABLE IF NOT EXISTS receipts (user_id INTEGER, key TEXT, PRIMARY KEY (user_id, key));
        """)

    def profile(self, user_id: int) -> dict:
        row = self.db.execute("SELECT data FROM profiles WHERE user_id=?", (user_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def save_profile(self, user_id: int, data: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO profiles VALUES (?, ?)", (user_id, json.dumps(data, ensure_ascii=False)))
        self.db.commit()

    def add_items(self, user_id: int, items: list[dict], receipt_date: str | None) -> bool:
        """Сохраняет покупки; False, если этот чек уже загружали."""
        key = f"{receipt_date}|{round(sum(i['total'] for i in items), 2)}|{len(items)}"
        try:
            self.db.execute("INSERT INTO receipts VALUES (?, ?)", (user_id, key))
        except sqlite3.IntegrityError:
            return False
        now = time.time()
        self.db.executemany(
            "INSERT INTO purchases VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(user_id, now, receipt_date, i["name"], i["category"], i["confidence"], i["price"],
              i["quantity"], i["total"], i["grams"]) for i in items])
        self.db.commit()
        return True

    def items(self, user_id: int, days: int = 90) -> list[dict]:
        rows = self.db.execute(
            "SELECT name, category, confidence, price, quantity, total, grams FROM purchases "
            "WHERE user_id=? AND ts>=? ORDER BY ts", (user_id, time.time() - days * 86400)).fetchall()
        keys = ["name", "category", "confidence", "price", "quantity", "total", "grams"]
        return [_fix(dict(zip(keys, r))) for r in rows]

    def last_receipt(self, user_id: int) -> list[dict]:
        row = self.db.execute("SELECT MAX(ts) FROM purchases WHERE user_id=?", (user_id,)).fetchone()
        if not row or row[0] is None:
            return []
        rows = self.db.execute(
            "SELECT name, category, confidence, price, quantity, total, grams FROM purchases "
            "WHERE user_id=? AND ts=?", (user_id, row[0])).fetchall()
        keys = ["name", "category", "confidence", "price", "quantity", "total", "grams"]
        return [_fix(dict(zip(keys, r))) for r in rows]

    def reclassify(self, predict) -> int:
        """Заново определяет категории всех сохранённых покупок (после улучшения классификатора)."""
        rows = self.db.execute("SELECT rowid, name, category FROM purchases").fetchall()
        if not rows:
            return 0
        preds = predict([r[1] for r in rows])
        changed = [(cat, conf, rid) for (rid, _, old), (cat, conf) in zip(rows, preds) if cat != old]
        self.db.executemany("UPDATE purchases SET category=?, confidence=? WHERE rowid=?", changed)
        self.db.commit()
        return len(changed)

    def forget(self, user_id: int) -> None:
        self.db.execute("DELETE FROM purchases WHERE user_id=?", (user_id,))
        self.db.execute("DELETE FROM profiles WHERE user_id=?", (user_id,))
        self.db.execute("DELETE FROM receipts WHERE user_id=?", (user_id,))
        self.db.commit()
