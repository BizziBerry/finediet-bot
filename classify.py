"""Классификация названий из чека: чистка шума + модель + правила-страховки.

    РН ГП ТЫКВА ВЕС 776  →  «тыква вес»  →  модель  →  если «несъедобное» неуверенно, а в названии
    есть слово-продукт из справочника («тыква») — берём лучшую пищевую категорию.
"""
import os
import re

import numpy as np
import pandas as pd

from nutrition import is_non_food, pretty
from receipt import PET_CATEGORY, is_pet_food

MISMATCH_BELOW = 0.80       # ответ модели не содержит главного слова названия — поправляем, если она не очень уверена
RESCUE_BELOW = 0.60          # «не еда» с уверенностью ниже этой — проверяем, нет ли в названии продукта
# признаки непродуктовых товаров: «мыло с молоком», «шампунь с мёдом» остаются «не едой»
NON_FOOD_MARKERS = re.compile(r"мыло|шампун|бальзам д|кондиционер|крем|гель|скраб|маска д|лосьон|дезодор|салфет|"
                              r"пакет|корм|наполнит|игрушк|свеч|освежит|ароматиз|средство|порошок д|чистящ|моющ|"
                              r"зубн|ополаскив|прокладк|подгузн|бумаг|губк|перчат|фольг|плёнк|пленк|батарейк|"
                              r"лампочк|косметич|туалетн|влажн|ватн|памперс|станок|бритв|носки|колготк", re.I)
GENERIC = {"для", "или", "без", "при", "все", "так", "вес", "шт", "муки", "сорт", "кат", "продукт", "продукты", "изделия", "изделие", "другой", "другие", "разный", "разные", "ассорти",
           "смесь", "набор", "напиток", "напитки", "консервы", "нераспознанные", "несъедобное", "добавками",
           "жирности", "приготовления", "быстрого", "сухой", "сухая", "свежий", "свежая", "мытая", "весовой"}


def load_rules(path: str = os.environ.get("CATEGORY_RULES", "category_rules.csv")) -> list:
    """Правила «шаблон в названии → категория». Срабатывают раньше модели и могут назначать категории,
    которых у модели нет, но которые есть в справочнике («тыква 9.3.3»)."""
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return [(re.compile(p, re.I), c.strip()) for p, c in zip(df.iloc[:, 0], df.iloc[:, 1]) if p.strip() and c.strip()]


def clean_name(name: str) -> str:
    s = str(name)
    s = re.sub(r"^(?:[А-ЯЁA-Z]{1,3}\.?\s+){1,3}(?=[А-ЯЁA-Zа-яё]{3,})", "", s)    # «РН ГП ТЫКВА» → «ТЫКВА»
    s = re.sub(r"(?<![\d.,])\b\d{3,}\b(?!\s*(?:г|гр|кг|мл|л|шт|%|x|х))", " ", s, flags=re.I)  # «776», артикулы
    return re.sub(r"\s+", " ", s).strip()


class Classifier:
    def __init__(self, model, table=None):
        self.model = model
        self.rules = load_rules()
        self.classes = np.asarray([str(c) for c in model.classes_])
        self.food_mask = np.array([not is_non_food(c) for c in self.classes])
        words = set()
        for c in (table.rows if table else self.classes):
            for w in re.findall(r"[а-яё]{3,}", pretty(c).lower()):
                if w not in GENERIC:
                    words.add(w[:5])
        self.food_stems = words

    def _food_word(self, name: str) -> bool:
        return any(w[:5] in self.food_stems for w in re.findall(r"[а-яё]{3,}", name.lower()))

    def predict(self, names: list[str]) -> list[tuple[str, float]]:
        """[(категория, уверенность)] для каждого названия."""
        cleaned = [clean_name(n) for n in names]
        proba = self.model.predict_proba(cleaned)
        out = []
        for name, cl, p in zip(names, cleaned, proba):
            i = int(p.argmax())
            cat, conf = str(self.classes[i]), float(p[i])
            rule = next((c for rx, c in self.rules if rx.search(name.lower())), None)
            if is_pet_food(name):                                   # корм для животных — правило сильнее модели
                cat, conf = PET_CATEGORY, 1.0
            elif rule:                                              # явное правило из category_rules.csv
                cat, conf = rule, 1.0
            elif is_non_food(cat) and self._food_word(cl) and not NON_FOOD_MARKERS.search(name):
                stems = {w[:5] for w in re.findall(r"[а-яё]{3,}", cl.lower())} & self.food_stems
                # категории, в названии которых есть это же слово («тыква» → «тыква 9.3.3»): прямое совпадение
                # с продуктом из справочника сильнее модели, даже уверенной
                has = np.array([any(st in c.lower() for st in stems) for c in self.classes]) & self.food_mask
                if has.any():
                    j = int(np.where(has, p, -1).argmax())
                    cat, conf = str(self.classes[j]), max(float(p[j]), 0.5)   # слово-продукт в названии — сильный признак
                elif conf < RESCUE_BELOW:                                       # только похожее слово — если модель сомневается
                    j = int(np.where(self.food_mask, p, -1).argmax())
                    cat, conf = str(self.classes[j]), float(p[j])
            elif conf < MISMATCH_BELOW and not is_non_food(cat) and not NON_FOOD_MARKERS.search(name):
                first = self._first_food_stem(cl)
                if first and first not in cat.lower():
                    # «САЛАТ "АФИЦИОН" ВЫБОРЖЕЦ» → «огурцы», «ШИПОВНИК … РАСТВОРИМЫЙ» → «кофе растворимый»:
                    # главное слово названия в ответе модели отсутствует → категории с этим словом;
                    # из правдоподобных берём самую базовую (самое короткое название: «салат», а не «салат из капусты»)
                    has = np.array([first in c.lower() for c in self.classes]) & self.food_mask
                    if has.any():
                        best = float(np.where(has, p, 0).max())
                        cands = [k for k in np.where(has)[0] if p[k] >= 0.3 * best]
                        j = min(cands, key=lambda k: (len(pretty(self.classes[k])), -p[k]))
                        cat, conf = str(self.classes[j]), 0.5
            out.append((cat, conf))
        return out

    def _first_food_stem(self, name: str) -> str | None:
        for w in re.findall(r"[а-яё]{3,}", name.lower()):
            if w[:5] in self.food_stems:
                return w[:5]
        return None

    def top(self, name: str, k: int = 3) -> list[tuple[str, float]]:
        p = self.model.predict_proba([clean_name(name)])[0]
        order = p.argsort()[::-1][:k]
        return [(str(self.classes[i]), float(p[i])) for i in order]
