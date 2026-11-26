"""Разбор JSON-чека ФНС и оценка веса каждой позиции в граммах."""
import re
from dataclasses import dataclass, asdict

# «930мл», «0,5 л», «1кг», «400 гр», «6х0.5л», «2*450г»
_UNIT_G = {"кг": 1000.0, "г": 1.0, "гр": 1.0, "л": 1000.0, "мл": 1.0}
_NUM = r"(\d+(?:[.,]\d+)?)"
_RE_MULTI = re.compile(rf"(\d+)\s*[xх×*]\s*{_NUM}\s*(кг|гр|г|мл|л)(?![а-яa-z])", re.I)
_RE_SINGLE = re.compile(rf"{_NUM}\s*(кг|гр|г|мл|л)(?![а-яa-z])", re.I)
_RE_EGGS = re.compile(r"(\d+)\s*шт", re.I)


def pack_grams(name: str) -> float | None:
    """Вес/объём упаковки из названия (жидкости: 1 мл ≈ 1 г). None, если в названии его нет."""
    s = name.lower()
    m = _RE_MULTI.search(s)
    if m:
        return int(m.group(1)) * float(m.group(2).replace(",", ".")) * _UNIT_G[m.group(3).lower()]
    m = _RE_EGGS.search(s)
    if m and ("яйц" in s or "яйко" in s):
        return int(m.group(1)) * 55.0                # среднее яйцо С1 без скорлупы ≈ 55 г
    found = _RE_SINGLE.findall(s)
    if found:
        num, unit = found[-1]                      # последнее упоминание обычно и есть вес упаковки
        g = float(num.replace(",", ".")) * _UNIT_G[unit.lower()]
        return g if 5 <= g <= 20000 else None
    return None


# марки кормов для животных: классификатор путает «Cesar паштет с говядиной» с мясным паштетом
PET_FOOD = re.compile(r"(?<![а-яa-z])(cesar|whiskas|felix|sheba|pedigree|purina|kitekat|friskies|royal\s*canin|"
                      r"chappi|darling|perfect\s*fit|pro\s*plan|hill'?s|acana|monge|meglium|"
                      r"мнямс|китекат|вискас|феликс|шеба|педигри|чаппи|фрискис|корм\s+для\s+(кош|соб|животн|птиц|рыб))",
                      re.I)
PET_CATEGORY = "корм для животных"


def is_pet_food(name: str) -> bool:
    return bool(PET_FOOD.search(name))


@dataclass
class Item:
    name: str
    price: float          # ₽ за единицу
    quantity: float
    total: float          # ₽ за позицию
    grams: float | None   # оценка веса позиции, None - неизвестно
    category: str = ""
    confidence: float = 0.0

    def to_dict(self):
        return asdict(self)


def item_grams(name: str, quantity: float, measure=None) -> float | None:
    # мера количества 11 = килограмм (ФФД 1.2); дробное количество - тоже весовой товар
    if measure == 11 or (quantity and abs(quantity - round(quantity)) > 1e-6):
        return quantity * 1000.0
    g = pack_grams(name)
    return g * quantity if g else None


def parse_receipt(data) -> tuple[list[Item], str | None]:
    """Возвращает позиции и дату чека. Принимает выгрузку «Проверки чеков» ФНС (или список таких чеков)."""
    if isinstance(data, list):
        data = data[0] if data else {}
    rec = data.get("ticket", {}).get("document", {}).get("receipt") or data.get("receipt") or data
    items = []
    for it in rec.get("items", []) or []:
        name = str(it.get("name") or "").strip()
        if not name:
            continue
        q = float(it.get("quantity") or 1)
        price = float(it.get("price") or 0) / 100      # в ФНС суммы в копейках
        total = float(it.get("sum") or 0) / 100 or price * q
        items.append(Item(name, price, q, total, item_grams(name, q, it.get("itemsQuantityMeasure"))))
    return items, rec.get("dateTime")
