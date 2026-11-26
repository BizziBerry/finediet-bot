"""Приём чека в любом виде: JSON ФНС, PDF, фото/скан, Word, Excel/CSV, текст.

Схема: файл → текст → позиции.
  1. JSON ФНС разбирается напрямую (receipt.parse_receipt).
  2. PDF / Word / Excel / текст → извлекаем текст; фото и сканы → распознавание (Ollama vision или Tesseract).
  3. Текст → позиции правилами для известных раскладок чеков (ОФД, магазины).
  4. Если правила не нашли позиции или сумма не сходится с «ИТОГО», а LLM подключена, — позиции достаёт LLM.
"""
import base64
import io
import json
import logging
import os
import re

from receipt import Item, item_grams, parse_receipt

log = logging.getLogger("finediet.readers")

TEXT_EXT = {".txt", ".text", ".html", ".htm", ".md"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}
TABLE_EXT = {".xlsx", ".xls", ".csv"}

_M = r"\d{1,3}(?:[  ]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"   # 1 179,98 / 486.07 / 78
_Q = r"\d+(?:[.,]\d+)?"


def money(s: str) -> float:
    return float(re.sub(r"[  ]", "", s).replace(",", "."))


# ---------- 1. файл → текст ----------
def text_from_pdf(data: bytes) -> tuple[str, list]:
    """Текстовый слой PDF; если его нет (скан) — картинки страниц для распознавания."""
    import pdfplumber
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        text = "\n".join((p.extract_text(layout=True) or "") for p in pdf.pages)
        images = []
        if len(text.strip()) < 30:
            for p in pdf.pages[:3]:
                buf = io.BytesIO()
                p.to_image(resolution=250).original.save(buf, format="PNG")
                images.append(buf.getvalue())
    return text, images


def text_from_docx(data: bytes) -> str:
    import docx
    d = docx.Document(io.BytesIO(data))
    lines = [p.text for p in d.paragraphs]
    for t in d.tables:
        lines += ["    ".join(c.text for c in row.cells) for row in t.rows]
    return "\n".join(lines)


def text_from_table(data: bytes, ext: str) -> str:
    import pandas as pd
    df = pd.read_csv(io.BytesIO(data), header=None, sep=None, engine="python") if ext == ".csv" \
        else pd.read_excel(io.BytesIO(data), header=None)
    return "\n".join("    ".join(str(v) for v in row if str(v) != "nan") for row in df.itertuples(index=False))


def text_from_html(data: bytes) -> str:
    s = data.decode("utf-8", errors="ignore")
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", s)
    s = re.sub(r"(?i)</t[dh]>", "    ", s)
    return re.sub(r"<[^>]+>", " ", s).replace("&nbsp;", " ")


def ocr_tesseract(image: bytes) -> str | None:
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except ImportError:
        return None
    cmd = os.environ.get("TESSERACT_CMD") or (r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                                              if os.name == "nt" else None)
    if cmd and os.path.exists(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        img = ImageOps.grayscale(Image.open(io.BytesIO(image)))
        if img.width < 1200:                                  # мелкое фото — увеличим
            img = img.resize((img.width * 2, img.height * 2))
        return pytesseract.image_to_string(img, lang="rus+eng", config="--psm 6")
    except Exception as e:                                    # нет tesseract.exe или языка rus
        log.warning("tesseract: %s", e)
        return None


# ---------- 2. текст → позиции (правила) ----------
def _clean_name(s: str) -> str:
    s = re.sub(r"\[M\]|\[М\]|,?\s*шт\.?\s*шт\.?|^\s*\d+\.\s+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.;:-")
    return s


def parse_text(text: str) -> tuple[list[Item], dict]:
    """Позиции из текста чека. Понимает три частые раскладки; meta: дата, итог, какой шаблон сработал."""
    text = text.replace("–", "-").replace("−", "-").replace(" ", " ")
    lines = [l.rstrip() for l in text.split("\n")]
    meta = {"date": None, "total": None, "layout": None}
    m = re.search(r"(\d{2}\.\d{2}\.(?:\d{4}|\d{2}))[ ,]+(\d{1,2}:\d{2})", text)
    if m:
        d, t = m.groups()
        dd, mm, yy = d.split(".")
        meta["date"] = f"{'20' + yy if len(yy) == 2 else yy}-{mm}-{dd}T{t}"
    m = re.search(rf"(?:ИТОГО?|Итого|итого|ВСЕГО|Всего)\s*:?\s*[=]?\s*({_M})", text)
    if m:
        meta["total"] = money(m.group(1))

    items: list[Item] = []

    # A) «1. Название ... 1.0 x 78.00 ₽ = 78.00 ₽» — всё в одной строке (1-ofd и др.)
    # знак ₽ на фото распознаётся как «P» или «Р»
    pat_a = re.compile(rf"^\s*(?P<name>\S.*?)\s+(?P<q>{_Q})\s*[xхX×*]\s*(?P<p>{_M})\s*(?:₽|руб\.?|р\.?|Р|P)?\s*=\s*(?P<s>{_M})")
    for l in lines:
        m = pat_a.match(l)
        if m and not re.search(r"НДС|ИТОГ", m["name"], re.I):
            items.append(Item(_clean_name(m["name"]), money(m["p"]), float(m["q"].replace(",", ".")),
                              money(m["s"]), None))
    if items:
        meta["layout"] = "одна строка: кол-во x цена = сумма"
        return _finish(items), meta

    # B) ofd.ru: «Название    5 X 60.00» / продолжение названия / «= 300.00»
    pat_b = re.compile(rf"^\s*(?P<name>\S*[A-Za-zА-Яа-яЁё].*?)\s+(?P<q>{_Q})\s*[XxХх×]\s*(?P<p>{_M})\s*$")
    pat_sum = re.compile(rf"^\s*=\s*(?P<s>{_M})\s*$")
    i = 0
    while i < len(lines):
        m = pat_b.match(lines[i])
        if not m:
            i += 1
            continue
        name, q, p = m["name"], m["q"], m["p"]
        j, total = i + 1, None
        while j < len(lines) and j < i + 8:
            ms = pat_sum.match(lines[j])
            if ms:
                total = money(ms["s"])
                break
            part = re.sub(r"в\s*т[.\-]?\s*ч.*$", "", lines[j]).strip()
            if part and not re.match(r"^(ПРИЗНАК|Размер|Мера|Код|Результат|Идентиф|Дата|Отрасл|UUID)", part):
                name += " " + part
            j += 1
        if total is not None:
            items.append(Item(_clean_name(name), money(p), float(q.replace(",", ".")), total, None))
            i = j + 1
        else:
            i += 1
    if items:
        meta["layout"] = "ofd: название, кол-во X цена, = сумма"
        return _finish(items), meta

    # C) магазинное приложение: «Арт.: …» / НАЗВАНИЕ / «0.868 кг   486,07 руб.» / «Акция … –50,00 руб.»
    pat_c = re.compile(rf"^\s*(?:(?P<q>{_Q})\s*(?P<u>кг|шт|л|г|уп)\.?\s+)?(?P<s>{_M})\s*(?:руб\.?|₽|р\.)\s*$", re.I)
    pat_disc = re.compile(rf"^(?:(?:Акция|Скидка|Скидки|Купон|Бонус)[^\d-]*)?-\s*(?P<s>{_M})\s*(?:руб\.?|₽|р\.)?\s*$", re.I)
    stop = re.compile(r"^(Итого|ИТОГ|Оплата|Ваша выгода|Начислено|Безнал|Наличн|Сдача|Всего)", re.I)
    prev, discounts = [], []          # скидки по позициям: (индекс, сумма)
    for l in lines:
        s = l.strip()
        if not s:
            continue
        if stop.match(s):
            break
        md = pat_disc.match(s)
        m = None if md else pat_c.match(l)
        if m:
            name = next((x for x in reversed(prev) if not re.match(r"^(Арт|Код|Штрих)", x, re.I)
                         and re.search(r"[A-Za-zА-Яа-я]{3}", x)), "")
            q = float(m["q"].replace(",", ".")) if m["q"] else 1.0
            unit = (m["u"] or "шт").lower()
            grams = q * 1000 if unit in ("кг", "л") else (q if unit == "г" else None)
            total = money(m["s"])
            items.append(Item(_clean_name(name), round(total / q, 2) if q else total, q, total, grams))
            prev = []
        elif md and items:
            discounts.append((len(items) - 1, money(md["s"])))
        else:
            prev.append(s)
    if items:
        # в одних чеках сумма позиции уже со скидкой (строка «Акция» — справочная), в других — до скидки.
        # Вычитаем скидки, только если так сходится с ИТОГО.
        raw = sum(i.total for i in items)
        after = raw - sum(d for _, d in discounts)
        if discounts and meta["total"] is not None and abs(after - meta["total"]) < abs(raw - meta["total"]):
            for k, d in discounts:
                items[k].total = round(items[k].total - d, 2)
        meta["layout"] = "приложение магазина: название, кол-во, сумма"
        return _finish(items), meta
    return [], meta


def _finish(items: list[Item]) -> list[Item]:
    out = []
    for it in items:
        if not it.name or it.total <= 0:
            continue
        if it.grams is None:
            it.grams = item_grams(it.name, it.quantity)
        out.append(it)
    return out


def total_ok(items: list[Item], meta: dict) -> bool:
    if not items:
        return False
    if meta.get("total") is None:
        return True
    return abs(sum(i.total for i in items) - meta["total"]) <= max(1.0, 0.01 * meta["total"])


# ---------- 3. LLM: текст или фото → позиции ----------
LLM_PROMPT = ("Это кассовый чек. Выпиши все купленные позиции. Ответь ТОЛЬКО JSON без пояснений: "
              '{"date": "ГГГГ-ММ-ДДTчч:мм или null", "total": число или null, "items": '
              '[{"name": "название как в чеке", "quantity": число, "price": цена за единицу, "sum": итог по позиции '
              "с учётом скидки}]}. Служебные строки (НДС, ИТОГО, оплата, кассир) не включай.")


async def llm_extract(text: str | None = None, image: bytes | None = None) -> tuple[list[Item], dict]:
    import httpx
    base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    if not base:
        return [], {}
    model = os.environ.get("LLM_VISION_MODEL") if image else os.environ.get("LLM_MODEL", "qwen2.5:7b")
    if not model:
        return [], {}
    content = [{"type": "text", "text": LLM_PROMPT}]
    if image:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}})
    else:
        content[0]["text"] += "\n\nТекст чека:\n" + text[:12000]
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{base}/chat/completions",
                              headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'ollama')}"},
                              json={"model": model, "temperature": 0, "messages": [{"role": "user", "content": content}],
                                    "response_format": {"type": "json_object"}})
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
    data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    items = []
    for it in data.get("items", []):
        try:
            q = float(it.get("quantity") or 1)
            s = float(it.get("sum") or 0) or float(it.get("price") or 0) * q
            items.append(Item(str(it["name"]).strip(), float(it.get("price") or s / q), q, s, None))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
    meta = {"date": data.get("date"), "total": data.get("total"), "layout": "LLM"}
    return _finish(items), meta


# ---------- 4. общий вход ----------
class ReadError(Exception):
    pass


async def read_receipt(data: bytes, filename: str = "", is_photo: bool = False) -> tuple[list[Item], dict]:
    """Главная функция: байты файла → (позиции, meta). meta['source'] — как прочитали."""
    ext = os.path.splitext(filename.lower())[1]
    images, text = [], None

    if ext == ".json" or (not ext and data.lstrip()[:1] in (b"{", b"[")):
        items, date = parse_receipt(json.loads(data.decode("utf-8-sig")))
        return items, {"date": date, "source": "JSON ФНС"}
    if is_photo or ext in IMAGE_EXT:
        images = [data]
    elif ext == ".pdf":
        text, images = text_from_pdf(data)
    elif ext == ".docx":
        text = text_from_docx(data)
    elif ext in TABLE_EXT:
        text = text_from_table(data, ext)
    elif ext in {".html", ".htm"}:
        text = text_from_html(data)
    elif ext in TEXT_EXT or not ext:
        text = data.decode("utf-8", errors="ignore")
    elif ext == ".doc":
        raise ReadError("старый формат .doc не читаю — сохрани как .docx или PDF")
    else:
        raise ReadError(f"формат {ext} не поддерживается")

    if text and text.strip():
        items, meta = parse_text(text)
        if total_ok(items, meta):
            meta["source"] = f"{ext[1:].upper()}, разбор правилами ({meta['layout']})"
            return items, meta
        try:
            li, lm = await llm_extract(text=text)
            if li:
                lm["source"] = f"{ext[1:].upper()}, разбор нейросетью"
                return li, lm
        except Exception as e:
            log.warning("LLM-разбор текста не удался: %s", e)
        if items:                       # правила нашли позиции, но сумма не сошлась — лучше, чем ничего
            meta["source"] = f"{ext[1:].upper()}, разбор правилами (сумма не сошлась с ИТОГО)"
            return items, meta
        raise ReadError("не нашла в тексте позиций чека")

    if images:
        if os.environ.get("LLM_VISION_MODEL") and os.environ.get("LLM_BASE_URL"):
            try:
                items, meta = await llm_extract(image=images[0])
                if items:
                    meta["source"] = "фото, распознавание нейросетью"
                    return items, meta
            except Exception as e:
                log.warning("vision-модель: %s", e)
        ocr = "\n".join(t for t in (ocr_tesseract(im) for im in images) if t)
        if ocr.strip():
            items, meta = parse_text(ocr)
            if not total_ok(items, meta):
                try:
                    li, lm = await llm_extract(text=ocr)
                    if li:
                        items, meta = li, lm
                except Exception as e:
                    log.warning("LLM по OCR: %s", e)
            if items:
                meta["source"] = "фото, OCR Tesseract"
                return items, meta
        raise ReadError("не смогла распознать фото. Лучше всего читаются PDF и JSON; для фото включи "
                        "распознавание (LLM_VISION_MODEL в .env или Tesseract, см. README)")
    raise ReadError("файл пустой")
