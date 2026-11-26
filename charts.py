"""Постеры для Telegram: чек как артефакт, который расшифровали в данные.

Визуальный язык:
  Archive Era    — тёплая бумага с зерном и сгибами, кассовая лента с рваным краем, моноширинный шрифт;
  Slow Burn      — линии «от руки» с лёгким дрожанием, штриховка вместо заливки;
  Codebreaking   — «штрихкод рациона»: каждая полоса — продукт, ширина — калории, цвет — группа;
  Rebel by Design — крупный заголовок, красные штампы поверх графиков;
  Wonderstruck   — удивительные цифры крупно: «40 378 ккал = 19 дней жизни».

Палитра групп проверена валидатором (различимость в т. ч. при дальтонизме); цвет следует за группой,
каждый цветной элемент подписан. Красный «штамп» — только статус «мало», всегда с текстом.
"""
import io
import math
import random
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

PAPER, PAPER_LIGHT, INK, INK2, FAINT = "#F2EBDD", "#FBF8F1", "#1B1B1B", "#6B655A", "#D8CFBD"
STAMP = "#B3261E"
GROUP_COLORS = {                               # фиксированно: цвет принадлежит группе, а не месту в рейтинге
    "Рыба и морепродукты": "#2D5BD7", "Мясо и птица": "#E2482D", "Молочное": "#00917F",
    "Хлеб, крупы, макароны": "#E8A317", "Сладости": "#7A3E6B", "Овощи и грибы": "#4E8F14",
    "Фрукты и ягоды": "#D85A8E",
}
OTHER = "#A39E93"
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"

plt.rcParams.update({"font.family": SANS, "text.color": INK, "axes.edgecolor": INK})


def short_name(name: str, limit: int = 26) -> str:
    """«РН ГП ТЫКВА ВЕС 776» → «тыква»; «Семечки тыквы очищенные 150ГР ШТ» → «семечки тыквы очищенные»."""
    import re
    from classify import clean_name
    s = clean_name(name).lower()
    s = re.sub(r"\b\d+[.,]?\d*\s*(г|гр|кг|мл|л|шт|%)\b\.?|\b(вес|шт|уп|м/у|пэт|ст/б)\b\.?|['\"«»]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.-")
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.-") + "…"


def _trunc(s: str, n: int) -> str:
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0].rstrip(" ,.-/") + "…"


def color_of(group: str) -> str:
    return GROUP_COLORS.get(group, OTHER)


def _num(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=PAPER)
    plt.close(fig)
    return buf.getvalue()


# ---------- «ручная работа» ----------
def _wobble(xs, ys, amp=0.004, n=60, seed=0):
    """Линия «от руки»: интерполяция + сглаженный шум."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    x = np.interp(t, np.linspace(0, 1, len(xs)), xs)
    y = np.interp(t, np.linspace(0, 1, len(ys)), ys)
    noise = np.convolve(rng.normal(0, amp, n + 8), np.ones(9) / 9, mode="valid")[:n]
    return x, y + noise


def _paper(fig, seed=7):
    """Бумага: зерно, два сгиба, кофейное кольцо."""
    bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
    rng = np.random.default_rng(seed)
    base = np.array(matplotlib.colors.to_rgb(PAPER))
    grain = rng.normal(0, 0.018, (320, 180, 1))
    img = np.clip(base + grain + np.linspace(0.01, -0.025, 320)[:, None, None], 0, 1)
    bg.imshow(img, aspect="auto", extent=[0, 1, 0, 1], interpolation="bilinear")
    for y in (0.5,):
        bg.plot([0, 1], [y, y], color="white", alpha=0.35, lw=1.2)
        bg.plot([0, 1], [y - 0.0015, y - 0.0015], color=INK2, alpha=0.08, lw=1)
    bg.plot([0.5, 0.5], [0, 1], color="white", alpha=0.25, lw=1)
    from matplotlib.patches import Ellipse
    w_, h_ = fig.get_size_inches()
    bg.add_patch(Ellipse((0.88, 0.93), 0.13, 0.13 * w_ / h_, transform=bg.transAxes, fill=False, lw=6,
                         color="#8B5E34", alpha=0.07))
    bg.set_xlim(0, 1), bg.set_ylim(0, 1), bg.axis("off")
    return bg


def _stamp(ax, x, y, text, rot=-12, size=11, color=STAMP, transform=None):
    """Резиновый штамп: рамка со скруглением, неровная прозрачность."""
    ax.text(x, y, text, rotation=rot, fontsize=size, fontweight="bold", color=color, alpha=0.88,
            ha="center", va="center", family=SANS, transform=transform or ax.transAxes,
            bbox={"boxstyle": "round,pad=0.35,rounding_size=0.25", "fc": "none", "ec": color, "lw": 2, "alpha": 0.8})


def _section(fig, x, y, title, caption=""):
    fig.text(x, y, title, family=MONO, fontsize=12.5, fontweight="bold", color=INK)
    if caption:
        fig.text(x, y - 0.017, caption, family=MONO, fontsize=9, color=INK)


# ---------- круговые диаграммы в форме фруктов ----------
FRUIT = {"apple": "#C8372D", "strawberry": "#D7263D", "orange": "#F08A24", "leaf": "#4E8F14", "stem": "#6B4423"}


def _fruit_outline(kind: str, n: int = 240) -> np.ndarray:
    """Силуэт фрукта в полярной форме вокруг (0, 0), радиус ≈ 1."""
    th = np.linspace(-np.pi / 2, 3 * np.pi / 2, n)
    if kind == "apple":            # ямка сверху, мягкая ямка снизу, чуть шире, чем выше
        r = 1 - 0.2 * np.exp(-((th - np.pi / 2) / 0.32) ** 2) - 0.035 * np.exp(-((th + np.pi / 2) / 0.6) ** 2) \
            + 0.04 * np.cos(2 * th)
        x, y = 1.04 * r * np.cos(th), 0.98 * r * np.sin(th)
    elif kind == "strawberry":     # широкие «плечи» сверху, к низу сужается в скруглённый носик
        down = np.clip(-np.sin(th), 0, None)
        up = np.clip(np.sin(th), 0, None)
        x = np.cos(th) * (1.02 - 0.55 * down ** 1.3)
        y = np.sin(th) * (0.78 - 0.1 * up + 0.62 * down) + 0.28
    else:                          # апельсин: почти круг с лёгкой неровностью кожуры
        r = 1 + 0.012 * np.sin(9 * th)
        x, y = r * np.cos(th), r * np.sin(th)
    return np.column_stack([x, y])


def _fruit_pie(ax, kind: str, parts: list, colors: list, hatches=None, label_min=0.07, text_colors=None):
    """Круговая диаграмма, обрезанная по силуэту фрукта. parts — доли (сумма 1), по часовой стрелке от 12 часов.
    Между долями — зазор цвета бумаги; проценты крупно внутри крупных долей, подписи групп — в легенде рядом."""
    from matplotlib.patches import PathPatch, Wedge, Ellipse
    from matplotlib.path import Path
    import matplotlib.patheffects as pe
    ax.set_aspect("equal"), ax.axis("off")
    ax.set_xlim(-1.35, 1.35), ax.set_ylim(-1.32, 1.5)
    pts = _fruit_outline(kind)
    clip = PathPatch(Path(pts, closed=True), transform=ax.transData, fc="none", ec="none")
    ax.add_patch(clip)
    ang = 90.0
    for k, (v, c) in enumerate(zip(parts, colors)):
        if v <= 0:
            continue
        w = Wedge((0, 0.02), 1.6, ang - v * 360, ang, fc=c, ec=PAPER_LIGHT, lw=2.2,
                  hatch=(hatches[k] if hatches else None), zorder=2)
        w.set_clip_path(clip)
        ax.add_patch(w)
        if v >= label_min:
            mid = math.radians(ang - v * 180)
            rr = 0.6 if kind != "strawberry" else 0.52
            tc = (text_colors[k] if text_colors else "white")
            ax.text(rr * math.cos(mid), rr * math.sin(mid) + (0.05 if kind == "strawberry" else 0), f"{v:.0%}",
                    ha="center", va="center", fontsize=13, fontweight="bold", color=tc, zorder=5,
                    path_effects=[pe.withStroke(linewidth=3, foreground=INK if tc == "white" else PAPER_LIGHT)])
        ang -= v * 360
    # контур «от руки»
    xs, ys = pts[:, 0] + np.random.default_rng(1).normal(0, 0.006, len(pts)), pts[:, 1]
    ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), color=INK, lw=1.8, zorder=4, solid_capstyle="round")
    if kind == "apple":
        ax.plot([0, 0.06, 0.1], [0.78, 1.0, 1.17], color=FRUIT["stem"], lw=4, solid_capstyle="round", zorder=4)
        ax.add_patch(Ellipse((0.36, 1.1), 0.5, 0.2, angle=25, fc=FRUIT["leaf"], ec=INK, lw=1.4, zorder=4))
        ax.plot([0.15, 0.56], [1.01, 1.19], color=INK, lw=0.8, zorder=5)
    elif kind == "strawberry":
        rng = np.random.default_rng(5)
        for _ in range(46):                           # семечки — по краю, чтобы не мешать цифрам
            a = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0.72, 0.92)
            i = int((a + np.pi / 2) % (2 * np.pi) / (2 * np.pi) * len(pts)) % len(pts)
            x, y = pts[i] * rad
            ax.add_patch(Ellipse((x, y), 0.045, 0.075, angle=math.degrees(a), fc="#F7E27A", ec=INK, lw=0.4,
                                 alpha=0.9, zorder=3))
        top = pts[:, 1].max()
        for a in (-150, -115, -90, -65, -30):         # чашелистик: зелёные язычки свисают вниз
            t = math.radians(a)
            tip = (0.42 * math.cos(t), top - 0.08 + 0.3 * math.sin(t))
            ax.add_patch(Polygon([(-0.09, top - 0.02), (0.09, top - 0.02), tip], closed=True,
                                 fc=FRUIT["leaf"], ec=INK, lw=1, zorder=5))
        ax.plot([0, 0.04], [top - 0.04, top + 0.22], color=FRUIT["leaf"], lw=4, solid_capstyle="round", zorder=6)
    else:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(0.97 * np.cos(th), 0.97 * np.sin(th) + 0.0, color=FRUIT["orange"], lw=7, alpha=0.95, zorder=3)
        ax.add_patch(Ellipse((0, 0.02), 0.09, 0.09, fc=PAPER_LIGHT, ec=INK, lw=0.8, zorder=5))
        ax.plot([0.05, 0.14], [1.0, 1.12], color=FRUIT["leaf"], lw=3, zorder=4)


def _legend_list(fig, x, y, rows, title, step=0.034, total_label=None):
    """Легенда-«чек» рядом с фруктом: цветная метка · название · сумма · доля (текст — чернилами, не цветом)."""
    fig.text(x, y, title, family=MONO, fontsize=10.5, fontweight="bold", color=INK)
    y -= 0.03
    for name, color, val, share, hatch in rows:
        fig.patches.append(FancyBboxPatch((x, y - 0.002), 0.022, 0.013, boxstyle="round,pad=0,rounding_size=0.003",
                                          transform=fig.transFigure, fc=color, ec=INK, lw=0.6, hatch=hatch))
        fig.text(x + 0.032, y, name, family=MONO, fontsize=10, color=INK)
        fig.text(x + 0.032, y - 0.015, f"{val} · {share:.0%}", family=MONO, fontsize=9, color=INK2)
        y -= step
    if total_label:
        fig.text(x, y + 0.004, total_label, family=MONO, fontsize=10, fontweight="bold", color=INK)


def _short_group(g: str) -> str:
    return (g.split(",")[0].replace(" и морепродукты", "").replace(" и птица", "").replace(" и грибы", "")
            .replace(" и ягоды", "").lower())


HATCH_OF = {"Напитки": "\\\\\\", "Бобовые, орехи, семена": "..."}    # 9-я группа цвета не получает: серый + штриховка, чтобы не слиться с «другим»


def _top_groups(pairs: list, k: int = 6) -> list:
    pairs = [(g, v) for g, v in pairs if v > 0]
    top, rest = pairs[:k], pairs[k:]
    if rest:
        top.append(("Другое", sum(v for _, v in rest)))
    return top


FOOTER = ("химсостав — USDA FoodData Central и справочник Тутельяна · нормы — МР 2.3.1.0253-21\n"
          "это ориентир, не назначение врача")


# ---------- отчёт: два слайда ----------
def report_charts(r: dict) -> list[bytes]:
    """«Мой рацион» двумя слайдами: 1) деньги и цифры, 2) из чего рацион и чего не хватает."""
    out = [report_slide_money(r), report_slide_diet(r)]
    return [p for p in out if p]


def report_chart(r: dict) -> bytes | None:          # совместимость: первый слайд
    return report_slide_money(r)


def report_slide_money(r: dict) -> bytes | None:
    products = r.get("products") or []
    totals = r.get("totals") or {}
    if not products or not totals.get("kcal"):
        return None
    random.seed(3)
    fig = plt.figure(figsize=(7.2, 12.8))
    bg = _paper(fig)
    fig.text(0.06, 0.965, "FINEDIET  ·  РАСШИФРОВКА ЧЕКОВ  ·  90 ДНЕЙ  ·  1/2", family=MONO, fontsize=10,
             color=INK2)
    fig.text(0.06, 0.912, "ЧТО СКРЫВАЕТ", fontsize=34, fontweight="bold", color=INK)
    t2 = fig.text(0.06, 0.864, "ТВОЙ ЧЕК", fontsize=34, fontweight="bold", color=INK)
    fig.canvas.draw()
    x_end = t2.get_window_extent().transformed(fig.transFigure.inverted()).x1
    fig.text(x_end + 0.008, 0.864, "?", fontsize=34, fontweight="bold", color=GROUP_COLORS["Мясо и птица"])
    x, y = _wobble([0.06, x_end + 0.045], [0.857, 0.854], amp=0.0012, seed=4)
    bg.plot(x, y, color=GROUP_COLORS["Мясо и птица"], lw=5, alpha=0.9, solid_capstyle="round")

    days = r.get("days_covered", {})
    kg = sum(p["grams"] for p in products) / 1000
    stats = [(_num(totals["kcal"]), "ккал", f"это {days.get('kcal', 0):.0f} дней\nпо твоей норме"),
             (_num(r["spend_rub"]), "руб на еду", f"{r.get('rub_per_1000_kcal') or 0:.0f} руб/1000 ккал\n"
                                                  f"{kg:.0f} кг · {r['items']} покупок"),
             (_num(totals.get("protein", 0)), "г белка", f"на {days.get('protein', 0):.0f} дней —\n"
                                                       f"{'с запасом' if days.get('protein', 0) > days.get('kcal', 0) else 'впритык'}")]
    for i, (big, unit, cap) in enumerate(stats):
        cx = 0.06 + i * 0.31
        fig.text(cx, 0.8, big, fontsize=24, fontweight="bold", color=INK)
        fig.text(cx, 0.782, unit, family=MONO, fontsize=10.5, fontweight="bold", color=INK)
        fig.text(cx, 0.745, cap, family=MONO, fontsize=9.2, color=INK, linespacing=1.35)
        if i:
            xs, ys = _wobble([cx - 0.025, cx - 0.025], [0.745, 0.83], amp=0.0015, seed=10 + i)
            bg.plot(xs, ys, color=INK, lw=1, alpha=0.5)

    # штрихкод рациона
    _section(fig, 0.06, 0.7, "ШТРИХКОД ТВОЕГО РАЦИОНА", "")
    fig.text(0.06, 0.683, "полоса — продукт · ширина — калории · цвет — группа", family=MONO, fontsize=9.2,
             color=INK)
    ax = fig.add_axes([0.06, 0.545, 0.88, 0.12])
    order = list(GROUP_COLORS) + ["другое"]
    prods = sorted(products, key=lambda p: (order.index(p["group"]) if p["group"] in GROUP_COLORS else len(order),
                                           -p["kcal"]))
    total = sum(p["kcal"] for p in prods)
    gap = 0.0025
    usable = 1 - gap * (len(prods) - 1)
    left, spans = 0, []
    for p in prods:
        w = usable * p["kcal"] / total
        top = 1 - random.uniform(0, 0.06)
        ax.add_patch(plt.Rectangle((left, 0), w, top, color=color_of(p["group"]), lw=0))
        spans.append((left, w, p))
        left += w + gap
    picked, seen_g = [], set()
    for sp in sorted(spans, key=lambda s: -s[1]):
        if sp[2]["group"] not in seen_g:
            picked.append(sp)
            seen_g.add(sp[2]["group"])
        if len(picked) == 3:
            break
    for k, (l0, w, p) in enumerate(sorted(picked, key=lambda s: s[0])):
        cx = l0 + w / 2
        words = p["name"].split()
        short = " ".join(words[:3] if len(words) > 2 and len(words[1]) <= 3 else words[:2])[:20]
        tx = min(max(cx, 0.1), 0.9)
        ax.annotate(f"{short}\n{_num(p['kcal'])} ккал", xy=(cx, 0.98), xytext=(tx, 1.15 + 0.25 * (k % 2)),
                    ha="center", va="bottom", fontsize=8.8, family=MONO, color=INK,
                    arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8})
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.95), ax.axis("off")
    present = [g for g in GROUP_COLORS if any(p["group"] == g for p in prods)]
    if any(p["group"] not in GROUP_COLORS for p in prods):
        present.append("Другое")
    lx, ly = 0.06, 0.52
    for g in present:
        label = _short_group(g)
        fig.patches.append(FancyBboxPatch((lx, ly), 0.014, 0.009, boxstyle="round,pad=0,rounding_size=0.002",
                                          transform=fig.transFigure, fc=color_of(g), ec="none"))
        fig.text(lx + 0.02, ly, label, family=MONO, fontsize=9.2, color=INK, va="bottom")
        lx += 0.045 + 0.0125 * len(label)
        if lx > 0.84:
            lx, ly = 0.06, ly - 0.02

    # яблоко: куда ушли деньги
    _section(fig, 0.06, 0.455, "КУДА УШЛИ ДЕНЬГИ", "")
    fig.text(0.06, 0.438, "доля расходов на еду по группам", family=MONO, fontsize=9.2, color=INK)
    spend = _top_groups(list((r.get("spend_by_group") or {}).items()))
    tot = sum(v for _, v in spend) or 1
    axa = fig.add_axes([0.02, 0.07, 0.56, 0.36])
    _fruit_pie(axa, "apple", [v / tot for _, v in spend], [color_of(g) for g, _ in spend],
               hatches=[HATCH_OF.get(g) for g, _ in spend])
    _legend_list(fig, 0.6, 0.41, [(_short_group(g), color_of(g), f"{_num(v)} руб", v / tot, HATCH_OF.get(g))
                                  for g, v in spend],
                 "ЧЕК", step=0.04, total_label=f"итого {_num(r['spend_rub'])} руб")
    if spend:
        g, v = spend[0]
        _stamp(bg, 0.3, 0.075, f"№1: {_short_group(g).upper()} · {v / tot:.0%}", rot=-5, size=10)
    fig.text(0.06, 0.012, FOOTER, family=MONO, fontsize=7.6, color=INK2, linespacing=1.4)
    return _png(fig)


def report_slide_diet(r: dict) -> bytes | None:
    products = r.get("products") or []
    if not products:
        return None
    fig = plt.figure(figsize=(7.2, 12.8))
    bg = _paper(fig, seed=11)
    fig.text(0.06, 0.965, "FINEDIET  ·  РАСШИФРОВКА ЧЕКОВ  ·  90 ДНЕЙ  ·  2/2", family=MONO, fontsize=10,
             color=INK2)
    fig.text(0.06, 0.912, "ИЗ ЧЕГО", fontsize=34, fontweight="bold", color=INK)
    t2 = fig.text(0.06, 0.864, "ТВОЙ РАЦИОН", fontsize=34, fontweight="bold", color=INK)
    fig.canvas.draw()
    x_end = t2.get_window_extent().transformed(fig.transFigure.inverted()).x1
    x, y = _wobble([0.06, x_end + 0.01], [0.857, 0.854], amp=0.0012, seed=6)
    bg.plot(x, y, color=GROUP_COLORS["Овощи и грибы"], lw=5, alpha=0.9, solid_capstyle="round")

    # клубника: калории по группам
    _section(fig, 0.06, 0.81, "КАЛОРИИ ПО ГРУППАМ", "")
    fig.text(0.06, 0.793, "откуда берётся энергия", family=MONO, fontsize=9.2, color=INK)
    by = {}
    for p in products:
        by[p["group"]] = by.get(p["group"], 0) + p["kcal"]
    kc = _top_groups(sorted(by.items(), key=lambda kv: -kv[1]))
    tot = sum(v for _, v in kc) or 1
    axs = fig.add_axes([0.02, 0.5, 0.56, 0.29])
    _fruit_pie(axs, "strawberry", [v / tot for _, v in kc], [color_of(g) for g, _ in kc],
               hatches=[HATCH_OF.get(g) for g, _ in kc])
    _legend_list(fig, 0.6, 0.77, [(_short_group(g), color_of(g), f"{_num(v)} ккал", v / tot, HATCH_OF.get(g))
                                  for g, v in kc],
                 "ГРУППЫ", step=0.037)

    # апельсин: белки / жиры / углеводы — ты и норма
    split = r.get("energy_split") or {}
    if split:
        _section(fig, 0.06, 0.475, "БЕЛКИ · ЖИРЫ · УГЛЕВОДЫ", "")
        fig.text(0.06, 0.458, "доля калорий; норма: белки 15%, жиры 30%, углеводы 55%", family=MONO, fontsize=9.2,
                 color=INK)
        names = ["белки", "жиры", "углеводы"]
        cols = ["#8A3B12", "#F08A24", "#FBD3A0"]               # тёмная, средняя, светлая мякоть + штриховка
        hatches = [None, "///", "..."]
        tcols = ["white", "white", INK]
        axo = fig.add_axes([0.02, 0.24, 0.42, 0.21])
        _fruit_pie(axo, "orange", [split.get(n, 0) for n in names], cols, hatches=hatches, text_colors=tcols)
        fig.text(0.23, 0.235, "ТЫ", family=MONO, fontsize=10.5, fontweight="bold", ha="center")
        axn = fig.add_axes([0.45, 0.3, 0.2, 0.1])
        _fruit_pie(axn, "orange", [0.15, 0.30, 0.55], cols, hatches=hatches, label_min=2)
        fig.text(0.55, 0.287, "НОРМА", family=MONO, fontsize=10, fontweight="bold", ha="center")
        ly = 0.43
        for n, c, h in zip(names, cols, hatches):
            fig.patches.append(FancyBboxPatch((0.7, ly - 0.002), 0.022, 0.013, boxstyle="round,pad=0,rounding_size=0.003",
                                              transform=fig.transFigure, fc=c, ec=INK, lw=0.6, hatch=h))
            fig.text(0.732, ly, f"{n} {split.get(n, 0):.0%}", family=MONO, fontsize=10, color=INK)
            ly -= 0.03
        fat = split.get("жиры", 0)
        if fat > 0.35:
            _stamp(bg, 0.8, 0.3, f"ЖИРОВ\n{fat:.0%}", rot=8, size=11)

    _balance(fig, bg, r.get("balance_vs_calories") or {}, [0.06, 0.04, 0.88, 0.15])
    fig.text(0.06, 0.012, FOOTER, family=MONO, fontsize=7.6, color=INK2, linespacing=1.4)
    return _png(fig)


def _zigzag_strip(ax, w=1.0, h=1.0, teeth=18, depth=0.018):
    xs = np.linspace(0, w, teeth * 2 + 1)
    top = [(x, h - (depth if i % 2 else 0)) for i, x in enumerate(xs)]
    bottom = [(x, (depth if i % 2 else 0)) for i, x in enumerate(xs)][::-1]
    ax.add_patch(Polygon(top + bottom, closed=True, fc=PAPER_LIGHT, ec=FAINT, lw=0.8, zorder=1))


def _receipt_money(fig, r, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    _zigzag_strip(ax)
    spend = list((r.get("spend_by_group") or {}).items())
    top, rest = spend[:7], spend[7:]
    if rest:
        top.append(("Другое", sum(v for _, v in rest)))
    ax.text(0.5, 0.93, "КАССОВЫЙ ЧЕК", family=MONO, fontsize=9.5, fontweight="bold", ha="center", zorder=2)
    ax.text(0.5, 0.875, "куда ушли деньги на еду", family=MONO, fontsize=7.5, ha="center", color=INK2, zorder=2)
    ax.text(0.5, 0.835, "- " * 21, family=MONO, fontsize=7, ha="center", color=INK2, zorder=2)
    total = r["spend_rub"] or 1
    y = 0.775
    step = 0.6 / max(len(top), 1)
    for g, v in top:
        name = g.lower().replace("морепродукты", "морепр.")[:24]
        ax.text(0.06, y, name, family=MONO, fontsize=7.6, zorder=2)
        ax.text(0.94, y, f"{_num(v)}", family=MONO, fontsize=7.6, ha="right", zorder=2)
        ax.add_patch(FancyBboxPatch((0.06, y - 0.03), 0.88 * v / total, 0.011,
                                    boxstyle="round,pad=0,rounding_size=0.006", fc=color_of(g), ec="none", zorder=2))
        y -= step
    ax.text(0.5, 0.115, "- " * 21, family=MONO, fontsize=7, ha="center", color=INK2, zorder=2)
    ax.text(0.06, 0.07, "ИТОГО НА ЕДУ", family=MONO, fontsize=8.5, fontweight="bold", zorder=2)
    ax.text(0.94, 0.07, f"{_num(total)} руб", family=MONO, fontsize=8.5, fontweight="bold", ha="right", zorder=2)
    if spend:
        g, v = spend[0]
        _stamp(ax, 0.6, 0.165, f"№1: {g.split(',')[0].upper()} · {v / total:.0%}", rot=-5, size=8)


def _energy_tapes(fig, split, rect):
    if not split:
        return
    fig.text(rect[0], rect[1] + rect[3] + 0.012, "ИЗ ЧЕГО КАЛОРИИ", family=MONO, fontsize=11, fontweight="bold")
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    rec = {"белки": 0.15, "жиры": 0.30, "углеводы": 0.55}
    styles = {"белки": dict(fc=INK, hatch=None), "жиры": dict(fc=PAPER_LIGHT, hatch="////"),
              "углеводы": dict(fc=PAPER_LIGHT, hatch="....")}
    for row, (label, data, yb) in enumerate([("ты", split, 0.6), ("норма", rec, 0.12)]):
        ax.text(0, yb + 0.33, label, family=MONO, fontsize=8, color=INK2)
        left, tilt = 0.0, (0.02 if row == 0 else -0.015)
        for n in ("белки", "жиры", "углеводы"):
            w = data.get(n, 0)
            if w <= 0:
                continue
            x0, x1 = left, left + w
            pts = [(x0, yb + tilt * x0), (x1, yb + tilt * x1), (x1 + 0.006, yb + 0.26 + tilt * x1),
                   (x0 - 0.004, yb + 0.25 + tilt * x0)]
            st = styles[n]
            ax.add_patch(Polygon(pts, closed=True, fc=st["fc"], ec=INK, lw=0.9, hatch=st["hatch"]))
            if w >= 0.1:
                ax.text((x0 + x1) / 2, yb + 0.13 + tilt * (x0 + x1) / 2, f"{w:.0%}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="white" if n == "белки" else INK, family=MONO,
                        bbox=None if n == "белки" else {"fc": PAPER_LIGHT, "ec": "none", "pad": 0.6})
            left = x1
    ax.text(0, -0.08, "■ белки   ▨ жиры   ░ углеводы", family=MONO, fontsize=7.4, color=INK)
    fat = split.get("жиры", 0)
    if fat > 0.35:
        _stamp(ax, 0.72, 1.02, f"ЖИРОВ {fat:.0%}", rot=8, size=8)


def _balance(fig, bg, balance, rect):
    if not balance:
        return
    _section(fig, rect[0], rect[1] + rect[3] + 0.022, "ЧЕГО НЕ ХВАТАЕТ И ЧЕГО С ИЗБЫТКОМ",
             "100% — нутриента ровно столько, сколько калорий · красный — меньше 80%")
    ax = fig.add_axes(rect)
    items = sorted(balance.items(), key=lambda kv: kv[1])
    if len(items) > 8:                       # для слайда: самое низкое и самое высокое
        low = [kv for kv in items if kv[1] < 1.0][:5]
        high = [kv for kv in items if kv[1] >= 1.0][-(8 - len(low)):]
        items = low + high
    n = len(items)
    ax.set_xlim(-0.02, 3.35), ax.set_ylim(-1.3, n - 0.3), ax.axis("off")
    x100 = 0.62 + 0.8                                   # стебли начинаются в 0.62, 100% = 0.8 по оси
    ys, xs = _wobble([-0.6, n - 0.4], [x100, x100], amp=0.012, seed=21)
    ax.plot(xs, ys, color=INK, lw=1, ls=(0, (4, 3)), alpha=0.7)
    ax.text(x100, -0.95, "в балансе", family=MONO, fontsize=9, color=INK, ha="center", va="top")
    for i, (name, v) in enumerate(items):
        vv = min(v, 3.0)
        low = v < 0.8
        sx, sy = _wobble([0.62, 0.62 + vv * 0.8], [i, i], amp=0.025, seed=100 + i)
        ax.plot(sx, sy, color=INK, lw=1.4, alpha=0.85, solid_capstyle="round")
        ax.plot(sx[-1], sy[-1], "o", ms=7.5, mfc=STAMP if low else INK, mec=PAPER, mew=1.4, zorder=3)
        ax.text(0.58, i, name, family=MONO, fontsize=9.5, ha="right", va="center")
        ax.text(sx[-1] + 0.08, i, f"{v:.0%}" + (" ›" if v > 3 else ""), family=MONO, fontsize=9,
                va="center", color=INK2, zorder=4, bbox={"fc": PAPER, "ec": "none", "pad": 0.3})
    lows = [k for k, v in items if v < 0.8]
    if lows:
        _stamp(ax, 2.75, max(len(lows) - 1, 0) / 2, "МАЛО:\n" + "\n".join(lows[:4]).upper(), rot=-10, size=10,
               transform=ax.transData)


# ---------- постер к корзине: «чек на неделю» ----------
def basket_chart(b: dict) -> bytes | None:
    basket = b.get("basket") or []
    if not basket:
        return None
    rows = basket[:16]
    h = 4.0 + 0.27 * (len(rows) + 5)
    fig = plt.figure(figsize=(7.2, h))
    _paper(fig, seed=11)
    ax = fig.add_axes([0.14, 0.03, 0.72, 0.94])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    _zigzag_strip(ax, depth=0.008, teeth=22)
    line = 1.0 / (len(rows) + 22)
    y = 1 - line * 1.4
    mode = "ближе к привычному" if b.get("mode") == "привычная" else "самая дешёвая"
    for txt, kw in [(f"ПЛАН ПОКУПОК НА {b['days']} ДНЕЙ", dict(fontsize=12, fontweight="bold")),
                    (f"что купить, чтобы неделю есть по норме · {mode}", dict(fontsize=7.6, color=INK2)),
                    ("- " * 26, dict(fontsize=7, color=INK2))]:
        ax.text(0.5, y, txt, family=MONO, ha="center", va="center", zorder=2, **kw)
        y -= line
    groups = {}
    for p in rows:
        g = p.get("group") or "Другое"
        groups.setdefault(g, 0)
    from optimizer import by_sections
    for sec, items in by_sections(rows):
        ax.text(0.05, y, sec, family=MONO, fontsize=7.8, fontweight="bold", va="center", zorder=2)
        y -= line
        for p in items:
            g = p["grams_total"]
            amount = f"{g / 1000:.1f} кг" if g >= 1000 else f"{g} г"
            mark = {"+": " ↑", "-": " ↓"}.get(p.get("vs_habit", ""), "")
            name = short_name(p["example"])
            ax.add_patch(FancyBboxPatch((0.05, y - 0.006), 0.012, 0.012, boxstyle="round,pad=0,rounding_size=0.003",
                                        fc=color_of(p.get("group", "")), ec="none", zorder=2))
            ax.text(0.075, y, name, family=MONO, fontsize=7.6, va="center", zorder=2)
            ax.text(0.66, y, amount, family=MONO, fontsize=7.6, va="center", ha="right", color=INK2, zorder=2)
            ax.text(0.95, y, f"{_num(p['rub'])}{mark}", family=MONO, fontsize=7.6, va="center", ha="right", zorder=2)
            y -= line
    if len(basket) > len(rows):
        ax.text(0.05, y, f"… и ещё {len(basket) - len(rows)}", family=MONO, fontsize=7.4, color=INK2, va="center")
        y -= line
    ax.text(0.5, y, "- " * 26, family=MONO, fontsize=7, ha="center", va="center", color=INK2, zorder=2)
    y -= line
    ax.text(0.05, y, "ИТОГО", family=MONO, fontsize=10, fontweight="bold", va="center", zorder=2)
    ax.text(0.95, y, f"{_num(b['total_rub'])} руб", family=MONO, fontsize=10, fontweight="bold", ha="right",
            va="center", zorder=2)
    y -= line
    if b.get("habit_rub"):
        ax.text(0.05, y, "обычный набор", family=MONO, fontsize=7.6, color=INK2, va="center", zorder=2)
        ax.text(0.95, y, f"{_num(b['habit_rub'])} руб", family=MONO, fontsize=7.6, color=INK2, ha="right",
                va="center", zorder=2)
        y -= line
    pd_ = b.get("per_day", {})
    ax.text(0.5, y, f"в день: {_num(pd_.get('kcal', 0))} ккал · белок {pd_.get('protein', 0)} г · "
            f"жиры {pd_.get('fat', 0)} г", family=MONO, fontsize=7.2, ha="center", va="center", color=INK2, zorder=2)
    y -= line * 1.2
    mp = b.get("micro_percent_of_norm") or {}
    if mp:
        full = sum(v >= 95 for v in mp.values())
        ax.text(0.5, y, f"витамины и минералы: {full} из {len(mp)} на 100%", family=MONO, fontsize=7.2,
                ha="center", va="center", color=INK2, zorder=2)
        y -= line * 1.4
    # полоса «состав корзины по группам» — те же цвета, что в штрихкоде рациона
    groups = {}
    for p in basket:
        g = p.get("group", "Другое") if p.get("group") in GROUP_COLORS else "Другое"
        groups[g] = groups.get(g, 0) + p["rub"]
    tot = sum(groups.values()) or 1
    left = 0.05
    for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        w = 0.9 * v / tot
        ax.add_patch(plt.Rectangle((left, y - line * 0.35), max(w - 0.004, 0.001), line * 0.7,
                                   color=color_of(g), lw=0, zorder=2))
        label = g.split(",")[0].split(" и ")[0].lower() if g in GROUP_COLORS else "другое"
        if w > 0.075:
            ax.text(left + w / 2, y - line * 0.95, label, family=MONO,
                    fontsize=6.6, ha="center", va="center", color=INK2, zorder=2)
        left += w
    y -= line * 1.6
    diff = (b.get("habit_rub") or 0) - b["total_rub"]
    if diff > 0:
        _stamp(ax, 0.68, max(y - line * 1.2, 0.08), f"ЭКОНОМИЯ {_num(diff)} руб", rot=-6, size=11)
    ax.text(0.05, 0.035, "правила тарелки: овощи и фрукты от 400 г/день · крупы и хлеб 15–35% ккал\n"
            "белковые продукты от 20% · молочное от 150 г · приправы до 50 г", family=MONO, fontsize=6.2,
            color=INK2, va="bottom", zorder=2, linespacing=1.4)
    return _png(fig)


# ---------- карточки остальных экранов ----------
def _card(h_in=9.0, seed=5):
    fig = plt.figure(figsize=(7.2, h_in))
    bg = _paper(fig, seed=seed)
    return fig, bg


def _headline(fig, bg, y, lines, accent=None, size=30):
    """Крупный заголовок с подчёркиванием маркером; accent — символ в конце красным."""
    t = None
    for i, line in enumerate(lines):
        t = fig.text(0.06, y - i * 0.055 * 9 / fig.get_size_inches()[1] * 1.0, line, fontsize=size,
                     fontweight="bold", color=INK)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(fig.transFigure.inverted())
    if accent:
        fig.text(bb.x1 + 0.008, bb.y0, accent, fontsize=size, fontweight="bold", color=GROUP_COLORS["Мясо и птица"])
    x, yy = _wobble([0.06, bb.x1 + (0.05 if accent else 0.01)], [bb.y0 - 0.008, bb.y0 - 0.011], amp=0.0012, seed=4)
    bg.plot(x, yy, color=GROUP_COLORS["Мясо и птица"], lw=5, alpha=0.9, solid_capstyle="round")
    return bb.y0


def welcome_card(name: str = "") -> bytes:
    fig, bg = _card(9.0, seed=2)
    fig.text(0.06, 0.945, "FINEDIET  ·  ДНЕВНИК ПОКУПОК", family=MONO, fontsize=8.5, color=INK2)
    if name:
        fig.text(0.06, 0.905, f"привет, {name}!", family=MONO, fontsize=11, color=INK)
    y0 = _headline(fig, bg, 0.83, ["ЧЕК —", "ЭТО ДАННЫЕ"], accent="!")
    fig.text(0.06, y0 - 0.075, "пришли чек — и я расскажу, что ты на самом деле ешь,\n"
             "чего не хватает и как собрать корзину дешевле", family=MONO, fontsize=8.6, color=INK2,
             linespacing=1.4)
    ax = fig.add_axes([0.06, 0.13, 0.88, 0.5])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    _zigzag_strip(ax, depth=0.012, teeth=22)
    ax.text(0.5, 0.92, "КАК ЭТО РАБОТАЕТ", family=MONO, fontsize=10, fontweight="bold", ha="center", zorder=2)
    ax.text(0.5, 0.86, "- " * 30, family=MONO, fontsize=7, ha="center", color=INK2, zorder=2)
    steps = [("01", "ПРИШЛИ ЧЕК", "PDF из письма · фото · JSON ФНС", "Рыба и морепродукты"),
             ("02", "РАСШИФРУЮ", "1 053 категории · ккал, БЖУ, 17 витаминов", "Молочное"),
             ("03", "ПОКАЖУ", "на сколько дней хватит · куда ушли деньги", "Хлеб, крупы, макароны"),
             ("04", "ПОДСКАЖУ", "где взять кальций · корзина на неделю", "Овощи и грибы")]
    y = 0.75
    for num, title, sub, g in steps:
        ax.add_patch(FancyBboxPatch((0.05, y - 0.035), 0.075, 0.07, boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=color_of(g), ec="none", zorder=2))
        ax.text(0.0875, y, num, family=MONO, fontsize=11, fontweight="bold", color="white", ha="center",
                va="center", zorder=3)
        ax.text(0.16, y + 0.018, title, family=MONO, fontsize=10.5, fontweight="bold", va="center", zorder=2)
        ax.text(0.16, y - 0.032, sub, family=MONO, fontsize=7.8, color=INK2, va="center", zorder=2)
        y -= 0.165
    ax.text(0.5, 0.09, "- " * 30, family=MONO, fontsize=7, ha="center", color=INK2, zorder=2)
    # штрихкод-подпись
    bx = fig.add_axes([0.06, 0.05, 0.5, 0.05])
    bx.axis("off"), bx.set_xlim(0, 1), bx.set_ylim(0, 1)
    rng = random.Random(1053)
    xx = 0.0
    while xx < 1:
        w = rng.choice([0.006, 0.006, 0.012, 0.018])
        bx.add_patch(plt.Rectangle((xx, 0.3), w, 0.7, color=INK, lw=0))
        xx += w + rng.choice([0.006, 0.01, 0.014])
    bx.text(0, 0.0, "1 0 5 3  ·  1 7  ·  9 8 %", family=MONO, fontsize=8, va="bottom")
    _stamp(bg, 0.78, 0.075, "Я НЕ ВРАЧ\nНОРМЫ — ОРИЕНТИР", rot=-9, size=8)
    return _png(fig)


def receipt_card(items: list, meta: dict, groups: dict, low_conf: float = 0.30) -> bytes | None:
    """Разобранный чек: лента, у позиции — метка группы и категория; «не еда» зачёркнута."""
    from nutrition import is_non_food, pretty
    if not items:
        return None
    rows = items[:22]
    h = 2.6 + 0.36 * len(rows)
    fig, bg = _card(h, seed=13)
    ax = fig.add_axes([0.08, 0.02, 0.84, 0.96])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    _zigzag_strip(ax, depth=0.006, teeth=26)
    line = 1.0 / (len(rows) * 1.55 + 10)
    y = 1 - line * 1.5
    date = str(meta.get("date") or "")[:10]
    ax.text(0.5, y, "ЧЕК РАСШИФРОВАН", family=MONO, fontsize=12, fontweight="bold", ha="center", va="center",
            zorder=2)
    y -= line
    ax.text(0.5, y, f"{date}  ·  {len(items)} позиций" if date else f"{len(items)} позиций", family=MONO,
            fontsize=8, ha="center", va="center", color=INK2, zorder=2)
    y -= line * 0.9
    ax.text(0.5, y, "- " * 32, family=MONO, fontsize=7, ha="center", va="center", color=INK2, zorder=2)
    y -= line
    food = other = 0.0
    for it in rows:
        nf = is_non_food(it["category"])
        g = groups.get(it["category"], "Другое")
        if nf:
            other += it["total"]
        else:
            food += it["total"]
        nm = short_name(it["name"], 30)
        ax.add_patch(FancyBboxPatch((0.04, y - 0.004), 0.014, 0.008 * 1, boxstyle="round,pad=0,rounding_size=0.003",
                                    fc=(FAINT if nf else color_of(g)), ec="none", zorder=2))
        t = ax.text(0.07, y, nm, family=MONO, fontsize=8, va="center", zorder=2, color=(INK2 if nf else INK))
        ax.text(0.96, y, _num(it["total"]), family=MONO, fontsize=8, va="center", ha="right", zorder=2,
                color=(INK2 if nf else INK))
        cat = "не еда" if nf else _trunc(pretty(it["category"]), 40)
        q = " ?" if (not nf and it.get("confidence", 1) < low_conf) else ""
        ax.text(0.07, y - line * 0.55, f"→ {cat}{q}", family=MONO, fontsize=6.8, va="center", zorder=2,
                color=(STAMP if q else INK2))
        if nf:                                                  # зачёркнуто маркером
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transData.inverted())
            xs, ys = _wobble([0.065, bb.x1 + 0.01], [y, y + 0.001], amp=line * 0.08, seed=int(y * 1000))
            ax.plot(xs, ys, color=STAMP, lw=1.6, alpha=0.75, zorder=3)
        y -= line * 1.55
    if len(items) > len(rows):
        ax.text(0.07, y, f"… и ещё {len(items) - len(rows)}", family=MONO, fontsize=7.6, color=INK2, va="center")
        y -= line
    ax.text(0.5, y, "- " * 32, family=MONO, fontsize=7, ha="center", va="center", color=INK2, zorder=2)
    y -= line
    ax.text(0.04, y, "НА ЕДУ", family=MONO, fontsize=9.5, fontweight="bold", va="center", zorder=2)
    ax.text(0.96, y, f"{_num(food)} руб", family=MONO, fontsize=9.5, fontweight="bold", va="center", ha="right",
            zorder=2)
    if other:
        y -= line
        ax.text(0.04, y, "не еда", family=MONO, fontsize=8, color=INK2, va="center", zorder=2)
        ax.text(0.96, y, f"{_num(other)} руб", family=MONO, fontsize=8, color=INK2, va="center", ha="right",
                zorder=2)
    src = str(meta.get("source") or "").split(",")[0].split("(")[0].strip().upper()
    if src:
        _stamp(ax, 0.72, max(y - line * 1.6, 0.04), f"ПРОЧИТАНО: {src}", rot=-6, size=9)
    return _png(fig)


def _label_frame(ax):
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=True, fc=PAPER_LIGHT, ec=INK, lw=2.2, zorder=1))


def profile_card(profile: dict, targets: dict, micro_targets: dict) -> bytes:
    """«Твоя норма» в стиле этикетки «Пищевая ценность»."""
    fig, bg = _card(8.6, seed=17)
    fig.text(0.06, 0.95, "FINEDIET  ·  ПРОФИЛЬ", family=MONO, fontsize=8.5, color=INK2)
    _headline(fig, bg, 0.885, ["ТВОЯ НОРМА"], size=28)
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.76])
    ax.set_xlim(0, 1), ax.set_ylim(0.17, 1), ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.19), 0.96, 0.79, fill=True, fc=PAPER_LIGHT, ec=INK, lw=2.2, zorder=1))
    ax.text(0.06, 0.93, "Пищевая ценность", fontsize=20, fontweight="bold", va="center", zorder=2)
    who = []
    if profile.get("sex"):
        who.append("женщина" if profile["sex"] == "ж" else "мужчина")
    for k, u in (("age", "лет"), ("height", "см"), ("weight", "кг")):
        if profile.get(k):
            who.append(f"{float(profile[k]):.0f} {u}")
    ax.text(0.06, 0.875, "на день · " + (", ".join(who) if who else "по умолчанию"), family=MONO, fontsize=8.5,
            va="center", zorder=2)
    ax.plot([0.05, 0.95], [0.85, 0.85], color=INK, lw=7, zorder=2)
    ax.text(0.06, 0.79, "Калории", fontsize=17, fontweight="bold", va="center", zorder=2)
    ax.text(0.94, 0.79, _num(targets["kcal"]), fontsize=26, fontweight="bold", va="center", ha="right", zorder=2)
    ax.plot([0.05, 0.95], [0.745, 0.745], color=INK, lw=4, zorder=2)
    rows = [("Белки", f"{targets['protein']:.0f} г"),
            ("Жиры", f"{targets['kcal'] * 0.2 / 9:.0f}–{targets['kcal'] * 0.35 / 9:.0f} г"),
            ("Углеводы", f"{targets['carbs']:.0f} г"), ("Клетчатка", f"{targets.get('fiber', 25):.0f} г")]
    y = 0.705
    for n, v in rows:
        ax.text(0.06, y, n, fontsize=11, fontweight="bold", va="center", zorder=2)
        ax.text(0.94, y, v, family=MONO, fontsize=11, va="center", ha="right", zorder=2)
        ax.plot([0.05, 0.95], [y - 0.025, y - 0.025], color=INK, lw=0.8, zorder=2)
        y -= 0.05
    ax.plot([0.05, 0.95], [y + 0.02, y + 0.02], color=INK, lw=7, zorder=2)
    y -= 0.02
    items = list(micro_targets.items())
    half = (len(items) + 1) // 2
    step = min(0.042, (y - 0.27) / max(half, 1))          # 17 нутриентов должны влезть над подписью
    fs = 8.6 if step >= 0.04 else 7.8
    for col, chunk in enumerate((items[:half], items[half:])):
        yy = y
        x0 = 0.06 + col * 0.46
        for n, v in chunk:
            unit = n.split(",")[-1].strip() if "," in n else ""
            nm = n.split(",")[0]
            ax.text(x0, yy, nm, family=MONO, fontsize=fs, va="center", zorder=2)
            ax.text(x0 + 0.42, yy, f"{v:g} {unit}", family=MONO, fontsize=fs, va="center", ha="right", zorder=2)
            yy -= step
    ax.text(0.06, 0.225, "КБЖУ — формула Миффлина–Сан Жеора · витамины — МР 2.3.1.0253-21\nэто ориентир, "
            "не назначение врача", family=MONO, fontsize=6.8, color=INK2, va="center", zorder=2, linespacing=1.4)
    goal = profile.get("goal")
    if goal:
        _stamp(bg, 0.77, 0.89, f"ЦЕЛЬ:\n{goal.upper()}", rot=-10, size=9)
    excl = profile.get("exclude") or []
    if excl:
        _stamp(bg, 0.3, 0.03, "НЕ ЕМ: " + ", ".join(e + "…" for e in excl).upper()[:40], rot=3, size=8, color=INK)
    return _png(fig)


def product_card(text: str, category: str, conf: float, per100: dict | None, alts: list) -> bytes:
    """«Определить продукт»: этикетка на 100 г + уверенность модели."""
    from nutrition import pretty
    fig, bg = _card(7.0, seed=23)
    fig.text(0.06, 0.94, "FINEDIET  ·  ЧТО ЭТО ЗА ПРОДУКТ", family=MONO, fontsize=8.5, color=INK2)
    fig.text(0.06, 0.885, f"«{text[:40]}»", family=MONO, fontsize=10, color=INK2)
    import textwrap
    title = textwrap.wrap(pretty(category).upper(), 20)[:2]
    if len(textwrap.wrap(pretty(category).upper(), 20)) > 2:
        title[-1] = _trunc(title[-1] + " …", 20)
    _headline(fig, bg, 0.80 if len(title) == 1 else 0.82, title, size=22)
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.62 if len(title) == 1 else 0.58])
    ax.set_xlim(0, 1), ax.axis("off")
    ax.text(0.06, 0.9, "Пищевая ценность", fontsize=18, fontweight="bold", va="center", zorder=2)
    ax.text(0.06, 0.835, "на 100 г · справочник Тутельяна", family=MONO, fontsize=8, va="center", zorder=2)
    ax.plot([0.05, 0.95], [0.8, 0.8], color=INK, lw=6, zorder=2)
    if per100:
        ax.text(0.06, 0.73, "Калории", fontsize=15, fontweight="bold", va="center", zorder=2)
        ax.text(0.94, 0.73, f"{per100.get('kcal', 0):.0f}", fontsize=22, fontweight="bold", va="center",
                ha="right", zorder=2)
        ax.plot([0.05, 0.95], [0.67, 0.67], color=INK, lw=3, zorder=2)
        y = 0.62
        for k, n in (("protein", "Белки"), ("fat", "Жиры"), ("carbs", "Углеводы"), ("fiber", "Клетчатка")):
            if k in per100:
                ax.text(0.06, y, n, fontsize=11, fontweight="bold", va="center", zorder=2)
                ax.text(0.94, y, f"{per100[k]:.1f} г", family=MONO, fontsize=11, va="center", ha="right", zorder=2)
                ax.plot([0.05, 0.95], [y - 0.035, y - 0.035], color=INK, lw=0.8, zorder=2)
                y -= 0.07
    else:
        ax.text(0.5, 0.6, "химсостава для этой категории\nв справочнике нет", family=MONO, fontsize=10,
                ha="center", va="center", color=INK2, zorder=2)
        y = 0.4
    if alts:
        ax.text(0.06, y - 0.02, "ещё похоже на:", family=MONO, fontsize=8, color=INK2, va="center", zorder=2)
        for i, (c, p) in enumerate(alts[:2]):
            ax.text(0.06, y - 0.075 - i * 0.055, f"· {pretty(c)[:36]}", family=MONO, fontsize=8, va="center", zorder=2)
            ax.text(0.94, y - 0.075 - i * 0.055, f"{p:.0%}", family=MONO, fontsize=8, va="center", ha="right",
                    zorder=2)
    bottom = (y - 0.075 - 0.055 * (min(len(alts), 2) - 1) - 0.05) if alts else y - 0.02
    ax.set_ylim(bottom - 0.02, 1)                       # этикетка ровно по содержимому
    ax.add_patch(plt.Rectangle((0.02, bottom), 0.96, 0.98 - bottom, fill=True, fc=PAPER_LIGHT, ec=INK, lw=2.2,
                               zorder=1))
    _stamp(bg, 0.78, 0.9, f"УВЕРЕН\n{conf:.0%}" if conf >= 0.3 else "НЕ УВЕРЕН", rot=-10, size=10,
           color=(INK if conf >= 0.3 else STAMP))
    return _png(fig)


# ---------- карточка с текстом: любой ответ бота в том же стиле, что и постеры ----------
EMOJI_MAP = {"☀️": "☀", "☀": "☀", "🌙": "☾", "🍲": "♨", "🍽": "✦", "🛒": "✦", "👩‍🍳": "✎", "🔥": "▲", "💰": "◆",
             "💪": "●", "🥦": "✿", "🐟": "≈", "🔄": "↻", "✅": "✓", "❗": "!", "⚠️": "!", "📌": "▸", "👉": "▸",
             "🍎": "♣", "🤔": "", "👋": "", "👩‍⚕️": "", "🍳": "✦", "₽": "руб", " ": " "}
HEADER_ICONS = ("☀", "☾", "♨", "✦", "✎", "♣")


def _clean(s: str) -> str:
    """DejaVu не рисует цветные эмодзи и знак рубля: заменяем на символы, которые есть в шрифте, остальное убираем."""
    s = re.sub(r"\*\*|__|^#+\s*", "", s, flags=re.M)          # markdown от модели: **жирный**, ### заголовки
    for a, b in EMOJI_MAP.items():
        s = s.replace(a, b)
    s = "".join(ch for ch in s if ord(ch) < 0x1F000 and ch not in "️‍")
    return s


def text_card(title: str, text: str, kicker: str = "", stamp: str = "") -> bytes:
    """Ответ бота карточкой: бумага, заголовок маркером, моноширинный текст. Высота — по длине текста."""
    import textwrap
    width_chars = 58
    rows = []                                    # (текст, жирный, отступ)
    for raw in _clean(text).split("\n"):
        line = raw.rstrip()
        if not line.strip():
            rows.append(("", False, 0))
            continue
        st = re.sub(r"^[-*]\s+", "• ", line.strip())             # «- пункт» → «• пункт»
        bold = st.startswith(HEADER_ICONS) or (st.endswith(":") and len(st) < 60)
        bullet = st[:1] in "•-▸*" or re.match(r"\d+[.)] ", st) is not None
        indent = 2 if bullet else 0
        for k, part in enumerate(textwrap.wrap(st, width=width_chars - indent) or [""]):
            rows.append((part if k == 0 else " " * indent + part, bold, indent))
    line_h = 0.255                               # дюймов на строку
    top, head = 0.55, 0.95                       # отступ сверху и место под заголовок
    h = min(max(top + head + len(rows) * line_h + 0.7, 4.0), 40)
    fig = plt.figure(figsize=(7.2, h))
    bg = _paper(fig, seed=hash(title) % 50)
    fy = lambda inch: 1 - inch / h                # дюймы от верха → доля высоты
    fig.text(0.06, fy(0.42), "FINEDIET" + (f"  ·  {kicker}" if kicker else ""), family=MONO, fontsize=10, color=INK2)
    t = fig.text(0.06, fy(top + 0.62), _clean(title).upper(), fontsize=24, fontweight="bold", color=INK)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(fig.transFigure.inverted())
    x, y = _wobble([0.06, min(bb.x1 + 0.01, 0.94)], [bb.y0 - 0.06 / h, bb.y0 - 0.075 / h], amp=0.012 / h, seed=4)
    bg.plot(x, y, color=GROUP_COLORS["Мясо и птица"], lw=4.5, alpha=0.9, solid_capstyle="round")
    y_in = top + head + 0.15
    for txt, bold, _ in rows:
        if txt:
            fig.text(0.06, fy(y_in), txt, family=MONO, fontsize=11.5, color=INK,
                     fontweight="bold" if bold else "normal", va="top")
        y_in += line_h
    if stamp:
        _stamp(bg, 0.78, fy(top + 0.45), _clean(stamp).upper(), rot=-8, size=10)
    fig.text(0.06, fy(h - 0.3), "ориентир, не назначение врача · FineDiet", family=MONO, fontsize=7.6, color=INK2)
    return _png(fig)
