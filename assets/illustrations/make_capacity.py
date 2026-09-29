"""Decision capacity skyline (log scale): how many options each setup has been measured deciding over.

Heights are measured option counts only:
  Laya alone        150 options (CLINC150, measured); at 10,000 and above it cannot run (options exceed its context)
  Laya + retrieval  5,903,530 pages, with a MiniLM embedding index + shortlist built and hosted separately
  MahaBodi          5,903,530 pages out of the box (in process to 100K; PostgreSQL store at 5.9M)
Accuracy is stated in the caption, never implied by height (research/results/bench_el*.json, BENCHMARKS.md).

    python3 assets/illustrations/make_capacity.py
"""
import math, os

W, H = 1200, 640
X0, Y0, Y1 = 150, 520, 70          # axis origin and top of the plot area
LOGMAX = 7.0                        # 10^7 at the top
INK, INK2, MUTE, LINE = "#0d3240", "#46655f", "#7d958f", "#e1ece7"
LEAF, SEA, ROOT, WARN = "#179a5f", "#0e5b73", "#b67a18", "#c0661c"


def y(v):
    return Y0 - (Y0 - Y1) * math.log10(v) / LOGMAX


def windows(o, x, w, top, color, step=22):
    for yy in range(int(top) + 14, Y0 - 10, step):
        for xx in range(int(x) + 12, int(x + w) - 16, 22):
            o.append('<rect x="%d" y="%d" width="10" height="9" rx="1.5" fill="%s" fill-opacity=".85"/>' % (xx, yy, color))


def main():
    o = []
    a = o.append
    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" font-family="Inter, Helvetica Neue, Arial, sans-serif">' % (W, H, W, H))
    a('<title>Candidate pool searched: Laya alone reads up to 150 measured options and cannot run at 10,000; Laya with your own index and MahaBodi both search 5.9 million pages and decide among 20</title>')
    a('<rect width="%d" height="%d" fill="#fbfdfc"/>' % (W, H))
    # axis and gridlines (log scale)
    for e in range(0, 8):
        yy = y(10 ** e)
        a('<line x1="%d" y1="%.0f" x2="%d" y2="%.0f" stroke="%s" stroke-width="1"/>' % (X0, yy, W - 40, yy, LINE))
        lab = {0: "1", 1: "10", 2: "100", 3: "1K", 4: "10K", 5: "100K", 6: "1M", 7: "10M"}[e]
        a('<text x="%d" y="%.0f" text-anchor="end" font-size="13" fill="%s">%s</text>' % (X0 - 10, yy + 4, MUTE, lab))
    a('<text x="42" y="%d" font-size="13" fill="%s" transform="rotate(-90 42 %d)" text-anchor="middle">candidate pool searched (log scale, measured)</text>' % ((Y0 + Y1) // 2, INK2, (Y0 + Y1) // 2))
    a('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="2"/>' % (X0, Y0, W - 40, Y0, INK))
    # 1. Laya alone: a house at 150, "cannot run" at 10K
    hx, hw, top = 230, 150, y(150)
    a('<rect x="%d" y="%.0f" width="%d" height="%.0f" fill="%s" fill-opacity=".85"/>' % (hx, top, hw, Y0 - top, SEA))
    a('<path d="M %d %.0f L %d %.0f L %d %.0f z" fill="%s"/>' % (hx - 10, top, hx + hw / 2, top - 38, hx + hw + 10, top, SEA))
    windows(o, hx, hw, top, "#e8fff3")
    a('<rect x="%d" y="%.0f" width="%d" height="%.0f" fill="none" stroke="%s" stroke-width="2" stroke-dasharray="6 6"/>' % (hx, y(1e4), hw, top - 38 - y(1e4), WARN))
    a('<text x="%d" y="%.0f" text-anchor="middle" font-size="14" font-weight="700" fill="%s">✕ cannot run at 10K</text>' % (hx + hw / 2, y(1e4) - 10, WARN))
    a('<text x="%d" y="%.0f" text-anchor="middle" font-size="12.5" fill="%s">options exceed its context</text>' % (hx + hw / 2, y(1e4) + 18, INK2))
    a('<text x="%d" y="%.0f" text-anchor="middle" font-size="15" font-weight="800" fill="%s">150 options</text>' % (hx + hw / 2, top - 46, SEA))
    # 2 and 3: the same pool, the same 20-way decision; the difference is what ships
    def tower(x, w, color, label_top, bi_split):
        t = y(5903530)
        a('<rect x="%d" y="%.0f" width="%d" height="%.0f" fill="%s"/>' % (x, t, w, Y0 - t, color))
        windows(o, x, w, t, "#e8fff3")
        a('<rect x="%d" y="%.0f" width="%d" height="%.0f" fill="%s"/>' % (x, y(20), w, Y0 - y(20), ROOT))
        a('<text x="%d" y="%.0f" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">decides among 20</text>' % (x + w / 2, (y(20) + Y0) / 2 + 4))
        a('<text x="%d" y="%.0f" text-anchor="middle" font-size="16" font-weight="800" fill="%s">%s</text>' % (x + w / 2, t - 12, color, label_top))
        if bi_split:
            a('<line x1="%d" y1="%.0f" x2="%d" y2="%.0f" stroke="#ffffff" stroke-width="2" stroke-dasharray="4 4"/>' % (x, y(1e5), x + w + 14, y(1e5)))
            a('<text x="%d" y="%.0f" font-size="12" font-weight="700" fill="%s">PostgreSQL store</text>' % (x + w + 18, y(1e5) - 8, SEA))
            a('<text x="%d" y="%.0f" font-size="12" font-weight="700" fill="%s">in process</text>' % (x + w + 18, y(1e5) + 16, SEA))
    sx, sw = 520, 180
    tower(sx, sw, "#5b7f86", "5.9M pages", False)
    mx, mw = 850, 180
    tower(mx, mw, LEAF, "5.9M pages", True)
    # labels under the baseline
    for cx, t1, t2 in ((hx + hw / 2, "Laya alone", "all options in one pass; measured on CLINC150"),
                       (sx + sw / 2, "Laya + your own index", "bring your own embeddings, index and glue"),
                       (mx + mw / 2, "MahaBodi", "memory + retrieval built in (PostgreSQL at 5.9M, loaded with scripts)")):
        a('<text x="%d" y="%d" text-anchor="middle" font-size="16" font-weight="800" fill="%s">%s</text>' % (cx, Y0 + 28, INK, t1))
        a('<text x="%d" y="%d" text-anchor="middle" font-size="12.5" fill="%s">%s</text>' % (cx, Y0 + 47, INK2, t2))
    # caption: accuracy, never implied by height
    a('<text x="%d" y="%d" text-anchor="middle" font-size="13" fill="%s">Height = candidate pool searched, not accuracy; both towers decide among 20 retrieved candidates. At 100K MahaBodi beats Laya + dense (0.177 vs 0.121);</text>' % (W / 2, H - 38, INK2))
    a('<text x="%d" y="%d" text-anchor="middle" font-size="13" fill="%s">at 5.9M it ties it (0.103 vs 0.122, 8.3 s per decision), and a remembered alias prior beats both (0.77). Details: BENCHMARKS.md.</text>' % (W / 2, H - 18, INK2))
    a('</svg>')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision-capacity.svg")
    open(p, "w").write("\n".join(o))
    print(p)


if __name__ == "__main__":
    main()
