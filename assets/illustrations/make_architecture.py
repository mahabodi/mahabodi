"""End-to-end MahaBodi architecture, animated (SVG + SMIL): agents -> engine pipeline -> typed answer, with memory,
the learn loop and the PostgreSQL scale path. Every box names a component that exists in crates/mahabodi-core
(system1/decide.rs, script.rs, embedder.rs; query.rs, density.rs, graph.rs, louvain.rs, memory.rs) or in deploy/.
Packets move along the request path; the animation is decorative. prefers-reduced-motion hides the packets.

    python3 assets/illustrations/make_architecture.py
"""
import os

W, H = 1200, 720
INK, INK2, MUTE, LINE = "#0d3240", "#46655f", "#7d958f", "#e1ece7"
LEAF, SEA, ROOT, WARN = "#179a5f", "#0e5b73", "#b67a18", "#c0661c"
AGENTS = [("Support", LEAF), ("Sales", SEA), ("Marketing", "#1b8a6a"), ("Finance", ROOT),
          ("HR", WARN), ("IT", SEA), ("Operations", LEAF), ("Legal", ROOT)]
STAGES = [("Typed request", "state · question", "· options"), ("Guard + cache", "script check", "· answer cache"),
          ("Query cascade", "lexical → typo", "→ dense (grounded)"), ("Shortlist", "top-k retrieval", "· tournament"),
          ("Laya decision", "ONNX Runtime", "· one pass"), ("Experience", "labelled cases", "· calibrate")]
SX0, SW, SG, SY, SH = 262, 104, 12, 196, 96   # stage boxes
MID = SY + SH / 2
OUT_X = 1000


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    o = []
    a = o.append
    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" font-family="Inter, Helvetica Neue, Arial, sans-serif">' % (W, H, W, H))
    a('<title>MahaBodi end-to-end architecture: agents send typed requests through the engine pipeline and get typed answers back</title>')
    a('<style>@media (prefers-reduced-motion: reduce){.pk{display:none}} text{fill:%s}</style>' % INK)
    a('<defs><marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
      '<path d="M0 0 L10 5 L0 10 z" fill="%s"/></marker>'
      '<linearGradient id="eng" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#eef6f2"/><stop offset="1" stop-color="#ffffff"/></linearGradient>'
      '<filter id="sh" x="-5%%" y="-5%%" width="110%%" height="120%%"><feDropShadow dx="0" dy="4" stdDeviation="7" flood-color="%s" flood-opacity=".10"/></filter></defs>' % (MUTE, INK))
    a('<rect width="%d" height="%d" fill="#fbfdfc"/>' % (W, H))
    # ---- agents column
    a('<text x="110" y="92" text-anchor="middle" font-size="15" font-weight="800">AI agents</text>')
    ay = []
    for i, (name, hue) in enumerate(AGENTS):
        y = 118 + i * 44
        ay.append(y)
        a('<rect x="38" y="%d" width="144" height="32" rx="16" fill="%s" fill-opacity=".09" stroke="%s" stroke-opacity=".35"/>' % (y - 16, hue, hue))
        a('<circle cx="58" cy="%d" r="6" fill="%s"/>' % (y, hue))
        a('<text x="72" y="%d" font-size="13" font-weight="600" fill="%s">%s</text>' % (y + 5, INK2, name))
    # ---- engine
    ex1, ex2 = SX0 - 22, SX0 + 6 * SW + 5 * SG + 22
    a('<rect x="%d" y="110" width="%d" height="250" rx="20" fill="url(#eng)" stroke="%s" stroke-width="1.5" filter="url(#sh)"/>' % (ex1, ex2 - ex1, LEAF))
    a('<text x="%d" y="140" font-size="17" font-weight="800">MahaBodi engine</text>' % (ex1 + 20))
    # request packets drawn under the stage boxes: they vanish into each stage and reappear between stages
    main_d = []
    for i, y in enumerate(ay):
        d = "M 58 %d L 182 %d C 215 %d, 225 %.0f, %d %.0f L %d %.0f L %d %.0f" % (y, y, y, MID, SX0, MID, ex2 - 22, MID, OUT_X + 30, MID)
        main_d.append(d)
        a('<circle class="pk" r="6" fill="%s" stroke="#fff" stroke-width="2" opacity="0"><animateMotion dur="7s" begin="%.2fs" repeatCount="indefinite" path="%s"/>'
          '<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.05;0.9;1" dur="7s" begin="%.2fs" repeatCount="indefinite"/></circle>' % (AGENTS[i][1], i * 0.875, d, i * 0.875))
    a('<text x="%d" y="140" font-size="13" fill="%s">one Rust core</text>' % (ex1 + 170, INK2))
    for i, (t, l1, l2) in enumerate(STAGES):
        x = SX0 + i * (SW + SG)
        a('<rect x="%d" y="%d" width="%d" height="%d" rx="12" fill="#ffffff" stroke="%s" stroke-opacity=".45"/>' % (x, SY, SW, SH, SEA))
        a('<circle cx="%d" cy="%d" r="11" fill="%s"/><text x="%d" y="%d" text-anchor="middle" font-size="12" font-weight="800" fill="#ffffff" style="fill:#fff">%d</text>' % (x + 16, SY + 18, SEA, x + 16, SY + 22, i + 1))
        a('<text x="%d" y="%d" text-anchor="middle" font-size="13" font-weight="700">%s</text>' % (x + SW / 2, SY + 48, esc(t)))
        a('<text x="%d" y="%d" text-anchor="middle" font-size="11" fill="%s" style="fill:%s">%s</text>' % (x + SW / 2, SY + 66, INK2, INK2, esc(l1)))
        a('<text x="%d" y="%d" text-anchor="middle" font-size="11" fill="%s" style="fill:%s">%s</text>' % (x + SW / 2, SY + 81, INK2, INK2, esc(l2)))
        if i < 5:
            a('<line x1="%d" y1="%.0f" x2="%d" y2="%.0f" stroke="%s" stroke-width="1.6" marker-end="url(#ar)"/>' % (x + SW + 1, MID, x + SW + SG - 1, MID, MUTE))
    a('<text x="%d" y="334" text-anchor="middle" font-size="12.5" fill="%s" style="fill:%s">No text generated · a probability for every option · grounded answers report their memory stage, match and handoff</text>' % ((ex1 + ex2) / 2, INK2, INK2))
    # agents -> engine
    for y in ay:
        a('<path d="M182 %d C 215 %d, 225 %.0f, %d %.0f" stroke="%s" stroke-opacity=".35" stroke-width="1.4" fill="none"/>' % (y, y, MID, SX0, MID, MUTE))
    # ---- output
    a('<rect x="%d" y="176" width="176" height="136" rx="16" fill="%s" filter="url(#sh)"/>' % (OUT_X, LEAF))
    for k, (t, fs, fw) in enumerate([("Typed answer", 15, 800), ("choice · score · yes/no", 12, 500), ("+ probabilities", 12, 500),
                                     ("+ memory citation", 12, 500), ("and handoff (grounded)", 12, 500)]):
        a('<text x="%d" y="%d" text-anchor="middle" font-size="%d" font-weight="%d" style="fill:#ffffff">%s</text>' % (OUT_X + 88, 204 + k * 24, fs, fw, t))
    a('<line x1="%d" y1="%.0f" x2="%d" y2="%.0f" stroke="%s" stroke-width="1.8" marker-end="url(#ar)"/>' % (ex2 - 22 + 1, MID, OUT_X - 2, MID, MUTE))
    # back to the agent (over the top)
    back = "M %d 176 C %d 60, 600 52, 300 60 S 40 64, 26 104" % (OUT_X + 88, OUT_X + 88)
    a('<path d="%s" stroke="%s" stroke-width="1.8" stroke-dasharray="6 6" fill="none" marker-end="url(#ar)"/>' % (back, LEAF))
    a('<text x="640" y="48" text-anchor="middle" font-size="13" font-weight="600" fill="%s" style="fill:%s">the typed decision goes back to the agent</text>' % (LEAF, LEAF))
    # ---- memory layer
    a('<rect x="%d" y="392" width="%d" height="118" rx="18" fill="#fffaf0" stroke="%s" stroke-opacity=".55"/>' % (ex1, ex2 - ex1, ROOT))
    a('<text x="%d" y="418" font-size="15" font-weight="800" fill="%s" style="fill:%s">Memory</text>' % (ex1 + 20, ROOT, ROOT))
    mems = [("fastmemory graph", "records · Louvain concept blocks", "density guard keeps all reachable"),
            ("Dense vectors", "MiniLM embedder, ONNX", "meaning search"),
            ("Experience memory", "labelled past decisions", "learns instantly")]
    mx = []
    for k, (t, l1, l2) in enumerate(mems):
        x = ex1 + 30 + k * 236
        mx.append(x + 100)
        a('<rect x="%d" y="430" width="200" height="66" rx="12" fill="#ffffff" stroke="%s" stroke-opacity=".5"/>' % (x, ROOT))
        a('<text x="%d" y="452" text-anchor="middle" font-size="13" font-weight="700">%s</text>' % (x + 100, t))
        a('<text x="%d" y="469" text-anchor="middle" font-size="11" style="fill:%s">%s</text>' % (x + 100, INK2, l1))
        a('<text x="%d" y="484" text-anchor="middle" font-size="11" style="fill:%s">%s</text>' % (x + 100, INK2, l2))
    s3 = SX0 + 2 * (SW + SG) + SW / 2; s4 = SX0 + 3 * (SW + SG) + SW / 2; s6 = SX0 + 5 * (SW + SG) + SW / 2
    for sx, tx in ((s3, mx[0]), (s3, mx[1]), (s6, mx[2])):
        a('<path d="M %.0f %d C %.0f 360, %.0f 400, %.0f 430" stroke="%s" stroke-opacity=".55" stroke-width="1.5" stroke-dasharray="4 4" fill="none"/>' % (sx, SY + SH, sx, tx, tx, ROOT))
    # learn loop: outcomes -> experience memory
    learn = "M %d 312 C %d 470, %d 470, %.0f 496" % (OUT_X + 88, OUT_X + 88, OUT_X - 40, mx[2] + 100)
    a('<path d="%s" stroke="%s" stroke-width="1.8" stroke-dasharray="6 6" fill="none" marker-end="url(#ar)"/>' % (learn, ROOT))
    a('<text x="985" y="532" font-size="12.5" font-weight="600" style="fill:%s">learn(): labelled outcomes</text>' % ROOT)
    a('<text x="985" y="549" font-size="12.5" font-weight="600" style="fill:%s">update memory in seconds,</text>' % ROOT)
    a('<text x="985" y="566" font-size="12.5" font-weight="600" style="fill:%s">no retraining</text>' % ROOT)
    # ---- scale path (blueprint)
    a('<rect x="%d" y="532" width="%d" height="96" rx="16" fill="none" stroke="%s" stroke-width="1.5" stroke-dasharray="7 6"/>' % (ex1, ex2 - ex1, SEA))
    a('<text x="%d" y="558" font-size="14" font-weight="800" style="fill:%s">Scale path</text>' % (ex1 + 20, SEA))
    for k, line in enumerate(["PostgreSQL + pgvector (+ Apache AGE) holds memory by namespace, and each",
                              "agent hydrates a small working set; models can be served separately (Triton).",
                              "Tested: 5.9M Wikipedia pages loaded on one Mac mini. Decisions at that scale",
                              "are still being measured; TB–PB sizes are a blueprint."]):
        a('<text x="%d" y="%d" font-size="12.5" style="fill:%s">%s</text>' % (ex1 + 112, 558 + k * 18, INK2, line))
    a('<path d="M %d 510 L %d 532" stroke="%s" stroke-width="1.5" stroke-dasharray="3 4" marker-end="url(#ar)"/>' % ((ex1 + ex2) / 2, (ex1 + ex2) / 2, SEA))
    # ---- bindings
    a('<text x="%d" y="664" text-anchor="middle" font-size="13.5" font-weight="700">One core, six languages: Rust · Python · Node.js · Java · C#/.NET · Go (+ C ABI)</text>' % (W / 2))
    a('<text x="%d" y="690" text-anchor="middle" font-size="11.5" style="fill:%s">Illustrative flow; component names match the source code. Measured results are in BENCHMARKS.md.</text>' % (W / 2, MUTE))
    # ---- animated packets (decorative)
    for k, (sx, tx) in enumerate(((s3, mx[0]), (s3, mx[1]), (s6, mx[2]))):
        d = "M %.0f %d C %.0f 360, %.0f 400, %.0f 430" % (sx, SY + SH, sx, tx, tx)
        a('<circle class="pk" r="4.5" fill="%s" opacity="0"><animateMotion dur="2.2s" begin="%.1fs" repeatCount="indefinite" keyPoints="0;1;0" keyTimes="0;0.5;1" calcMode="linear" path="%s"/>'
          '<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="2.2s" begin="%.1fs" repeatCount="indefinite"/></circle>' % (ROOT, 0.7 * k, d, 0.7 * k))
    a('<circle class="pk" r="6" fill="%s" stroke="#fff" stroke-width="2" opacity="0"><animateMotion dur="4.5s" begin="1s" repeatCount="indefinite" path="%s"/>'
      '<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.08;0.9;1" dur="4.5s" begin="1s" repeatCount="indefinite"/></circle>' % (LEAF, back))
    a('<circle class="pk" r="5" fill="%s" stroke="#fff" stroke-width="2" opacity="0"><animateMotion dur="5s" begin="2.5s" repeatCount="indefinite" path="%s"/>'
      '<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.08;0.9;1" dur="5s" begin="2.5s" repeatCount="indefinite"/></circle>' % (ROOT, learn))
    a('</svg>')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture-flow.svg")
    open(p, "w").write("\n".join(o))
    print(p)


if __name__ == "__main__":
    main()
