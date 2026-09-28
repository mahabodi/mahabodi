"""Agents scattered across business functions, all routed through MahaBodi (illustration for the one-pager).

Deterministic (seed 7): every run writes the same agents-routing.svg next to this script.

    python3 assets/illustrations/make_agents_routing.py
"""
import math, os, random

W, H = 1200, 780
CX, CY = 600, 392
FUNCS = [  # name, hue, what its agents decide
    ("Customer support", "#179a5f", "intents · known issues"),
    ("Sales", "#0e5b73", "leads · products"),
    ("Marketing", "#1b8a6a", "segments · offers"),
    ("Finance", "#b67a18", "GL codes · vendors"),
    ("HR", "#c0661c", "policies · cases"),
    ("IT &amp; security", "#0e5b73", "tickets · tools"),
    ("Operations", "#179a5f", "parts · routes"),
    ("Legal &amp; compliance", "#b67a18", "clauses · rules"),
]


def main():
    rnd = random.Random(7)
    out = []
    a = out.append
    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" font-family="Inter, Helvetica Neue, Arial, sans-serif">' % (W, H, W, H))
    a('<title>AI agents scattered across business functions, all routed through MahaBodi</title>')
    a('<defs>'
      '<radialGradient id="hub" cx="50%" cy="40%" r="60%"><stop offset="0" stop-color="#27b877"/><stop offset="1" stop-color="#0e5b73"/></radialGradient>'
      '<radialGradient id="halo" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#16a36a" stop-opacity=".22"/><stop offset="1" stop-color="#16a36a" stop-opacity="0"/></radialGradient>'
      '<filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="6" stdDeviation="10" flood-color="#0d3240" flood-opacity=".12"/></filter>'
      '</defs>')
    a('<rect width="%d" height="%d" fill="#fbfdfc"/>' % (W, H))
    # memory ring around the hub
    a('<circle cx="%d" cy="%d" r="126" fill="none" stroke="#b67a18" stroke-opacity=".45" stroke-width="2" stroke-dasharray="3 6"/>' % (CX, CY))
    a('<path id="ring" d="M %d %d A 140 140 0 0 1 %d %d" fill="none"/>' % (CX - 140, CY, CX + 140, CY))
    a('<text font-size="13" font-weight="600" fill="#b67a18" letter-spacing=".5"><textPath href="#ring" startOffset="4">MEMORY · documents · catalogues · decisions</textPath></text>')
    agents = []
    zones = []
    for i, (name, hue, what) in enumerate(FUNCS):
        ang = -math.pi / 2 + 2 * math.pi * i / len(FUNCS)
        zx, zy = CX + 440 * math.cos(ang), CY + 270 * math.sin(ang)
        zones.append((zx, zy, name, hue, what))
        for _ in range(rnd.randint(7, 11)):
            r, t = 62 * math.sqrt(rnd.random()), rnd.random() * 2 * math.pi
            agents.append((zx + r * math.cos(t) * 1.35, zy + r * math.sin(t) * 0.8, hue))
    # routes: every agent to the hub
    for x, y, hue in agents:
        mx, my = (x + CX) / 2 + (CY - y) * 0.12, (y + CY) / 2 + (x - CX) * 0.06
        a('<path d="M%.1f %.1f Q %.1f %.1f %d %d" stroke="%s" stroke-opacity=".22" stroke-width="1.4" fill="none"/>' % (x, y, mx, my, CX, CY, hue))
    for zx, zy, name, hue, what in zones:
        a('<ellipse cx="%.0f" cy="%.0f" rx="104" ry="66" fill="%s" fill-opacity=".07" stroke="%s" stroke-opacity=".25" stroke-dasharray="4 5"/>' % (zx, zy, hue, hue))
        ty = zy - 78 if zy < CY else zy + 92
        a('<text x="%.0f" y="%.0f" text-anchor="middle" font-size="19" font-weight="700" fill="#0d3240">%s</text>' % (zx, ty, name))
        a('<text x="%.0f" y="%.0f" text-anchor="middle" font-size="14" fill="#46655f">%s</text>' % (zx, ty + 19, what))
    for x, y, hue in agents:
        a('<circle cx="%.1f" cy="%.1f" r="7" fill="%s" stroke="#ffffff" stroke-width="2"/>' % (x, y, hue))
    # hub
    a('<circle cx="%d" cy="%d" r="150" fill="url(#halo)"/>' % (CX, CY))
    a('<circle cx="%d" cy="%d" r="92" fill="url(#hub)" filter="url(#soft)"/>' % (CX, CY))
    a('<text x="%d" y="%d" text-anchor="middle" font-size="25" font-weight="800" fill="#ffffff">MahaBodi</text>' % (CX, CY + 2))
    a('<text x="%d" y="%d" text-anchor="middle" font-size="12" fill="#e8fff3">remember · shortlist · decide</text>' % (CX, CY + 24))
    # legend
    a('<circle cx="40" cy="40" r="7" fill="#179a5f" stroke="#fff" stroke-width="2"/><text x="56" y="45" font-size="15" fill="#46655f">an AI agent making a decision</text>')
    a('<path d="M33 70 Q 45 62 57 70" stroke="#179a5f" stroke-opacity=".5" stroke-width="2" fill="none"/><text x="66" y="75" font-size="15" fill="#46655f">routed through MahaBodi: one typed decision, no text generated</text>')
    a('</svg>')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents-routing.svg")
    open(p, "w").write("\n".join(out))
    print(p, len(agents), "agents")


if __name__ == "__main__":
    main()
