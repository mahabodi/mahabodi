"""Generate the MahaBodi logo SVGs (assets/logo/). Run: python assets/logo/make_logo.py

The mark is an AI Bodhi tree:
  roots    - glowing memory graph (fastmemory topology)
  trunk    - the MahaBodi core
  branches - the language bindings
  leaves   - heart-shaped peepal (Bodhi) leaves, each carrying a neural node (Laya decisions),
             linked into a network canopy
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Peepal leaf, stem at (0,0), tip pointing up (-y). Length ~ 88 units at scale 1.
LEAF = ("M0 0 C -5 -4 -26 -12 -28 -34 C -30 -54 -14 -64 -5 -70 C -2 -74 -1 -80 0 -90 "
        "C 1 -80 2 -74 5 -70 C 14 -64 30 -54 28 -34 C 26 -12 5 -4 0 0 Z")
NODE_Y = 36  # neural node sits in the broad part of the leaf

# (stem x, stem y, angle deg (0 = up, + = clockwise), scale)
LEAVES = [
    (256, 112, 0, 0.78),
    (206, 124, -16, 0.70), (306, 124, 16, 0.70),
    (160, 146, -34, 0.68), (352, 146, 34, 0.68),
    (122, 184, -56, 0.64), (390, 184, 56, 0.64),
    (100, 230, -80, 0.56), (412, 230, 80, 0.56),
    (230, 168, -8, 0.54), (282, 168, 8, 0.54),
    (184, 196, -28, 0.52), (328, 196, 28, 0.52),
    (146, 222, -52, 0.48), (366, 222, 52, 0.48),
]

# branches: (path d, width)
BRANCHES = [
    ("M256 300 C 252 250 256 190 256 112", 13),
    ("M256 296 C 246 250 222 190 206 124", 10),
    ("M256 296 C 266 250 290 190 306 124", 10),
    ("M250 290 C 226 240 190 186 160 146", 10),
    ("M262 290 C 286 240 322 186 352 146", 10),
    ("M246 292 C 214 250 160 214 122 184", 9),
    ("M266 292 C 298 250 352 214 390 184", 9),
    ("M244 296 C 200 272 140 248 100 230", 7),
    ("M268 296 C 312 272 372 248 412 230", 7),
    ("M252 236 C 240 206 232 186 230 168", 5),
    ("M260 236 C 272 206 280 186 282 168", 5),
    ("M236 250 C 214 228 196 210 184 196", 5),
    ("M276 250 C 298 228 316 210 328 196", 5),
    ("M226 266 C 196 246 168 232 146 222", 5),
    ("M286 266 C 316 246 344 232 366 222", 5),
]

ROOT_NODES = [(150, 408), (362, 408), (196, 444), (316, 444), (256, 468), (226, 420), (286, 420)]
ROOT_PATHS = [
    ("M242 352 C 224 382 188 396 150 408", 6),
    ("M270 352 C 288 382 324 396 362 408", 6),
    ("M248 352 C 236 392 216 424 196 444", 5),
    ("M264 352 C 276 392 296 424 316 444", 5),
    ("M256 352 C 256 400 256 440 256 468", 5),
]
ROOT_LINKS = [(0, 5), (5, 2), (2, 4), (4, 3), (3, 6), (6, 1), (5, 6), (5, 4), (6, 4), (0, 2), (1, 3)]


def leaf_node(x, y, a, s):
    r = math.radians(a)
    return x + math.sin(r) * NODE_Y * s, y - math.cos(r) * NODE_Y * s


def canopy_links(nodes):
    # link each leaf node to its 2 nearest neighbours: a light neural mesh through the canopy
    out = set()
    for i, (x, y) in enumerate(nodes):
        near = sorted(range(len(nodes)), key=lambda j: (nodes[j][0] - x) ** 2 + (nodes[j][1] - y) ** 2)[1:3]
        for j in near:
            out.add(tuple(sorted((i, j))))
    return sorted(out)


def mark(standalone=True, size=512, backdrop=True):
    nodes = [leaf_node(*l) for l in LEAVES]
    g = []
    g.append('''<defs>
    <radialGradient id="mb-halo" cx="50%" cy="40%" r="60%">
      <stop offset="0" stop-color="#143f52"/><stop offset="0.65" stop-color="#0b2432"/><stop offset="1" stop-color="#06131b"/>
    </radialGradient>
    <linearGradient id="mb-bark" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#45c291"/><stop offset="1" stop-color="#1c6a5e"/>
    </linearGradient>
    <linearGradient id="mb-leaf" x1="0" y1="1" x2="0" y2="0">
      <stop offset="0" stop-color="#28a468"/><stop offset="1" stop-color="#c4f58a"/>
    </linearGradient>
    <linearGradient id="mb-root" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#f7cf6a"/><stop offset="1" stop-color="#e4842c"/>
    </linearGradient>
    <filter id="mb-glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <g id="mb-peepal">
      <path d="__LEAF__" fill="url(#mb-leaf)"/>
      <path d="M0 -2 C 0 -30 0 -60 0 -84" stroke="#1d7a52" stroke-width="2.2" fill="none" stroke-linecap="round"/>
      <path d="M0 -26 L -14 -40 M0 -26 L 14 -40 M0 -46 L -12 -58 M0 -46 L 12 -58" stroke="#1d7a52" stroke-width="1.3" fill="none" stroke-linecap="round" opacity="0.8"/>
    </g>
  </defs>'''.replace("__LEAF__", LEAF))
    if backdrop:
        g.append('<circle cx="256" cy="256" r="248" fill="url(#mb-halo)"/>')
        g.append('<circle cx="256" cy="256" r="246" fill="none" stroke="#45c291" stroke-opacity="0.4" stroke-width="3"/>')
    # canopy mesh (behind leaves)
    links = canopy_links(nodes)
    g.append('<g stroke="#c4f58a" stroke-opacity="0.32" stroke-width="1.5">' +
             "".join('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (*nodes[i], *nodes[j]) for i, j in links) + "</g>")
    # branches
    g.append('<g stroke="url(#mb-bark)" stroke-linecap="round" fill="none">' +
             "".join('<path d="%s" stroke-width="%s"/>' % (d, w) for d, w in BRANCHES) + "</g>")
    # trunk with flared base
    g.append('<path d="M222 354 C 236 338 242 318 243 298 C 244 282 249 270 256 258 C 263 270 268 282 269 298 '
             'C 270 318 276 338 290 354 Z" fill="url(#mb-bark)"/>')
    # leaves + nodes
    g.append('<g filter="url(#mb-glow)">' +
             "".join('<use href="#mb-peepal" transform="translate(%s %s) rotate(%s) scale(%s)"/>' % l for l in LEAVES) + "</g>")
    g.append('<g fill="#f2fff6" filter="url(#mb-glow)">' +
             "".join('<circle cx="%.1f" cy="%.1f" r="%.1f"/>' % (x, y, 3.2 + 2.2 * l[3]) for (x, y), l in zip(nodes, LEAVES)) + "</g>")
    # ground
    g.append('<path d="M112 356 Q 256 340 400 356" stroke="#45c291" stroke-opacity="0.5" stroke-width="3" fill="none" stroke-linecap="round"/>')
    # roots: memory graph
    g.append('<g stroke="url(#mb-root)" stroke-linecap="round" fill="none" filter="url(#mb-glow)">' +
             "".join('<path d="%s" stroke-width="%s"/>' % (d, w) for d, w in ROOT_PATHS) + "</g>")
    g.append('<g stroke="#f7cf6a" stroke-opacity="0.5" stroke-width="1.5">' +
             "".join('<line x1="%s" y1="%s" x2="%s" y2="%s"/>' % (*ROOT_NODES[i], *ROOT_NODES[j]) for i, j in ROOT_LINKS) + "</g>")
    g.append('<g fill="#ffe7ad" filter="url(#mb-glow)">' +
             "".join('<circle cx="%s" cy="%s" r="%s"/>' % (x, y, 6.5 if k == 4 else 5.5) for k, (x, y) in enumerate(ROOT_NODES)) + "</g>")
    body = "\n  ".join(g)
    if not standalone:
        return body
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="%d" height="%d" role="img" aria-label="MahaBodi">\n'
            '  <title>MahaBodi</title>\n  %s\n</svg>\n' % (size, size, body))


def full_logo(dark=True):
    fg1, fg2, sub = ("#e8fff3", "#8fe3a4", "#9cc9bd") if dark else ("#0e3b4a", "#1f9a63", "#4a6f68")
    bg = '<rect width="1400" height="512" rx="36" fill="#081a24"/>' if dark else '<rect width="1400" height="512" rx="36" fill="#ffffff"/>'
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 512" width="1400" height="512" role="img" aria-label="MahaBodi">\n'
            '  <title>MahaBodi</title>\n  %s\n  <g transform="translate(24 24) scale(0.906)">\n  %s\n  </g>\n'
            '  <text x="520" y="268" font-family="Inter, \'Helvetica Neue\', Arial, sans-serif" font-size="150" font-weight="800" letter-spacing="-3">'
            '<tspan fill="%s">Maha</tspan><tspan fill="%s">Bodi</tspan></text>\n'
            '  <text x="526" y="346" font-family="Inter, \'Helvetica Neue\', Arial, sans-serif" font-size="34" font-weight="500" fill="%s" letter-spacing="1">'
            'System-1 memory and decisions for AI agents</text>\n</svg>\n') % (bg, mark(standalone=False), fg1, fg2, sub)


def banner():
    """Long horizontal README illustration: title | tree | what each part of the tree is."""
    font = "Inter, 'Helvetica Neue', Arial, sans-serif"
    parts = [  # (y on tree to point at, colour, heading, detail)
        (150, "#b8f07e", "Leaves: Laya decisions", "typed choice / score / yes-no answers, calibrated, with handoff"),
        (215, "#45c291", "Branches: six languages", "Rust core; Python, Node.js, Java, C# and Go bindings"),
        (300, "#45c291", "Trunk: the MahaBodi engine", "query cascade, density guard, experience memory, tournament"),
        (430, "#f7cf6a", "Roots: fastmemory graph", "topology memory in RAM, or on PostgreSQL + Apache AGE at scale"),
    ]
    tree_x, tree_y, s = 560, 20, 0.92  # tree drawn at translate(tree_x, tree_y) scale(s)
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 500" width="1600" height="500" role="img" '
           'aria-label="MahaBodi: an AI Bodhi tree. Roots are fastmemory graph memory, the trunk is the MahaBodi engine, '
           'branches are six language bindings, leaves are Laya decisions.">',
           '<title>MahaBodi</title>',
           '<defs><linearGradient id="bn-bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#06131b"/>'
           '<stop offset="0.55" stop-color="#0b2634"/><stop offset="1" stop-color="#07161f"/></linearGradient>'
           '<radialGradient id="bn-spot" cx="50%" cy="45%" r="50%"><stop offset="0" stop-color="#1b5064" stop-opacity="0.9"/>'
           '<stop offset="1" stop-color="#0b2634" stop-opacity="0"/></radialGradient></defs>',
           '<rect width="1600" height="500" rx="28" fill="url(#bn-bg)"/>',
           '<ellipse cx="%d" cy="250" rx="300" ry="240" fill="url(#bn-spot)"/>' % (tree_x + 256 * s),
           # title block
           '<text x="64" y="215" font-family="%s" font-size="92" font-weight="800" letter-spacing="-2">'
           '<tspan fill="#e8fff3">Maha</tspan><tspan fill="#8fe3a4">Bodi</tspan></text>' % font,
           '<text x="68" y="268" font-family="%s" font-size="27" font-weight="500" fill="#9cc9bd">System-1 memory and decisions</text>' % font,
           '<text x="68" y="304" font-family="%s" font-size="27" font-weight="500" fill="#9cc9bd">for AI agents</text>' % font,
           '<text x="68" y="360" font-family="%s" font-size="20" fill="#5f8f86">fastmemory + Laya, in Rust</text>' % font,
           '<g transform="translate(%d %d) scale(%s)">%s</g>' % (tree_x, tree_y, s, mark(standalone=False, backdrop=False))]
    # callouts on the right, with leader lines back to the tree
    for i, (ty, col, head, detail) in enumerate(parts):
        y = 118 + i * 88
        x0 = tree_x + 256 * s + {150: 150, 215: 170, 300: 16, 430: 100}.get(ty, 80) * s
        out.append('<path d="M%.0f %.0f C %.0f %.0f %.0f %.0f 1000 %d" stroke="%s" stroke-opacity="0.55" stroke-width="1.6" fill="none"/>'
                   % (x0, tree_y + ty * s, x0 + 60, tree_y + ty * s, 950, y - 8, y - 8, col))
        out.append('<circle cx="1000" cy="%d" r="5" fill="%s"/>' % (y - 8, col))
        out.append('<text x="1020" y="%d" font-family="%s" font-size="26" font-weight="700" fill="%s">%s</text>' % (y, font, col, head))
        out.append('<text x="1020" y="%d" font-family="%s" font-size="18" fill="#a9cfc5">%s</text>' % (y + 28, font, detail))
    out.append("</svg>\n")
    return "\n".join(out)


if __name__ == "__main__":
    open(os.path.join(HERE, "mahabodi-banner.svg"), "w").write(banner())
    open(os.path.join(HERE, "mahabodi-mark.svg"), "w").write(mark())
    open(os.path.join(HERE, "mahabodi-logo-dark.svg"), "w").write(full_logo(True))
    open(os.path.join(HERE, "mahabodi-logo-light.svg"), "w").write(full_logo(False))
    print("wrote mahabodi-mark.svg, mahabodi-logo-dark.svg, mahabodi-logo-light.svg")
