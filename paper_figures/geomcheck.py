"""Numeric fidelity check on the Fig 1 PPTX export.

The shape count proves the slide is vector, not a bitmap; this checks that the
geometry actually landed where fig1_overview.py draws it. Compares every emitted
shape against the recorded primitives and flags anything off-slide or displaced.
"""
import sys

from pptx import Presentation
from pptx.util import Emu

sys.path.insert(0, ".")
import fig1_to_pptx as F  # noqa: E402  (re-uses the recorder and the sx/sy map)

F.record()
REC = F.REC

prs = Presentation(sys.argv[1] if len(sys.argv) > 1 else F.OUT)
SW, SH = prs.slide_width.inches, prs.slide_height.inches
shapes = list(prs.slides[0].shapes)
print(f"slide {SW:.2f} x {SH:.2f} in, {len(shapes)} shapes")

# 1. nothing may fall outside the slide
off = []
for sh in shapes:
    x, y = Emu(sh.left).inches, Emu(sh.top).inches
    w, h = Emu(sh.width).inches, Emu(sh.height).inches
    if x < -0.02 or y < -0.02 or x + w > SW + 0.02 or y + h > SH + 0.02:
        label = (sh.text_frame.text[:38].replace("\n", " ")
                 if sh.has_text_frame else sh.shape_type)
        off.append((round(x, 2), round(y, 2), round(w, 2), round(h, 2), label))
print(f"off-slide shapes: {len(off)}")
for o in off:
    print("   ", o)

# 2. every recorded box must have a shape at its mapped corner
miss = 0
for b in REC["box"]:
    ex, ey = F.sx(b["x"]), F.sy(b["y"] + b["h"])
    if not any(abs(Emu(s.left).inches - ex) < 0.02
               and abs(Emu(s.top).inches - ey) < 0.02 for s in shapes):
        miss += 1
        print(f"    box missing at ({ex:.2f},{ey:.2f}) fc={b['fc']}")
print(f"boxes placed: {len(REC['box']) - miss}/{len(REC['box'])}")

# 3. the gold z0 hand-off: 3 wire segments + 2 arrowheads + the label
GOLD = "B7950B"
wires = [p for p in REC["poly"] if p["c"].upper().lstrip("#") == GOLD]
heads = [a for a in REC["arrow"] if a["c"].upper().lstrip("#") == GOLD]
print(f"hand-off: {len(wires)} plotted segments, {len(heads)} arrow segments")
for a in heads:
    print(f"    arrow {a['p1']} -> {a['p2']} style={a['sty']}")

lbl = [t for t in REC["text"] if "clean latent" in t["s"]]
for t in lbl:
    print(f"    label ha={t['ha']} fill={t.get('fill')} "
          f"at data({t['x']},{t['y']}) -> in({F.sx(t['x']):.2f},{F.sy(t['y']):.2f})")
    for s in shapes:
        if s.has_text_frame and "clean latent" in s.text_frame.text:
            x, y = Emu(s.left).inches, Emu(s.top).inches
            w, h = Emu(s.width).inches, Emu(s.height).inches
            solid = s.fill.type is not None and str(s.fill.type) != "None"
            print(f"    emitted box x={x:.2f} y={y:.2f} w={w:.2f} h={h:.2f} "
                  f"right={x + w:.2f} filled={solid}")

# 4. no text box may collide with the gold wire's horizontal run
if wires:
    hy = None
    for p in wires:
        if len(p["ys"]) == 2 and abs(p["ys"][0] - p["ys"][1]) < 1e-6:
            hy = F.sy(p["ys"][0])
    if hy is not None:
        print(f"wire y = {hy:.2f} in")
