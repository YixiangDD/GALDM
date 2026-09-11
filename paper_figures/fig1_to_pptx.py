# -*- coding: utf-8 -*-
"""Fig 1 (v5) -> fully editable PPTX.

Rather than hardcoding the layout (which would drift from the figure), this
instruments the v5 script's own drawing helpers, records every primitive, and
re-emits each one as a native PowerPoint shape. Every box, label, arrow and
curve is separately selectable and editable; no bitmap is embedded.
"""
import os
import re
import sys
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig1_overview.py')
# override with GALDM_PPTX_DIR if you want the decks somewhere else
OUT = os.path.join(os.environ.get("GALDM_PPTX_DIR",
                                  os.path.join(_P.FIGURES_DIR, "pptx")),
                   'Fig1_overview.pptx')

# v5 axes: x in [-0.5, 21.6], y in [-0.8, 15.5]; slide 21 x 14 in.
# Must stay in sync with ax.set_xlim/set_ylim in fig1_overview.py, otherwise every
# shape lands displaced (the 8-step inference row widened the x range).
X_MIN, X_MAX = -0.5, 21.6
Y_MIN, Y_MAX = -0.8, 15.5
SLIDE_W, SLIDE_H = 21.0, 14.0
MARGIN = 0.3
SX = (SLIDE_W - 2 * MARGIN) / (X_MAX - X_MIN)
SY = (SLIDE_H - 2 * MARGIN) / (Y_MAX - Y_MIN)


def sx(x):
    return MARGIN + (x - X_MIN) * SX


def sy(y):
    return MARGIN + (Y_MAX - y) * SY


def rgb(h):
    if not isinstance(h, str):
        return RGBColor(0x2C, 0x3E, 0x50)
    h = h.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


_SUB = {'0': '\u2080', '1': '\u2081', '2': '\u2082', '3': '\u2083',
        '4': '\u2084', '5': '\u2085', '6': '\u2086', '7': '\u2087',
        '8': '\u2088', '9': '\u2089', 't': '\u209c', 'T': '\u1d40'}
_SUP = {'0': '\u2070', '1': '\u00b9', '2': '\u00b2', '3': '\u00b3', '4': '\u2074', '5': '\u2075', '6': '\u2076', '7': '\u2077', '8': '\u2078', '9': '\u2079'}
_SYM = [
    ('\\mathcal{L}', 'L'), ('\\mathbb{R}', '\u211d'),
    ('\\lambda', '\u03bb'), ('\\gamma', '\u03b3'),
    ('\\epsilon', '\u03b5'), ('\\eta', '\u03b7'),
    ('\\sigma', '\u03c3'), ('\\mu', '\u03bc'),
    ('\\alpha', '\u03b1'), ('\\beta', '\u03b2'),
    ('\\delta', '\u03b4'), ('\\odot', '\u2299'),
    ('\\rightarrow', '\u2192'), ('\\dashv', '\u22a3'),
    ('\\sim', '~'), ('\\in', '\u2208'),
    # \cdots must precede \cdot: replacement is sequential, so the shorter token
    # would otherwise match the prefix and leave a stray 's' ("\u00b7s").
    ('\\cdots', '\u22ef'), ('\\cdot', '\u00b7'),
    ('\\times', '\u00d7'), ('\\|', '\u2016'), ('\\,', ' '),
]


def _hat(txt):
    """\hat{x} -> x + combining circumflex (renders as x-hat in PowerPoint)."""
    def rep(m):
        inner = m.group(1)
        return inner[0] + '\u0302' + inner[1:] if inner else inner
    return re.sub(r'\\hat\{([^}]*)\}', rep, txt)


def _subscript(txt):
    """x_{abc} -> unicode subscript when all chars map, else x_abc."""
    def rep(m):
        base, sub = m.group(1), m.group(2)
        if sub and all(c in _SUB for c in sub):
            return base + ''.join(_SUB[c] for c in sub)
        return base + '_' + sub
    txt = re.sub(r'([\w}])_\{([^}]*)\}', rep, txt)
    return re.sub(r'([\w}\u2016)])_(\w)(?![\w])',
                  lambda m: m.group(1) + _SUB.get(m.group(2),
                                                 '_' + m.group(2)), txt)


def _superscript(txt):
    def rep(m):
        base, sup = m.group(1), m.group(2)
        if sup and all(c in _SUP for c in sup):
            return base + ''.join(_SUP[c] for c in sup)
        return base + '^' + sup
    txt = re.sub(r'([\w}\u2016)])\^\{([^}]*)\}', rep, txt)
    return re.sub(r'([\w}\u2016)])\^(\w)(?![\w])',
                  lambda m: m.group(1) + _SUP.get(m.group(2),
                                                 '^' + m.group(2)), txt)


def demath(s):
    """matplotlib mathtext -> plain unicode (PowerPoint has no mathtext)."""
    if not isinstance(s, str):
        return str(s)
    out = s
    for k, v in _SYM:
        out = out.replace(k, v)
    out = _superscript(_subscript(out))
    out = _hat(out)
    out = out.replace('$', '')
    out = re.sub(r'\{([^{}]*)\}', r'\1', out)
    return out.replace('\\', '')


REC = {'box': [], 'text': [], 'arrow': [], 'poly': []}


def _rbox(xy, w, h, fc, ec=None, lw=1.2, alpha=1.0, zo=2, r=0.15):
    REC['box'].append(dict(x=xy[0], y=xy[1], w=w, h=h, fc=fc, ec=ec,
                           lw=lw, alpha=alpha, zo=zo, r=r, dashed=False))


def _T(x, y, s, fs=14, c='#2C3E50', ha='center', va='center', w='normal',
       zo=5, **kw):
    # a matplotlib bbox becomes a real solid fill on the text box, otherwise
    # lines drawn underneath show through the label in PowerPoint
    bb = kw.get('bbox') or {}
    fill = bb.get('fc') or bb.get('facecolor')
    REC['text'].append(dict(x=x, y=y, s=demath(s), fs=fs, c=c, ha=ha,
                            bold=(w == 'bold'), zo=zo, fill=fill))


def _A(p1, p2, c='#2C3E50', lw=1.5, sty='->', ls='-', zo=3):
    REC['arrow'].append(dict(p1=p1, p2=p2, c=c, lw=lw, sty=sty, ls=ls, zo=zo))


class _FakeAx:
    def plot(self, xs, ys, color='#2C3E50', lw=1.5, ls='-', zorder=3,
             alpha=1.0, **kw):
        REC['poly'].append(dict(xs=list(np.atleast_1d(xs)),
                                ys=list(np.atleast_1d(ys)),
                                c=color, lw=lw, ls=ls))

    def add_patch(self, p):
        try:
            xy = p.get_xy() if hasattr(p, 'get_xy') else (0, 0)
            bb = p.get_extents()
            fc = p.get_facecolor()
            ec = p.get_edgecolor()
            import matplotlib.colors as mc
            REC['box'].append(dict(
                x=xy[0], y=xy[1],
                w=getattr(p, '_width', 1.0), h=getattr(p, '_height', 1.0),
                fc=mc.to_hex(fc), ec=mc.to_hex(ec),
                lw=p.get_linewidth(), alpha=1.0, zo=p.get_zorder() or 2,
                r=0.08, dashed=(p.get_linestyle() in ('--', 'dashed'))))
        except Exception:
            pass

    def text(self, x, y, s, fontsize=12, color='#2C3E50', ha='center',
             va='center', **kw):
        _T(x, y, s, fs=fontsize, c=color, ha=ha)

    def annotate(self, *a, **k):
        ap = k.get('arrowprops') or {}
        xy, xytext = k.get('xy'), k.get('xytext')
        if xy is None or xytext is None:
            return
        REC['arrow'].append(dict(p1=xytext, p2=xy,
                                 c=ap.get('color', '#2C3E50'),
                                 lw=ap.get('lw', 1.5),
                                 sty=ap.get('arrowstyle', '->'),
                                 ls='-', zo=k.get('zorder', 3)))

    def __getattr__(self, name):
        return lambda *a, **k: None


def record():
    src = open(SRC, encoding='utf8').read()
    lines = src.split(chr(10))
    i0 = next(i for i, l in enumerate(lines) if l.startswith('C = {'))
    DROP = ('plt.savefig', 'plt.tight_layout', 'plt.close', 'print(',
            'ax.set_', 'ax.axis', 'fig,', 'fig ')
    keep = []
    skip_def = False
    depth = 0
    dropping = False
    for l in lines[i0:]:
        st = l.strip()
        if dropping:
            depth += l.count('(') - l.count(')')
            if depth <= 0:
                dropping = False
                depth = 0
            continue
        if st.startswith('def '):
            skip_def = True
            continue
        if skip_def:
            if l and not l[0].isspace():
                skip_def = False
            else:
                continue
        if st.startswith(DROP) or 'FancyBboxPatch' in st \
           or st == 'ax.add_patch(p_)':
            d = l.count('(') - l.count(')')
            if d > 0:
                dropping = True
                depth = d
            continue
        keep.append(l)
    # cut everything from the output/save block onward
    cut = len(keep)
    for i, l in enumerate(keep):
        if l.startswith('try:') or l.startswith('plt.savefig'):
            cut = i
            break
    code = chr(10).join(keep[:cut])
    # side-panel FancyBboxPatch in v5 -> record as a dashed box
    _rbox((15.1, 4.05), 5.0, 1.05, '#FDFEFE', '#7F8C8D', lw=1.0, zo=2, r=0.08)
    REC['box'][-1]['dashed'] = True
    g = {'rbox': _rbox, 'T': _T, 'A': _A, 'ax': _FakeAx(), 'np': np,
         'FancyBboxPatch': lambda *a, **k: None, '__name__': '__rec__'}
    exec(compile(code, '<v5rec>', 'exec'), g)
    print('recorded: %d boxes, %d texts, %d arrows, %d polylines'
          % (len(REC['box']), len(REC['text']), len(REC['arrow']),
             len(REC['poly'])))


# ------------------------------------------------------------------ emit pass
def _dash(lnpr, kind):
    d = lnpr.find(qn('a:prstDash'))
    if d is None:
        d = lnpr.makeelement(qn('a:prstDash'), {})
        lnpr.append(d)
    d.set('val', kind)


def add_box(slide, b):
    x, y = sx(b['x']), sy(b['y'] + b['h'])
    w, h = b['w'] * SX, b['h'] * SY
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    shp.shadow.inherit = False
    f = shp.fill
    f.solid()
    f.fore_color.rgb = rgb(b['fc'])
    if b.get('alpha', 1.0) < 1.0:
        f.fore_color._xFill.find(qn('a:srgbClr')).append(
            f.fore_color._xFill.makeelement(
                qn('a:alpha'), {'val': str(int(b['alpha'] * 100000))}))
    ln = shp.line
    if b.get('ec'):
        ln.color.rgb = rgb(b['ec'])
        ln.width = Pt(max(0.5, b['lw']))
        if b.get('dashed'):
            _dash(ln._get_or_add_ln(), 'dash')
    else:
        ln.fill.background()
    # corner radius -> adjustment value (fraction of half the short side)
    try:
        short = min(w, h)
        adj = min(0.5, max(0.0, (b['r'] * SX) / short)) if short else 0.1
        shp.adjustments[0] = adj
    except Exception:
        pass
    shp.text_frame.text = ''
    return shp


def add_text(slide, t):
    fs = t['fs']
    lines = str(t['s']).split('\n')
    # Arial averages ~0.52 em per character; size the box to the widest line so
    # the frame tracks the glyphs instead of running off the slide
    ncols = max(4, max(len(ln) for ln in lines))
    est_w = max(0.35, 0.52 * ncols * fs / 72.0)
    est_h = max(0.20, 1.28 * len(lines) * fs / 72.0)
    cx, cy = sx(t['x']), sy(t['y'])
    if t['ha'] == 'left':
        left = cx
    elif t['ha'] == 'right':
        left = cx - est_w
    else:
        left = cx - est_w / 2.0
    top = cy - est_h / 2.0
    tb = slide.shapes.add_textbox(Inches(left), Inches(top),
                                  Inches(est_w), Inches(est_h))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if t.get('fill'):
        tb.fill.solid()
        tb.fill.fore_color.rgb = rgb(t['fill'])
        tb.line.fill.background()
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = {'left': PP_ALIGN.LEFT, 'right': PP_ALIGN.RIGHT}.get(
            t['ha'], PP_ALIGN.CENTER)
        run = para.add_run()
        run.text = line
        run.font.size = Pt(fs)
        run.font.bold = t['bold']
        run.font.color.rgb = rgb(t['c'])
        run.font.name = 'Arial'
    return tb


def add_arrow(slide, a):
    x1, y1 = sx(a['p1'][0]), sy(a['p1'][1])
    x2, y2 = sx(a['p2'][0]), sy(a['p2'][1])
    cxn = slide.shapes.add_connector(2, Inches(x1), Inches(y1),
                                     Inches(x2), Inches(y2))  # STRAIGHT
    ln = cxn.line
    ln.color.rgb = rgb(a['c'])
    ln.width = Pt(max(0.75, a['lw']))
    lnpr = ln._get_or_add_ln()
    if a['ls'] in ('--', 'dashed'):
        _dash(lnpr, 'dash')
    if '>' in (a['sty'] or ''):
        he = lnpr.makeelement(qn('a:headEnd'), {'type': 'none'})
        te = lnpr.makeelement(qn('a:tailEnd'), {'type': 'triangle',
                                               'w': 'med', 'len': 'med'})
        lnpr.append(he)
        lnpr.append(te)
    return cxn


def add_poly(slide, p):
    xs, ys = p['xs'], p['ys']
    if len(xs) < 2:
        return None
    b = slide.shapes.build_freeform(Inches(sx(xs[0])), Inches(sy(ys[0])))
    b.add_line_segments([(Inches(sx(x)), Inches(sy(y)))
                         for x, y in zip(xs[1:], ys[1:])], close=False)
    shp = b.convert_to_shape()
    shp.fill.background()
    ln = shp.line
    ln.color.rgb = rgb(p['c'] if isinstance(p['c'], str) else '#2C3E50')
    ln.width = Pt(max(0.75, p['lw']))
    if p['ls'] in ('--', 'dashed'):
        _dash(ln._get_or_add_ln(), 'dash')
    return shp


def emit():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    n = 0
    for b in sorted(REC['box'], key=lambda d: d.get('zo', 2)):
        add_box(slide, b); n += 1
    for p in REC['poly']:
        if add_poly(slide, p) is not None:
            n += 1
    for a in REC['arrow']:
        add_arrow(slide, a); n += 1
    for t in sorted(REC['text'], key=lambda d: d.get('zo', 5)):
        add_text(slide, t); n += 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT)
    print('shapes emitted:', n)
    print('saved:', OUT)
    return OUT


if __name__ == '__main__':
    record()
    emit()
