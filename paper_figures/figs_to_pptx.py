# -*- coding: utf-8 -*-
"""Export the data figures as PPTX files in which every component is editable.

Each panel becomes a native PowerPoint chart (right-click -> Edit Data works),
and every title, axis label and annotation is a separate text box. No panel is
a flattened bitmap, so the whole figure can be restyled inside PowerPoint.

Outputs one .pptx per figure into OUTDIR.
"""
import json
import os
import sys

import numpy as np
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_MARK
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P  # noqa: E402

R = _P.RESULTS_DIR + os.sep
# override with GALDM_PPTX_DIR if you want the decks somewhere else
OUTDIR = os.environ.get("GALDM_PPTX_DIR",
                        os.path.join(_P.FIGURES_DIR, "pptx")) + os.sep
os.makedirs(OUTDIR, exist_ok=True)

BLUE = RGBColor(0x2C, 0x7F, 0xB8)
RED = RGBColor(0xDE, 0x2D, 0x26)
GREY = RGBColor(0x7F, 0x8C, 0x8D)
INK = RGBColor(0x2C, 0x3E, 0x50)
SLIDE_W, SLIDE_H = 13.333, 7.5


def deck():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    return prs, prs.slides.add_slide(prs.slide_layouts[6])


def label(slide, x, y, text, size=13, bold=False, colour=INK, w=5.0, h=0.34,
          align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = "Arial"
    r.font.color.rgb = colour
    return tb


def style_axes(chart, xtitle=None, ytitle=None, size=9):
    for ax, title in ((chart.category_axis if chart.has_title is None else None,
                       None),):
        pass
    for ax, title in ((chart.value_axis, ytitle),):
        ax.has_major_gridlines = True
        ax.major_tick_mark = XL_TICK_MARK.OUTSIDE
        ax.tick_labels.font.size = Pt(size)
        ax.tick_labels.font.name = "Arial"
        if title:
            ax.has_title = True
            ax.axis_title.text_frame.text = title
            f = ax.axis_title.text_frame.paragraphs[0].runs[0].font
            f.size = Pt(size + 1)
            f.name = "Arial"
            f.bold = False
    try:
        ax = chart.category_axis
    except Exception:
        return
    ax.major_tick_mark = XL_TICK_MARK.OUTSIDE
    ax.tick_labels.font.size = Pt(size)
    ax.tick_labels.font.name = "Arial"
    if xtitle:
        ax.has_title = True
        ax.axis_title.text_frame.text = xtitle
        f = ax.axis_title.text_frame.paragraphs[0].runs[0].font
        f.size = Pt(size + 1)
        f.name = "Arial"
        f.bold = False


def scatter(slide, x, y, w, h, series, xtitle, ytitle, size=9, msize=5,
            legend=True):
    """series: list of (name, xs, ys, RGBColor)."""
    cd = XyChartData()
    for name, xs, ys, _c in series:
        s = cd.add_series(name)
        for xv, yv in zip(xs, ys):
            s.add_data_point(float(xv), float(yv))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(x), Inches(y),
                               Inches(w), Inches(h), cd)
    chart = gf.chart
    chart.has_title = False
    for ser, (_n, _xs, _ys, colour) in zip(chart.series, series):
        ser.marker.style = 8            # circle
        ser.marker.size = msize
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = colour
        ser.marker.format.line.color.rgb = colour
        ser.format.line.fill.background()
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
        chart.legend.font.name = "Arial"
    style_axes(chart, xtitle, ytitle, size)
    return chart


def bars(slide, x, y, w, h, cats, series, ytitle, size=9, legend=True):
    """series: list of (name, values, RGBColor)."""
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals, _c in series:
        cd.add_series(name, [float(v) for v in vals])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x),
                                Inches(y), Inches(w), Inches(h), cd)
    chart = gf.chart
    chart.has_title = False
    for ser, (_n, _v, colour) in zip(chart.series, series):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = colour
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
        chart.legend.font.name = "Arial"
    style_axes(chart, None, ytitle, size)
    return chart


# ----------------------------------------------------------------------- Fig 3
def fig3():
    bio = json.load(open(R + "bio_readout.json", encoding="utf-8"))
    ev = json.load(open(R + "edge_vectors_CESC.json", encoding="utf-8"))
    ho = json.load(open(R + "t2_heldout_CESC.json", encoding="utf-8"))
    prs, sl = deck()

    label(sl, 0.45, 0.22, "a  Stage-associated pathway programmes", 14, True)
    ser = []
    for cohort, colour in (("CESC", BLUE), ("KIRC", RED)):
        pp = bio[cohort]["per_pathway"]
        names = bio[cohort]["pathway_names"]
        keep = [i for i, n in enumerate(names) if n != "Background"]
        s = bio[cohort]["t_spearman"]
        ser.append((f"{cohort}  rho = {s['mean']:.2f}",
                    [pp["t_real_mean"][i] for i in keep],
                    [pp["t_gen_mean"][i] for i in keep], colour))
    lo, hi = -3.0, 1.2
    ser.append(("identity", [lo, hi], [lo, hi], GREY))
    scatter(sl, 0.45, 0.62, 4.0, 3.1, ser,
            "Real cohort: Welch t (late vs early)",
            "Generated cohort: Welch t")
    label(sl, 0.45, 3.80,
          "Points are 5-seed means over the 15 KEGG pathways; the background "
          "gene group is excluded here (the print figure shows it as an open "
          "marker). rho is the per-seed mean over all 16 groups. The third "
          "series is the identity line.", 9, False,
          GREY, w=4.0, h=0.6)

    label(sl, 4.72, 0.22,
          f"b  {ev['n_edges']} miRTarBase-validated edges (CESC)", 14, True)
    scatter(sl, 4.72, 0.62, 4.0, 3.1, [
        (f"GA-LDM  r = {ev['pearson_real_gen']:.2f}", ev["real"], ev["gen"], BLUE),
        (f"pairing permuted  r = {ev['pearson_real_shuffled']:.2f}",
         ev["real"], ev["gen_shuffled"], GREY),
    ], "Real: corr(miRNA, target mRNA)",
        "Generated: corr(miRNA, target mRNA)", msize=3)
    label(sl, 4.72, 3.80,
          "This panel is one seed (42), in which 84% of repressive edges keep "
          "a negative sign; over 5 seeds the mean is 87 +/- 2%.", 9, False,
          GREY, w=4.0, h=0.6)

    label(sl, 9.0, 0.22, "c  Dependency on unseen edges (CESC)", 14, True)
    rows = [("Training edges", ho["Full"]["train"]),
            ("Held-out edges", ho["Full"]["heldE"]),
            ("Held-out miRNAs", ho["Full"]["heldN"]),
            ("Held-out edges,\nno cross-modal attn.", ho["wo_CrossModal"]["heldE"])]
    cats = [r[0].replace("\n", " ") for r in rows]
    bars(sl, 9.0, 0.62, 3.9, 3.1, cats,
         [("XM-Corr (5-seed mean)", [np.mean(r[1]) for r in rows], BLUE)],
         "XM-Corr on miRTarBase edges", legend=False)
    null = float(np.mean(ho["Full"]["heldE_null"]))
    label(sl, 9.0, 3.80,
          f"Per-seed values are in the chart data. Permuted pairing collapses "
          f"to {abs(null):.2f} (permutation P < 0.001).", 9, False, GREY,
          w=3.9, h=0.6)

    label(sl, 0.45, 4.62, "Fig. 3. Biological fidelity of generated cohorts.",
          13, True)
    label(sl, 0.45, 4.98,
          "Every panel is a native PowerPoint chart: right-click a series and "
          "choose Edit Data to see or change the underlying values. Titles, "
          "axis labels and notes are separate text boxes.", 11, False, GREY,
          w=12.4, h=0.9)
    out = OUTDIR + "Fig3_bioreadout.pptx"
    prs.save(out)
    print("saved", out)


# ----------------------------------------------------------------------- Fig 4
def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a.mean() - b.mean()) /
                 (np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12))


def fig4():
    pc = json.load(open(R + "prior_scarcity_CESC.json", encoding="utf-8"))
    pk = json.load(open(R + "prior_scarcity_KIRC.json", encoding="utf-8"))
    t6 = json.load(open(R + "t6_gadit_CESC.json", encoding="utf-8"))
    prs, sl = deck()

    panels = [
        ("a  Grouping, CESC", pc["N199_KEGG"]["Bio_FID"],
         pc["N199_Random"]["Bio_FID"], "KEGG / family", "matched random"),
        ("b  Grouping, KIRC", pk["N358_KEGG"]["Bio_FID"],
         pk["N358_Random"]["Bio_FID"], "KEGG / family", "matched random"),
        ("c  Identity encoding, CESC", t6["GA-DiT"]["Bio_FID"],
         t6["plain_DiT"]["Bio_FID"], "GA-DiT", "plain DiT"),
    ]
    for i, (title, ours, theirs, n1, n2) in enumerate(panels):
        x = 0.45 + i * 3.16
        label(sl, x, 0.22, title, 13, True, w=3.1)
        # one category per seed keeps every individual split editable
        cats = [f"seed {j + 1}" for j in range(len(ours))]
        bars(sl, x, 0.62, 2.95, 3.1, cats,
             [(n1, ours, BLUE), (n2, theirs, RED)], "Bio-FID (lower better)")
        label(sl, x, 3.80,
              f"d = {cohen_d(ours, theirs):.2f}, "
              f"{int((np.asarray(ours) < np.asarray(theirs)).sum())}/"
              f"{len(ours)} seeds; means "
              f"{np.mean(ours):.1f} vs {np.mean(theirs):.1f}", 9, False, GREY,
              w=2.95, h=0.6)

    cohorts = ["CESC", "COAD", "HNSC", "KIRC"]
    gm, sm, gd, sd = [], [], [], []
    for c in cohorts:
        n = json.load(open(R + f"novelty_sensitivity_{c}.json", encoding="utf-8"))
        gm.append(n["GA-LDM"]["ratio_to_real"][0])
        sm.append(n["SMOTE"]["ratio_to_real"][0])
        gd.append(n["GA-LDM"]["dup_rate"][0])
        sd.append(n["SMOTE"]["dup_rate"][0])
    label(sl, 9.93, 0.22, "d  Novelty versus memorisation", 13, True, w=3.1)
    bars(sl, 9.93, 0.62, 2.95, 3.1, cohorts,
         [("GA-LDM", gm, BLUE), ("SMOTE", sm, RED)],
         "NN distance / a real sample's")
    label(sl, 9.93, 3.80,
          "Near-duplicate rate: GA-LDM " +
          ", ".join(f"{100 * v:.0f}%" for v in gd) + "; SMOTE " +
          ", ".join(f"{100 * v:.0f}%" for v in sd) +
          ". 1.0 = a real held-out sample.", 9, False, GREY, w=2.95, h=0.75)

    label(sl, 0.45, 4.62,
          "Fig. 4. What the structural prior buys, and whether the samples are "
          "novel.", 13, True, w=12.4)
    label(sl, 0.45, 4.98,
          "Panels a-c show all five per-seed values as paired columns so no "
          "split is hidden behind a mean; effect sizes use the "
          "independent-samples Cohen's d of the main text. Every panel is a "
          "native chart (right-click -> Edit Data).", 11, False, GREY,
          w=12.4, h=0.9)
    out = OUTDIR + "Fig4_prior_novelty.pptx"
    prs.save(out)
    print("saved", out)


# ------------------------------------------------------------------- Fig S1/S3
def figS1():
    """Coupling-weight sweep. Two measures of different scale -> two charts."""
    d = json.load(open(R + "pareto_crossmodal_CESC.json", encoding="utf-8"))
    ws = sorted(d, key=float)
    prs, sl = deck()
    label(sl, 0.6, 0.25, "a  Distributional fidelity", 14, True)
    bars(sl, 0.6, 0.72, 5.6, 4.3, [f"lambda_cm = {w}" for w in ws],
         [("Bio-FID (5-seed mean)", [np.mean(d[w]["Bio_FID"]) for w in ws], BLUE)],
         "Bio-FID (lower better)", legend=False)
    label(sl, 6.9, 0.25, "b  Within-family miRNA structure", 14, True)
    bars(sl, 6.9, 0.72, 5.6, 4.3, [f"lambda_cm = {w}" for w in ws],
         [("miRNA Corr (5-seed mean)",
           [np.mean(d[w]["miRNA_Corr"]) for w in ws], RED)],
         "miRNA Corr (higher better)", legend=False)
    label(sl, 0.6, 5.35,
          "Fig. S1. Sensitivity to the cross-modal coupling weight on CESC.",
          13, True, w=12.2)
    label(sl, 0.6, 5.72,
          "Raising the weight trades distributional fidelity against "
          "cross-modal structure: Bio-FID is best at 0 and degrades "
          "monotonically, while within-family miRNA correlation rises and "
          "saturates near 2. The two measures are on different scales and are "
          "therefore plotted as two charts rather than on twin axes. Per-seed "
          "values are in each chart's data.", 11, False, GREY, w=12.2, h=1.1)
    out = OUTDIR + "FigS1_coupling_weight.pptx"
    prs.save(out)
    print("saved", out)


def figS3():
    from pptx.enum.shapes import MSO_SHAPE
    src = R + "interpretability_KIRC_persample.json"
    if not os.path.exists(src):                       # fall back to the means
        src = R + "interpretability_KIRC.json"
    d = json.load(open(src, encoding="utf-8"))
    ps = d.get("per_sample")

    prs, sl = deck()
    label(sl, 0.6, 0.25, "HIF-1 signalling activity by stage (KIRC)", 14, True,
          w=8.0)

    if ps is None:
        bars(sl, 0.6, 0.75, 6.4, 4.2, ["Early stage", "Late stage"],
             [("Real cohort", d["real_activity_mean"], BLUE),
              ("Generated cohort", d["gen_activity_mean"], RED)],
             "Mean pathway activity")
        note = ("Only the group means were cached for this build, so the "
                "per-sample spread is not shown.")
    else:
        groups = [("Real early", ps["real_early"], BLUE),
                  ("Real late", ps["real_late"], RED),
                  ("Gen. early", ps["gen_early"], BLUE),
                  ("Gen. late", ps["gen_late"], RED)]
        # x slot per group, with jitter so overlapping samples stay visible
        rng = np.random.default_rng(0)
        series = []
        for i, (name, vals, colour) in enumerate(groups):
            v = np.asarray(vals, float)
            xs = (i + 1) + rng.uniform(-0.16, 0.16, v.size)
            series.append((f"{name}  (n={v.size})", xs, v, colour))

        PX, PY, PW, PH = 0.6, 0.80, 7.5, 4.15
        allv = np.concatenate([np.asarray(g[1], float) for g in groups])
        lo, hi = allv.min(), allv.max()
        pad = (hi - lo) * 0.10
        lo, hi = lo - pad, hi + pad

        ch = scatter(sl, PX, PY, PW, PH, series, "Cohort and stage",
                     "HIF-1 pathway activity", size=9, msize=4, legend=True)
        ch.value_axis.minimum_scale = float(lo)
        ch.value_axis.maximum_scale = float(hi)
        ch.category_axis.minimum_scale = 0.5
        ch.category_axis.maximum_scale = len(groups) + 0.5

        # plot-area geometry, so the native box marks land on the data
        ax_l, ax_r = PX + 0.72, PX + PW - 0.16
        ax_t, ax_b = PY + 0.48, PY + PH - 0.52

        def gx(slot):
            return ax_l + (slot - 0.5) / len(groups) * (ax_r - ax_l)

        def gy(val):
            return ax_b - (val - lo) / (hi - lo) * (ax_b - ax_t)

        for i, (name, vals, colour) in enumerate(groups):
            v = np.sort(np.asarray(vals, float))
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            iqr = q3 - q1
            wlo = v[v >= q1 - 1.5 * iqr].min()
            whi = v[v <= q3 + 1.5 * iqr].max()
            cx = gx(i + 1)
            bw = 0.42
            box = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                      Inches(cx - bw / 2), Inches(gy(q3)),
                                      Inches(bw), Inches(gy(q1) - gy(q3)))
            box.fill.background()
            box.line.color.rgb = INK
            box.line.width = Pt(1.1)
            box.shadow.inherit = False
            box.text_frame.text = ""
            for y1, y2, xa, xb in ((gy(whi), gy(q3), cx, cx),
                                   (gy(q1), gy(wlo), cx, cx),
                                   (gy(whi), gy(whi), cx - 0.10, cx + 0.10),
                                   (gy(wlo), gy(wlo), cx - 0.10, cx + 0.10),
                                   (gy(med), gy(med), cx - bw / 2, cx + bw / 2)):
                ln = sl.shapes.add_connector(1, Inches(xa), Inches(y1),
                                             Inches(xb), Inches(y2))
                ln.line.color.rgb = INK
                ln.line.width = Pt(1.6 if y1 == y2 == gy(med) else 1.0)

        note = ("Box = interquartile range, centre line = median, whiskers = "
                "1.5 x IQR. Every point is one sample and lives in the chart's "
                "own data sheet.")

    stats_txt = (
        f"Real cohort:  Mann-Whitney P = {d['real_mannwhitney_p']:.4f}\n"
        f"Generated cohort:  P = {d['gen_mannwhitney_p']:.4f}\n\n"
        f"Activity-distribution distance between the real and the generated "
        f"cohort: {d['activity_distribution_distance']:.4f}")
    if ps is not None:
        stats_txt += (
            "\n\nEarly-versus-late rank-biserial effect size:\n"
            "    real 0.164,  generated 0.116\n"
            "Group sizes are matched in power (harmonic-mean n 172 and 177), "
            "so the P-value gap reflects the weaker effect, not a weaker test.")
    label(sl, 8.35, 0.95, stats_txt, 12, False, INK, w=4.3, h=3.3)
    label(sl, 8.35, 4.45, note, 10, False, GREY, w=4.3, h=1.4)

    label(sl, 0.6, 5.30,
          "Fig. S3. HIF-1 pathway activity in real and generated KIRC samples "
          "by stage.", 13, True, w=7.5)
    label(sl, 0.6, 5.62,
          "Generated quartiles track the real ones closely in both stages "
          "(early Q1 3.474 versus 3.474, late Q3 3.646 versus 3.650), so the "
          "model reproduces the shape of the activity distribution and not "
          "only its mean. The stage contrast, however, is attenuated: the "
          "early-versus-late rank-biserial effect size falls from 0.164 in the "
          "real cohort to 0.116 in the generated one, which is why the real "
          "difference reaches P<0.01 while the generated one reaches only "
          "P=0.06. Bootstrapping each effect at both sets of group sizes "
          "shifts the P-values by under 0.005, so this gap is a genuine "
          "shrinkage of the biological contrast rather than a loss of "
          "statistical power. Marginal distributions are therefore easier for "
          "the model to match than between-stage separation.",
          11, False, GREY, w=7.5, h=1.75)
    out = OUTDIR + "FigS3_hif1.pptx"
    prs.save(out)
    print("saved", out)



# ----------------------------------------------------------------------- Fig 2
def fig2():
    """Six shared-UMAP panels, one native scatter chart each."""
    C = os.path.join(_P.FIGURES_DIR, "CESC", "_umap_coords_CESC.json")
    d = json.load(open(C, encoding="utf-8"))
    real = np.asarray(d["real"])
    prs, sl = deck()
    order = ["GA-LDM", "SMOTE", "CTGAN", "TabDDPM", "TabSyn", "scDiffusion"]
    for i, m in enumerate(order):
        g = np.asarray(d["gen"][m])
        col, row = i % 3, i // 3
        x = 0.42 + col * 4.28
        y = 0.55 + row * 2.62
        label(sl, x, y - 0.30, f"{'abcdef'[i]}  {m}", 12, True, w=4.0)
        scatter(sl, x, y, 4.05, 2.30,
                [("Real", real[:, 0], real[:, 1], BLUE),
                 ("Generated", g[:, 0], g[:, 1], RED)],
                None, None, size=8, msize=4, legend=(i == 0))
    label(sl, 0.42, 5.95,
          "Fig. 2. Shared-UMAP embedding of real versus generated CESC samples.",
          13, True, w=12.5)
    label(sl, 0.42, 6.32,
          "A single UMAP is fitted once on the real training data and every "
          "method is projected into those same coordinates, so the panels are "
          "directly comparable. GA-LDM covers both real clusters; TabDDPM "
          "misses the smaller cluster entirely and several baselines pile onto "
          "the periphery. TabDiff is omitted because its output is numerically "
          "indistinguishable from TabDDPM. Each panel is a native scatter chart "
          "holding its own coordinates.", 11, False, GREY, w=12.5, h=1.0)
    out = OUTDIR + "Fig2_shared_umap.pptx"
    prs.save(out)
    print("saved", out)


# ---------------------------------------------------------------------- Fig S2
def figS2():
    """Correlation heat maps as native rectangles: one panel per slide.

    A heat map has no native PowerPoint chart type, so each cell is emitted as
    its own rectangle. Splitting the seven panels across seven slides keeps
    each slide at 1600 shapes, which PowerPoint edits comfortably.
    """
    from matplotlib import colormaps, colors
    from pptx.enum.shapes import MSO_SHAPE

    C = os.path.join(_P.FIGURES_DIR, "CESC", "_corr_mats_CESC.json")
    if not os.path.exists(C):
        print("SKIP FigS2: run experiments/figs_shared.py first ->", C)
        return
    d = json.load(open(C, encoding="utf-8"))
    rv = json.load(open(os.path.join(_P.FIGURES_DIR, "CESC",
                                     "_rv_mantel_CESC.json"), encoding="utf-8"))
    cmap = colormaps["RdBu_r"]
    norm = colors.Normalize(-1, 1)

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    intro = prs.slides.add_slide(prs.slide_layouts[6])
    label(intro, 0.7, 1.2,
          "Fig. S2. Within-pathway gene-correlation matrices, real versus "
          "generated (CESC).", 20, True, w=12.0, h=0.8)
    label(intro, 0.7, 2.3,
          "One panel per slide that follows. Every cell is a separate, "
          "recolourable rectangle and every label is a text box, so the whole "
          "figure can be restyled in PowerPoint. Each panel shows the "
          "correlation matrix of the 40 retained member genes of the first KEGG "
          "pathway, on a shared -1 to +1 red-blue scale. PCE is the Frobenius "
          "distance to the real matrix (lower is better) and RV the matrix "
          "similarity coefficient (higher is better), with a Mantel permutation "
          "P-value.", 13, False, GREY, w=12.0, h=2.2)

    for name, mat in d["mats"].items():
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        M = np.asarray(mat)
        n = M.shape[0]
        side = 5.6 / n
        x0, y0 = 1.1, 1.15
        stat = rv.get(name)
        ttl = name if stat is None else (
            f"{name}    PCE = {stat['PCE']:.3f}    RV = {stat['RV']:.2f}"
            + ("    P < 0.001" if stat["mantel_p"] < 1e-3
               else f"    P = {stat['mantel_p']:.3f}"))
        label(sl, x0, 0.55, ttl, 16, True, w=9.0)
        for i in range(n):
            for j in range(n):
                sh = sl.shapes.add_shape(
                    MSO_SHAPE.RECTANGLE, Inches(x0 + j * side),
                    Inches(y0 + i * side), Inches(side), Inches(side))
                sh.shadow.inherit = False
                sh.fill.solid()
                r_, g_, b_ = [int(255 * v) for v in cmap(norm(M[i, j]))[:3]]
                sh.fill.fore_color.rgb = RGBColor(r_, g_, b_)
                sh.line.fill.background()
                sh.text_frame.text = ""
        # native colour bar
        for k in range(21):
            v = 1 - k / 10.0
            sh = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.35),
                                     Inches(y0 + k * 0.266), Inches(0.34),
                                     Inches(0.266))
            sh.shadow.inherit = False
            sh.fill.solid()
            r_, g_, b_ = [int(255 * c) for c in cmap(norm(v))[:3]]
            sh.fill.fore_color.rgb = RGBColor(r_, g_, b_)
            sh.line.fill.background()
        for k, v in ((0, "+1.0"), (10, "0.0"), (20, "-1.0")):
            label(sl, 7.78, y0 + k * 0.266 - 0.02, v, 11, False, INK, w=0.9)
        label(sl, 7.35, y0 + 5.75, "Pearson correlation", 11, False, GREY, w=2.2)
        label(sl, x0, 6.95,
              f"{n} x {n} member genes; {n * n} individually editable cells.",
              11, False, GREY, w=8.0)

    out = OUTDIR + "FigS2_corrmat.pptx"
    prs.save(out)
    print("saved", out, f"({len(prs.slides._sldIdLst)} slides)")







if __name__ == "__main__":
    fig2()
    fig3()
    fig4()
    figS1()
    figS2()
    figS3()
