# -*- coding: utf-8 -*-
"""Runs the novelty analysis (NN distance plus duplicate rate at several thresholds) on COAD,
HNSC and KIRC, so the comparison against SMOTE covers all four cohorts rather than CESC alone.
Reuses novelty_eval.run(), which is already parameterised by cohort. Writes
results/novelty_sensitivity_{cancer}.json in the same format as CESC."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.novelty_eval import run

if __name__ == "__main__":
    t0 = time.time()
    for c in ["COAD", "HNSC", "KIRC"]:
        print("\n########## %s ##########" % c, flush=True)
        run(c)
    print("\n[ALL done] %.0fs" % (time.time() - t0))
