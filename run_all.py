"""Single entry point for the whole experiment suite.

Usage:
  # smoke test (~10 min)
  python run_all.py --quick

  # medium scale (~30 min)
  python run_all.py --medium

  # full protocol (~60-80 GPU-hours)
  python run_all.py --full

  # a single experiment
  python run_all.py --exp comparison --cancers CESC --seeds 42
  python run_all.py --exp ablation --cancers CESC,KIRC --mode both
  python run_all.py --exp sensitivity
  python run_all.py --exp grn_ablation
  python run_all.py --exp interpretability
"""
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="GA-LDM experiment runner")
    ap.add_argument("--exp", default="all",
                    choices=["all", "comparison", "ablation", "sensitivity",
                             "grn_ablation", "interpretability"])
    ap.add_argument("--cancers", default="CESC,COAD,HNSC,KIRC")
    ap.add_argument("--seeds", default="42,123,456,789,1024")
    ap.add_argument("--quick", action="store_true", help="smoke test, ~10 min")
    ap.add_argument("--medium", action="store_true", help="medium, ~30 min per cohort")
    ap.add_argument("--full", action="store_true", help="full protocol from the paper")
    ap.add_argument("--mode", default="both", choices=["progressive", "leave_one_out", "both"])
    args = ap.parse_args()

    cancers = [c.strip() for c in args.cancers.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    # epoch budget
    if args.quick:
        vae_ep, dit_ep, base_ep = 20, 30, 30
    elif args.medium:
        vae_ep, dit_ep, base_ep = 200, 400, 300
    elif args.full:
        vae_ep, dit_ep, base_ep = 500, 800, 500
    else:
        vae_ep, dit_ep, base_ep = 500, 800, 500  # full protocol default

    t0 = time.time()
    print(f"{'='*60}")
    print(f"GA-LDM experiments")
    print(f"experiment: {args.exp} | cohorts: {cancers} | seeds: {seeds}")
    print(f"budget: VAE {vae_ep} ep / DiT {dit_ep} ep / baselines {base_ep} ep")
    print(f"{'='*60}\n")

    if args.exp in ("all", "comparison"):
        print("\n" + "="*40 + " comparison " + "="*40)
        from experiments.comparison import run_comparison
        run_comparison(cancers, seeds, quick=args.quick,
                       vae_ep=vae_ep, dit_ep=dit_ep, base_ep=base_ep)

    if args.exp in ("all", "ablation"):
        print("\n" + "="*40 + " ablation " + "="*40)
        from experiments.ablation import run_ablation
        abl_cancers = ["CESC", "KIRC"] if args.exp == "all" else cancers
        modes = ["progressive", "leave_one_out"] if args.mode == "both" else [args.mode]
        for mode in modes:
            run_ablation(abl_cancers, seeds, mode, quick=args.quick,
                         vae_ep=vae_ep, dit_ep=dit_ep)

    if args.exp in ("all", "sensitivity"):
        print("\n" + "="*40 + " hyper-parameter sensitivity " + "="*40)
        from experiments.sensitivity import run_sensitivity
        run_sensitivity("CESC", seed=42, quick=args.quick,
                        vae_ep=vae_ep, dit_ep=dit_ep)

    if args.exp in ("all", "grn_ablation"):
        print("\n" + "="*40 + " GRN ablation " + "="*40)
        from experiments.grn_ablation import run_grn_ablation
        run_grn_ablation("CESC", seed=42, quick=args.quick,
                         vae_ep=vae_ep, dit_ep=dit_ep)

    if args.exp in ("all", "interpretability"):
        print("\n" + "="*40 + " interpretability case study " + "="*40)
        from experiments.interpretability import run_interpretability
        run_interpretability("KIRC", seed=42, quick=args.quick,
                             vae_ep=vae_ep, dit_ep=dit_ep)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"done. total {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")
    print(f"results written to: {_P.RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
