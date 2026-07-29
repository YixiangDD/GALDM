"""Serial queue: once the comparison run finishes, runs (1) the GRN ablation over 5 seeds and
then (2) the cross-modal Pareto sweep. Both use the current code and the main experiment epoch
budget, writing into results/. Start this by hand after the comparison completes.

Usage: python tools/run_gpu_queue.py 2>&1 | tee results/_gpu_queue.log
"""
import subprocess, sys, time, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
env = dict(os.environ, PYTHONIOENCODING="utf-8")

JOBS = [
    ("GRN ablation, 5 seeds (CESC)",
     [PY, "-u", "-m", "experiments.grn_ablation",
      "--cancer", "CESC", "--seeds", "42,123,456,789,1024"]),
    ("cross-modal Pareto sweep (CESC, lambda_cm)",
     [PY, "-u", "-m", "experiments.pareto_crossmodal",
      "--cancer", "CESC", "--seeds", "42,123,456,789,1024",
      "--lambdas", "0,0.25,0.5,1.0,2.0,4.0"]),
]

for name, cmd in JOBS:
    print(f"\n{'='*70}\n[GPU QUEUE] START: {name}\n  cmd: {' '.join(cmd)}\n{'='*70}",
          flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    print(f"[GPU QUEUE] DONE: {name}  (rc={rc}, {time.time()-t0:.0f}s)", flush=True)
    if rc != 0:
        print(f"[GPU QUEUE] non-zero exit; stopping the queue so it can be investigated.", flush=True)
        sys.exit(rc)

print("\n[GPU QUEUE] all done.", flush=True)
