"""Every training run in the project plan, grouped by phase. Finished runs are skipped, so it resumes.

  python -m ext.grid --dry_run                       # list runs
  python -m ext.grid --phase 3,4 --parallel 3        # lab GPU: 3 runs at a time
  python -m ext.grid --phase 3 --seeds 1             # quick pass, one seed

Phases: 3 = paper model with/without early popularity, 4 = Extension A (CLIP),
5 = Extension C ablations + channel split, 7 = days of early popularity (optional).
Extra arguments after `--` go to every ext.train call, e.g. `-- --in_memory --device cuda`.
"""
import argparse
import os
import subprocess
import sys
import time

from ext.common import DEFAULT_RUNS, REPO_DIR


def runs(seeds):
    R = []
    S = range(seeds)
    for ep in (0, 1):
        R += [(3, f"rand_ep{ep}_resnet_s{s}", ["--ep_days", str(ep)]) for s in S]
    for tag, extra in [("clip", ["--image", "clip"]), ("both", ["--image", "both"]),
                       ("noimg", ["--image", "none"]), ("clipsim", ["--image", "clip", "--clip_sim"])]:
        R += [(4, f"rand_ep0_{tag}_s{s}", extra) for s in S]
    R += [(4, f"rand_ep1_clip_s{s}", ["--ep_days", "1", "--image", "clip"]) for s in S]
    for tag, extra in [("notext", ["--no_text"]), ("nocatlang", ["--no_catlang"]), ("nonumeric", ["--no_numeric"])]:
        R += [(5, f"rand_ep0_{tag}_s{s}", extra) for s in S]
    for ep in (0, 1):
        for img in ("resnet", "clip"):
            R += [(5, f"chan_ep{ep}_{img}_s{s}", ["--split", "channel", "--ep_days", str(ep), "--image", img])
                  for s in S]
    for k in (2, 3, 7):
        R += [(7, f"rand_ep{k}_resnet_s{s}", ["--ep_days", str(k)]) for s in S]
    # seed is the trailing digit of the name
    return [(ph, name, extra + ["--seed", name.rsplit("_s", 1)[1]]) for ph, name, extra in R]


def main():
    argv = sys.argv[1:]
    passthrough = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", default="3,4,5,7")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--runs_dir", default=DEFAULT_RUNS)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)

    phases = {int(x) for x in args.phase.split(",")}
    todo = [(n, e) for ph, n, e in runs(args.seeds) if ph in phases
            and not os.path.exists(os.path.join(args.runs_dir, n, "metrics.json"))]
    print(f"{len(todo)} runs to do in phases {sorted(phases)}")
    if args.dry_run:
        for n, e in todo:
            print(f"  {n:28s} {' '.join(e)}")
        return
    # Create the split files once, before parallel runs could race to write them.
    from ext.common import DEFAULT_FEATURES
    from ext.data import make_split
    feats = passthrough[passthrough.index("--features") + 1] if "--features" in passthrough else DEFAULT_FEATURES
    for kind in ("random", "channel"):
        make_split(feats, kind, 23)
    active, t0 = [], time.time()
    while todo or active:
        while todo and len(active) < args.parallel:
            n, e = todo.pop(0)
            os.makedirs(os.path.join(args.runs_dir, n), exist_ok=True)
            log = open(os.path.join(args.runs_dir, n, "train.log"), "w")
            cmd = [sys.executable, "-m", "ext.train", "--name", n, "--runs_dir", args.runs_dir] + e + passthrough
            active.append((n, subprocess.Popen(cmd, cwd=REPO_DIR, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"[{(time.time() - t0) / 60:6.1f} min] started {n}", flush=True)
        for item in list(active):
            n, proc, log = item
            if proc.poll() is not None:
                log.close(); active.remove(item)
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode}), see {n}/train.log"
                print(f"[{(time.time() - t0) / 60:6.1f} min] finished {n}: {status}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
