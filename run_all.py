#!/usr/bin/env python3
"""Run the tests and all scripts; every table and figure is written to --outdir (default: results).

  python run_all.py            full run
  python run_all.py --quick    reduced run (small n, coarse grids)
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def steps(out, quick):
    q = ["--quick"] if quick else []
    return [
        ("tests", ["tests/test_all.py"]),
        ("collective_entanglement", ["collective_entanglement.py", "--outdir", out] + q),
        ("uniform_drive", ["uniform_drive.py", "--outdir", out]),
        ("effective_model", ["effective_model.py", "--outdir", out, "--max-n", "5" if quick else "6"]),
        ("scaling_symmetric", ["scaling_symmetric.py", "--outdir", out] + (["--nmax-eff", "12", "--nmax-full", "9"] if quick else [])),
        ("extended_results", ["extended_results.py", "--outdir", out] + q),
        ("independent_decay", ["independent_decay.py", "--outdir", out] + q),
        ("peak_map", ["peak_map.py", "--outdir", out] + q),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, a.outdir), exist_ok=True)
    t0 = time.time()
    for name, cmd in steps(a.outdir, a.quick):
        if name == "tests" and a.skip_tests:
            continue
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable] + cmd, cwd=HERE, check=True)
        print(f"--- {name}: {time.time() - t:.0f} s", flush=True)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs: {os.path.join(HERE, a.outdir)}")


if __name__ == "__main__":
    main()
