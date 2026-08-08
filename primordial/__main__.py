import argparse, sys
from .pipeline import run

def _show_ledger(manifest_path):
    import json, os
    from .pipeline import LEDGER_PATH
    from .judge import Ledger
    label = {}
    if manifest_path:
        from .universe import Manifest
        mf = Manifest.load(manifest_path)
        label[mf.fingerprint()] = mf.name
    if not os.path.exists(LEDGER_PATH):
        print(f"no ledger yet at {LEDGER_PATH}")
        return
    with open(LEDGER_PATH) as f:
        d = json.load(f)
    print(f"ledger: {LEDGER_PATH}")
    for fp, entries in d.items():
        eff = Ledger(LEDGER_PATH, fp).effective_trials()
        name = label.get(fp, "")
        cands = sum(e.get("n_candidates", 0) for e in entries)
        peeks = sum(e.get("holdout_peeks", 0) for e in entries)
        print(f"  {fp}  {name:<20} entries {len(entries):>3} | candidates "
              f"{cands:>4} | peeks {peeks:>3} | EFFECTIVE {eff}")


def _doctor():
    import subprocess, sys, tempfile, os
    print("1/2 unit tests...")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       capture_output=True, text=True)
    print("   ", *(r.stdout or r.stderr).strip().splitlines()[-1:])
    if r.returncode != 0:
        print("DOCTOR: FAIL (tests)")
        return
    print("2/2 negative control (short)... ~1-2 min")
    env = dict(os.environ)
    tmp = tempfile.mkdtemp(prefix="prim_doctor_")
    env["PRIMORDIAL_CACHE"] = os.path.join(tmp, "cache")
    env["PRIMORDIAL_LEDGER"] = os.path.join(tmp, "ledger.json")
    r = subprocess.run([sys.executable, "-m", "primordial", "run",
                        "--manifest", "manifests/smoke_synthetic.yaml",
                        "--generations", "2", "--pop", "8", "--islands", "2"],
                       capture_output=True, text=True, env=env)
    ok = "0 survivor(s)" in (r.stdout or "")
    print("    0 survivors on random data:", "yes" if ok else "NO <-- problem")
    print("DOCTOR:", "OK" if ok else "FAIL (negative control leaked)")


def main():
    p = argparse.ArgumentParser(prog="primordial")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a search from a manifest")
    r.add_argument("--manifest", required=True)
    r.add_argument("--generations", type=int, default=12)
    r.add_argument("--pop", type=int, default=30)
    r.add_argument("--islands", type=int, default=3)
    r.add_argument("--seed", type=int, default=None)
    rp = sub.add_parser("report", help="write report.html for a run dir")
    rp.add_argument("run_dir")
    lg = sub.add_parser("ledger", help="show trial ledger per universe")
    lg.add_argument("--manifest", default=None)
    sub.add_parser("version", help="print version")
    sub.add_parser("doctor", help="verify install: tests + negative control")
    a = p.parse_args()
    from . import __version__
    print(f"primordial v{__version__}")
    if a.cmd == "version":
        return
    if a.cmd == "ledger":
        _show_ledger(a.manifest)
        return
    if a.cmd == "doctor":
        _doctor()
        return
    if a.cmd == "run":
        run(a.manifest, generations=a.generations, pop_size=a.pop,
            n_islands=a.islands, seed=a.seed)
    elif a.cmd == "report":
        from .report import generate
        print("wrote", generate(a.run_dir))

if __name__ == "__main__":
    main()
