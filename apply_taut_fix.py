#!/usr/bin/env python3
"""
v0.1.9 -> tautology-guard fix. Run from the primordial repo root.

WHY: on the 5m ETF run, gen 3's best genome was lte(body_frac, body_frac) --
a tautology -- scoring -0.042 instead of the -9.99 floor. The v0.1.8
behavioral constant-signal guard missed it for two compounding reasons:

  1. A comparison against NaN is False. Real 5m/1h bars contain zero-range
     bars (high == low), which make body_frac NaN, which flips the
     tautology's signal 1 -> 0. It "varies", so the guard passed it.
  2. The guard probed genome.signal(), which applies the "confirm" rolling
     min AFTERWARDS -- so each NaN-induced 0 propagates forward into bars
     where the atoms ARE defined, defeating a naive finite-value mask.

FIX: probe the ENTRY TREE directly (before entry-style smoothing), and
evaluate it only on bars where every atom it reads is finite. A tautology is
then constant on its own domain and gets floored.

This is the 5th appearance of the always-fire species. It was benign here
(negative fitness, never reached finals) but it burns population slots.
"""
import re, sys, pathlib

root = pathlib.Path(".")
tree_p, judge_p = root / "primordial/tree.py", root / "primordial/judge.py"
for p in (tree_p, judge_p):
    if not p.exists():
        sys.exit(f"run me from the primordial repo root -- missing {p}")

t = tree_p.read_text()
if "def terminal_names(" in t:
    print("tree.py: already patched, skipping")
else:
    anchor = "def has_terminal(tree) -> bool:"
    assert anchor in t, "tree.py: anchor not found -- wrong version?"
    add = '''def terminal_names(tree) -> list:
    """Every atom column this tree reads. Used by the fitness guard to
    evaluate a signal only where its inputs are actually defined."""
    out = []
    for nd in _all_nodes(tree):
        if nd["op"] == "term":
            out.append(nd["name"])
    return sorted(set(out))


'''
    tree_p.write_text(t.replace(anchor, add + anchor))
    print("tree.py: added terminal_names()")

j = judge_p.read_text()
if "probe the ENTRY TREE directly" in j:
    print("judge.py: already patched, skipping")
else:
    old = """        probe = genome.signal(data[syms[0]])
        # skip atom warmup (max rolling n is 200): NaN->0 padding makes an
        # always-true tree look like it "varies" from 0 to 1 once
        tail = probe[250:]
        tail = tail[np.isfinite(tail)]"""
    assert old in j, "judge.py: guard block not found -- are you on v0.1.9?"
    new = '''        frame = data[syms[0]]
        # probe the ENTRY TREE directly, not genome.signal(): the "confirm"
        # style applies a rolling min AFTERWARDS, so a single NaN-induced 0
        # propagates forward into bars where the atoms ARE defined and a
        # tautology looks like it varies even under the finite mask below.
        probe = _T.bool_signal(genome.entry_tree, frame)
        # skip atom warmup (max rolling n is 200): NaN->0 padding makes an
        # always-true tree look like it "varies" from 0 to 1 once
        tail = probe[250:]
        # ...and look ONLY where every atom this tree reads is defined.
        # Without this, lte(x, x) escapes: comparison against NaN is False,
        # so one zero-range bar (high == low -> body_frac NaN) makes the
        # tautology flip 1 -> 0. That is how lte(body_frac, body_frac)
        # reached gen-3 best on the 5m ETF run instead of being floored.
        names = [n for n in _T.terminal_names(genome.entry_tree)
                 if n in frame.columns]
        if names:
            ok = np.ones(len(frame), dtype=bool)
            for nm in names:
                ok &= np.isfinite(frame[nm].to_numpy(dtype=float))
            tail = tail[ok[250:]]
        tail = tail[np.isfinite(tail)]'''
    judge_p.write_text(j.replace(old, new))
    print("judge.py: guard now probes the entry tree on its defined domain")

print("\ndone. now run:")
print("  python3.10 -m pytest tests -q")
print("  python3.10 scripts/run_controls.py")
