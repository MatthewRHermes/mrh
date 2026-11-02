#!/usr/bin/env python3
"""Replay the gemm/gemm_batch calls recorded by LIBGPU's PROFILE_ML and rank them
by how much of the total gemm time each one accounts for.

Input lines look like (the name= field is verbatim -replay syntax):

  LIBGPU :: PROFILE_ML :: count= 3312  name= gemm N N 180 180 43200 180 43200 180 1 0
  LIBGPU :: PROFILE_ML :: count= 306   name= gemm_batch T N 16 16 180 180 180 16 1 0 240

Usage:
  ./replay_profile.py run.log                 # or: ... | ./replay_profile.py
  ./replay_profile.py run.log -n 20           # fewer timing iterations (faster)
  ./replay_profile.py run.log --exe ./a.out --threads 8
"""

import argparse
import re
import subprocess
import sys
from collections import OrderedDict

MARKER = "PROFILE_ML ::"
REC = re.compile(r"^\s*count=\s*(\d+)\s+name=\s*(\S.*?)\s*$")
TIME = re.compile(r"time=\s*([0-9.eE+-]+)\s*\[ms\]")
GFLOPS = re.compile(r"flops=\s*([0-9.eE+-]+)\s*\[GFlops/s\]")


# field layouts, mirroring ProfileML in mathlib/mathlib.h and -replay in main.cpp
#   gemm        ta tb m n k lda ldb ldc alpha beta                          -> 11
#   gemm_batch  ... batch [strideA strideB strideC]                         -> 12 or 15
#   gemv        ta m n lda incx incy alpha beta                             -> 9
#   gemv_batch  ... batch strideA strideX strideY                           -> 13
LAYOUT = {"gemm": (11,), "gemm_batch": (12, 15), "gemv": (9,), "gemv_batch": (13,)}


def valid_replay(name):
    """True if name is a well-formed -replay argument string.

    Guards against three things seen in real logs: other LIBGPU printfs that also
    carry a count= field, MPI ranks interleaving mid-line, and truncated output.
    """
    f = name.split()
    if not f or f[0] not in LAYOUT or len(f) not in LAYOUT[f[0]]:
        return False
    gemv = f[0].startswith("gemv")
    ntrans = 1 if gemv else 2
    if any(t not in ("N", "T", "C") for t in f[1:1+ntrans]):
        return False
    if gemv:
        pos, nz, scal, nonneg = f[2:5], f[5:7], f[7:9], []      # m n lda | incx incy | a b
        if len(f) == 13:
            pos, nonneg = pos + f[9:10], f[10:13]               # batch | strides
    else:
        pos, nz, scal = f[3:9], [], f[9:11]
        nonneg = f[12:15] if len(f) == 15 else []
        if len(f) >= 12:
            pos = pos + f[11:12]
    try:
        if any(int(v) <= 0 for v in pos): return False
        if any(int(v) == 0 for v in nz): return False           # inc may be negative
        if any(int(v) < 0 for v in nonneg): return False        # stride 0 = broadcast
        for v in scal: float(v)
    except ValueError:
        return False
    return True


def parse(stream):
    """-> (OrderedDict {replay_args: total_count}, {name: n_records}, [rejected]).

    Anchors on the PROFILE_ML marker and splits on it, so a line mangled by two
    ranks writing at once still yields the intact records it contains.
    """
    calls, seen, rejected, skipped = OrderedDict(), {}, [], 0
    for line in stream:
        if MARKER not in line:
            continue
        for frag in line.split(MARKER)[1:]:
            m = REC.match(frag)
            if not m:
                continue
            count, name = int(m.group(1)), m.group(2)
            if not valid_replay(name):
                rejected.append((count, name))
                continue
            calls[name] = calls.get(name, 0) + count
            seen[name] = seen.get(name, 0) + 1
    return calls, seen, rejected, skipped


def shape(name):
    """(mode, m, n, k, batch); k=1 for gemv so the flop count works out as 2*m*n."""
    f = name.split()
    mode = f[0]
    if mode.startswith("gemv"):
        m, n, k = int(f[2]), int(f[3]), 1
        batch = int(f[9]) if len(f) == 13 else 1
    else:
        m, n, k = int(f[3]), int(f[4]), int(f[5])
        batch = int(f[11]) if len(f) > 11 else 1
    return mode, m, n, k, batch


def bench(exe, name, num_iter, threads, timeout):
    """-> (per_call_ms, gflops) or (None, err) on failure."""
    cmd = [exe, "-replay"] + name.split() + ["-num_iter", str(num_iter)]
    env_prefix = {"OMP_NUM_THREADS": str(threads)} if threads else {}
    try:
        import os
        env = dict(os.environ, **env_prefix)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()
        return None, tail[-1][:60] if tail else f"exit {p.returncode}"
    t = TIME.search(p.stdout)
    g = GFLOPS.search(p.stdout)
    if not t:
        return None, "no time= line"
    return float(t.group(1)) / num_iter, (float(g.group(1)) if g else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="file with PROFILE_ML lines (default: stdin)")
    ap.add_argument("-n", "--num-iter", type=int, default=100)
    ap.add_argument("--exe", default="./a.out")
    ap.add_argument("--threads", type=int, default=0, help="OMP_NUM_THREADS (0 = inherit)")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    with open(args.log) if args.log else sys.stdin as fh:
        calls, seen, rejected, skipped = parse(fh)
    if not calls:
        sys.exit(f"no valid '{MARKER} count= ... name= ...' records found")

    if skipped:
        print(f"note: ignored {skipped} gemv/gemv_batch record(s) — not replayable", file=sys.stderr)
    for count, name in rejected:
        print(f"skipped malformed record: count={count} name={name!r}", file=sys.stderr)
    dup = {k: v for k, v in seen.items() if v > 1}
    if dup:
        r = max(dup.values())
        print(f"note: {len(dup)} shape(s) appeared in up to {r} records "
              f"(multiple MPI ranks or concatenated runs); counts are summed",
              file=sys.stderr)

    rows, failed = [], []
    for i, (name, count) in enumerate(calls.items(), 1):
        print(f"[{i}/{len(calls)}] {name}", file=sys.stderr, flush=True)
        per_call_ms, gflops = bench(args.exe, name, args.num_iter, args.threads, args.timeout)
        if per_call_ms is None:
            failed.append((name, count, gflops))
            continue
        mode, m, n, k, batch = shape(name)
        rows.append({
            "name": name, "count": count, "mode": mode,
            "mnk": (f"{m}x{n}" if mode.startswith("gemv") else f"{m}x{n}x{k}")
                   + (f" x{batch}" if batch > 1 else ""),
            "per_call_ms": per_call_ms,
            "total_s": per_call_ms * count / 1000.0,
            "gflops": gflops,
            "gflop_call": 2.0 * m * n * k * batch / 1e9,
            "family": "gemv" if mode.startswith("gemv") else "gemm",
            "strided": len(name.split()) in (13, 15),   # gemv_batch=13, gemm_batch=15
        })

    if not rows:
        sys.exit("every replay failed; nothing to report")

    rows.sort(key=lambda r: -r["total_s"])
    grand = sum(r["total_s"] for r in rows)
    w = max(len(r["mnk"]) for r in rows)
    rule = "-" * (2 + 12 + w + 9 + 11 + 10 + 8 + 8 + 9)

    def table(family, label):
        """One table per BLAS family: share/cum are relative to that family, since
        a gemv is not competing with a gemm for the same work."""
        fam = [r for r in rows if r["family"] == family]
        if not fam:
            return
        sub_t = sum(r["total_s"] for r in fam)
        print(f"\n{label}  --  {len(fam)} shape(s), {sub_t:.3f}s "
              f"({sub_t/grand:.1%} of all BLAS time)")
        print(f"{'':2} {'mode':10} {'m x n [x k] [xbatch]':{w}}  {'count':>7} "
              f"{'per call':>10} {'total':>9} {'share':>7} {'cum':>7} {'GF/s':>7}")
        print(rule)
        cum = 0.0
        for i, r in enumerate(fam, 1):
            frac = r["total_s"] / sub_t
            cum += frac
            print(f"{i:2} {r['mode']:10} {r['mnk']:{w}}  {r['count']:7} "
                  f"{r['per_call_ms']:9.4f}m {r['total_s']:8.3f}s {frac:6.1%} {cum:6.1%} "
                  f"{r['gflops']:7.1f}")
        print(rule)
        print(f"{'':2} {'TOTAL':10} {'':{w}}  {sum(r['count'] for r in fam):7} "
              f"{'':10} {sub_t:8.3f}s")
        cum = 0.0
        for i, r in enumerate(fam, 1):
            cum += r["total_s"] / sub_t
            if cum >= 0.9:
                print(f"   top {i} of {len(fam)} = {cum:.1%} of {label.lower()} time")
                break

    table("gemm", "GEMM")
    table("gemv", "GEMV")
    print(f"\nall BLAS: {sum(r['count'] for r in rows)} calls, {grand:.3f}s")

    guessed = [r for r in rows if r["mode"] == "gemm_batch" and not r["strided"]]
    if guessed:
        print(f"\nnote: {len(guessed)} batched shape(s) carried no strides in the log, so "
              f"the replay assumed packed batches.\n      Rebuild libgpu with -D_PROFILE_ML "
              f"from a current tree to record them; without them the timing for any\n"
              f"      broadcast operand (stride 0) is pessimistic.")

    if failed:
        print(f"\n{len(failed)} shape(s) did not run:")
        for name, count, why in failed:
            print(f"  count={count:<7} {name}   [{why}]")


if __name__ == "__main__":
    main()
