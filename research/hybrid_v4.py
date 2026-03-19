"""
Hybrid Strategy V4 — Ultra-fast
================================
Giảm dan=1000, 3 configs, 199 draws. Flush output.
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# Force unbuffered
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from collections import Counter
import numpy as np
from scraper.data_manager import get_mega645_numbers
from math import comb

print("Loading...", flush=True)
mega = get_mega645_numbers()
n = len(mega)
print(f"Data: {n} ky", flush=True)

# === STEP 1: Pool size accuracy ===
print(f"\n{'='*70}", flush=True)
print("  ACCURACY / COL / POOL SIZE (1383 ky walk-forward)", flush=True)
print(f"{'='*70}", flush=True)

pick = 6
min_hist = 100
total = n - min_hist
pool_sizes = [5, 6, 7, 8, 10, 12, 15, 18, 20]
hits = [[0]*len(pool_sizes) for _ in range(pick)]

for i in range(min_hist, n):
    actual = sorted(mega[i])
    freq = [Counter() for _ in range(pick)]
    for d in mega[max(0,i-100):i]:
        sd = sorted(d)
        for p in range(pick):
            freq[p][sd[p]] += 1
    for p in range(pick):
        for si, ps in enumerate(pool_sizes):
            pool = set(v for v, _ in freq[p].most_common(ps))
            if actual[p] in pool:
                hits[p][si] += 1

print(f"  {'':6s}", end="")
for ps in pool_sizes:
    print(f" | {'top'+str(ps):>6s}", end="")
print(flush=True)
for p in range(pick):
    print(f"  Cot {p+1} ", end="")
    for si in range(len(pool_sizes)):
        print(f" | {hits[p][si]/total*100:5.1f}%", end="")
    print(flush=True)

# === STEP 2: Dan backtest - 3 configs only ===
print(f"\n{'='*70}", flush=True)
print("  DAN BACKTEST (1000 dan, 199 ky)", flush=True)
print(f"{'='*70}", flush=True)

configs = {
    'flat42':    [42]*6,
    'all10':     [10,10,10,10,10,10],
    'hybrid':    [10,12,15,15,12,10],
}

dan_size = 1000
start = n - 200
total_tests = n - start - 1

results = {c: {'3':0,'4':0,'5':0,'6':0,'t':0} for c in configs}
t0 = time.time()

for i in range(start, n - 1):
    history = mega[max(0,i-100):i+1]
    actual = set(mega[i+1])
    
    pos_freq = [Counter() for _ in range(pick)]
    for d in history:
        sd = sorted(d)
        for p in range(pick):
            pos_freq[p][sd[p]] += 1
    
    flat_freq = Counter(num for d in history for num in d)
    flat42 = [v for v, _ in flat_freq.most_common(42)]

    for cfg_name, col_pools in configs.items():
        dan = []
        seen = set()
        
        if col_pools == [42]*6:
            for _ in range(dan_size * 2):
                if len(dan) >= dan_size: break
                c = sorted(np.random.choice(flat42, pick, replace=False).tolist())
                if tuple(c) not in seen:
                    dan.append(c)
                    seen.add(tuple(c))
        else:
            plists = [[v for v, _ in pos_freq[p].most_common(col_pools[p])] for p in range(pick)]
            for _ in range(dan_size * 8):
                if len(dan) >= dan_size: break
                combo = []
                used = set()
                ok = True
                for p in range(pick):
                    cands = [v for v in plists[p] if v not in used]
                    if not cands:
                        ok = False
                        break
                    idx = np.random.randint(len(cands))
                    combo.append(cands[idx])
                    used.add(cands[idx])
                if ok:
                    combo = sorted(combo)
                    if tuple(combo) not in seen:
                        dan.append(combo)
                        seen.add(tuple(combo))
        
        r = results[cfg_name]
        r['t'] += 1
        if dan:
            best = max(len(set(d) & actual) for d in dan)
            if best >= 3: r['3'] += 1
            if best >= 4: r['4'] += 1
            if best >= 5: r['5'] += 1
            if best >= 6: r['6'] += 1

    done = i - start + 1
    if done % 20 == 0 or done == total_tests:
        elapsed = time.time() - t0
        print(f"  [{done}/{total_tests}] hyb={results['hybrid']['5']}/{done} flat={results['flat42']['5']}/{done} | {elapsed:.0f}s", flush=True)

elapsed = time.time() - t0

print(f"\n  RESULTS ({total_tests} ky, {dan_size} dan, {elapsed:.0f}s):", flush=True)
print(f"  {'Config':12s} | {'Pools':25s} | {'3+':>6s} | {'4+':>6s} | {'5+':>6s} | {'6/6':>6s}", flush=True)
print(f"  {'─'*12}-+-{'─'*25}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}", flush=True)
for c in configs:
    r = results[c]
    t = r['t']
    print(f"  {c:12s} | {str(configs[c]):25s} | {r['3']/t*100:5.1f}% | {r['4']/t*100:5.1f}% | {r['5']/t*100:5.1f}% | {r['6']/t*100:5.1f}%", flush=True)

# Random comparison
p5 = comb(pick,5) * comb(45-pick,1) / comb(45, pick)
p5_any = 1 - (1 - p5)**dan_size
t = results['flat42']['t']
rnd = p5_any * t
print(f"\n  Random 5+ (1000 dan): {rnd:.1f} = {rnd/t*100:.1f}%", flush=True)
for c in configs:
    r = results[c]
    print(f"    {c}: {r['5']}/{t} = {r['5']/rnd:.2f}x random", flush=True)

print(f"\n{'#'*60}", flush=True)
print("  DONE", flush=True)
