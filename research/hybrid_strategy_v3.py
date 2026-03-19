"""
Hybrid Strategy V3 — Tối ưu tốc độ, fix generation bottleneck
================================================================
Dùng product of column pools thay vì random generation → không bao giờ stuck.
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from collections import Counter
import numpy as np
from scraper.data_manager import get_mega645_numbers
from math import comb


def find_optimal_pools(data, pick=6, min_history=100):
    """Tìm pool tối ưu cho mỗi cột."""
    n = len(data)
    total = n - min_history
    pool_sizes = [5, 6, 7, 8, 10, 12, 15, 18, 20]
    hits = [[0]*len(pool_sizes) for _ in range(pick)]

    for i in range(min_history, n):
        actual = sorted(data[i])
        freq = [Counter() for _ in range(pick)]
        for d in data[max(0,i-100):i]:
            sd = sorted(d)
            for p in range(pick):
                freq[p][sd[p]] += 1
        for p in range(pick):
            for si, ps in enumerate(pool_sizes):
                pool = set(v for v, _ in freq[p].most_common(ps))
                if actual[p] in pool:
                    hits[p][si] += 1

    print(f"\n  ACCURACY / COL / POOL SIZE ({total} ky, walk-forward):")
    print(f"  {'':6s}", end="")
    for ps in pool_sizes:
        print(f" | {'top'+str(ps):>6s}", end="")
    print()
    for p in range(pick):
        print(f"  Cot {p+1} ", end="")
        for si in range(len(pool_sizes)):
            pct = hits[p][si]/total*100
            mark = " <<" if pct >= 95 else ""
            print(f" | {pct:5.1f}%", end="")
        print()

    # Find sweet spot per col
    print(f"\n  MIN POOL FOR >= 90% ACCURACY:")
    for p in range(pick):
        for si, ps in enumerate(pool_sizes):
            if hits[p][si]/total*100 >= 90:
                print(f"    Cot {p+1}: top-{ps} = {hits[p][si]/total*100:.1f}%")
                break
        else:
            print(f"    Cot {p+1}: top-{pool_sizes[-1]} = {hits[p][-1]/total*100:.1f}% (max tested)")


def fast_dan_from_pools(pool_lists, pos_freq, n_dan, pick=6):
    """Nhanh: random sample từ column pools, ensure valid (sorted, unique)."""
    dan = []
    seen = set()
    max_attempts = n_dan * 10
    
    for _ in range(max_attempts):
        if len(dan) >= n_dan:
            break
        
        combo = []
        used = set()
        ok = True
        
        for p in range(pick):
            cands = [v for v in pool_lists[p] if v not in used]
            if not cands:
                ok = False
                break
            # Weighted random
            w = np.array([pos_freq[p].get(c, 1) for c in cands], dtype=float)
            w += np.random.random(len(w)) * 0.5  # add noise
            w /= w.sum()
            chosen = int(np.random.choice(cands, p=w))
            combo.append(chosen)
            used.add(chosen)
        
        if ok:
            combo = sorted(combo)
            key = tuple(combo)
            if key not in seen:
                dan.append(combo)
                seen.add(key)
    
    return dan


def backtest(data, max_num=45, pick=6, test_count=200):
    """Backtest hybrid strategies."""
    n = len(data)
    start = max(100, n - test_count)
    total_tests = n - start - 1

    configs = {
        'flat_pool42':     [42]*6,       # V14-style
        'all_10':          [10,10,10,10,10,10],
        'hybrid_A':        [10,12,15,15,12,10],
        'hybrid_B':        [8,12,18,18,12,8],
        'hybrid_C':        [10,15,20,20,15,10],
    }
    
    dan_size = 5000
    
    results = {c: {'3':0,'4':0,'5':0,'6':0,'total':0} for c in configs}
    t0 = time.time()

    for i in range(start, n - 1):
        history = data[max(0,i-100):i+1]
        actual = set(data[i+1])

        # Build per-position frequency
        pos_freq = [Counter() for _ in range(pick)]
        for d in history:
            sd = sorted(d)
            for p in range(pick):
                pos_freq[p][sd[p]] += 1
        
        flat_freq = Counter(num for d in history for num in d)

        for cfg_name, col_pools in configs.items():
            if col_pools == [42]*6:
                # Flat: random from top-42
                pool42 = [v for v, _ in flat_freq.most_common(42)]
                dan = []
                seen = set()
                for _ in range(dan_size * 3):
                    if len(dan) >= dan_size: break
                    c = sorted(np.random.choice(pool42, pick, replace=False).tolist())
                    if tuple(c) not in seen:
                        dan.append(c)
                        seen.add(tuple(c))
            else:
                pool_lists = [[v for v, _ in pos_freq[p].most_common(col_pools[p])] for p in range(pick)]
                dan = fast_dan_from_pools(pool_lists, pos_freq, dan_size, pick)

            r = results[cfg_name]
            r['total'] += 1
            
            if dan:
                matches = [len(set(d) & actual) for d in dan]
                best = max(matches)
                if best >= 3: r['3'] += 1
                if best >= 4: r['4'] += 1
                if best >= 5: r['5'] += 1
                if best >= 6: r['6'] += 1

        done = i - start + 1
        if done % 20 == 0 or done == total_tests:
            elapsed = time.time() - t0
            eta = (elapsed/done) * (total_tests - done)
            rh = results['hybrid_A']
            rv = results['flat_pool42']
            t = rh['total']
            print(f"  [{done}/{total_tests}] hybA: 5+={rh['5']}/{t} ({rh['5']/t*100:.1f}%) | "
                  f"flat42: 5+={rv['5']}/{t} ({rv['5']/t*100:.1f}%) | {elapsed:.0f}s ETA {eta:.0f}s")

    elapsed = time.time() - t0
    
    print(f"\n{'='*85}")
    print(f"  BACKTEST RESULTS — {total_tests} ky, {dan_size} dan, {elapsed:.0f}s")
    print(f"{'='*85}")
    
    print(f"\n  {'Config':20s} | {'Pools':25s} | {'3+':>6s} | {'4+':>6s} | {'5+':>6s} | {'6/6':>6s}")
    print(f"  {'─'*20}-+-{'─'*25}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}")
    for cfg_name, col_pools in configs.items():
        r = results[cfg_name]
        t = r['total']
        print(f"  {cfg_name:20s} | {str(col_pools):25s} | {r['3']/t*100:5.1f}% | {r['4']/t*100:5.1f}% | {r['5']/t*100:5.1f}% | {r['6']/t*100:5.1f}%")

    # Random comparison
    p5 = comb(pick,5) * comb(max_num-pick,1) / comb(max_num, pick)
    p5_any = 1 - (1 - p5)**dan_size
    t = results['hybrid_A']['total']
    rnd_5 = p5_any * t

    print(f"\n  Random chance 5+ (5000 dan from 45): {rnd_5:.1f} = {rnd_5/t*100:.1f}%")
    for c in configs:
        r = results[c]
        ratio = r['5'] / rnd_5 if rnd_5 > 0 else 0
        print(f"    {c}: {r['5']}/{t} ({r['5']/t*100:.1f}%) = {ratio:.2f}x random")

    return results


if __name__ == '__main__':
    print("Loading data...")
    mega = get_mega645_numbers()
    print(f"Mega 6/45: {len(mega)} ky")

    find_optimal_pools(mega)
    print(f"\n>>> Backtest 200 ky...")
    backtest(mega, test_count=200)

    print(f"\n{'#'*60}")
    print(f"  DONE")
    print(f"{'#'*60}")
