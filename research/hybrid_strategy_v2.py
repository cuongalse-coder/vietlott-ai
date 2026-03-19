"""
Hybrid Strategy V2 — Tối ưu tốc độ
=====================================
- Cột dễ (1,6): thu nhỏ pool → chính xác cao
- Cột khó (3,4): mở rộng pool → cover nhiều hơn
- Test 200 kỳ gần nhất, tối ưu generation speed
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from collections import Counter
import numpy as np
from scraper.data_manager import get_mega645_numbers


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

    print(f"\n  ACCURACY PER COL / POOL SIZE ({total} ky):")
    print(f"  {'':6s}", end="")
    for ps in pool_sizes:
        print(f" | {'top'+str(ps):>6s}", end="")
    print()
    for p in range(pick):
        print(f"  Cot {p+1} ", end="")
        for si in range(len(pool_sizes)):
            print(f" | {hits[p][si]/total*100:5.1f}%", end="")
        print()
    return hits, pool_sizes, total


def hybrid_backtest(data, max_num=45, pick=6, test_count=200):
    """Backtest hybrid strategy."""
    n = len(data)
    start = max(100, n - test_count)
    total_tests = n - start - 1

    configs = {
        'V14_random42':    {'pools': [42]*6, 'type': 'flat'},
        'baseline_10':     {'pools': [10,10,10,10,10,10], 'type': 'col'},
        'hybrid_A':        {'pools': [10,12,15,15,12,10], 'type': 'col'},
        'hybrid_B':        {'pools': [8,10,18,18,10,8],   'type': 'col'},
        'max_compress':    {'pools': [6,8,15,15,8,6],     'type': 'col'},
    }
    dan_sizes = [100, 500, 1000, 5000]
    
    results = {}
    for cfg in configs:
        for ds in dan_sizes:
            results[f"{cfg}_d{ds}"] = {'3': 0, '4': 0, '5': 0, '6': 0, 'total': 0}

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
        
        # Flat pool (for V14-style)
        flat_freq = Counter(num for d in history for num in d)
        flat_pool = [v for v, _ in flat_freq.most_common(42)]

        for cfg_name, cfg in configs.items():
            if cfg['type'] == 'flat':
                # V14-style: random from flat pool
                max_ds = max(dan_sizes)
                dan = []
                seen = set()
                while len(dan) < max_ds:
                    combo = sorted(np.random.choice(flat_pool, pick, replace=False).tolist())
                    if tuple(combo) not in seen:
                        dan.append(combo)
                        seen.add(tuple(combo))
            else:
                # Column-constrained generation
                col_pools = cfg['pools']
                pool_lists = [[v for v, _ in pos_freq[p].most_common(col_pools[p])] for p in range(pick)]
                
                max_ds = max(dan_sizes)
                dan = []
                seen = set()
                for _ in range(max_ds * 5):
                    if len(dan) >= max_ds:
                        break
                    combo = []
                    used = set()
                    ok = True
                    for p in range(pick):
                        cands = [v for v in pool_lists[p] if v not in used]
                        if not cands:
                            ok = False
                            break
                        w = np.array([pos_freq[p].get(c,1) + np.random.random()*2 for c in cands], dtype=float)
                        w /= w.sum()
                        chosen = int(np.random.choice(cands, p=w))
                        combo.append(chosen)
                        used.add(chosen)
                    if ok and len(combo) == pick:
                        combo = sorted(combo)
                        if tuple(combo) not in seen:
                            dan.append(combo)
                            seen.add(tuple(combo))

            for ds in dan_sizes:
                key = f"{cfg_name}_d{ds}"
                r = results[key]
                r['total'] += 1
                sub = dan[:ds]
                matches = [len(set(d) & actual) for d in sub]
                best = max(matches) if matches else 0
                if best >= 3: r['3'] += 1
                if best >= 4: r['4'] += 1
                if best >= 5: r['5'] += 1
                if best >= 6: r['6'] += 1

        done = i - start + 1
        if done % 20 == 0 or done == total_tests:
            elapsed = time.time() - t0
            eta = (elapsed/done) * (total_tests - done) if done > 0 else 0
            k = 'hybrid_A_d5000'
            r = results[k]
            t = r['total']
            print(f"  [{done}/{total_tests}] hybA_5k: 5+={r['5']}/{t} | V14_5k: 5+={results['V14_random42_d5000']['5']}/{t} | {elapsed:.0f}s ETA {eta:.0f}s")

    elapsed = time.time() - t0
    
    # Results
    print(f"\n{'='*95}")
    print(f"  HYBRID BACKTEST — {total_tests} ky, {elapsed:.0f}s")
    print(f"{'='*95}")
    
    for ds in dan_sizes:
        print(f"\n  === {ds} DAN ===")
        print(f"  {'Config':20s} | {'Pools':25s} | {'3+':>6s} | {'4+':>6s} | {'5+':>6s} | {'6/6':>6s}")
        print(f"  {'─'*20}-+-{'─'*25}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}")
        for cfg_name in configs:
            key = f"{cfg_name}_d{ds}"
            r = results[key]
            t = r['total']
            p_str = str(configs[cfg_name]['pools'])
            print(f"  {cfg_name:20s} | {p_str:25s} | {r['3']/t*100:5.1f}% | {r['4']/t*100:5.1f}% | {r['5']/t*100:5.1f}% | {r['6']/t*100:5.1f}%")

    # Best overall
    print(f"\n  BEST vs WORST (5000 dan):")
    best5k = max([f"{c}_d5000" for c in configs], key=lambda k: results[k]['5'])
    worst5k = min([f"{c}_d5000" for c in configs], key=lambda k: results[k]['5'])
    rb = results[best5k]; rw = results[worst5k]
    t = rb['total']
    print(f"    BEST:  {best5k:25s} → 5+={rb['5']/t*100:.1f}% | 6/6={rb['6']/t*100:.1f}%")
    print(f"    WORST: {worst5k:25s} → 5+={rw['5']/t*100:.1f}% | 6/6={rw['6']/t*100:.1f}%")

    return results


if __name__ == '__main__':
    print("Loading data...")
    mega = get_mega645_numbers()
    print(f"Mega 6/45: {len(mega)} ky")

    find_optimal_pools(mega)
    print("\n>>> Backtest 200 ky gan nhat...")
    hybrid_backtest(mega, test_count=200)

    print(f"\n{'#'*60}")
    print(f"  DONE")
    print(f"{'#'*60}")
