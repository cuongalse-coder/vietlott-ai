"""
Chiến Lược Hybrid: Max Accuracy Cột Dễ + Default Cột 3/4
==========================================================
- Cột 1,2,5,6: tìm pool size nhỏ nhất để đạt ~100% accuracy
- Cột 3,4: fix top-15 hay về nhất (default)
- Ghép lại → generate dan → backtest toàn bộ kỳ
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from collections import Counter
from itertools import combinations
import numpy as np
from scraper.data_manager import get_mega645_numbers


def find_optimal_pool_per_col(data, pick=6, min_history=100):
    """Tìm pool size tối ưu cho mỗi cột để đạt target accuracy."""
    n = len(data)
    total = n - min_history
    
    print(f"\n{'='*80}")
    print(f"  TIM POOL SIZE TOI UU MOI COT (walk-forward {total} ky)")
    print(f"{'='*80}")
    
    col_names = ['Cot 1', 'Cot 2', 'Cot 3', 'Cot 4', 'Cot 5', 'Cot 6']
    pool_sizes = [5, 6, 7, 8, 10, 12, 15, 18, 20, 25]
    
    # results[col][pool_size] = hit_count
    results = [[0]*len(pool_sizes) for _ in range(pick)]
    
    for i in range(min_history, n):
        actual = sorted(data[i])
        history = data[max(0, i-100):i]
        
        pos_freq = [Counter() for _ in range(pick)]
        for d in history:
            sd = sorted(d)
            for p in range(pick):
                pos_freq[p][sd[p]] += 1
        
        for p in range(pick):
            for si, ps in enumerate(pool_sizes):
                pool = set(v for v, _ in pos_freq[p].most_common(ps))
                if actual[p] in pool:
                    results[p][si] += 1
    
    print(f"\n  {'Col':6s}", end="")
    for ps in pool_sizes:
        print(f" | {'top'+str(ps):>6s}", end="")
    print()
    print(f"  {'─'*6}", end="")
    for _ in pool_sizes:
        print(f"-+-{'─'*6}", end="")
    print()
    
    optimal = {}
    for p in range(pick):
        print(f"  {col_names[p]:6s}", end="")
        for si, ps in enumerate(pool_sizes):
            pct = results[p][si] / total * 100
            print(f" | {pct:5.1f}%", end="")
        print()
        
        # Find smallest pool with >= 95%
        for si, ps in enumerate(pool_sizes):
            pct = results[p][si] / total * 100
            if pct >= 95:
                optimal[p] = (ps, pct)
                break
        if p not in optimal:
            optimal[p] = (pool_sizes[-1], results[p][-1] / total * 100)
    
    print(f"\n  POOL SIZE CHO ~95% ACCURACY:")
    for p in range(pick):
        ps, pct = optimal[p]
        print(f"    {col_names[p]}: top-{ps} = {pct:.1f}%")
    
    return optimal, results


def hybrid_dan_backtest(data, max_num=45, pick=6, min_history=100, test_count=None):
    """Backtest: dùng hybrid pool → generate dan → check match."""
    n = len(data)
    if test_count:
        start = max(min_history, n - test_count)
    else:
        start = min_history
    total_tests = n - start - 1
    
    # Config: pool sizes per column
    # Easy cols: push to ~95-100%
    # Hard cols: fix top-15
    configs = {
        'baseline_10': [10, 10, 10, 10, 10, 10],
        'hybrid_A': [10, 12, 15, 15, 12, 10],       # expand middle
        'hybrid_B': [8, 10, 18, 18, 10, 8],          # compress easy, expand hard
        'hybrid_C': [10, 15, 20, 20, 15, 10],        # more aggressive expansion
        'max_easy': [6, 8, 15, 15, 8, 6],            # minimal easy, expanded hard
    }
    
    dan_sizes = [100, 500, 1000, 5000]
    
    results = {}
    for cfg_name, col_pools in configs.items():
        for ds in dan_sizes:
            key = f"{cfg_name}_d{ds}"
            results[key] = {'any3': 0, 'any4': 0, 'any5': 0, 'any6': 0, 'total': 0,
                           'cfg': cfg_name, 'dan_size': ds, 'pools': col_pools}
    
    t0 = time.time()
    
    for i in range(start, n - 1):
        history = data[max(0, i-100):i+1]
        actual = set(data[i + 1])
        
        # Build position frequency
        pos_freq = [Counter() for _ in range(pick)]
        for d in history:
            sd = sorted(d)
            for p in range(pick):
                pos_freq[p][sd[p]] += 1
        
        for cfg_name, col_pools in configs.items():
            # Build pools
            pools = [set(v for v, _ in pos_freq[p].most_common(col_pools[p])) for p in range(pick)]
            pool_lists = [[v for v, _ in pos_freq[p].most_common(col_pools[p])] for p in range(pick)]
            
            # Generate dan — position-constrained combos
            max_dan = max(dan_sizes)
            dan = []
            seen = set()
            
            for attempt in range(max_dan * 4):
                if len(dan) >= max_dan:
                    break
                combo = []
                used = set()
                valid = True
                
                for p in range(pick):
                    candidates = [v for v in pool_lists[p] if v not in used]
                    if not candidates:
                        valid = False
                        break
                    # Weighted by frequency
                    weights = np.array([pos_freq[p].get(c, 1) + np.random.random() * 2 
                                       for c in candidates], dtype=float)
                    weights = weights / weights.sum()
                    chosen = int(np.random.choice(candidates, p=weights))
                    combo.append(chosen)
                    used.add(chosen)
                
                if valid and len(combo) == pick:
                    combo = sorted(combo)
                    if tuple(combo) not in seen:
                        dan.append(combo)
                        seen.add(tuple(combo))
            
            # Check matches for each dan size
            for ds in dan_sizes:
                key = f"{cfg_name}_d{ds}"
                r = results[key]
                r['total'] += 1
                
                sub_dan = dan[:ds]
                matches = [len(set(d) & actual) for d in sub_dan]
                best = max(matches) if matches else 0
                
                if best >= 3: r['any3'] += 1
                if best >= 4: r['any4'] += 1
                if best >= 5: r['any5'] += 1
                if best >= 6: r['any6'] += 1
        
        done = i - start + 1
        if done % 50 == 0 or done == total_tests:
            elapsed = time.time() - t0
            # Show hybrid_A with 5000 dan
            k = 'hybrid_A_d5000'
            r = results[k]
            t = r['total']
            print(f"  [{done}/{total_tests}] hybrid_A_5000: 5+={r['any5']}/{t} ({r['any5']/t*100:.1f}%) | {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    
    # === RESULTS ===
    print(f"\n{'='*90}")
    print(f"  HYBRID DAN BACKTEST — {total_tests} ky, {elapsed:.0f}s")
    print(f"{'='*90}")
    
    # Table per dan size
    for ds in dan_sizes:
        print(f"\n  --- {ds} dan ---")
        print(f"  {'Config':20s} | {'Pools':25s} | {'3+':>6s} | {'4+':>6s} | {'5+':>6s} | {'6/6':>6s}")
        print(f"  {'─'*20}-+-{'─'*25}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}")
        for cfg_name in configs:
            key = f"{cfg_name}_d{ds}"
            r = results[key]
            t = r['total']
            pools_str = str(r['pools'])
            print(f"  {cfg_name:20s} | {pools_str:25s} | {r['any3']/t*100:5.1f}% | {r['any4']/t*100:5.1f}% | {r['any5']/t*100:5.1f}% | {r['any6']/t*100:5.1f}%")
    
    # Best config
    print(f"\n  BEST CONFIG (5000 dan):")
    best_key = max([f"{c}_d5000" for c in configs], key=lambda k: results[k]['any5'])
    r = results[best_key]
    t = r['total']
    print(f"    {best_key}: 5+={r['any5']/t*100:.1f}% | 6/6={r['any6']/t*100:.1f}%")
    
    # Random comparison
    from math import comb
    p5 = comb(pick,5) * comb(max_num-pick, 1) / comb(max_num, pick)
    p5_any = 1 - (1 - p5)**5000
    rnd_5 = p5_any * t
    print(f"    Random expected 5+ (5000 dan): {rnd_5:.1f}")
    print(f"    Best vs Random: {r['any5']/rnd_5:.2f}x")
    
    return results


if __name__ == '__main__':
    print("Loading data...")
    mega = get_mega645_numbers()
    print(f"Mega 6/45: {len(mega)} ky quay")
    
    # Step 1: Find optimal pool sizes per column
    optimal, _ = find_optimal_pool_per_col(mega)
    
    # Step 2: Backtest hybrid strategies (ALL draws)
    print(f"\n>>> Backtest with ALL draws...")
    results = hybrid_dan_backtest(mega, test_count=None)
    
    print(f"\n{'#'*60}")
    print(f"  DONE")
    print(f"{'#'*60}")
