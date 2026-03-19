"""
Phase 4: Coverage Math Optimization
=====================================
1. Turán covering design: minimum sets to cover all t-subsets
2. Conditional coverage: fix pos1/pos6, optimize pos2-5
3. Progressive dan sizing experiment
4. Compare: optimized 100-dan vs random 100-dan vs 5000-dan
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from collections import Counter
from itertools import combinations
from math import comb
import numpy as np
from scraper.data_manager import get_mega645_numbers
from models.master_predictor import MasterPredictor


def analyze_position_ranges(data, pick=6):
    """Analyze natural ranges for each sorted position."""
    print("\n" + "="*60)
    print("  POSITION RANGE ANALYSIS")
    print("="*60)
    
    pos_vals = [[] for _ in range(pick)]
    for d in data[-500:]:
        sd = sorted(d)
        for p in range(pick):
            pos_vals[p].append(sd[p])
    
    for p in range(pick):
        vals = pos_vals[p]
        print(f"\n  Pos{p+1}: min={min(vals)}, max={max(vals)}, "
              f"mean={np.mean(vals):.1f}, median={np.median(vals):.0f}")
        # Most common values
        freq = Counter(vals)
        top5 = freq.most_common(8)
        print(f"    Top-8: {', '.join(f'{v}({c})' for v, c in top5)}")
        # 80% range
        sorted_vals = sorted(vals)
        lo = sorted_vals[int(len(vals) * 0.1)]
        hi = sorted_vals[int(len(vals) * 0.9)]
        print(f"    80% range: [{lo}, {hi}] (width={hi-lo+1})")
    
    return pos_vals


def covering_design_analysis(pool_size, pick=6):
    """Analyze covering design bounds."""
    print("\n" + "="*60)
    print("  COVERING DESIGN ANALYSIS")
    print("="*60)
    
    for t in [2, 3, 4]:
        # Schönheim bound: C(v,k,t) >= ceil(v/k * C(v-1,k-1,t-1) / ...)
        total_t_subsets = comb(pool_size, t)
        t_in_k = comb(pick, t)
        lower_bound = total_t_subsets / t_in_k
        
        print(f"\n  t={t}: Cover all {t}-subsets of {pool_size} numbers")
        print(f"    Total {t}-subsets: {total_t_subsets:,}")
        print(f"    Each {pick}-set covers: {t_in_k} of them")
        print(f"    Lower bound (Schönheim): {lower_bound:.0f} sets")
        
        if t <= 3:
            print(f"    Coverage per set: {t_in_k/total_t_subsets*100:.4f}%")
            print(f"    100 sets cover: ~{min(100, 100*t_in_k/total_t_subsets*100):.2f}% of {t}-subsets")
            print(f"    5000 sets cover: ~{min(100, 5000*t_in_k/total_t_subsets*100*0.63):.2f}% of {t}-subsets (with overlap)")


def conditional_coverage_experiment(data, max_num=45, pick=6, test_count=100):
    """Test conditional coverage: fix easy positions, optimize hard ones."""
    print("\n" + "="*60)
    print("  CONDITIONAL COVERAGE EXPERIMENT")
    print("  Fix pos1 (easy) & pos6 (easy), optimize pos2-5")
    print("="*60)
    
    n = len(data)
    start = max(60, n - test_count)
    
    # Strategy: build per-position pools, fix pos1 & pos6
    results_100 = {'3': 0, '4': 0, '5': 0, '6': 0}
    results_500 = {'3': 0, '4': 0, '5': 0, '6': 0}
    results_random = {'3': 0, '4': 0, '5': 0, '6': 0}
    total = 0
    t0 = time.time()
    
    for i in range(start, n - 1):
        history = data[:i + 1]
        actual = set(data[i + 1])
        total += 1
        
        # Build position pools from history
        pos_freq = [Counter() for _ in range(pick)]
        for d in history[-200:]:
            sd = sorted(d)
            for p in range(pick):
                pos_freq[p][sd[p]] += 1
        
        # Position pools: top values per position
        pos_pools = []
        for p in range(pick):
            top = [num for num, _ in pos_freq[p].most_common(15)]
            pos_pools.append(top)
        
        # === Strategy 1: Conditional coverage (100 dan) ===
        dan_100 = []
        seen = set()
        for _ in range(300):  # over-generate then take 100
            combo = []
            used = set()
            for p in range(pick):
                candidates = [n for n in pos_pools[p] if n not in used]
                if not candidates:
                    candidates = [n for n in range(1, max_num+1) if n not in used]
                weights = np.array([pos_freq[p].get(c, 1) + np.random.random()*2 for c in candidates])
                weights = weights / weights.sum()
                chosen = int(np.random.choice(candidates, p=weights))
                combo.append(chosen)
                used.add(chosen)
            combo = sorted(combo)
            if tuple(combo) not in seen:
                dan_100.append(combo)
                seen.add(tuple(combo))
            if len(dan_100) >= 100:
                break
        
        # === Strategy 2: Same but 500 dan ===
        dan_500 = list(dan_100)
        for _ in range(2000):
            combo = []
            used = set()
            for p in range(pick):
                candidates = [n for n in pos_pools[p] if n not in used]
                if not candidates:
                    candidates = [n for n in range(1, max_num+1) if n not in used]
                weights = np.array([pos_freq[p].get(c, 1) + np.random.random()*3 for c in candidates])
                weights = weights / weights.sum()
                chosen = int(np.random.choice(candidates, p=weights))
                combo.append(chosen)
                used.add(chosen)
            combo = sorted(combo)
            if tuple(combo) not in seen:
                dan_500.append(combo)
                seen.add(tuple(combo))
            if len(dan_500) >= 500:
                break
        
        # === Strategy 3: Pure random 100 dan from pool of 42 ===
        predictor = MasterPredictor(max_num, pick)
        predictor.data = [d[:pick] for d in history]
        pool, scores = predictor._get_pool(history)
        dan_random = []
        seen_r = set()
        while len(dan_random) < 100:
            combo = sorted(np.random.choice(pool, pick, replace=False).tolist())
            if tuple(combo) not in seen_r:
                dan_random.append(combo)
                seen_r.add(tuple(combo))
        
        # Score all strategies
        for dan, res in [(dan_100, results_100), (dan_500, results_500), (dan_random, results_random)]:
            matches = [len(set(d) & actual) for d in dan]
            best = max(matches)
            if best >= 3: res['3'] += 1
            if best >= 4: res['4'] += 1
            if best >= 5: res['5'] += 1
            if best >= 6: res['6'] += 1
        
        if total % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{total}/{n-start-1}] cond100: 5+={results_100['5']} | cond500: 5+={results_500['5']} | rand100: 5+={results_random['5']} | {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    print(f"\n  RESULTS ({total} draws, {elapsed:.0f}s):")
    print(f"  {'Strategy':25s} | {'3+':>8s} | {'4+':>8s} | {'5+':>8s} | {'6/6':>8s}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for name, res in [("Conditional 100-dan", results_100), ("Conditional 500-dan", results_500), ("Random 100-dan", results_random)]:
        print(f"  {name:25s} | {res['3']/total*100:7.1f}% | {res['4']/total*100:7.1f}% | {res['5']/total*100:7.1f}% | {res['6']/total*100:7.1f}%")
    
    return results_100, results_500, results_random


def progressive_dan_experiment(data, max_num=45, pick=6, test_count=50):
    """Test: does increasing dan size linearly improve hit rate?"""
    print("\n" + "="*60)
    print("  PROGRESSIVE DAN SIZE EXPERIMENT")
    print("="*60)
    
    n = len(data)
    start = max(60, n - test_count)
    
    dan_sizes = [100, 500, 1000, 2000, 5000]
    results = {s: {'3': 0, '4': 0, '5': 0, '6': 0} for s in dan_sizes}
    total = 0
    t0 = time.time()
    
    for i in range(start, n - 1):
        history = data[:i + 1]
        actual = set(data[i + 1])
        total += 1
        
        predictor = MasterPredictor(max_num, pick)
        predictor.data = [d[:pick] for d in history]
        pool, scores = predictor._get_pool(history)
        
        # Generate max dan once, then slice
        predictor.n_dan = max(dan_sizes)
        full_dan = predictor._ultra_hybrid_dan(pool, scores, history)
        
        for size in dan_sizes:
            dan = full_dan[:size]
            matches = [len(set(d) & actual) for d in dan]
            best = max(matches)
            if best >= 3: results[size]['3'] += 1
            if best >= 4: results[size]['4'] += 1
            if best >= 5: results[size]['5'] += 1
            if best >= 6: results[size]['6'] += 1
        
        if total % 10 == 0:
            elapsed = time.time() - t0
            s5000 = results[5000]['5']
            s100 = results[100]['5']
            print(f"  [{total}/{n-start-1}] 100dan:5+={s100} | 5000dan:5+={s5000} | {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    print(f"\n  PROGRESSIVE DAN RESULTS ({total} draws, {elapsed:.0f}s):")
    print(f"  {'Dan Size':>10s} | {'3+':>8s} | {'4+':>8s} | {'5+':>8s} | {'6/6':>8s} | {'Cost (VND)':>12s}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}")
    for size in dan_sizes:
        res = results[size]
        cost = size * 10000  # 10k VND per ticket
        print(f"  {size:10d} | {res['3']/total*100:7.1f}% | {res['4']/total*100:7.1f}% | "
              f"{res['5']/total*100:7.1f}% | {res['6']/total*100:7.1f}% | {cost:>10,} VND")
    
    return results


if __name__ == '__main__':
    print("Loading data...")
    mega = get_mega645_numbers()
    print(f"Mega 6/45: {len(mega)} draws")
    
    # 1. Position range analysis
    analyze_position_ranges(mega)
    
    # 2. Covering design bounds
    covering_design_analysis(pool_size=42, pick=6)
    
    # 3. Conditional coverage experiment (100 draws for speed)
    conditional_coverage_experiment(mega, test_count=100)
    
    # 4. Progressive dan sizing (50 draws for speed)
    progressive_dan_experiment(mega, test_count=50)
    
    print(f"\n{'#'*60}")
    print(f"  PHASE 4 COMPLETE")
    print(f"{'#'*60}")
