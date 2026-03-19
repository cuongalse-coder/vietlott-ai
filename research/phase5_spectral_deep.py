"""
Deep-Dive: Spectral Period ~9.8 at Pos2-4
============================================
Phase 1 found FFT peaks at period ~9.8 draws for positions 2-4.
This script tests if this periodicity is EXPLOITABLE for prediction.
Also: Phase 4 coverage optimization for 100-dan target.
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'e:\\VietlottAI')

import numpy as np
from collections import Counter
from scipy import stats
from scraper.data_manager import get_mega645_numbers


def deep_spectral_analysis(data):
    """Deep-dive the spectral anomaly."""
    print(f"\n{'='*60}")
    print(f"  DEEP SPECTRAL ANALYSIS")
    print(f"{'='*60}")
    
    for pos in range(6):
        vals = np.array([sorted(d)[pos] for d in data], dtype=float)
        vals_centered = vals - np.mean(vals)
        
        fft = np.fft.fft(vals_centered)
        power = np.abs(fft[1:len(fft)//2])**2
        freqs = np.fft.fftfreq(len(vals_centered))[1:len(fft)//2]
        
        # Top 5 peaks
        top5 = np.argsort(power)[-5:][::-1]
        avg_power = np.mean(power)
        
        print(f"\n  Pos{pos+1} top-5 spectral peaks:")
        for rank, idx in enumerate(top5):
            period = 1.0 / abs(freqs[idx]) if freqs[idx] != 0 else float('inf')
            ratio = power[idx] / avg_power
            print(f"    #{rank+1}: period={period:.1f} draws, power={ratio:.2f}x avg")
    
    # Key question: can we USE the period ~9.8 for prediction?
    print(f"\n  --- EXPLOITATION TEST ---")
    
    # Strategy: for pos3 (period ~9.8), predict based on value ~10 draws ago
    pos_to_test = [2, 3]  # pos3 and pos4 (0-indexed)
    
    for pos in pos_to_test:
        vals = [sorted(d)[pos] for d in data]
        
        # Test: does val[i] correlate with val[i-10]?
        for lag in [9, 10, 11]:
            x = np.array(vals[lag:])
            y = np.array(vals[:-lag])
            r, p = stats.pearsonr(x, y)
            print(f"  Pos{pos+1} lag-{lag}: r={r:.4f}, p={p:.4f} {'***' if p < 0.05 else ''}")
    
    # Backtest: use lag-10 correlation for prediction
    print(f"\n  --- BACKTEST: Lag-10 Prediction ---")
    
    correct = 0
    total = 0
    within_3 = 0  # predicted value within ±3 of actual
    
    for i in range(max(20, len(data)-100), len(data)-1):
        for pos in pos_to_test:
            actual_val = sorted(data[i+1])[pos]
            predicted_val = sorted(data[i-9])[pos]  # lag-10
            
            diff = abs(actual_val - predicted_val)
            if diff <= 3:
                within_3 += 1
            total += 1
    
    if total > 0:
        hit_pct = within_3 / total * 100
        # Expected if random: P(|val - predicted| <= 3) depends on position range
        # Rough estimate: 7/range ≈ 7/35 ≈ 20%
        expected_pct = 20
        print(f"  Within ±3: {within_3}/{total} = {hit_pct:.1f}% (random expected: ~{expected_pct}%)")
        
        p_val = stats.binomtest(within_3, total, expected_pct/100).pvalue
        print(f"  p-value: {p_val:.4f}")
    
    return within_3, total


def test_coverage_optimization(data):
    """Phase 4: Test optimal 100-dan coverage strategies."""
    print(f"\n{'='*60}")
    print(f"  PHASE 4: Coverage Optimization for 100-Dan")
    print(f"{'='*60}")
    
    from models.master_predictor import MasterPredictor
    
    n = len(data)
    predictor = MasterPredictor(45, 6)
    
    # Strategy: Fix pos1 (top-3) and pos6 (top-3), vary middle
    # This reduces search space dramatically
    test_count = 100
    start = max(60, n - test_count)
    
    any_3 = any_4 = any_5 = 0
    t0 = time.time()
    
    for i in range(start, n - 1):
        history = data[:i + 1]
        actual = set(data[i + 1])
        look = min(200, len(history))
        
        # Build position pools
        pos_freq = [Counter() for _ in range(6)]
        for d in history[-look:]:
            sd = sorted(d)
            for p in range(6):
                pos_freq[p][sd[p]] += 1
        
        # Pos1: top-3, Pos6: top-3 → fix these
        pos1_top = [n for n, _ in pos_freq[0].most_common(3)]
        pos6_top = [n for n, _ in pos_freq[5].most_common(3)]
        # Pos2-5: top-5 each
        mid_pools = []
        for p in range(1, 5):
            mid_pools.append([n for n, _ in pos_freq[p].most_common(5)])
        
        # Generate 100 sets: 3 x (combos from middle) x 3
        dan = []
        seen = set()
        
        for p1 in pos1_top:
            for p6 in pos6_top:
                for _ in range(12):  # ~100 sets total
                    combo = [p1]
                    used = {p1, p6}
                    for mid_p in range(4):
                        candidates = [n for n in mid_pools[mid_p] if n not in used]
                        if candidates:
                            chosen = int(np.random.choice(candidates))
                            combo.append(chosen)
                            used.add(chosen)
                    combo.append(p6)
                    
                    if len(combo) == 6:
                        combo = sorted(combo)
                        if tuple(combo) not in seen:
                            dan.append(combo)
                            seen.add(tuple(combo))
                    if len(dan) >= 100:
                        break
                if len(dan) >= 100:
                    break
            if len(dan) >= 100:
                break
        
        # Pad if needed
        predictor.data = [d[:6] for d in history]
        pool, scores = predictor._get_pool(history)
        while len(dan) < 100:
            combo = sorted(np.random.choice(pool, 6, replace=False).tolist())
            if tuple(combo) not in seen:
                dan.append(combo)
                seen.add(tuple(combo))
        
        set_matches = [len(set(d) & actual) for d in dan[:100]]
        best = max(set_matches) if set_matches else 0
        
        if best >= 3: any_3 += 1
        if best >= 4: any_4 += 1
        if best >= 5: any_5 += 1
    
    total = n - start - 1
    elapsed = time.time() - t0
    
    print(f"  Fixed Pos1/Pos6 + Middle pool strategy:")
    print(f"  100 dan, {total} tests, {elapsed:.0f}s")
    print(f"  3+: {any_3}/{total} = {any_3/total*100:.1f}%")
    print(f"  4+: {any_4}/{total} = {any_4/total*100:.1f}%")
    print(f"  5+: {any_5}/{total} = {any_5/total*100:.1f}%")
    
    return {'any_3_pct': any_3/total*100, 'any_4_pct': any_4/total*100, 'any_5_pct': any_5/total*100}


if __name__ == '__main__':
    data = get_mega645_numbers()
    print(f"Mega 6/45: {len(data)} draws")
    
    deep_spectral_analysis(data)
    coverage = test_coverage_optimization(data)
    
    print(f"\n{'#'*60}")
    print(f"  COMBINED RESULTS")
    print(f"{'#'*60}")
    print(f"  Phase 4 Coverage (100 dan): 3+={coverage['any_3_pct']:.1f}%, 4+={coverage['any_4_pct']:.1f}%, 5+={coverage['any_5_pct']:.1f}%")
    print(f"{'#'*60}")
