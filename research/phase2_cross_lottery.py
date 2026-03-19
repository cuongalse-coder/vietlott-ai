"""
Phase 2: Cross-Lottery Correlation Analysis
=============================================
Test if Mega 6/45 and Power 6/55 share RNG or have correlated outputs.
If they do → use one to predict the other.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'e:\\VietlottAI')

import numpy as np
from collections import Counter, defaultdict
from scipy import stats
from scraper.data_manager import get_mega645_numbers, get_power655_numbers
import sqlite3

DB_PATH = 'e:\\VietlottAI\\data\\vietlott.db'

def get_draws_with_dates():
    """Get draws with dates for both lotteries."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    mega = {}
    try:
        c.execute("SELECT draw_date, n1, n2, n3, n4, n5, n6 FROM mega645 ORDER BY draw_date")
        for row in c.fetchall():
            date = row[0]
            nums = sorted([row[1], row[2], row[3], row[4], row[5], row[6]])
            mega[date] = nums
    except:
        pass
    
    power = {}
    try:
        c.execute("SELECT draw_date, n1, n2, n3, n4, n5, n6 FROM power655 ORDER BY draw_date")
        for row in c.fetchall():
            date = row[0]
            nums = sorted([row[1], row[2], row[3], row[4], row[5], row[6]])
            power[date] = nums
    except:
        pass
    
    conn.close()
    return mega, power


def test_same_day_correlation(mega, power):
    """Test 2.1: Same-day draw correlation."""
    print(f"\n{'='*60}")
    print(f"  TEST 2.1: Same-Day Correlation (Mega vs Power)")
    print(f"{'='*60}")
    
    common_dates = sorted(set(mega.keys()) & set(power.keys()))
    print(f"  Common draw dates: {len(common_dates)}")
    
    if len(common_dates) < 20:
        print(f"  NOT ENOUGH common dates for analysis")
        return {'verdict': 'INSUFFICIENT_DATA', 'p_value': 1.0}
    
    # Position-wise correlation
    results = []
    for mp in range(6):
        for pp in range(6):
            m_vals = [mega[d][mp] for d in common_dates]
            p_vals = [power[d][pp] for d in common_dates]
            r, p = stats.pearsonr(m_vals, p_vals)
            results.append((mp+1, pp+1, r, p))
            if p < 0.05:
                print(f"  *** Mega_pos{mp+1} vs Power_pos{pp+1}: r={r:.4f}, p={p:.4f} SIGNIFICANT!")
    
    # Any significant?
    sig_count = sum(1 for _, _, _, p in results if p < 0.05)
    expected_sig = len(results) * 0.05  # false positive rate
    
    print(f"\n  Significant correlations: {sig_count}/{len(results)} (expected by chance: {expected_sig:.1f})")
    
    # Overall test: concatenate all numbers same day
    m_flat = []
    p_flat = []
    for d in common_dates:
        m_flat.extend(mega[d])
        p_flat.extend(power[d])
    r_overall, p_overall = stats.pearsonr(m_flat, p_flat)
    print(f"  Overall flat correlation: r={r_overall:.4f}, p={p_overall:.4f}")
    
    verdict = "PATTERN" if sig_count > expected_sig * 2 and p_overall < 0.01 else "RANDOM"
    print(f"  VERDICT: {verdict}")
    return {'verdict': verdict, 'p_value': p_overall, 'sig_count': sig_count}


def test_shared_rng(mega, power):
    """Test 2.2: Shared RNG detection — interleave draws and test for patterns."""
    print(f"\n{'='*60}")
    print(f"  TEST 2.2: Shared RNG Seed Detection")
    print(f"{'='*60}")
    
    common_dates = sorted(set(mega.keys()) & set(power.keys()))
    if len(common_dates) < 20:
        print(f"  NOT ENOUGH data")
        return {'verdict': 'INSUFFICIENT_DATA'}
    
    # Interleave: Mega numbers then Power numbers for same day
    interleaved = []
    for d in common_dates:
        interleaved.extend(mega[d])
        interleaved.extend(power[d])
    
    # Test 1: Serial correlation on interleaved sequence
    x = np.array(interleaved[:-1])
    y = np.array(interleaved[1:])
    r_serial, p_serial = stats.pearsonr(x, y)
    print(f"  Interleaved serial correlation: r={r_serial:.4f}, p={p_serial:.6f}")
    
    # Test 2: XOR pattern — if shared LCG, XOR of consecutive pairs should show bias
    xor_vals = [interleaved[i] ^ interleaved[i+1] for i in range(0, len(interleaved)-1, 2)]
    # Chi-square on XOR distribution
    xor_freq = Counter(xor_vals)
    observed = list(xor_freq.values())
    expected = [len(xor_vals) / len(xor_freq)] * len(xor_freq)
    if len(observed) > 5:
        chi2, p_xor = stats.chisquare(observed[:len(expected)], expected[:len(observed)])
        print(f"  XOR pattern chi-square: chi2={chi2:.2f}, p={p_xor:.6f}")
    else:
        p_xor = 1.0
    
    # Test 3: Sum correlation — sum of Mega draw vs sum of Power draw same day
    m_sums = [sum(mega[d]) for d in common_dates]
    p_sums = [sum(power[d]) for d in common_dates]
    r_sum, p_sum = stats.pearsonr(m_sums, p_sums)
    print(f"  Sum(Mega) vs Sum(Power) correlation: r={r_sum:.4f}, p={p_sum:.4f}")
    
    # Test 4: Difference sequence — if same RNG, differences should be structured
    diffs = [mega[d][i] - power[d][i] for d in common_dates for i in range(min(6,6))]
    # Runs test on differences
    median_diff = np.median(diffs)
    runs = [1 if d > median_diff else 0 for d in diffs]
    n_runs = sum(1 for i in range(1, len(runs)) if runs[i] != runs[i-1]) + 1
    n1 = sum(runs)
    n0 = len(runs) - n1
    if n0 > 0 and n1 > 0:
        expected_runs = 2 * n0 * n1 / (n0 + n1) + 1
        var_runs = 2 * n0 * n1 * (2*n0*n1 - n0 - n1) / ((n0+n1)**2 * (n0+n1-1))
        if var_runs > 0:
            z_runs = (n_runs - expected_runs) / np.sqrt(var_runs)
            p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
            print(f"  Difference runs test: z={z_runs:.3f}, p={p_runs:.4f}")
        else:
            p_runs = 1.0
    else:
        p_runs = 1.0
    
    min_p = min(p_serial, p_xor, p_sum, p_runs)
    verdict = "PATTERN" if min_p < 0.01 else "RANDOM"
    print(f"  Min p-value: {min_p:.6f}")
    print(f"  VERDICT: {verdict}")
    return {'verdict': verdict, 'min_p': min_p}


def test_cross_lag(mega, power):
    """Test 2.3: Cross-lag correlation between Mega and Power."""
    print(f"\n{'='*60}")
    print(f"  TEST 2.3: Cross-Lag Correlation")
    print(f"{'='*60}")
    
    mega_data = get_mega645_numbers()
    power_data = get_power655_numbers()
    
    # Use sum as signal (simplest aggregate)
    m_sums = [sum(d) for d in mega_data]
    p_sums = [sum(d) for d in power_data]
    
    min_len = min(len(m_sums), len(p_sums))
    m_sums = m_sums[:min_len]
    p_sums = p_sums[:min_len]
    
    print(f"  Mega draws: {len(mega_data)}, Power draws: {len(power_data)}, Using: {min_len}")
    
    best_lag = (0, 0, 1.0)
    for lag in range(-10, 11):
        if lag >= 0:
            x = m_sums[lag:]
            y = p_sums[:len(x)]
        else:
            x = p_sums[-lag:]
            y = m_sums[:len(x)]
        
        if len(x) < 50:
            continue
        r, p = stats.pearsonr(x, y)
        if p < best_lag[2]:
            best_lag = (lag, r, p)
        if abs(r) > 0.05:
            print(f"  Lag {lag:+3d}: r={r:.4f}, p={p:.4f} {'***' if p < 0.05 else ''}")
    
    print(f"\n  Best lag: {best_lag[0]:+d} (r={best_lag[1]:.4f}, p={best_lag[2]:.4f})")
    
    verdict = "PATTERN" if best_lag[2] < 0.01 else "RANDOM"
    print(f"  VERDICT: {verdict}")
    return {'verdict': verdict, 'best_lag': best_lag[0], 'p_value': best_lag[2]}


def test_shared_numbers(mega, power):
    """Test 2.4: Shared number frequency anomaly."""
    print(f"\n{'='*60}")
    print(f"  TEST 2.4: Shared Number Frequency")
    print(f"{'='*60}")
    
    common_dates = sorted(set(mega.keys()) & set(power.keys()))
    if len(common_dates) < 20:
        print(f"  NOT ENOUGH data")
        return {'verdict': 'INSUFFICIENT_DATA'}
    
    # How many numbers are shared between Mega and Power on same day?
    shared_counts = []
    for d in common_dates:
        shared = len(set(mega[d]) & set(power[d]))
        shared_counts.append(shared)
    
    avg_shared = np.mean(shared_counts)
    
    # Expected shared (hypergeometric): picking 6 from 45 and 6 from 55
    # Approximate: P(shared=k) for random
    # For Mega (1-45) and Power (1-55), shared pool = 1-45
    # Expected shared ≈ 6 * 6 / 45 ≈ 0.8
    expected = 6 * 6 / 45
    
    print(f"  Common dates: {len(common_dates)}")
    print(f"  Avg shared numbers: {avg_shared:.3f} (expected random: {expected:.3f})")
    
    dist = Counter(shared_counts)
    for k in sorted(dist.keys()):
        pct = dist[k] / len(shared_counts) * 100
        print(f"  Shared {k}: {dist[k]} ({pct:.1f}%)")
    
    # Z-test
    se = np.std(shared_counts) / np.sqrt(len(shared_counts))
    z = (avg_shared - expected) / se if se > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"  Z-test: z={z:.3f}, p={p_val:.4f}")
    
    verdict = "PATTERN" if p_val < 0.01 else "RANDOM"
    print(f"  VERDICT: {verdict}")
    return {'verdict': verdict, 'avg_shared': avg_shared, 'p_value': p_val}


if __name__ == '__main__':
    print("Loading data with dates...")
    mega, power = get_draws_with_dates()
    print(f"Mega dates: {len(mega)}, Power dates: {len(power)}")
    
    results = {}
    results['same_day'] = test_same_day_correlation(mega, power)
    results['shared_rng'] = test_shared_rng(mega, power)
    results['cross_lag'] = test_cross_lag(mega, power)
    results['shared_nums'] = test_shared_numbers(mega, power)
    
    print(f"\n{'#'*60}")
    print(f"  PHASE 2 SUMMARY")
    print(f"{'#'*60}")
    patterns_found = sum(1 for r in results.values() if r.get('verdict') == 'PATTERN')
    for name, r in results.items():
        v = r.get('verdict', 'N/A')
        p = r.get('p_value', r.get('min_p', 'N/A'))
        marker = " <<<< EXPLOITABLE!" if v == 'PATTERN' else ""
        print(f"  {name:15s}: {v:10s} (p={p}){marker}")
    print(f"\n  Patterns found: {patterns_found}/{len(results)}")
    if patterns_found > 0:
        print(f"  >>> WEAKNESS DETECTED! Cross-lottery correlation exists!")
    else:
        print(f"  >>> No cross-lottery weakness found. Mega and Power are independent.")
    print(f"{'#'*60}")
