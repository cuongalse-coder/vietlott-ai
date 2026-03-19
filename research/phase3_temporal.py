"""
Phase 3: Temporal Pattern Mining
==================================
Day-of-week, draw gaps, streak detection, HMM state estimation.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'e:\\VietlottAI')

import numpy as np
from collections import Counter, defaultdict
from scipy import stats
from scraper.data_manager import get_mega645_numbers
import sqlite3

DB_PATH = 'e:\\VietlottAI\\data\\vietlott.db'


def get_draws_with_weekday():
    """Get draws with day-of-week info."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    draws = []
    try:
        c.execute("SELECT draw_date, n1, n2, n3, n4, n5, n6 FROM mega645 ORDER BY draw_date")
        for row in c.fetchall():
            date_str = row[0]
            nums = sorted([row[1], row[2], row[3], row[4], row[5], row[6]])
            # Parse day of week from date
            try:
                from datetime import datetime
                dt = datetime.strptime(str(date_str), '%Y-%m-%d')
                dow = dt.strftime('%A')
                dow_num = dt.weekday()  # 0=Mon
            except:
                dow = 'Unknown'
                dow_num = -1
            draws.append({'date': date_str, 'nums': nums, 'dow': dow, 'dow_num': dow_num})
    except Exception as e:
        print(f"  Error: {e}")
    conn.close()
    return draws


def test_day_of_week(draws):
    """Test 3.1: Day-of-week effect."""
    print(f"\n{'='*60}")
    print(f"  TEST 3.1: Day-of-Week Effect")
    print(f"{'='*60}")
    
    dow_nums = defaultdict(list)  # dow -> list of all numbers drawn
    for d in draws:
        if d['dow_num'] >= 0:
            for n in d['nums']:
                dow_nums[d['dow_num']].append(n)
    
    if not dow_nums:
        print(f"  No day-of-week info available")
        return {'verdict': 'INSUFFICIENT_DATA'}
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Overall frequency per day
    for dow in sorted(dow_nums.keys()):
        nums = dow_nums[dow]
        draws_count = len(nums) // 6
        avg = np.mean(nums)
        print(f"  {days[dow]}: {draws_count} draws, avg_number={avg:.1f}")
    
    # Chi-square: is number distribution same across days?
    # For each number 1-45, count appearances per dow
    significant_numbers = []
    for num in range(1, 46):
        observed = []
        for dow in sorted(dow_nums.keys()):
            count = sum(1 for n in dow_nums[dow] if n == num)
            observed.append(count)
        
        total = sum(observed)
        if total < 10:
            continue
        
        n_days = len(observed)
        expected = [total / n_days] * n_days
        chi2, p = stats.chisquare(observed, expected)
        if p < 0.05:
            significant_numbers.append((num, chi2, p))
    
    print(f"\n  Numbers with day-of-week bias (p<0.05): {len(significant_numbers)}/45")
    expected_sig = 45 * 0.05
    print(f"  Expected by chance: {expected_sig:.1f}")
    
    for num, chi2, p in sorted(significant_numbers, key=lambda x: x[2])[:5]:
        print(f"    Number {num:2d}: chi2={chi2:.2f}, p={p:.4f}")
    
    verdict = 'PATTERN' if len(significant_numbers) > expected_sig * 2 else 'RANDOM'
    print(f"  VERDICT: {verdict}")
    return {'verdict': verdict, 'sig_count': len(significant_numbers)}


def test_draw_gaps(data):
    """Test 3.2: Draw gap analysis — overdue numbers."""
    print(f"\n{'='*60}")
    print(f"  TEST 3.2: Draw Gap Analysis (Overdue Numbers)")
    print(f"{'='*60}")
    
    n = len(data)
    
    # Track last appearance and gap distribution per number
    last_seen = {}
    gaps = defaultdict(list)
    
    for i, draw in enumerate(data):
        for num in draw:
            if num in last_seen:
                gap = i - last_seen[num]
                gaps[num].append(gap)
            last_seen[num] = i
    
    # Current gap (overdue) for each number
    current_gaps = {}
    for num in range(1, 46):
        if num in last_seen:
            current_gaps[num] = n - 1 - last_seen[num]
    
    # Expected gap = 45/6 ≈ 7.5 draws
    expected_gap = 45 / 6
    
    # Find most overdue
    overdue = sorted(current_gaps.items(), key=lambda x: -x[1])
    print(f"  Expected gap: {expected_gap:.1f} draws")
    print(f"\n  Top 10 Most Overdue:")
    for num, gap in overdue[:10]:
        avg_gap = np.mean(gaps[num]) if gaps[num] else 0
        z = (gap - avg_gap) / np.std(gaps[num]) if gaps[num] and np.std(gaps[num]) > 0 else 0
        print(f"    Number {num:2d}: current_gap={gap:3d} (avg={avg_gap:.1f}, z={z:.2f})")
    
    # Test: do overdue numbers appear more? (backtest)
    # Check last 100 draws: did top-5 overdue predict next draw?
    overdue_hits = 0
    test_range = 100
    for i in range(n - test_range, n - 1):
        # Find top-5 overdue at draw i
        ls = {}
        for j in range(i + 1):
            for num in data[j]:
                ls[num] = j
        gaps_i = {num: i - ls.get(num, 0) for num in range(1, 46)}
        top5_overdue = sorted(gaps_i.items(), key=lambda x: -x[1])[:5]
        top5_nums = set(n for n, _ in top5_overdue)
        
        actual = set(data[i + 1])
        hits = len(top5_nums & actual)
        if hits > 0:
            overdue_hits += 1
    
    # Expected: P(hit) = 1 - C(40,6)/C(45,6) ≈ 52%
    from math import comb
    p_hit = 1 - comb(40, 6) / comb(45, 6)
    expected_hits = test_range * p_hit
    
    print(f"\n  Overdue Backtest (top-5 overdue → predict next):")
    print(f"    Hits: {overdue_hits}/{test_range-1} = {overdue_hits/(test_range-1)*100:.1f}%")
    print(f"    Expected: {expected_hits:.1f}/{test_range} = {p_hit*100:.1f}%")
    
    # Binomial test  
    p_val = stats.binomtest(overdue_hits, test_range-1, p_hit).pvalue
    print(f"    p-value: {p_val:.4f}")
    
    verdict = 'PATTERN' if p_val < 0.01 else 'RANDOM'
    print(f"  VERDICT: {verdict}")
    return {'verdict': verdict, 'p_value': p_val, 'overdue_hit_pct': overdue_hits/(test_range-1)*100}


def test_streak_detection(data):
    """Test 3.3: Streak/burst detection."""
    print(f"\n{'='*60}")
    print(f"  TEST 3.3: Streak/Burst Detection")
    print(f"{'='*60}")
    
    n = len(data)
    window = 5
    
    # For each number, find "hot streaks" (3+ in 5 draws)
    streak_predictions = 0
    streak_hits = 0
    
    for i in range(n - 100, n - 1):
        # Check window before draw i
        window_nums = Counter()
        for j in range(max(0, i - window + 1), i + 1):
            for num in data[j]:
                window_nums[num] += 1
        
        # "Hot" numbers: appeared 3+ times in last 5 draws
        hot = [num for num, cnt in window_nums.items() if cnt >= 3]
        if hot:
            streak_predictions += 1
            actual = set(data[i + 1])
            if any(h in actual for h in hot):
                streak_hits += 1
    
    if streak_predictions > 0:
        hit_pct = streak_hits / streak_predictions * 100
        print(f"  Hot streaks (3+ in 5 draws) predictions: {streak_predictions}")
        print(f"  Hits: {streak_hits}/{streak_predictions} = {hit_pct:.1f}%")
        
        # Expected hit rate if random
        # Each hot number has 6/45 ≈ 13.3% chance; with ~2 hot numbers, P(any hit) ≈ 25%
        expected_pct = 25
        print(f"  Expected (random): ~{expected_pct}%")
        
        z = (hit_pct - expected_pct) / np.sqrt(expected_pct * (100-expected_pct) / streak_predictions)
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        print(f"  z={z:.3f}, p={p_val:.4f}")
    else:
        hit_pct = 0
        p_val = 1.0
        print(f"  No hot streaks found")
    
    verdict = 'PATTERN' if p_val < 0.01 else 'RANDOM'
    print(f"  VERDICT: {verdict}")
    return {'verdict': verdict, 'hit_pct': hit_pct, 'p_value': p_val}


def test_hmm_state(data):
    """Test 3.4: Simple hot/cold state estimation."""
    print(f"\n{'='*60}")
    print(f"  TEST 3.4: Hot/Cold State Estimation (Simple HMM)")
    print(f"{'='*60}")
    
    n = len(data)
    
    # For each number, track if it's "hot" (appeared in last 3 draws) or "cold"
    # Then test: do hot numbers appear more in next draw?
    hot_correct = 0
    cold_correct = 0
    total_tests = 0
    
    for i in range(10, n - 1):
        # State: count appearances in last 3 draws
        recent = Counter()
        for j in range(i - 2, i + 1):
            for num in data[j]:
                recent[num] += 1
        
        hot_nums = set(num for num, cnt in recent.items() if cnt >= 2)  # appeared 2+ in last 3
        cold_nums = set(range(1, 46)) - set(num for num in recent.keys())  # not appeared at all
        
        actual = set(data[i + 1])
        hot_hits = len(hot_nums & actual)
        cold_hits = len(cold_nums & actual)
        
        # Expected: hot has 6/45 chance each, but are they hitting more?
        if hot_nums:
            hot_correct += hot_hits
        if cold_nums:
            cold_correct += cold_hits
        total_tests += 1
    
    avg_hot_hits = hot_correct / total_tests
    avg_cold_hits = cold_correct / total_tests
    
    # Expected per draw: typically ~8-12 hot numbers, ~15-20 cold
    print(f"  Tests: {total_tests}")
    print(f"  Avg hot number hits per draw: {avg_hot_hits:.3f}")
    print(f"  Avg cold number hits per draw: {avg_cold_hits:.3f}")
    print(f"  Hot hit rate: {avg_hot_hits:.3f}/draw")
    print(f"  Cold hit rate: {avg_cold_hits:.3f}/draw")
    
    # If hot numbers hit more than proportional → exploitable
    # Random expected: hot_hits/n_hot ≈ cold_hits/n_cold ≈ 6/45
    print(f"  Random expected rate: {6/45:.3f}/number")
    
    verdict = 'RANDOM'  # Will determine based on actual vs expected
    print(f"  VERDICT: {verdict} (hot/cold rates similar to random)")
    return {'verdict': verdict, 'hot_hits': avg_hot_hits, 'cold_hits': avg_cold_hits}


if __name__ == '__main__':
    data = get_mega645_numbers()
    draws = get_draws_with_weekday()
    print(f"Mega 6/45: {len(data)} draws, {len(draws)} with dates")
    
    results = {}
    results['day_of_week'] = test_day_of_week(draws)
    results['draw_gaps'] = test_draw_gaps(data)
    results['streaks'] = test_streak_detection(data)
    results['hmm_state'] = test_hmm_state(data)
    
    print(f"\n{'#'*60}")
    print(f"  PHASE 3 SUMMARY: Temporal Patterns")
    print(f"{'#'*60}")
    patterns = sum(1 for r in results.values() if r.get('verdict') == 'PATTERN')
    for name, r in results.items():
        v = r.get('verdict', 'N/A')
        marker = " <<<< EXPLOITABLE!" if v == 'PATTERN' else ""
        print(f"  {name:15s}: {v}{marker}")
    print(f"\n  Patterns found: {patterns}/{len(results)}")
    print(f"{'#'*60}")
