"""
V16 Attack Vector Analysis
===========================
Analyze all 7 attack vectors on historical data to find optimal parameters.
Then combine into V16 model and backtest.
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from collections import Counter, defaultdict
from itertools import combinations
from math import comb
import numpy as np
from scraper.data_manager import get_mega645_numbers


def attack1_mandel_compression(data, max_num=45, pick=6):
    """Attack 1: Find minimum pool size that covers historical jackpots."""
    print("\n" + "="*60)
    print("  ATTACK 1: MANDEL COMPRESSION")
    print("  Find smallest pool covering most jackpots")
    print("="*60)
    
    # Count frequency of each number across ALL draws
    freq = Counter(n for d in data for n in d)
    ranked = [n for n, _ in freq.most_common()]
    
    # For each pool size, check: how many draws are FULLY contained?
    results = []
    for pool_size in range(15, 40):
        pool = set(ranked[:pool_size])
        covered = sum(1 for d in data if set(d).issubset(pool))
        pct = covered / len(data) * 100
        combos = comb(pool_size, pick)
        results.append((pool_size, covered, pct, combos))
        if pool_size <= 30 or pool_size % 5 == 0:
            print(f"  Pool={pool_size}: {covered}/{len(data)} draws fully covered ({pct:.1f}%) | C({pool_size},{pick})={combos:,}")
    
    # Find sweet spot: best coverage per combo
    print(f"\n  SWEET SPOT ANALYSIS:")
    for ps, cov, pct, combos in results:
        efficiency = pct / (combos / 1000)
        if ps <= 30 or ps % 5 == 0:
            print(f"  Pool={ps}: coverage={pct:.1f}% / {combos:,} combos = efficiency {efficiency:.4f}")
    
    # Test: sliding window analysis (last 200 draws)
    print(f"\n  SLIDING WINDOW (last 200 draws):")
    recent_freq = Counter(n for d in data[-200:] for n in d)
    recent_ranked = [n for n, _ in recent_freq.most_common()]
    for pool_size in [18, 20, 22, 25, 28, 30]:
        pool = set(recent_ranked[:pool_size])
        covered = sum(1 for d in data[-200:] if set(d).issubset(pool))
        print(f"  Pool={pool_size} (recent): {covered}/200 = {covered/200*100:.1f}%")
    
    return ranked, freq


def attack2_antipattern_filter(data, max_num=45, pick=6):
    """Attack 2: Identify patterns that NEVER appear → filter them out."""
    print("\n" + "="*60)
    print("  ATTACK 2: ANTI-PATTERN FILTER")
    print("  Find combination patterns that never win")
    print("="*60)
    
    n_draws = len(data)
    
    # Pattern 1: All odd or all even
    all_odd = sum(1 for d in data if all(n % 2 == 1 for n in d))
    all_even = sum(1 for d in data if all(n % 2 == 0 for n in d))
    print(f"\n  All odd: {all_odd}/{n_draws} = {all_odd/n_draws*100:.2f}%")
    print(f"  All even: {all_even}/{n_draws} = {all_even/n_draws*100:.2f}%")
    
    # Odd/even distribution
    odd_counts = Counter(sum(1 for n in d if n % 2 == 1) for d in data)
    print(f"\n  Odd count distribution:")
    for k in range(7):
        c = odd_counts.get(k, 0)
        print(f"    {k} odd: {c} ({c/n_draws*100:.1f}%)")
    
    # Pattern 2: Sum ranges
    sums = [sum(d) for d in data]
    print(f"\n  Sum range: [{min(sums)}, {max(sums)}], mean={np.mean(sums):.0f}, std={np.std(sums):.0f}")
    sum_ranges = [(60, 80), (80, 100), (100, 120), (120, 140), (140, 160), (160, 180), (180, 210)]
    for lo, hi in sum_ranges:
        c = sum(1 for s in sums if lo <= s < hi)
        print(f"    Sum [{lo}-{hi}): {c} ({c/n_draws*100:.1f}%)")
    
    # Pattern 3: Consecutive numbers
    consec_counts = Counter()
    for d in data:
        sd = sorted(d)
        max_consec = 1
        cur = 1
        for i in range(1, len(sd)):
            if sd[i] == sd[i-1] + 1:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 1
        consec_counts[max_consec] += 1
    print(f"\n  Max consecutive numbers in draw:")
    for k in sorted(consec_counts.keys()):
        c = consec_counts[k]
        print(f"    {k}: {c} ({c/n_draws*100:.1f}%)")
    
    # Pattern 4: Same decade (all numbers in same 10-range)
    decade_counts = Counter()
    for d in data:
        decades = set(n // 10 for n in d)
        decade_counts[len(decades)] += 1
    print(f"\n  Number of distinct decades:")
    for k in sorted(decade_counts.keys()):
        c = decade_counts[k]
        print(f"    {k} decades: {c} ({c/n_draws*100:.1f}%)")
    
    # Pattern 5: Gap between consecutive sorted numbers
    gaps = []
    for d in data:
        sd = sorted(d)
        draw_gaps = [sd[i+1] - sd[i] for i in range(len(sd)-1)]
        gaps.append(max(draw_gaps))
    max_gap_dist = Counter(gaps)
    print(f"\n  Max gap between consecutive sorted numbers:")
    print(f"    Range: [{min(gaps)}, {max(gaps)}], mean={np.mean(gaps):.1f}")
    
    # Pattern 6: Last digit diversity
    ld_diversity = Counter()
    for d in data:
        ld = set(n % 10 for n in d)
        ld_diversity[len(ld)] += 1
    print(f"\n  Last digit diversity:")
    for k in sorted(ld_diversity.keys()):
        c = ld_diversity[k]
        print(f"    {k} unique last digits: {c} ({c/n_draws*100:.1f}%)")
    
    # Compile filter rules
    print(f"\n  >>> FILTER RULES (reject combos matching these):")
    safe_sum_lo = np.percentile(sums, 2)
    safe_sum_hi = np.percentile(sums, 98)
    print(f"    Rule 1: Sum outside [{safe_sum_lo:.0f}, {safe_sum_hi:.0f}]")
    print(f"    Rule 2: All odd (0 even) or all even (0 odd)")
    print(f"    Rule 3: 0 or 1 odd (or 5-6 odd) → less than 2% each")
    print(f"    Rule 4: Only 1-2 distinct decades")
    print(f"    Rule 5: Max consecutive >= 4")
    
    return {
        'sum_lo': safe_sum_lo, 'sum_hi': safe_sum_hi,
        'min_odd': 2, 'max_odd': 4,
        'min_decades': 3,
        'max_consecutive': 3,
    }


def attack3_pair_locking(data, max_num=45, pick=6):
    """Attack 3: Find anomalously frequent pairs."""
    print("\n" + "="*60)
    print("  ATTACK 3: PAIR LOCKING")
    print("  Find pairs appearing anomalously often")
    print("="*60)
    
    n_draws = len(data)
    expected = n_draws * pick * (pick - 1) / (max_num * (max_num - 1))
    
    pair_freq = Counter()
    for d in data:
        for a, b in combinations(sorted(d), 2):
            pair_freq[(a, b)] += 1
    
    # Find anomalous pairs (> 2x expected)
    anomalous = [(p, c, c/expected) for p, c in pair_freq.most_common(50) if c > expected * 1.5]
    
    print(f"  Expected pair frequency: {expected:.2f}")
    print(f"  Anomalous pairs (>1.5x expected): {len(anomalous)}")
    for p, c, ratio in anomalous[:20]:
        print(f"    ({p[0]:2d}, {p[1]:2d}): {c} times ({ratio:.2f}x expected)")
    
    # Test: recent 200 draws
    recent_pairs = Counter()
    for d in data[-200:]:
        for a, b in combinations(sorted(d), 2):
            recent_pairs[(a, b)] += 1
    expected_recent = 200 * pick * (pick - 1) / (max_num * (max_num - 1))
    recent_anomalous = [(p, c, c/expected_recent) for p, c in recent_pairs.most_common(30) if c > expected_recent * 2]
    
    print(f"\n  Recent 200 draws — anomalous pairs (>2x):")
    for p, c, ratio in recent_anomalous[:10]:
        print(f"    ({p[0]:2d}, {p[1]:2d}): {c} times ({ratio:.2f}x)")
    
    return anomalous[:20], recent_anomalous[:10]


def attack4_position_ranges(data, max_num=45, pick=6):
    """Attack 4: Build tight per-position pools."""
    print("\n" + "="*60)
    print("  ATTACK 4: POSITION-RANGE COMPRESSION")
    print("  Build tight pools per sorted position")
    print("="*60)
    
    pos_vals = [[] for _ in range(pick)]
    for d in data:
        sd = sorted(d)
        for p in range(pick):
            pos_vals[p].append(sd[p])
    
    pos_pools = []
    for p in range(pick):
        vals = pos_vals[p]
        lo = int(np.percentile(vals, 5))
        hi = int(np.percentile(vals, 95))
        freq = Counter(vals)
        top_vals = [v for v, _ in freq.most_common(12)]
        coverage = sum(freq[v] for v in top_vals) / len(vals) * 100
        
        pool = sorted(set(range(lo, hi + 1)) & set(v for v in range(1, max_num + 1)))
        print(f"\n  Pos{p+1}: 90% range=[{lo},{hi}] (width={hi-lo+1})")
        print(f"    Top-12 values: {sorted(top_vals)} (cover {coverage:.1f}%)")
        pos_pools.append({'lo': lo, 'hi': hi, 'top': sorted(top_vals)})
    
    # Cross-position independence test
    total_space = 1
    for p in range(pick):
        total_space *= len(pos_pools[p]['top'])
    print(f"\n  Position-constrained search space: {total_space:,}")
    print(f"  vs unconstrained C(45,6): {comb(45,6):,}")
    print(f"  Compression ratio: {comb(45,6)/total_space:.0f}x")
    
    return pos_pools


def attack5_overdue_window(data, max_num=45, pick=6):
    """Attack 5: Detect 'window of opportunity' when multiple numbers are overdue."""
    print("\n" + "="*60)
    print("  ATTACK 5: OVERDUE WINDOW DETECTION")
    print("="*60)
    
    n = len(data)
    expected_gap = max_num / pick  # ~7.5 draws between appearances
    
    # Track gap for each number
    last_seen = {}
    current_gaps = {}
    
    for i, d in enumerate(data):
        for num in range(1, max_num + 1):
            if num in d:
                last_seen[num] = i
        if i == n - 1:
            for num in range(1, max_num + 1):
                current_gaps[num] = i - last_seen.get(num, -100)
    
    # Numbers currently overdue (gap > 1.5x expected)
    overdue = [(num, gap) for num, gap in current_gaps.items() if gap > expected_gap * 1.5]
    overdue.sort(key=lambda x: -x[1])
    
    print(f"  Expected gap: {expected_gap:.1f} draws")
    print(f"  Currently overdue (gap > {expected_gap*1.5:.0f}): {len(overdue)} numbers")
    for num, gap in overdue[:15]:
        print(f"    Number {num:2d}: gap={gap} draws ({gap/expected_gap:.1f}x expected)")
    
    # Test: does "overdue" predict return?
    hit_count = 0
    test_count = 0
    for i in range(100, n - 1):
        gaps_at_i = {}
        for num in range(1, max_num + 1):
            last = -100
            for j in range(i, max(0, i - 100), -1):
                if num in data[j]:
                    last = j
                    break
            gaps_at_i[num] = i - last
        
        # Get top-10 overdue numbers
        overdue_at_i = sorted(range(1, max_num + 1), key=lambda x: -gaps_at_i[x])[:10]
        actual_next = set(data[i + 1])
        
        if any(n in actual_next for n in overdue_at_i):
            hit_count += 1
        test_count += 1
    
    hit_rate = hit_count / test_count * 100
    random_rate = (1 - comb(max_num - 10, pick) / comb(max_num, pick)) * 100
    print(f"\n  OVERDUE PREDICTION TEST ({test_count} draws):")
    print(f"    Top-10 overdue hit next draw: {hit_rate:.1f}%")
    print(f"    Random expected (any 10 numbers): {random_rate:.1f}%")
    print(f"    Ratio: {hit_rate/random_rate:.2f}x")
    
    return overdue


def combined_backtest(data, max_num=45, pick=6, test_count=200):
    """Backtest V16: all attacks combined."""
    print("\n" + "="*60)
    print("  V16 COMBINED ATTACK BACKTEST")
    print("="*60)
    
    n = len(data)
    start = max(100, n - test_count)
    total_tests = n - start - 1
    
    any_3 = any_4 = any_5 = any_6 = 0
    total = 0
    five_plus_details = []
    t0 = time.time()
    
    for i in range(start, n - 1):
        history = data[:i + 1]
        actual = set(data[i + 1])
        total += 1
        
        # === BUILD COMPRESSED POOL ===
        # Recent frequency (aggressive: top-25 from last 100 draws)
        recent_freq = Counter(num for d in history[-100:] for num in d)
        pool_25 = [n for n, _ in recent_freq.most_common(25)]
        
        # Position pools (top-10 per position)
        pos_freq = [Counter() for _ in range(pick)]
        for d in history[-200:]:
            sd = sorted(d)
            for p in range(pick):
                pos_freq[p][sd[p]] += 1
        pos_pools = [[v for v, _ in pos_freq[p].most_common(12)] for p in range(pick)]
        
        # Pair anomaly bonus
        pair_freq = Counter()
        for d in history[-150:]:
            for a, b in combinations(sorted(d), 2):
                pair_freq[(a, b)] += 1
        top_pairs = pair_freq.most_common(30)
        pair_numbers = set()
        for (a, b), c in top_pairs:
            pair_numbers.add(a)
            pair_numbers.add(b)
        
        # Overdue numbers
        last_seen = {}
        for di, d in enumerate(history):
            for num in d:
                last_seen[num] = di
        hist_len = len(history)
        expected_gap = max_num / pick
        overdue_nums = [num for num in range(1, max_num + 1)
                       if hist_len - last_seen.get(num, 0) > expected_gap * 1.5]
        
        # === ANTI-PATTERN RULES ===
        sums = [sum(d) for d in history[-200:]]
        sum_lo = np.percentile(sums, 3)
        sum_hi = np.percentile(sums, 97)
        
        def is_valid_combo(combo):
            s = sum(combo)
            if s < sum_lo or s > sum_hi:
                return False
            odd = sum(1 for n in combo if n % 2 == 1)
            if odd < 2 or odd > 4:
                return False
            decades = len(set(n // 10 for n in combo))
            if decades < 3:
                return False
            sd = sorted(combo)
            for j in range(len(sd) - 2):
                if sd[j+1] == sd[j] + 1 and sd[j+2] == sd[j] + 2:
                    # 3 consecutive is OK, but 4+ is bad
                    if j + 3 < len(sd) and sd[j+3] == sd[j] + 3:
                        return False
            return True
        
        # === GENERATE DAN (5000 sets) ===
        dan = []
        seen = set()
        n_dan = 5000
        
        # Strategy A: Position-constrained (40%)
        for _ in range(n_dan * 4):
            if len(dan) >= n_dan * 40 // 100:
                break
            combo = []
            used = set()
            for p in range(pick):
                candidates = [n for n in pos_pools[p] if n not in used]
                if not candidates:
                    candidates = [n for n in pool_25 if n not in used]
                if not candidates:
                    candidates = [n for n in range(1, max_num + 1) if n not in used]
                weights = np.array([pos_freq[p].get(c, 1) + np.random.random() * 2 for c in candidates])
                weights = weights / weights.sum()
                chosen = int(np.random.choice(candidates, p=weights))
                combo.append(chosen)
                used.add(chosen)
            combo = sorted(combo)
            if is_valid_combo(combo) and tuple(combo) not in seen:
                dan.append(combo)
                seen.add(tuple(combo))
        
        # Strategy B: Pair-locked (25%)
        for _ in range(n_dan * 3):
            if len(dan) >= n_dan * 65 // 100:
                break
            # Start with a top pair
            pair_idx = np.random.randint(min(20, len(top_pairs)))
            (a, b), _ = top_pairs[pair_idx]
            combo = [a, b]
            used = {a, b}
            while len(combo) < pick:
                candidates = [n for n in pool_25 if n not in used]
                if not candidates:
                    candidates = [n for n in range(1, max_num + 1) if n not in used]
                weights = np.array([recent_freq.get(c, 1) + np.random.random() * 2 for c in candidates])
                weights = weights / weights.sum()
                chosen = int(np.random.choice(candidates, p=weights))
                combo.append(chosen)
                used.add(chosen)
            combo = sorted(combo)
            if is_valid_combo(combo) and tuple(combo) not in seen:
                dan.append(combo)
                seen.add(tuple(combo))
        
        # Strategy C: Overdue-seeded (15%)
        for _ in range(n_dan * 3):
            if len(dan) >= n_dan * 80 // 100:
                break
            # Seed with 1-2 overdue numbers
            n_overdue_seed = min(2, len(overdue_nums))
            if n_overdue_seed > 0:
                seeds = list(np.random.choice(overdue_nums, min(n_overdue_seed, len(overdue_nums)), replace=False))
            else:
                seeds = []
            combo = list(seeds)
            used = set(combo)
            while len(combo) < pick:
                candidates = [n for n in pool_25 if n not in used]
                if not candidates:
                    candidates = [n for n in range(1, max_num + 1) if n not in used]
                weights = np.array([recent_freq.get(c, 1) + np.random.random() * 3 for c in candidates])
                weights = weights / weights.sum()
                chosen = int(np.random.choice(candidates, p=weights))
                combo.append(chosen)
                used.add(chosen)
            combo = sorted(combo)
            if is_valid_combo(combo) and tuple(combo) not in seen:
                dan.append(combo)
                seen.add(tuple(combo))
        
        # Strategy D: Pure filtered random from pool_25 (20%)
        while len(dan) < n_dan:
            combo = sorted(np.random.choice(pool_25, pick, replace=False).tolist())
            if is_valid_combo(combo) and tuple(combo) not in seen:
                dan.append(combo)
                seen.add(tuple(combo))
        
        # Score
        matches = [len(set(d) & actual) for d in dan]
        best = max(matches) if matches else 0
        
        if any(m >= 3 for m in matches): any_3 += 1
        if any(m >= 4 for m in matches): any_4 += 1
        if any(m >= 5 for m in matches):
            any_5 += 1
            for j, m in enumerate(matches):
                if m >= 5:
                    five_plus_details.append({
                        'draw': i + 1, 'actual': sorted(actual),
                        'predicted': dan[j], 'match': m
                    })
                    break
        if any(m >= 6 for m in matches): any_6 += 1
        
        if total % 10 == 0 or total == total_tests:
            elapsed = time.time() - t0
            eta = (elapsed / total) * (total_tests - total)
            print(f"  [{total}/{total_tests}] best={best}/6 | 5+={any_5} | 6/6={any_6} | {elapsed:.0f}s | ETA {eta:.0f}s")
    
    elapsed = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"  V16 RESULTS ({total} draws, {n_dan} dan)")
    print(f"{'='*60}")
    print(f"  3+: {any_3}/{total} = {any_3/total*100:.1f}%")
    print(f"  4+: {any_4}/{total} = {any_4/total*100:.1f}%")
    print(f"  5+: {any_5}/{total} = {any_5/total*100:.1f}%  <<<")
    print(f"  6/6: {any_6}/{total} = {any_6/total*100:.1f}%  <<<")
    print(f"  Time: {elapsed:.0f}s")
    
    # Compare
    from math import comb as mcomb
    print(f"\n  COMPARISON:")
    print(f"  V14 (200 draws): 5+=12.6%, 6/6=0.0%")
    print(f"  V16 (200 draws): 5+={any_5/total*100:.1f}%, 6/6={any_6/total*100:.1f}%")
    
    p5_single = mcomb(pick, 5) * mcomb(max_num - pick, 1) / mcomb(max_num, pick)
    p5_any = 1 - (1 - p5_single) ** n_dan
    random_5 = p5_any * total
    print(f"  Random expected 5+: {random_5:.1f}")
    print(f"  V16 vs Random (5+): {any_5/random_5:.2f}x" if random_5 > 0 else "")
    
    if five_plus_details:
        print(f"\n  5+ DETAILS ({len(five_plus_details)}):")
        for d in five_plus_details[:15]:
            tag = "6/6!" if d['match'] == 6 else f"{d['match']}/6"
            print(f"    #{d['draw']}: actual={d['actual']} pred={d['predicted']} [{tag}]")
    
    return {'any_3': any_3/total*100, 'any_4': any_4/total*100,
            'any_5': any_5/total*100, 'any_6': any_6/total*100,
            'total': total}


if __name__ == '__main__':
    print("Loading data...")
    mega = get_mega645_numbers()
    print(f"Mega 6/45: {len(mega)} draws")
    
    # Run analysis
    ranked, freq = attack1_mandel_compression(mega)
    filters = attack2_antipattern_filter(mega)
    pairs_all, pairs_recent = attack3_pair_locking(mega)
    pos_pools = attack4_position_ranges(mega)
    overdue = attack5_overdue_window(mega)
    
    # V16 combined backtest
    results = combined_backtest(mega, test_count=200)
    
    print(f"\n{'#'*60}")
    print(f"  ALL ATTACKS COMPLETE")
    print(f"{'#'*60}")
