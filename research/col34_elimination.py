"""
Bước 2: Mô Hình Loại Trừ Cho Cột 3 & Cột 4
=============================================
Cột 3 (43.5%) và Cột 4 (43.0%) là "tường lửa" — cải thiện ở đây = tăng jackpot rate.
Thử nhiều kỹ thuật loại trừ/mở rộng pool thông minh cho 2 cột này.
Walk-forward test trên TOÀN BỘ kỳ quay.
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from collections import Counter, defaultdict
import numpy as np
from scraper.data_manager import get_mega645_numbers


def analyze_col34_patterns(data, pick=6):
    """Phân tích sâu tại sao Cột 3 & 4 khó đoán."""
    print(f"\n{'='*70}")
    print(f"  PHAN TICH SAU: COT 3 & COT 4")
    print(f"{'='*70}")

    for col in [2, 3]:  # 0-indexed: col 2 = Cột 3, col 3 = Cột 4
        col_name = f"Cot {col+1}"
        vals = [sorted(d)[col] for d in data]

        print(f"\n  --- {col_name} ---")
        print(f"  Range: [{min(vals)}, {max(vals)}], spread={max(vals)-min(vals)}")
        print(f"  Unique values used: {len(set(vals))}")
        
        # Entropy (how spread out)
        freq = Counter(vals)
        total = len(vals)
        probs = [c/total for c in freq.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(len(freq))
        print(f"  Entropy: {entropy:.2f} / {max_entropy:.2f} ({entropy/max_entropy*100:.1f}% of max)")

        # Volatility (how much it jumps between draws)
        diffs = [abs(vals[i] - vals[i-1]) for i in range(1, len(vals))]
        print(f"  Avg jump between draws: {np.mean(diffs):.1f}")
        print(f"  Max jump: {max(diffs)}")
        
        # Autocorrelation
        from numpy import corrcoef
        for lag in [1, 2, 3, 5, 10]:
            if len(vals) > lag:
                corr = corrcoef(vals[:-lag], vals[lag:])[0, 1]
                print(f"  Autocorrelation lag-{lag}: {corr:.4f}")


def method_markov_chain(history, col, top_n=10, order=1):
    """Markov chain: predict next value based on current value."""
    seq = [sorted(d)[col] for d in history]
    if len(seq) < order + 1:
        return set(range(1, 46))
    
    # Build transition counts
    trans = defaultdict(Counter)
    for i in range(order, len(seq)):
        state = tuple(seq[i-order:i])
        trans[state][seq[i]] += 1
    
    # Current state
    current = tuple(seq[-order:])
    if current in trans:
        predictions = trans[current].most_common(top_n)
        return set(v for v, _ in predictions)
    
    # Fallback: frequency
    freq = Counter(seq[-100:])
    return set(v for v, _ in freq.most_common(top_n))


def method_momentum(history, col, top_n=10):
    """Momentum: weight recent values higher, detect direction."""
    seq = [sorted(d)[col] for d in history]
    if len(seq) < 10:
        return set(range(1, 46))
    
    scores = Counter()
    # Recent frequency with decay
    for i, val in enumerate(seq[-30:]):
        weight = (i + 1) / 30
        scores[val] += weight * 3
    
    # Direction: if trending up, boost higher values
    recent_5 = seq[-5:]
    recent_20 = seq[-20:]
    diff = np.mean(recent_5) - np.mean(recent_20)
    for val in range(1, 46):
        if diff > 0 and val > np.mean(recent_5):
            scores[val] += 1
        elif diff < 0 and val < np.mean(recent_5):
            scores[val] += 1
    
    # Neighbor bonus: values near recent
    for val in seq[-3:]:
        for neighbor in range(max(1, val-2), min(46, val+3)):
            scores[neighbor] += 0.5
    
    return set(v for v, _ in scores.most_common(top_n))


def method_conditional(history, col, top_n=10):
    """Conditional: predict col3/4 based on col1/col6 of current draw's predecessors."""
    pick = 6
    seq = [sorted(d) for d in history]
    if len(seq) < 5:
        return set(range(1, 46))
    
    # What was col1 and col6 in last draw?
    last = seq[-1]
    col1_val = last[0]
    col6_val = last[5]
    
    # Find historical draws with similar col1/col6
    scores = Counter()
    for i in range(len(seq) - 1):
        d = seq[i]
        sim = 0
        if abs(d[0] - col1_val) <= 2: sim += 1
        if abs(d[5] - col6_val) <= 2: sim += 1
        if sim >= 1:
            # What followed at this column?
            if i + 1 < len(seq):
                scores[seq[i+1][col]] += sim
    
    if scores:
        return set(v for v, _ in scores.most_common(top_n))
    
    freq = Counter(d[col] for d in seq[-100:])
    return set(v for v, _ in freq.most_common(top_n))


def method_gap_return(history, col, top_n=10, max_num=45):
    """Gap return: numbers absent long from this column are more likely to return."""
    seq = [sorted(d)[col] for d in history]
    last_seen = {}
    for i, val in enumerate(seq):
        last_seen[val] = i
    
    n = len(seq)
    # Score by gap + frequency
    freq = Counter(seq[-100:])
    scores = Counter()
    for val in range(1, max_num + 1):
        gap = n - last_seen.get(val, 0)
        f = freq.get(val, 0)
        scores[val] = gap * 0.3 + f * 2  # balance overdue with frequency
    
    return set(v for v, _ in scores.most_common(top_n))


def method_union_smart(history, col, top_n=12):
    """Smart union: combine frequency + markov + momentum, take union of top candidates."""
    freq_pool = method_markov_chain(history, col, top_n=8, order=1)
    mom_pool = method_momentum(history, col, top_n=8)
    cond_pool = method_conditional(history, col, top_n=8)
    gap_pool = method_gap_return(history, col, top_n=8)
    
    # Union then trim to top_n by voting
    votes = Counter()
    for pool in [freq_pool, mom_pool, cond_pool, gap_pool]:
        for v in pool:
            votes[v] += 1
    
    # Frequency as tiebreaker
    seq = [sorted(d)[col] for d in history[-100:]]
    freq = Counter(seq)
    
    ranked = sorted(votes.items(), key=lambda x: (-x[1], -freq.get(x[0], 0)))
    return set(v for v, _ in ranked[:top_n])


def walk_forward_enhanced(data, max_num=45, pick=6, min_history=100):
    """Walk-forward test: compare baseline vs enhanced methods for col 3 & 4."""
    n = len(data)
    total_tests = n - min_history
    
    methods = {
        'baseline_freq': lambda h, c: set(v for v, _ in Counter(sorted(d)[c] for d in h[-100:]).most_common(10)),
        'markov_1': lambda h, c: method_markov_chain(h, c, top_n=10, order=1),
        'markov_2': lambda h, c: method_markov_chain(h, c, top_n=10, order=2),
        'momentum': lambda h, c: method_momentum(h, c, top_n=10),
        'conditional': lambda h, c: method_conditional(h, c, top_n=10),
        'gap_return': lambda h, c: method_gap_return(h, c, top_n=10),
        'smart_union_12': lambda h, c: method_union_smart(h, c, top_n=12),
        'smart_union_15': lambda h, c: method_union_smart(h, c, top_n=15),
        'freq_12': lambda h, c: set(v for v, _ in Counter(sorted(d)[c] for d in h[-100:]).most_common(12)),
        'freq_15': lambda h, c: set(v for v, _ in Counter(sorted(d)[c] for d in h[-100:]).most_common(15)),
    }
    
    # Track hits for Col3 (idx=2) and Col4 (idx=3) separately
    results = {name: {'col3_hits': 0, 'col4_hits': 0, 'both_hits': 0, 'total': 0,
                       'full6_hits': 0}
               for name in methods}
    
    t0 = time.time()
    
    for i in range(min_history, n):
        history = data[:i]
        actual = sorted(data[i])
        
        # Fixed: col1, col2, col5, col6 always use top-10 freq
        fixed_pools = {}
        for col in [0, 1, 4, 5]:  # col1, col2, col5, col6
            freq = Counter(sorted(d)[col] for d in history[-100:])
            fixed_pools[col] = set(v for v, _ in freq.most_common(10))
        
        # Check fixed cols
        fixed_ok = all(actual[col] in fixed_pools[col] for col in [0, 1, 4, 5])
        
        for name, method_fn in methods.items():
            r = results[name]
            r['total'] += 1
            
            pool3 = method_fn(history, 2)  # Col 3
            pool4 = method_fn(history, 3)  # Col 4
            
            c3_hit = actual[2] in pool3
            c4_hit = actual[3] in pool4
            
            if c3_hit: r['col3_hits'] += 1
            if c4_hit: r['col4_hits'] += 1
            if c3_hit and c4_hit: r['both_hits'] += 1
            if c3_hit and c4_hit and fixed_ok: r['full6_hits'] += 1
        
        done = i - min_history + 1
        if done % 200 == 0 or done == total_tests:
            elapsed = time.time() - t0
            bl = results['baseline_freq']
            su = results['smart_union_12']
            print(f"  [{done}/{total_tests}] baseline c3={bl['col3_hits']/bl['total']*100:.1f}% c4={bl['col4_hits']/bl['total']*100:.1f}% | "
                  f"smart12 c3={su['col3_hits']/su['total']*100:.1f}% c4={su['col4_hits']/su['total']*100:.1f}% | {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    
    print(f"\n{'='*90}")
    print(f"  KET QUA LOAI TRU COT 3 & COT 4 — {total_tests} ky")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*90}")
    
    print(f"\n  {'Method':20s} | {'Col3':>7s} | {'Col4':>7s} | {'Both':>7s} | {'6/6':>7s} | {'Pool':>5s}")
    print(f"  {'─'*20}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*5}")
    
    for name in ['baseline_freq', 'markov_1', 'markov_2', 'momentum', 'conditional',
                  'gap_return', 'smart_union_12', 'smart_union_15', 'freq_12', 'freq_15']:
        r = results[name]
        t = r['total']
        c3 = r['col3_hits']/t*100
        c4 = r['col4_hits']/t*100
        both = r['both_hits']/t*100
        f6 = r['full6_hits']/t*100
        pool_n = name.split('_')[-1] if any(c.isdigit() for c in name.split('_')[-1]) else '10'
        print(f"  {name:20s} | {c3:6.1f}% | {c4:6.1f}% | {both:5.1f}% | {f6:5.1f}% | {pool_n:>5s}")
    
    # Best method analysis
    best_name = max(results.keys(), key=lambda k: results[k]['full6_hits'])
    best = results[best_name]
    t = best['total']
    
    print(f"\n  BEST: {best_name}")
    print(f"    Col3: {best['col3_hits']/t*100:.1f}%")
    print(f"    Col4: {best['col4_hits']/t*100:.1f}%")
    print(f"    Both: {best['both_hits']/t*100:.1f}%")
    print(f"    6/6 match: {best['full6_hits']/t*100:.1f}%")
    
    # Compare with baseline
    bl = results['baseline_freq']
    print(f"\n  IMPROVEMENT vs baseline (freq top-10):")
    print(f"    Col3: {best['col3_hits']/t*100:.1f}% vs {bl['col3_hits']/bl['total']*100:.1f}% ({(best['col3_hits']-bl['col3_hits'])/bl['total']*100:+.1f}%)")
    print(f"    Col4: {best['col4_hits']/t*100:.1f}% vs {bl['col4_hits']/bl['total']*100:.1f}% ({(best['col4_hits']-bl['col4_hits'])/bl['total']*100:+.1f}%)")
    print(f"    6/6: {best['full6_hits']/t*100:.1f}% vs {bl['full6_hits']/bl['total']*100:.1f}% ({(best['full6_hits']-bl['full6_hits'])/bl['total']*100:+.1f}%)")
    
    return results


if __name__ == '__main__':
    print("Loading data...")
    mega = get_mega645_numbers()
    print(f"Mega 6/45: {len(mega)} ky quay")
    
    # Step 1: Analyze patterns
    analyze_col34_patterns(mega)
    
    # Step 2: Walk-forward test all elimination methods
    results = walk_forward_enhanced(mega)
    
    print(f"\n{'#'*60}")
    print(f"  DONE")
    print(f"{'#'*60}")
