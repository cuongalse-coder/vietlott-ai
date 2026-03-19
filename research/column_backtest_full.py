"""
Column Pool Walk-Forward Backtest — Test trên TOÀN BỘ kỳ quay
===============================================================
Với mỗi kỳ, dùng lịch sử trước đó để build column pools (top-N mỗi cột),
rồi check xem kỳ tiếp theo có nằm trong pool không.
Test: top-8, top-10, top-12, top-15 mỗi cột.
Đo: % kỳ mà 6/6 cột đều trúng pool, và trung bình bao nhiêu cột trúng.
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from collections import Counter
import numpy as np
from scraper.data_manager import get_mega645_numbers


def walk_forward_column_test(data, max_num=45, pick=6, min_history=60):
    """Walk-forward: tại mỗi kỳ i, dùng data[:i] để build pools, check data[i]."""
    n = len(data)
    total_tests = n - min_history
    
    top_ns = [8, 10, 12, 15]
    lookbacks = [50, 100, 200, 'all']
    
    # Results storage
    results = {}
    for top_n in top_ns:
        for lb in lookbacks:
            key = f"top{top_n}_lb{lb}"
            results[key] = {
                'full_match': 0,    # 6/6 cột trúng
                'pos_hits': [0]*pick,  # per-position hits
                'col_counts': Counter(),  # distribution of how many cols matched
                'total': 0,
            }
    
    t0 = time.time()
    
    for i in range(min_history, n):
        actual = sorted(data[i])
        
        for lb in lookbacks:
            if lb == 'all':
                history = data[:i]
            else:
                history = data[max(0, i - lb):i]
            
            # Build per-position frequency
            pos_freq = [Counter() for _ in range(pick)]
            for d in history:
                sd = sorted(d)
                for p in range(pick):
                    pos_freq[p][sd[p]] += 1
            
            for top_n in top_ns:
                key = f"top{top_n}_lb{lb}"
                r = results[key]
                r['total'] += 1
                
                # Top-N per position
                pools = [set(num for num, _ in pos_freq[p].most_common(top_n)) for p in range(pick)]
                
                cols_matched = 0
                for p in range(pick):
                    if actual[p] in pools[p]:
                        r['pos_hits'][p] += 1
                        cols_matched += 1
                
                r['col_counts'][cols_matched] += 1
                if cols_matched == pick:
                    r['full_match'] += 1
        
        done = i - min_history + 1
        if done % 200 == 0 or done == total_tests:
            elapsed = time.time() - t0
            # Show progress with one key metric
            k = f"top10_lb100"
            fm = results[k]['full_match']
            t = results[k]['total']
            print(f"  [{done}/{total_tests}] top10_lb100 6/6={fm}/{t} ({fm/t*100:.1f}%) | {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    
    # === PRINT RESULTS ===
    print(f"\n{'='*90}")
    print(f"  WALK-FORWARD COLUMN TEST — {total_tests} ky quay")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*90}")
    
    # Table 1: Full match rates
    print(f"\n  6/6 COT DEU TRUNG POOL (% ky):")
    print(f"  {'':15s}", end="")
    for lb in lookbacks:
        print(f" | {'lb='+str(lb):>10s}", end="")
    print()
    print(f"  {'─'*15}", end="")
    for _ in lookbacks:
        print(f"-+-{'─'*10}", end="")
    print()
    for top_n in top_ns:
        print(f"  {'Top-'+str(top_n)+'/cot':15s}", end="")
        for lb in lookbacks:
            key = f"top{top_n}_lb{lb}"
            r = results[key]
            pct = r['full_match'] / r['total'] * 100
            print(f" | {pct:9.1f}%", end="")
        print()
    
    # Table 2: Per-position hit rates (best config)
    print(f"\n  TY LE TRUNG TUNG COT (top-10, lookback=100):")
    best_key = "top10_lb100"
    r = results[best_key]
    for p in range(pick):
        pct = r['pos_hits'][p] / r['total'] * 100
        bar = '█' * int(pct / 2)
        print(f"    Cot {p+1}: {pct:5.1f}% {bar}")
    
    # Table 3: Distribution of matched columns
    print(f"\n  PHAN BO SO COT TRUNG (top-10, lookback=100):")
    for m in range(pick + 1):
        cnt = r['col_counts'].get(m, 0)
        pct = cnt / r['total'] * 100
        bar = '█' * int(pct)
        print(f"    {m}/6 cot: {cnt:5d} ({pct:5.1f}%) {bar}")
    
    # Table 4: All configs comparison (for dan generation potential)
    print(f"\n  SEARCH SPACE vs COVERAGE:")
    print(f"  {'Config':20s} | {'6/6 match':>10s} | {'Avg cols':>9s} | {'Search space':>15s} | {'Compression':>12s}")
    print(f"  {'─'*20}-+-{'─'*10}-+-{'─'*9}-+-{'─'*15}-+-{'─'*12}")
    for top_n in top_ns:
        key = f"top{top_n}_lb100"
        r = results[key]
        pct = r['full_match'] / r['total'] * 100
        avg_cols = sum(m * c for m, c in r['col_counts'].items()) / r['total']
        space = top_n ** pick
        compress = 8145060 / space
        print(f"  {'Top-'+str(top_n)+' lb100':20s} | {pct:9.1f}% | {avg_cols:8.2f} | {space:>13,} | {compress:>10.1f}x")
    
    # === CONCLUSION ===
    print(f"\n{'='*90}")
    print(f"  KET LUAN")
    print(f"{'='*90}")
    
    best = results["top10_lb100"]
    pct_6 = best['full_match'] / best['total'] * 100
    avg = sum(m * c for m, c in best['col_counts'].items()) / best['total']
    
    print(f"  Voi top-10 moi cot (lookback=100):")
    print(f"    - Trung binh {avg:.2f}/6 cot moi ky")
    print(f"    - {pct_6:.1f}% ky trung GON 6/6 cot")
    print(f"    - Search space: 1,000,000 combos (8x nho hon C(45,6))")
    print(f"    - Neu dung 5000 dan trong search space nay -> cover {5000/1000000*100:.1f}%")
    
    # Compare with random
    # Random top-10 per col: probability of hitting  
    print(f"\n  So voi RANDOM (chon 10 so bat ky moi cot):")
    p_random_col = 10 / max_num  # ~22% per col
    p_random_6 = p_random_col ** pick
    print(f"    Random 1 cot: {p_random_col*100:.1f}%")
    print(f"    Random 6/6: {p_random_6*100:.4f}%")
    print(f"    Model 6/6: {pct_6:.1f}%")
    print(f"    Model vs Random: {pct_6 / (p_random_6*100):.1f}x")
    
    return results


if __name__ == '__main__':
    print("Loading data...")
    mega = get_mega645_numbers()
    print(f"Mega 6/45: {len(mega)} ky quay")
    print(f"Test tu ky 61 den ky {len(mega)} ({len(mega)-60} tests)")
    
    results = walk_forward_column_test(mega)
    
    print(f"\n{'#'*60}")
    print(f"  DONE")
    print(f"{'#'*60}")
