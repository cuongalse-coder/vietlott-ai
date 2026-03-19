"""
Column Frequency Analysis — Bước 1: Thống kê nền tảng
======================================================
Phân tích tần suất từng số ở từng cột (sorted position 1-6).
Dùng làm cơ sở cho bước loại trừ tiếp theo.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from collections import Counter, defaultdict
import numpy as np
from scraper.data_manager import get_mega645_numbers


def column_frequency_analysis(data, max_num=45, pick=6):
    n = len(data)
    print(f"\n{'='*80}")
    print(f"  THONG KE TAN SUAT THEO TUNG COT — {n} ky quay")
    print(f"{'='*80}")

    # === PER-COLUMN FREQUENCY ===
    pos_freq = [Counter() for _ in range(pick)]
    for d in data:
        sd = sorted(d)
        for p in range(pick):
            pos_freq[p][sd[p]] += 1

    col_names = ['Cot 1 (nho nhat)', 'Cot 2', 'Cot 3', 'Cot 4', 'Cot 5', 'Cot 6 (lon nhat)']

    for p in range(pick):
        print(f"\n{'─'*80}")
        print(f"  {col_names[p].upper()}")
        print(f"{'─'*80}")
        freq = pos_freq[p]
        total = sum(freq.values())
        ranked = freq.most_common()

        # Top 15
        print(f"  Top 15 so xuat hien nhieu nhat:")
        print(f"  {'So':>4s} | {'Lan':>5s} | {'%':>6s} | {'Bar':30s}")
        print(f"  {'─'*4}-+-{'─'*5}-+-{'─'*6}-+-{'─'*30}")
        for num, cnt in ranked[:15]:
            pct = cnt / total * 100
            bar = '█' * int(pct * 2)
            print(f"  {num:4d} | {cnt:5d} | {pct:5.1f}% | {bar}")

        # Range stats
        vals = [num for num, _ in ranked]
        print(f"\n  Range thuc te: [{min(vals)}, {max(vals)}]")
        print(f"  Mean: {np.mean([num for d in data for i, num in enumerate(sorted(d)) if i == p]):.1f}")

        # Coverage: top-N numbers cover how much?
        for top_n in [5, 8, 10, 12, 15]:
            top_nums = set(num for num, _ in ranked[:top_n])
            cov = sum(cnt for num, cnt in ranked[:top_n]) / total * 100
            print(f"  Top-{top_n:2d} cover: {cov:.1f}%")

    # === OVERALL TOP NUMBERS PER COLUMN ===
    print(f"\n{'='*80}")
    print(f"  TOM TAT: TOP-10 SO MOI COT")
    print(f"{'='*80}")
    print(f"  {'Cot':8s} | {'Top-10 so (nhieu -> it)':50s} | {'Cover':>7s}")
    print(f"  {'─'*8}-+-{'─'*50}-+-{'─'*7}")
    col_pools = []
    for p in range(pick):
        top10 = pos_freq[p].most_common(10)
        nums_str = ', '.join(f"{num}" for num, _ in top10)
        total = sum(pos_freq[p].values())
        cov = sum(cnt for _, cnt in top10) / total * 100
        print(f"  Cot {p+1:2d}  | {nums_str:50s} | {cov:5.1f}%")
        col_pools.append([num for num, _ in top10])

    # === RECENT TRENDS (last 100 draws) ===
    print(f"\n{'='*80}")
    print(f"  XU HUONG GAN DAY (100 ky cuoi)")
    print(f"{'='*80}")
    recent_pos_freq = [Counter() for _ in range(pick)]
    for d in data[-100:]:
        sd = sorted(d)
        for p in range(pick):
            recent_pos_freq[p][sd[p]] += 1

    print(f"  {'Cot':8s} | {'Top-10 so (100 ky gan nhat)':50s} | {'Cover':>7s}")
    print(f"  {'─'*8}-+-{'─'*50}-+-{'─'*7}")
    recent_pools = []
    for p in range(pick):
        top10 = recent_pos_freq[p].most_common(10)
        nums_str = ', '.join(f"{num}" for num, _ in top10)
        total = sum(recent_pos_freq[p].values())
        cov = sum(cnt for _, cnt in top10) / total * 100
        print(f"  Cot {p+1:2d}  | {nums_str:50s} | {cov:5.1f}%")
        recent_pools.append([num for num, _ in top10])

    # === CROSS-COLUMN: unique numbers total ===
    all_pool = set()
    for pool in col_pools:
        all_pool.update(pool)
    print(f"\n  Tong so unique (top-10 moi cot): {len(all_pool)} / 45 so")
    print(f"  Pool: {sorted(all_pool)}")

    all_recent = set()
    for pool in recent_pools:
        all_recent.update(pool)
    print(f"  Tong so unique (recent top-10): {len(all_recent)} / 45 so")
    print(f"  Pool recent: {sorted(all_recent)}")

    # === VALIDATION: How many draws fully covered by column pool? ===
    print(f"\n{'='*80}")
    print(f"  KIEM TRA: BAO NHIEU KY TRUNG GON TRONG POOL?")
    print(f"{'='*80}")

    # Method: for each draw, check if number at pos p is in col_pools[p]
    for label, pools, test_data in [
        ("All data (top-10)", col_pools, data),
        ("Recent 100 (top-10)", recent_pools, data[-100:]),
        ("All data on recent 200", col_pools, data[-200:]),
    ]:
        full_match = 0
        pos_match_counts = [0] * pick
        for d in test_data:
            sd = sorted(d)
            all_in = True
            for p in range(pick):
                if sd[p] in pools[p]:
                    pos_match_counts[p] += 1
                else:
                    all_in = False
            if all_in:
                full_match += 1
        total = len(test_data)
        print(f"\n  {label}:")
        print(f"    6/6 cot deu trung pool: {full_match}/{total} = {full_match/total*100:.1f}%")
        for p in range(pick):
            print(f"    Cot {p+1} trung: {pos_match_counts[p]}/{total} = {pos_match_counts[p]/total*100:.1f}%")

    # === COMBINATION SEARCH SPACE ===
    print(f"\n{'='*80}")
    print(f"  KHONG GIAN TIM KIEM")
    print(f"{'='*80}")
    for top_n in [8, 10, 12, 15]:
        pools = [set(num for num, _ in pos_freq[p].most_common(top_n)) for p in range(pick)]
        space = 1
        for p in range(pick):
            space *= len(pools[p])
        # But need to account for ordering constraint (pos1 < pos2 < ... < pos6)
        # Rough estimate: space / pick! for ordering
        from math import factorial
        ordered = space  # already constrained by position
        print(f"  Top-{top_n:2d}/cot: {' x '.join(str(len(p)) for p in pools)} = {space:>12,} combos")

    return col_pools, recent_pools, pos_freq


if __name__ == '__main__':
    print("Loading data...")
    mega = get_mega645_numbers()
    print(f"Mega 6/45: {len(mega)} ky quay")
    col_pools, recent_pools, pos_freq = column_frequency_analysis(mega)
