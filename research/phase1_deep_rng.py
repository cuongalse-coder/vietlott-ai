"""
Phase 1: Deep RNG Forensics
==============================
Advanced tests beyond basic chi-square/serial correlation.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'e:\\VietlottAI')

import numpy as np
from collections import Counter
from scipy import stats
from scraper.data_manager import get_mega645_numbers


def test_bit_level(data, max_num=45):
    """Test 1.1-1.2: Bit-level analysis."""
    print(f"\n{'='*60}")
    print(f"  TEST 1.1: Bit-Level Analysis")
    print(f"{'='*60}")
    
    bits_needed = int(np.ceil(np.log2(max_num + 1)))
    print(f"  Numbers 1-{max_num} need {bits_needed} bits each")
    
    # Convert all numbers to bit string
    bitstream = []
    for draw in data:
        for n in sorted(draw):
            bits = format(n, f'0{bits_needed}b')
            bitstream.extend([int(b) for b in bits])
    
    total_bits = len(bitstream)
    print(f"  Total bits: {total_bits}")
    
    # Monobit test
    ones = sum(bitstream)
    zeros = total_bits - ones
    s = abs(ones - zeros) / np.sqrt(total_bits)
    p_mono = 2 * (1 - stats.norm.cdf(s))
    mono_pass = p_mono > 0.01
    print(f"  Monobit: ones={ones}, zeros={zeros}, s={s:.3f}, p={p_mono:.4f} {'PASS' if mono_pass else 'FAIL <<<'}")
    
    # Block frequency (block=128)
    block_size = 128
    n_blocks = total_bits // block_size
    block_props = []
    for i in range(n_blocks):
        block = bitstream[i*block_size:(i+1)*block_size]
        block_props.append(sum(block) / block_size)
    chi2 = 4 * block_size * sum((p - 0.5)**2 for p in block_props)
    p_block = 1 - stats.chi2.cdf(chi2, n_blocks)
    block_pass = p_block > 0.01
    print(f"  Block freq: chi2={chi2:.2f}, p={p_block:.4f} {'PASS' if block_pass else 'FAIL <<<'}")
    
    # Runs test
    runs = 1
    for i in range(1, total_bits):
        if bitstream[i] != bitstream[i-1]:
            runs += 1
    pi = ones / total_bits
    expected_runs = 2 * total_bits * pi * (1-pi) + 1
    var_runs = 2 * total_bits * pi * (1-pi) * (2*pi*(1-pi) - 1/total_bits)
    if var_runs > 0:
        z_runs = (runs - expected_runs) / np.sqrt(var_runs)
        p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
    else:
        z_runs = 0
        p_runs = 1.0
    runs_pass = p_runs > 0.01
    print(f"  Runs: {runs} (expected {expected_runs:.0f}), z={z_runs:.3f}, p={p_runs:.4f} {'PASS' if runs_pass else 'FAIL <<<'}")
    
    # Per-bit-position analysis
    print(f"\n  Per-bit-position frequency (should be ~50% each):")
    for bit_pos in range(bits_needed):
        bit_vals = [bitstream[i*bits_needed + bit_pos] for i in range(total_bits // bits_needed)]
        ones_pct = sum(bit_vals) / len(bit_vals) * 100
        _, p = stats.binomtest(sum(bit_vals), len(bit_vals), 0.5).statistic, stats.binomtest(sum(bit_vals), len(bit_vals), 0.5).pvalue
        sig = "BIAS <<<" if p < 0.01 else ""
        print(f"    Bit {bit_pos}: {ones_pct:.1f}% ones (p={p:.4f}) {sig}")
    
    all_pass = mono_pass and block_pass and runs_pass
    return {'verdict': 'RANDOM' if all_pass else 'PATTERN', 'p_mono': p_mono, 'p_block': p_block, 'p_runs': p_runs}


def test_spectral(data):
    """Test 1.2: FFT spectral analysis per position."""
    print(f"\n{'='*60}")
    print(f"  TEST 1.2: Spectral (FFT) Analysis")
    print(f"{'='*60}")
    
    results = {}
    for pos in range(6):
        vals = np.array([sorted(d)[pos] for d in data], dtype=float)
        vals = vals - np.mean(vals)  # detrend
        
        fft = np.fft.fft(vals)
        power = np.abs(fft[1:len(fft)//2])**2  # skip DC
        
        # Find dominant frequency
        peak_idx = np.argmax(power) + 1
        peak_freq = peak_idx / len(vals)
        peak_period = len(vals) / peak_idx if peak_idx > 0 else float('inf')
        peak_power = power[peak_idx-1]
        
        # Compare peak with average power (should be ~uniform for random)
        avg_power = np.mean(power)
        ratio = peak_power / avg_power
        
        # Threshold: for random, peak should be < 3x average (roughly)
        is_sig = ratio > 4.0
        sig = "PEAK <<<" if is_sig else ""
        print(f"  Pos{pos+1}: peak_period={peak_period:.1f} draws, power_ratio={ratio:.2f}x avg {sig}")
        results[f'pos{pos+1}'] = {'period': peak_period, 'ratio': ratio, 'significant': is_sig}
    
    any_sig = any(r['significant'] for r in results.values())
    verdict = 'PATTERN' if any_sig else 'RANDOM'
    print(f"  VERDICT: {verdict}")
    return {'verdict': verdict, 'results': results}


def test_birthday_spacing(data, max_num=45):
    """Test 1.3: Birthday spacings test."""
    print(f"\n{'='*60}")
    print(f"  TEST 1.3: Birthday Spacings Test")
    print(f"{'='*60}")
    
    results = {}
    for pos in range(6):
        vals = [sorted(d)[pos] for d in data]
        
        # Compute spacings between consecutive occurrences of each value
        last_seen = {}
        spacings = []
        for i, v in enumerate(vals):
            if v in last_seen:
                spacings.append(i - last_seen[v])
            last_seen[v] = i
        
        if len(spacings) < 30:
            continue
        
        # For random: spacings should follow geometric distribution
        # with p = freq/total for each value
        avg_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)
        
        # Kolmogorov-Smirnov test against exponential
        _, p_ks = stats.kstest(spacings, 'expon', args=(0, avg_spacing))
        sig = "ANOMALY <<<" if p_ks < 0.01 else ""
        print(f"  Pos{pos+1}: avg_spacing={avg_spacing:.1f}, std={std_spacing:.1f}, KS p={p_ks:.4f} {sig}")
        results[f'pos{pos+1}'] = {'p_ks': p_ks}
    
    any_sig = any(r['p_ks'] < 0.01 for r in results.values())
    verdict = 'PATTERN' if any_sig else 'RANDOM'
    print(f"  VERDICT: {verdict}")
    return {'verdict': verdict, 'results': results}


def test_pair_spacing(data):
    """Test 1.4: Pair spacing analysis."""
    print(f"\n{'='*60}")
    print(f"  TEST 1.4: Pair Spacing Test")
    print(f"{'='*60}")
    
    # Track when each pair appears
    pair_positions = {}
    for i, draw in enumerate(data):
        sd = sorted(draw)
        for a in range(len(sd)):
            for b in range(a+1, len(sd)):
                pair = (sd[a], sd[b])
                if pair not in pair_positions:
                    pair_positions[pair] = []
                pair_positions[pair].append(i)
    
    # Compute spacings for frequently occurring pairs
    all_spacings = []
    anomalous_pairs = []
    for pair, positions in pair_positions.items():
        if len(positions) < 5:
            continue
        spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        all_spacings.extend(spacings)
        
        avg = np.mean(spacings)
        std = np.std(spacings)
        expected_avg = len(data) / len(positions)
        
        # Z-test: is this pair's spacing different from expected?
        if std > 0:
            z = (avg - expected_avg) / (std / np.sqrt(len(spacings)))
            if abs(z) > 3:
                anomalous_pairs.append((pair, avg, expected_avg, z))
    
    # Overall distribution
    if all_spacings:
        _, p_ks = stats.kstest(all_spacings, 'expon', args=(0, np.mean(all_spacings)))
        print(f"  Total pair spacings: {len(all_spacings)}")
        print(f"  Overall KS test: p={p_ks:.4f}")
        print(f"  Anomalous pairs (|z|>3): {len(anomalous_pairs)}")
        for pair, avg, exp, z in anomalous_pairs[:5]:
            print(f"    Pair {pair}: avg_spacing={avg:.1f} vs expected={exp:.1f}, z={z:.2f}")
    else:
        p_ks = 1.0
    
    verdict = 'PATTERN' if len(anomalous_pairs) > len(pair_positions)*0.01 else 'RANDOM'
    print(f"  VERDICT: {verdict}")
    return {'verdict': verdict, 'anomalous': len(anomalous_pairs), 'p_ks': p_ks}


if __name__ == '__main__':
    data = get_mega645_numbers()
    print(f"Mega 6/45: {len(data)} draws")
    
    results = {}
    results['bit_level'] = test_bit_level(data)
    results['spectral'] = test_spectral(data)
    results['birthday'] = test_birthday_spacing(data)
    results['pair_spacing'] = test_pair_spacing(data)
    
    print(f"\n{'#'*60}")
    print(f"  PHASE 1 SUMMARY: Deep RNG Forensics")
    print(f"{'#'*60}")
    patterns = sum(1 for r in results.values() if r.get('verdict') == 'PATTERN')
    for name, r in results.items():
        v = r.get('verdict', 'N/A')
        marker = " <<<< EXPLOITABLE!" if v == 'PATTERN' else ""
        print(f"  {name:15s}: {v}{marker}")
    print(f"\n  Weaknesses found: {patterns}/{len(results)}")
    print(f"{'#'*60}")
