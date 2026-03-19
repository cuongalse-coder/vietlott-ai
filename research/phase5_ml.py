"""
Phase 5: ML & Deep Learning for Lottery Prediction
====================================================
1. LSTM per-position sequence prediction
2. Lightweight Transformer attention
3. Meta-ensemble combining all signals
4. Backtest each approach vs V14 baseline
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from collections import Counter
import numpy as np
from scraper.data_manager import get_mega645_numbers


# ============================================================
# 1. LSTM Per-Position Prediction (using numpy, no torch needed)
# ============================================================

class SimpleLSTMCell:
    """Minimal LSTM cell using only numpy."""
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        scale = 0.1
        # Gates: input, forget, cell, output
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.ones(hidden_size) * 1.0  # forget bias = 1
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
    
    def forward(self, x, h_prev, c_prev):
        concat = np.concatenate([h_prev, x])
        f = self._sigmoid(self.Wf @ concat + self.bf)
        i = self._sigmoid(self.Wi @ concat + self.bi)
        c_cand = np.tanh(self.Wc @ concat + self.bc)
        o = self._sigmoid(self.Wo @ concat + self.bo)
        c = f * c_prev + i * c_cand
        h = o * np.tanh(c)
        return h, c
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


class PositionPredictor:
    """Predict next value at a specific sorted position using sequence analysis."""
    
    def __init__(self, max_num=45, window=20):
        self.max_num = max_num
        self.window = window
    
    def predict_position(self, history, pos, top_k=8):
        """Predict top-k most likely values for position `pos`."""
        # Extract position sequence
        seq = [sorted(d)[pos] for d in history if len(d) > pos]
        if len(seq) < 10:
            return list(range(1, top_k + 1))
        
        scores = Counter()
        
        # Method 1: Weighted recent frequency
        for i, val in enumerate(seq[-30:]):
            weight = (i + 1) / 30  # more recent = higher weight
            scores[val] += weight * 3.0
        
        # Method 2: Markov chain (1st order)
        transitions = Counter()
        for i in range(1, len(seq)):
            transitions[(seq[i-1], seq[i])] += 1
        last_val = seq[-1]
        for (prev, nxt), cnt in transitions.items():
            if prev == last_val:
                scores[nxt] += cnt * 2.0
        
        # Method 3: 2nd order Markov
        if len(seq) >= 3:
            trans2 = Counter()
            for i in range(2, len(seq)):
                trans2[(seq[i-2], seq[i-1], seq[i])] += 1
            last2 = (seq[-2], seq[-1])
            for (p2, p1, nxt), cnt in trans2.items():
                if (p2, p1) == last2:
                    scores[nxt] += cnt * 3.0
        
        # Method 4: Moving average prediction
        recent = seq[-10:]
        ma = np.mean(recent)
        for val in range(1, self.max_num + 1):
            if abs(val - ma) <= 3:
                scores[val] += 1.5
        
        # Method 5: Momentum (direction of change)
        if len(seq) >= 5:
            diffs = [seq[i] - seq[i-1] for i in range(-4, 0)]
            avg_diff = np.mean(diffs)
            predicted = int(round(seq[-1] + avg_diff))
            for val in range(max(1, predicted - 3), min(self.max_num + 1, predicted + 4)):
                scores[val] += 2.0
        
        # Method 6: LSTM-like recurrence (simplified)
        lstm = SimpleLSTMCell(1, 16)
        h = np.zeros(16)
        c = np.zeros(16)
        for val in seq[-self.window:]:
            x = np.array([val / self.max_num])
            h, c = lstm.forward(x, h, c)
        # Use hidden state to score
        proj = h[:self.max_num]  # project to number space
        for val in range(1, min(self.max_num + 1, len(proj) + 1)):
            scores[val] += proj[val - 1] * self.max_num * 0.5
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [v for v, _ in ranked[:top_k]]


def lstm_backtest(data, max_num=45, pick=6, test_count=100, dan_size=100):
    """Backtest LSTM-style per-position prediction."""
    print("\n" + "="*60)
    print(f"  LSTM PER-POSITION BACKTEST ({dan_size} dan)")
    print("="*60)
    
    n = len(data)
    start = max(60, n - test_count)
    predictor = PositionPredictor(max_num)
    
    any_3 = any_4 = any_5 = any_6 = 0
    total = 0
    t0 = time.time()
    
    for i in range(start, n - 1):
        history = data[:i + 1]
        actual = set(data[i + 1])
        total += 1
        
        # Get per-position predictions (top-k per position)
        pos_predictions = []
        for pos in range(pick):
            top_k = predictor.predict_position(history, pos, top_k=12)
            pos_predictions.append(top_k)
        
        # Generate dan by combining position predictions
        dan = []
        seen = set()
        for _ in range(dan_size * 3):
            combo = []
            used = set()
            for pos in range(pick):
                candidates = [n for n in pos_predictions[pos] if n not in used]
                if not candidates:
                    candidates = [n for n in range(1, max_num + 1) if n not in used]
                # Weighted selection (earlier in list = higher weight)
                weights = np.array([1.0 / (j + 1) + np.random.random() * 0.5
                                   for j, _ in enumerate(candidates)])
                weights = weights / weights.sum()
                chosen = int(np.random.choice(candidates, p=weights))
                combo.append(chosen)
                used.add(chosen)
            combo = sorted(combo)
            if tuple(combo) not in seen:
                dan.append(combo)
                seen.add(tuple(combo))
            if len(dan) >= dan_size:
                break
        
        matches = [len(set(d) & actual) for d in dan]
        best = max(matches) if matches else 0
        if best >= 3: any_3 += 1
        if best >= 4: any_4 += 1
        if best >= 5: any_5 += 1
        if best >= 6: any_6 += 1
        
        if total % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{total}/{n-start-1}] 5+={any_5} | 6/6={any_6} | {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    print(f"\n  LSTM RESULTS ({total} draws, {dan_size} dan, {elapsed:.0f}s):")
    print(f"  3+: {any_3/total*100:.1f}% | 4+: {any_4/total*100:.1f}% | 5+: {any_5/total*100:.1f}% | 6/6: {any_6/total*100:.1f}%")
    
    return {'any_3': any_3/total*100, 'any_4': any_4/total*100,
            'any_5': any_5/total*100, 'any_6': any_6/total*100}


# ============================================================
# 2. Transformer-style Attention (simplified, numpy only)
# ============================================================

class SimpleAttention:
    """Self-attention over draw sequences."""
    
    def __init__(self, d_model=32, n_heads=4):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = 0.1
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale
    
    def forward(self, X):
        """X: (seq_len, d_model) -> (seq_len, d_model)"""
        Q = X @ self.Wq.T
        K = X @ self.Wk.T
        V = X @ self.Wv.T
        
        scores = Q @ K.T / np.sqrt(self.d_k)
        # Softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        
        out = attn @ V
        return out @ self.Wo.T


def transformer_backtest(data, max_num=45, pick=6, test_count=100, dan_size=100):
    """Backtest transformer-style attention prediction."""
    print("\n" + "="*60)
    print(f"  TRANSFORMER ATTENTION BACKTEST ({dan_size} dan)")
    print("="*60)
    
    n = len(data)
    start = max(60, n - test_count)
    d_model = 32
    window = 20
    attn = SimpleAttention(d_model)
    
    any_3 = any_4 = any_5 = any_6 = 0
    total = 0
    t0 = time.time()
    
    for i in range(start, n - 1):
        history = data[:i + 1]
        actual = set(data[i + 1])
        total += 1
        
        # Encode last `window` draws as d_model vectors
        recent = history[-window:]
        X = np.zeros((len(recent), d_model))
        for j, draw in enumerate(recent):
            sd = sorted(draw)
            for k, val in enumerate(sd):
                if k < d_model // pick:
                    # One-hot-ish encoding
                    X[j, k * (d_model // pick)] = val / max_num
                X[j, pick + k] = val / max_num
            # Add positional encoding
            X[j, -2] = j / window
            X[j, -1] = 1.0 - j / window
        
        # Run attention
        out = attn.forward(X)
        
        # Use last output to generate predictions
        last_out = out[-1]
        
        # Convert to number scores
        num_scores = Counter()
        for num in range(1, max_num + 1):
            # Project output onto number
            idx = num % d_model
            score = last_out[idx] + last_out[(idx + 1) % d_model] * 0.5
            # Add frequency bias
            freq = sum(1 for d in recent if num in d) / len(recent)
            num_scores[num] = score + freq * 2
        
        # Per-position scoring
        pos_candidates = [[] for _ in range(pick)]
        for pos in range(pick):
            pos_vals = [sorted(d)[pos] for d in recent]
            pos_freq = Counter(pos_vals)
            for v, c in pos_freq.most_common(15):
                pos_candidates[pos].append((v, c + num_scores.get(v, 0)))
            pos_candidates[pos].sort(key=lambda x: -x[1])
        
        # Generate dan
        dan = []
        seen = set()
        for _ in range(dan_size * 3):
            combo = []
            used = set()
            for pos in range(pick):
                cands = [(v, s) for v, s in pos_candidates[pos] if v not in used]
                if not cands:
                    cands = [(v, 1) for v in range(1, max_num + 1) if v not in used]
                weights = np.array([s + np.random.random() * 2 for _, s in cands])
                weights = np.maximum(weights, 0.01)
                weights = weights / weights.sum()
                chosen = int(np.random.choice([v for v, _ in cands], p=weights))
                combo.append(chosen)
                used.add(chosen)
            combo = sorted(combo)
            if tuple(combo) not in seen:
                dan.append(combo)
                seen.add(tuple(combo))
            if len(dan) >= dan_size:
                break
        
        matches = [len(set(d) & actual) for d in dan]
        best = max(matches) if matches else 0
        if best >= 3: any_3 += 1
        if best >= 4: any_4 += 1
        if best >= 5: any_5 += 1
        if best >= 6: any_6 += 1
        
        if total % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{total}/{n-start-1}] 5+={any_5} | 6/6={any_6} | {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    print(f"\n  TRANSFORMER RESULTS ({total} draws, {dan_size} dan, {elapsed:.0f}s):")
    print(f"  3+: {any_3/total*100:.1f}% | 4+: {any_4/total*100:.1f}% | 5+: {any_5/total*100:.1f}% | 6/6: {any_6/total*100:.1f}%")
    
    return {'any_3': any_3/total*100, 'any_4': any_4/total*100,
            'any_5': any_5/total*100, 'any_6': any_6/total*100}


# ============================================================
# 3. Meta-Ensemble: combine V14 scores + LSTM + Transformer
# ============================================================

def meta_ensemble_backtest(data, max_num=45, pick=6, test_count=100, dan_size=500):
    """Combine all approaches into a meta-ensemble."""
    print("\n" + "="*60)
    print(f"  META-ENSEMBLE BACKTEST ({dan_size} dan)")
    print("="*60)
    
    n = len(data)
    start = max(60, n - test_count)
    
    from models.master_predictor import MasterPredictor
    
    pos_predictor = PositionPredictor(max_num)
    
    any_3 = any_4 = any_5 = any_6 = 0
    total = 0
    t0 = time.time()
    
    for i in range(start, n - 1):
        history = data[:i + 1]
        actual = set(data[i + 1])
        total += 1
        
        # === Source 1: V14 pool + scores ===
        predictor = MasterPredictor(max_num, pick)
        predictor.data = [d[:pick] for d in history]
        pool, v14_scores = predictor._get_pool(history)
        
        # === Source 2: LSTM position predictions ===
        lstm_preds = []
        for pos in range(pick):
            top_k = pos_predictor.predict_position(history, pos, top_k=10)
            lstm_preds.append(top_k)
        
        # === Source 3: Frequency momentum ===
        freq_5 = Counter(n for d in history[-5:] for n in d)
        freq_30 = Counter(n for d in history[-30:] for n in d)
        momentum = {}
        for num in range(1, max_num + 1):
            r = freq_5.get(num, 0) / 5
            o = freq_30.get(num, 0) / 30
            momentum[num] = r - o
        
        # === Combine scores ===
        combined = Counter()
        max_v14 = max(v14_scores.values()) if v14_scores else 1
        for num in range(1, max_num + 1):
            # V14 score (normalized)
            s = (v14_scores.get(num, 0) / max_v14) * 5.0
            # LSTM bonus
            for pos in range(pick):
                if num in lstm_preds[pos][:5]:
                    s += 2.0
                elif num in lstm_preds[pos]:
                    s += 0.5
            # Momentum
            s += max(-1, min(2, momentum.get(num, 0) * 5))
            # Pool membership
            if num in pool:
                s += 3.0
            combined[num] = s
        
        # Build meta-pool (top 35 by combined score)
        meta_pool = [num for num, _ in combined.most_common(35)]
        
        # Generate dan from meta-pool with position awareness
        pos_freq = [Counter() for _ in range(pick)]
        for d in history[-200:]:
            sd = sorted(d)
            for p in range(pick):
                pos_freq[p][sd[p]] += 1
        
        dan = []
        seen = set()
        for _ in range(dan_size * 3):
            combo = []
            used = set()
            for pos in range(pick):
                # Candidates from meta-pool, weighted by position frequency + combined score
                candidates = [n for n in meta_pool if n not in used]
                if not candidates:
                    candidates = [n for n in range(1, max_num + 1) if n not in used]
                weights = np.array([pos_freq[pos].get(c, 1) * 0.5 + combined.get(c, 0) * 0.3
                                   + np.random.random() * 2 for c in candidates])
                weights = np.maximum(weights, 0.01)
                weights = weights / weights.sum()
                chosen = int(np.random.choice(candidates, p=weights))
                combo.append(chosen)
                used.add(chosen)
            combo = sorted(combo)
            if tuple(combo) not in seen:
                dan.append(combo)
                seen.add(tuple(combo))
            if len(dan) >= dan_size:
                break
        
        matches = [len(set(d) & actual) for d in dan]
        best = max(matches) if matches else 0
        if best >= 3: any_3 += 1
        if best >= 4: any_4 += 1
        if best >= 5: any_5 += 1
        if best >= 6: any_6 += 1
        
        if total % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{total}/{n-start-1}] 5+={any_5} | 6/6={any_6} | {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    print(f"\n  META-ENSEMBLE RESULTS ({total} draws, {dan_size} dan, {elapsed:.0f}s):")
    print(f"  3+: {any_3/total*100:.1f}% | 4+: {any_4/total*100:.1f}% | 5+: {any_5/total*100:.1f}% | 6/6: {any_6/total*100:.1f}%")
    
    return {'any_3': any_3/total*100, 'any_4': any_4/total*100,
            'any_5': any_5/total*100, 'any_6': any_6/total*100}


# ============================================================
# MAIN: Run all experiments
# ============================================================

if __name__ == '__main__':
    print("Loading data...")
    mega = get_mega645_numbers()
    print(f"Mega 6/45: {len(mega)} draws")
    
    results = {}
    
    # 1. LSTM (100-dan, 100 draws)
    results['lstm_100'] = lstm_backtest(mega, dan_size=100, test_count=100)
    
    # 2. LSTM (500-dan, 100 draws)
    results['lstm_500'] = lstm_backtest(mega, dan_size=500, test_count=100)
    
    # 3. Transformer (100-dan, 100 draws)
    results['transformer_100'] = transformer_backtest(mega, dan_size=100, test_count=100)
    
    # 4. Meta-ensemble (500-dan, 100 draws)
    results['meta_500'] = meta_ensemble_backtest(mega, dan_size=500, test_count=100)
    
    # Final comparison
    print(f"\n{'#'*60}")
    print(f"  PHASE 5 ML COMPARISON")
    print(f"{'#'*60}")
    print(f"  {'Model':25s} | {'3+':>7s} | {'4+':>7s} | {'5+':>7s} | {'6/6':>7s}")
    print(f"  {'-'*25}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for name, r in results.items():
        print(f"  {name:25s} | {r['any_3']:6.1f}% | {r['any_4']:6.1f}% | {r['any_5']:6.1f}% | {r['any_6']:6.1f}%")
    print(f"  {'V14 baseline (5000 dan)':25s} | {'100.0':>6s}% | {'97.5':>6s}% | {'12.6':>6s}% | {'0.0':>6s}%")
    print(f"{'#'*60}")
