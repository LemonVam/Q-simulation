import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import os
import cirq
from collections import Counter
from functools import lru_cache
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ensures 3D projection is registered)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.svm import LinearSVC
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# ---- Matplotlib aesthetics (bigger fonts, crisper figures) ----
plt.rcParams.update({
    'figure.figsize': (8.5, 6.5),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.dpi': 220,
    'figure.dpi': 120,
    'axes.grid': True,
    'grid.alpha': 0.25,
})

# ---- CPU-friendly hyperparameters ----
EMBEDDING_DIM = 4           # keep small for speed
MAXLEN = 256                # shorter sequences speed up CPU training
EPOCHS = 8                  # fewer epochs for quicker runs
BATCH_SIZE = 256            # larger batch improves throughput on CPU
CIRCUIT_DEPTH = 1           # shallow circuit
TOP_K_WORDS = 2000          # tighter vocabulary for faster Bloch sampling
ROUND_KEY = 4               # round vectors for cache key stability
QUANTUM_DIM = 3 * EMBEDDING_DIM  # Bloch sphere features: <X>, <Y>, <Z> per qubit
ANGLE_SCALE = np.pi / 2     # scale tanh-normalized values into [-pi/2, pi/2]
REPETITIONS_TOP = 192       # precision/speed trade-off per request
REPETITIONS_OTHER = 64      # (kept for future per-word adaptive use)
ENTANGLER = 'CZ_RING'       # ring entangler for more uniform propagation

# ---- weighting knobs for review-level Bloch aggregation ----
BLOCH_WEIGHT_BETA = 0.5     # bonus from Bloch vector magnitude ||xyz||
EMO_BONUS_GAMMA   = 0.5     # bonus if word appears in emotion lexicon
AXIS_P_LO, AXIS_P_HI = 2, 98  # percentile bounds for tight axes
PROTOTYPE_ARROW_LENGTH = 0.8  # shorter unit length for prototype arrows
QUIVER_LENGTH_SCALE    = 0.8  # shorter quiver arrows in 3D plots
INCLUDE_P1090 = True  # if True, extend stats from 18D to 24D by adding p10/p90 per axis
# ---- SVM/XGB tuning knobs ----
SVM_C = 1.0
SVM_MAX_ITER = 10000
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 4
XGB_LR = 0.05
# --------------------------------------

# ---- Emotion seeds & lexicon (defined early so helpers can reference) ----
EMOTION_SEEDS = {
    'joy':        ['good', 'great', 'love', 'happy', 'enjoy', 'fun'],
    'anger':      ['angry', 'hate', 'furious', 'rage', 'annoy'],
    'sadness':    ['sad', 'cry', 'tragic', 'lonely', 'depress'],
    'fear':       ['fear', 'scary', 'terrify', 'horror', 'afraid'],
    'surprise':   ['surprise', 'shocking', 'unexpected', 'twist'],
    'disgust':    ['disgust', 'gross', 'nasty', 'offend', 'vile'],
    'trust':      ['trust', 'reliable', 'honest', 'loyal'],
    'anticipation':['anticipate', 'expect', 'await', 'predict']
}
EMO_LEXICON = set([w for ws in EMOTION_SEEDS.values() for w in ws])
# -------------------------------------------------------------------------

@lru_cache(maxsize=20000)
def quantum_feature_extraction_cached(word_tuple):
    """Compute Bloch-sphere features for a word vector (tuple) using sampling.
    For each qubit, we estimate <X>, <Y>, <Z> via basis-rotation measurements:
      <Z> from Z-basis; <X> by H then measure; <Y> by S^† then H then measure.
    Returns a flat array of shape (3 * num_qubits,) in the order [X1, Y1, Z1, X2, Y2, Z2, ...].
    """
    word_vector = np.array(word_tuple, dtype=np.float32)
    num_qubits = len(word_vector)
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]

    # Build encoding + entangling circuit up to (but not including) measurement
    base = cirq.Circuit()
    # Angle Encoding
    for i, value in enumerate(word_vector):
        theta = ANGLE_SCALE * np.tanh(value)
        base.append(cirq.ry(theta).on(qubits[i]))
    # Entangling + simple rx layer
    def entangle_layer(layer_idx: int):
        if ENTANGLER == 'CZ_RING':
            # Brickwork on a ring: even-odd then odd-even pairs to spread entanglement uniformly
            # Sublayer A: (0,1), (2,3), ... with wrap-around
            for i in range(0, num_qubits, 2):
                q1 = qubits[i]
                q2 = qubits[(i+1) % num_qubits]
                base.append(cirq.CZ(q1, q2))
            # Sublayer B: (1,2), (3,4), ... with wrap-around
            for i in range(1, num_qubits, 2):
                q1 = qubits[i]
                q2 = qubits[(i+1) % num_qubits]
                base.append(cirq.CZ(q1, q2))
        else:
            # Default: linear CNOT chain (no wrap)
            for i in range(num_qubits - 1):
                base.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        # Simple mixer to avoid symmetries; depth-dependent angle
        for i in range(num_qubits):
            base.append(cirq.rx(np.pi/(layer_idx+1)).on(qubits[i]))

    for d in range(CIRCUIT_DEPTH):
        entangle_layer(d)

    simulator = cirq.Simulator()

    # Helper to run with a given pre-measurement rotation and return <Z> estimates
    def measure_expectation(pre_ops):
        c = base.copy()
        for op in pre_ops:
            c.append(op)
        c.append(cirq.measure(*qubits, key='m'))
        result = simulator.run(c, repetitions=REPETITIONS_TOP)
        bits = result.measurements['m']  # shape (reps, num_qubits)
        # Expectation of Z from bit outcomes: <Z> = 1 - 2*P(bit=1)
        p1 = bits.mean(axis=0)
        return 1.0 - 2.0 * p1

    # <Z>: no rotation
    expZ = measure_expectation(pre_ops=[])
    # <X>: apply H to each qubit
    expX = measure_expectation(pre_ops=[cirq.H.on(q) for q in qubits])
    # <Y>: apply S^† then H to each qubit
    expY = measure_expectation(pre_ops=[cirq.S.on(q)**-1 for q in qubits] + [cirq.H.on(q) for q in qubits])

    # Interleave per qubit as [X1, Y1, Z1, X2, Y2, Z2, ...]
    bloch = np.empty(3 * num_qubits, dtype=np.float32)
    for i in range(num_qubits):
        bloch[3*i + 0] = expX[i]
        bloch[3*i + 1] = expY[i]
        bloch[3*i + 2] = expZ[i]
    return bloch

# 融合 Word2Vec 向量和量子特征
def combine_classical_quantum_features(word_vector):
    key = tuple(np.round(np.asarray(word_vector, dtype=np.float32), ROUND_KEY))
    quantum_features = quantum_feature_extraction_cached(key)
    combined = np.concatenate([np.asarray(word_vector, dtype=np.float32), quantum_features])
    return combined

# ---- Bloch aggregation helpers (review-level) ----

def bloch_mean_xyz(bloch_vec):
    """Given a flat bloch vector [X1,Y1,Z1,...], return per-axis means (Xbar,Ybar,Zbar)."""
    n = len(bloch_vec) // 3
    if n == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x = np.array([bloch_vec[3*i + 0] for i in range(n)], dtype=np.float32)
    y = np.array([bloch_vec[3*i + 1] for i in range(n)], dtype=np.float32)
    z = np.array([bloch_vec[3*i + 2] for i in range(n)], dtype=np.float32)
    return np.array([x.mean(), y.mean(), z.mean()], dtype=np.float32)


def review_bloch_xyz_means(review_ids):
    """Average (weighted) Bloch features over tokens in one review (only words present in Word2Vec).
    Returns (Xbar, Ybar, Zbar). Uses cached per-word Bloch features.
    """
    words = [reverse_word_index.get(i - 3, '?') for i in review_ids]
    tf = Counter(words)
    weighted = np.zeros(3, dtype=np.float32)
    weight_sum = 0.0
    for w, cnt in tf.items():
        if w not in word2vec_model.wv:
            continue
        vec = word2vec_model.wv[w]
        key = tuple(np.round(np.asarray(vec, dtype=np.float32), ROUND_KEY))
        bloch = quantum_feature_extraction_cached(key)
        n = len(bloch)//3
        X = np.array([bloch[3*i+0] for i in range(n)], dtype=np.float32)
        Y = np.array([bloch[3*i+1] for i in range(n)], dtype=np.float32)
        Z = np.array([bloch[3*i+2] for i in range(n)], dtype=np.float32)
        xyz = np.array([X.mean(), Y.mean(), Z.mean()], dtype=np.float32)
        # weights: TF-IDF * (1 + beta * ||xyz||) * (1 + gamma * 1_{word in emotion lexicon})
        idf = IDF.get(w, 1.0)
        bloch_bonus = 1.0 + BLOCH_WEIGHT_BETA * float(np.linalg.norm(xyz))
        emo_bonus = 1.0 + (EMO_BONUS_GAMMA if w in EMO_LEXICON else 0.0)
        wgt = float(cnt) * float(idf) * bloch_bonus * emo_bonus
        weighted += xyz * wgt
        weight_sum += wgt
    if weight_sum == 0.0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return weighted / weight_sum

def review_bloch_xyz_stats(review_ids):
    """Return weighted Bloch stats per review.
    Base 18D: [Xbar,Ybar,Zbar,  Xstd,Ystd,Zstd,  Xskew,Yskew,Zskew,  Xkurt,Ykurt,Zkurt,  corrXY,corrXZ,corrYZ].
    If INCLUDE_P1090=True, append p10/p90 per axis to make 24D (total = 18 + 6).

    All statistics are computed **weighted over tokens** with weight:
        w = TF(w)*IDF(w) * (1 + BLOCH_WEIGHT_BETA*||xyz||) * (1 + EMO_BONUS_GAMMA*1_{w in EMO_LEXICON}).
    Here xyz is the per-token Bloch axis means (X̄_token, Ȳ_token, Z̄_token).
    """
    # 1) Gather per-token xyz means and weights
    words = [reverse_word_index.get(i - 3, '?') for i in review_ids]
    tf = Counter(words)
    xs, ys, zs, ws = [], [], [], []
    for w, cnt in tf.items():
        if w not in word2vec_model.wv:
            continue
        vec = word2vec_model.wv[w]
        key = tuple(np.round(np.asarray(vec, dtype=np.float32), ROUND_KEY))
        bloch = quantum_feature_extraction_cached(key)
        n = len(bloch)//3
        X = np.array([bloch[3*i+0] for i in range(n)], dtype=np.float32)
        Y = np.array([bloch[3*i+1] for i in range(n)], dtype=np.float32)
        Z = np.array([bloch[3*i+2] for i in range(n)], dtype=np.float32)
        xyz = np.array([X.mean(), Y.mean(), Z.mean()], dtype=np.float32)
        # token weight
        idf = IDF.get(w, 1.0)
        bloch_bonus = 1.0 + BLOCH_WEIGHT_BETA * float(np.linalg.norm(xyz))
        emo_bonus   = 1.0 + (EMO_BONUS_GAMMA if w in EMO_LEXICON else 0.0)
        wt = float(cnt) * float(idf) * bloch_bonus * emo_bonus
        xs.append(xyz[0]); ys.append(xyz[1]); zs.append(xyz[2]); ws.append(wt)

    if not ws:
        base18 = np.zeros(18, dtype=np.float32)
        if INCLUDE_P1090:
            return np.concatenate([base18, np.zeros(6, dtype=np.float32)])
        return base18

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    ws = np.asarray(ws, dtype=np.float64)

    # 2) Weighted means
    def wmean(a, w):
        return float((a*w).sum() / (w.sum() + 1e-12))
    mx = wmean(xs, ws); my = wmean(ys, ws); mz = wmean(zs, ws)

    # 3) Weighted std (population) and standardized moments
    def wstd(a, w, mu):
        v = (w * (a - mu)**2).sum() / (w.sum() + 1e-12)
        return float(np.sqrt(max(v, 0.0)))
    sx = wstd(xs, ws, mx); sy = wstd(ys, ws, my); sz = wstd(zs, ws, mz)

    def wskew(a, w, mu, s):
        if s <= 1e-12:
            return 0.0
        z = (a - mu) / s
        return float((w * z**3).sum() / (w.sum() + 1e-12))
    def wkurt(a, w, mu, s):
        if s <= 1e-12:
            return 0.0
        z = (a - mu) / s
        return float((w * z**4).sum() / (w.sum() + 1e-12))  # non-Fisher
    skx = wskew(xs, ws, mx, sx); sky = wskew(ys, ws, my, sy); skz = wskew(zs, ws, mz, sz)
    kx  = wkurt(xs, ws, mx, sx); ky  = wkurt(ys, ws, my, sy); kz  = wkurt(zs, ws, mz, sz)

    # 4) Weighted correlations
    def wcov(a, b, w, ma, mb):
        return float((w * (a - ma) * (b - mb)).sum() / (w.sum() + 1e-12))
    def wcorr(a, b, w, ma, mb, sa, sb):
        denom = (sa*sb) if (sa>1e-12 and sb>1e-12) else 0.0
        return float(wcov(a,b,w,ma,mb)/(denom + 1e-12)) if denom>0 else 0.0
    cxy = wcorr(xs, ys, ws, mx, my, sx, sy)
    cxz = wcorr(xs, zs, ws, mx, mz, sx, sz)
    cyz = wcorr(ys, zs, ws, my, mz, sy, sz)

    base18 = np.array([mx, my, mz, sx, sy, sz, skx, sky, skz, kx, ky, kz, cxy, cxz, cyz], dtype=np.float32)

    if not INCLUDE_P1090:
        return base18

    # 5) Weighted p10/p90 (adds 6 dims -> 24D)
    def wpercentile(a, w, q):
        order = np.argsort(a)
        a_sorted = a[order]
        w_sorted = w[order]
        cw = np.cumsum(w_sorted)
        t = q * cw[-1]
        idx = np.searchsorted(cw, t, side='left')
        idx = np.clip(idx, 0, len(a_sorted)-1)
        return float(a_sorted[idx])

    p10x = wpercentile(xs, ws, 0.10); p90x = wpercentile(xs, ws, 0.90)
    p10y = wpercentile(ys, ws, 0.10); p90y = wpercentile(ys, ws, 0.90)
    p10z = wpercentile(zs, ws, 0.10); p90z = wpercentile(zs, ws, 0.90)
    extra6 = np.array([p10x, p90x, p10y, p90y, p10z, p90z], dtype=np.float32)

    return np.concatenate([base18, extra6])
# -----------------------------------------------

# Helper: names for Bloch stats features (must match review_bloch_xyz_stats)
def bloch_feature_names():
    base = ['mx','my','mz','sx','sy','sz','skx','sky','skz','kx','ky','kz','cxy','cxz','cyz']
    if INCLUDE_P1090:
        base += ['p10x','p90x','p10y','p90y','p10z','p90z']
    return base

# 1. 加载和准备 IMDB 数据集
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(text_ids):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_ids])

x_train_words = [decode_review(text_ids).split() for text_ids in x_train]
x_test_words  = [decode_review(text_ids).split()  for text_ids in x_test]

# Select most frequent words to receive expensive quantum features
word_freq = Counter(w for review in x_train_words for w in review)
most_common_words = set(w for w, _ in word_freq.most_common(TOP_K_WORDS))

# Build document frequencies and IDF for TF-IDF weighting
N_DOCS = len(x_train_words)
doc_freq = Counter()
for review in x_train_words:
    doc_freq.update(set(review))  # unique words per document
IDF = {w: np.log((N_DOCS + 1) / (df + 1)) + 1.0 for w, df in doc_freq.items()}

# 2. Word2Vec 训练
embedding_dim = EMBEDDING_DIM
word2vec_model = Word2Vec(sentences=x_train_words, vector_size=embedding_dim, window=5, min_count=1, workers=4)

# 3. 构建嵌入矩阵 (对高频词加入量子特征，其余只用经典向量并用0填充量子部分)
if __name__ == '__main__':
    embedding_matrix = np.zeros((vocab_size, embedding_dim + QUANTUM_DIM), dtype=np.float32)
    embedding_matrix_classic = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    wv = word2vec_model.wv

    def build_row(args):
        word, i = args
        if i >= vocab_size:
            return None
        if word not in wv:
            return None
        vec = wv[word]
        classical_vec = vec.astype(np.float32)
        if word in most_common_words:
            combined = combine_classical_quantum_features(vec)
        else:
            combined = np.concatenate([classical_vec, np.zeros(QUANTUM_DIM, dtype=np.float32)])
        return (i, combined)

    with ThreadPool(processes=max(1, cpu_count() - 1)) as pool:
        for res in pool.imap_unordered(build_row, word_index.items(), chunksize=256):
            if res is None:
                continue
            idx, comb = res
            embedding_matrix[idx] = comb

    # 填充经典-only 嵌入矩阵
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        if word in wv:
            embedding_matrix_classic[i] = wv[word].astype(np.float32)

    # 4. 构建两套模型：经典-only vs 经典+量子（CPU友好设置）
    def build_model(output_dim):
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=MAXLEN, trainable=False),
            LSTM(64, dropout=0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # 5. 处理输入数据
    x_train_padded = pad_sequences(x_train, maxlen=MAXLEN)
    x_test_padded  = pad_sequences(x_test,  maxlen=MAXLEN)

    # 6. 经典-only 训练与评估
    model_classic = build_model(embedding_dim)
    model_classic.layers[0].build((None,))
    model_classic.layers[0].set_weights([embedding_matrix_classic])
    history_classic = model_classic.fit(
        x_train_padded, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test_padded, y_test),
        verbose=1
    )
    loss_classic, acc_classic = model_classic.evaluate(x_test_padded, y_test, verbose=0)

    # 7. 经典+量子 训练与评估
    model_quantum = build_model(embedding_dim + QUANTUM_DIM)
    model_quantum.layers[0].build((None,))
    model_quantum.layers[0].set_weights([embedding_matrix])
    history_quantum = model_quantum.fit(
        x_train_padded, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test_padded, y_test),
        verbose=1
    )
    loss_quantum, acc_quantum = model_quantum.evaluate(x_test_padded, y_test, verbose=0)

    print(f"Classic Test Acc: {acc_classic:.4f}, Loss: {loss_classic:.4f}")
    print(f"Quantum Test Acc: {acc_quantum:.4f}, Loss: {loss_quantum:.4f}")

    # 8. 绘制训练过程对比图（accuracy 与 loss）
    # Accuracy 曲线
    plt.figure()
    plt.plot(history_classic.history['accuracy'], label='Classic Train Acc')
    plt.plot(history_classic.history['val_accuracy'], label='Classic Val Acc')
    plt.plot(history_quantum.history['accuracy'], label='Quantum Train Acc')
    plt.plot(history_quantum.history['val_accuracy'], label='Quantum Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training/Validation Accuracy: Classic vs Quantum')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_accuracy.png', dpi=150)

    # Loss 曲线
    plt.figure()
    plt.plot(history_classic.history['loss'], label='Classic Train Loss')
    plt.plot(history_classic.history['val_loss'], label='Classic Val Loss')
    plt.plot(history_quantum.history['loss'], label='Quantum Train Loss')
    plt.plot(history_quantum.history['val_loss'], label='Quantum Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss: Classic vs Quantum')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_loss.png', dpi=150)

    # 9. 测试集准确率柱状对比
    plt.figure()
    methods = ['Classic', 'Quantum']
    accs = [acc_classic, acc_quantum]
    plt.bar(methods, accs)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Comparison')
    for i, v in enumerate(accs):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig('test_accuracy_bar.png', dpi=150)

    print('Saved figures: comparison_accuracy.png, comparison_loss.png, test_accuracy_bar.png')

    # 10. 基于 Bloch 的情感可视化：按情感类别绘制 (Xbar, Zbar) 分布 & 轴向小提琴图
    def sample_indices_by_label(y, label, k):
        idx = np.where(y == label)[0]
        if len(idx) == 0:
            return np.array([], dtype=int)
        if len(idx) <= k:
            return idx
        rng = np.random.default_rng(42)
        return rng.choice(idx, size=k, replace=False)

    k_per_class = 800  # 可按需要调整（更密集，更稳定）
    pos_idx = sample_indices_by_label(y_test, 1, k_per_class)
    neg_idx = sample_indices_by_label(y_test, 0, k_per_class)

    Xbars, Ybars, Zbars, labels = [], [], [], []
    for i in pos_idx:
        xb, yb, zb = review_bloch_xyz_means(x_test[i])
        Xbars.append(xb); Ybars.append(yb); Zbars.append(zb); labels.append(1)
    for i in neg_idx:
        xb, yb, zb = review_bloch_xyz_means(x_test[i])
        Xbars.append(xb); Ybars.append(yb); Zbars.append(zb); labels.append(0)

    Xbars = np.array(Xbars); Ybars = np.array(Ybars); Zbars = np.array(Zbars); labels = np.array(labels)

    # 10.1 2D 散点：Xbar vs Zbar，颜色表示情感
    plt.figure()
    plt.scatter(Xbars[labels==0], Zbars[labels==0], alpha=0.45, s=10, linewidths=0.2, edgecolors='none', label='Negative')
    plt.scatter(Xbars[labels==1], Zbars[labels==1], alpha=0.45, s=10, linewidths=0.2, edgecolors='none', label='Positive')
    plt.xlabel('<X> mean per review')
    plt.ylabel('<Z> mean per review')
    plt.title('Bloch Means (X vs Z) — Test Subset')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('bloch_scatter_xz.png')

    # Density view: hexbin per class (side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    hb0 = axes[0].hexbin(Xbars[labels==0], Zbars[labels==0], gridsize=35, mincnt=1)
    axes[0].set_title('Negative density (hexbin)')
    axes[0].set_xlabel('<X> mean'); axes[0].set_ylabel('<Z> mean')
    fig.colorbar(hb0, ax=axes[0], fraction=0.046, pad=0.04)
    hb1 = axes[1].hexbin(Xbars[labels==1], Zbars[labels==1], gridsize=35, mincnt=1)
    axes[1].set_title('Positive density (hexbin)')
    axes[1].set_xlabel('<X> mean')
    fig.colorbar(hb1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('bloch_xz_density.png')

    # 10.2 小提琴图：正负样本的 Xbar/Ybar/Zbar 分布
    def violin_by_label(values, lbls, title, fname, ylabel):
        data = [values[lbls==0], values[lbls==1]]
        plt.figure()
        plt.violinplot(data, showmeans=True, showmedians=False)
        plt.xticks([1, 2], ['Negative', 'Positive'])
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)

    violin_by_label(Xbars, labels, 'Distribution of <X> Means by Sentiment', 'bloch_violin_x.png', '<X> mean')
    violin_by_label(Ybars, labels, 'Distribution of <Y> Means by Sentiment', 'bloch_violin_y.png', '<Y> mean')
    violin_by_label(Zbars, labels, 'Distribution of <Z> Means by Sentiment', 'bloch_violin_z.png', '<Z> mean')

    # 10.3 打印每轴的类间差异（Cohen's d）
    def cohens_d(a, b):
        a = np.asarray(a); b = np.asarray(b)
        na, nb = len(a), len(b)
        va, vb = a.var(ddof=1), b.var(ddof=1)
        sp = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2)) if (na+nb-2) > 0 else 0.0
        return (a.mean() - b.mean()) / sp if sp > 0 else 0.0

    d_x = cohens_d(Xbars[labels==1], Xbars[labels==0])
    d_y = cohens_d(Ybars[labels==1], Ybars[labels==0])
    d_z = cohens_d(Zbars[labels==1], Zbars[labels==0])
    print(f"Cohen's d (Positive - Negative): d_X={d_x:.3f}, d_Y={d_y:.3f}, d_Z={d_z:.3f}")

    print('Saved Bloch visualizations: bloch_scatter_xz.png, bloch_violin_x.png, bloch_violin_y.png, bloch_violin_z.png')

    # 11. 三维 Bloch 分布图：以 (Xbar, Ybar, Zbar) 为坐标，按情感上色，并叠加单位球
    from math import pi
    import numpy as _np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Colored scatter with slight jitter to reduce overplot
    jitter = 0.002
    Xj = Xbars + np.random.uniform(-jitter, jitter, size=Xbars.shape)
    Yj = Ybars + np.random.uniform(-jitter, jitter, size=Ybars.shape)
    Zj = Zbars + np.random.uniform(-jitter, jitter, size=Zbars.shape)
    ax.scatter(Xj[labels==0], Yj[labels==0], Zj[labels==0], alpha=0.45, s=8, label='Negative')
    ax.scatter(Xj[labels==1], Yj[labels==1], Zj[labels==1], alpha=0.45, s=8, label='Positive')

    # Add a translucent unit sphere surface for depth cue
    u_s = _np.linspace(0, 2*pi, 64)
    v_s = _np.linspace(0, pi, 32)
    xs_s = _np.outer(_np.cos(u_s), _np.sin(v_s))
    ys_s = _np.outer(_np.sin(u_s), _np.sin(v_s))
    zs_s = _np.outer(_np.ones_like(u_s), _np.cos(v_s))
    ax.plot_surface(xs_s, ys_s, zs_s, alpha=0.06, linewidth=0, shade=True)

    ax.set_xlabel('<X> mean per review')
    ax.set_ylabel('<Y> mean per review')
    ax.set_zlabel('<Z> mean per review')
    ax.set_title('3D Bloch Means per Review (Test Subset)')

    # Tighten axis ranges by percentile to reduce unit length and improve visual contrast
    def tight_bounds(a):
        lo = np.percentile(a, AXIS_P_LO)
        hi = np.percentile(a, AXIS_P_HI)
        m = max(abs(lo), abs(hi))
        return -m, m
    xlo,xhi = tight_bounds(Xbars)
    ylo,yhi = tight_bounds(Ybars)
    zlo,zhi = tight_bounds(Zbars)
    ax.set_xlim([xlo,xhi]); ax.set_ylim([ylo,yhi]); ax.set_zlim([zlo,zhi])

    try:
        ax.set_box_aspect((1,1,1))  # requires newer matplotlib
    except Exception:
        pass

    # ax.legend()
    plt.tight_layout()
    plt.savefig('bloch_scatter_xyz_3d.png', dpi=180)
    print('Saved 3D Bloch figure: bloch_scatter_xyz_3d.png')

    # 12. Bloch-only 线性分类器 + ROC/PR + 3D 决策边界
    # 使用加权统计特征 review_bloch_xyz_stats（18D/24D）→ LDA(1D) + PCA(2D) → 3D 可视化空间
    feats = []
    lbls = []
    for i in pos_idx:
        feats.append(review_bloch_xyz_stats(x_test[i])); lbls.append(1)
    for i in neg_idx:
        feats.append(review_bloch_xyz_stats(x_test[i])); lbls.append(0)
    X_feat = np.vstack(feats).astype(np.float32)
    y_lbl = np.array(lbls, dtype=np.int32)

    # 训练/验证划分
    X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y_lbl, test_size=0.35, random_state=123, stratify=y_lbl)

    # 标准化
    scaler = StandardScaler().fit(X_tr)
    X_trs = scaler.transform(X_tr)
    X_tes = scaler.transform(X_te)

    # 12.a 线性 SVM（直接用加权 18D/24D 特征）
    svm = LinearSVC(C=SVM_C, class_weight='balanced', max_iter=SVM_MAX_ITER)
    svm.fit(X_trs, y_tr)
    svm_scores = svm.decision_function(X_tes)
    fpr_svm, tpr_svm, _ = roc_curve(y_te, svm_scores)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    prec_svm, rec_svm, _ = precision_recall_curve(y_te, svm_scores)
    ap_svm = average_precision_score(y_te, svm_scores)
    plt.figure(); plt.plot(fpr_svm, tpr_svm, label=f'Linear SVM (AUC={roc_auc_svm:.3f})'); plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC — Linear SVM on Bloch Stats'); plt.legend(); plt.tight_layout(); plt.savefig('svm_linear_roc.png', dpi=150)
    plt.figure(); plt.plot(rec_svm, prec_svm, label=f'Linear SVM (AP={ap_svm:.3f})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR — Linear SVM on Bloch Stats'); plt.legend(); plt.tight_layout(); plt.savefig('svm_linear_pr.png', dpi=150)
    print(f'Linear SVM on Bloch stats: AUC={roc_auc_svm:.3f}, AP={ap_svm:.3f}')
    # Visualize SVM coefficients (importance by |weight|)
    feat_names = bloch_feature_names()
    svm_w = svm.coef_.ravel()
    order = np.argsort(np.abs(svm_w))[::-1]
    topk = min(len(order), 15)
    sel = order[:topk]
    plt.figure(figsize=(8, 5))
    plt.barh([feat_names[i] if i < len(feat_names) else f'f{i}' for i in sel][::-1], np.abs(svm_w[sel])[::-1])
    plt.xlabel('|weight|'); plt.title('Linear SVM Coefficient Importance (top)')
    plt.tight_layout(); plt.savefig('svm_coef_importance.png', dpi=180)

    # 12.b XGBoost（可选）
    if _HAS_XGB:
        # 合理的缺省参数；按需要可再调优
        pos_w = (y_tr==0).sum() / max(1,(y_tr==1).sum())
        xgb = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LR,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=max(1, cpu_count()-1),
            scale_pos_weight=pos_w,
            tree_method='hist',
            verbosity=0,
        )
        xgb.fit(X_trs, y_tr)
        xgb_scores = xgb.predict_proba(X_tes)[:,1]
        fpr_xgb, tpr_xgb, _ = roc_curve(y_te, xgb_scores)
        roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
        prec_xgb, rec_xgb, _ = precision_recall_curve(y_te, xgb_scores)
        ap_xgb = average_precision_score(y_te, xgb_scores)
        plt.figure(); plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={roc_auc_xgb:.3f})'); plt.plot([0,1],[0,1],'--')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC — XGBoost on Bloch Stats'); plt.legend(); plt.tight_layout(); plt.savefig('xgb_roc.png', dpi=150)
        plt.figure(); plt.plot(rec_xgb, prec_xgb, label=f'XGBoost (AP={ap_xgb:.3f})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR — XGBoost on Bloch Stats'); plt.legend(); plt.tight_layout(); plt.savefig('xgb_pr.png', dpi=150)
        print(f'XGBoost on Bloch stats: AUC={roc_auc_xgb:.3f}, AP={ap_xgb:.3f}')
        # XGBoost feature importance (weight-based)
        xgb_imp = getattr(xgb, 'feature_importances_', None)
        if xgb_imp is not None and len(xgb_imp)>0:
            k = min(15, len(xgb_imp))
            order = np.argsort(xgb_imp)[::-1][:k]
            names = [feat_names[i] if i < len(feat_names) else f'f{i}' for i in order]
            plt.figure(figsize=(8,5)); plt.barh(names[::-1], xgb_imp[order][::-1])
            plt.xlabel('importance'); plt.title('XGBoost Feature Importance (weight)')
            plt.tight_layout(); plt.savefig('xgb_feature_importance.png', dpi=180)
        # Gain-based importance via booster
        try:
            booster = xgb.get_booster()
            gain_map = booster.get_score(importance_type='gain')
            if gain_map:
                # keys like 'f0','f1',... map to gain
                items = sorted(((int(k[1:]), v) for k,v in gain_map.items()), key=lambda t: t[1], reverse=True)[:15]
                idxs = [i for i,_ in items]; gains = [g for _,g in items]
                names = [feat_names[i] if i < len(feat_names) else f'f{i}' for i in idxs]
                plt.figure(figsize=(8,5)); plt.barh(names[::-1], np.array(gains)[::-1])
                plt.xlabel('gain'); plt.title('XGBoost Feature Importance (gain)')
                plt.tight_layout(); plt.savefig('xgb_gain_importance.png', dpi=180)
        except Exception:
            pass
    else:
        print('Note: xgboost not installed; skipped XGBoost training. Install via `pip install xgboost`.')

    # LDA 取 1D 方向（Z'）
    lda = LinearDiscriminantAnalysis(n_components=1, priors=[0.5, 0.5])
    z_tr = lda.fit_transform(X_trs, y_tr).ravel()
    z_te = lda.transform(X_tes).ravel()

    # 残差空间（去除 LDA 方向分量）
    w = lda.scalings_.ravel()  # LDA 方向（在标准化空间）
    w = w / (np.linalg.norm(w) + 1e-9)
    def remove_component(X, w):
        proj = (X @ w[:, None]) * w[None, :]
        return X - proj
    X_tr_res = remove_component(X_trs, w)
    X_te_res = remove_component(X_tes, w)

    # PCA 两个主轴作为 X', Y'
    U, S, Vt = np.linalg.svd(X_tr_res, full_matrices=False)
    pc = Vt[:2].T
    x_tr = X_tr_res @ pc[:, 0]
    y_trp = X_tr_res @ pc[:, 1]
    x_te = X_te_res @ pc[:, 0]
    y_tep = X_te_res @ pc[:, 1]

    # 3D 坐标
    XYZ_tr = np.vstack([x_tr, y_trp, z_tr]).T
    XYZ_te = np.vstack([x_te, y_tep, z_te]).T

    # 在 3D 空间训练线性分类器（得到平面决策边界）
    clf3d = LogisticRegression(max_iter=200, solver='lbfgs', class_weight='balanced')
    clf3d.fit(XYZ_tr, y_tr)
    scores = clf3d.decision_function(XYZ_te)

    # ROC
    fpr, tpr, _ = roc_curve(y_te, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Bloch-only Linear Classifier ROC')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bloch_only_roc.png', dpi=150)

    # PR
    prec, rec, _ = precision_recall_curve(y_te, scores)
    ap = average_precision_score(y_te, scores)
    plt.figure()
    plt.plot(rec, prec, label=f'PR (AP={ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Bloch-only Linear Classifier PR')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bloch_only_pr.png', dpi=150)

    print(f'Bloch-only Linear Classifier: AUC={roc_auc:.3f}, AP={ap:.3f}')

    # 3D 决策边界可视化（在 (X',Y',Z') 空间中是一张平面）
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Color by decision score (coolwarm), symmetric around 0
    sc3 = ax.scatter(XYZ_te[:,0], XYZ_te[:,1], XYZ_te[:,2], c=scores, cmap='coolwarm', s=10, alpha=0.6)
    cbar = plt.colorbar(sc3, shrink=0.65, pad=0.02)
    cbar.set_label('Decision score')

    # 决策平面：w_x x + w_y y + w_z z + b = 0
    w3 = clf3d.coef_.ravel(); b3 = clf3d.intercept_[0]
    xx, yy = np.meshgrid(
        np.linspace(XYZ_te[:,0].min(), XYZ_te[:,0].max(), 25),
        np.linspace(XYZ_te[:,1].min(), XYZ_te[:,1].max(), 25)
    )
    if abs(w3[2]) > 1e-9:
        zz = -(w3[0]*xx + w3[1]*yy + b3) / w3[2]
        ax.plot_surface(xx, yy, zz, alpha=0.2)

    # Tight axes by 2–98 percentiles for clarity
    def tight3(a):
        lo = np.percentile(a, 2); hi = np.percentile(a, 98)
        m = max(abs(lo), abs(hi)); return -m, m
    xlo,xhi = tight3(XYZ_te[:,0]); ylo,yhi = tight3(XYZ_te[:,1]); zlo,zhi = tight3(XYZ_te[:,2])
    ax.set_xlim([xlo,xhi]); ax.set_ylim([ylo,yhi]); ax.set_zlim([zlo,zhi])

    ax.set_xlabel("X' (PCA on residual)")
    ax.set_ylabel("Y' (PCA on residual)")
    ax.set_zlabel("Z' (LDA)")
    ax.set_title("3D Decision Boundary in (X',Y',Z') Space")
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass
    ax.legend()
    plt.tight_layout()
    plt.savefig('bloch_only_decision_3d.png', dpi=180)
    print('Saved: bloch_only_roc.png, bloch_only_pr.png, bloch_only_decision_3d.png')
    print('Models compared on Bloch stats (std.):')
    print(f'  • Logistic (3D-projected): AUC={roc_auc:.3f}, AP={ap:.3f}')
    print(f'  • Linear SVM (18/24D):    AUC={roc_auc_svm:.3f}, AP={ap_svm:.3f}')
    if _HAS_XGB:
        print(f'  • XGBoost (18/24D):       AUC={roc_auc_xgb:.3f}, AP={ap_xgb:.3f}')
    print('Saved importance plots: svm_coef_importance.png' + (', xgb_feature_importance.png, xgb_gain_importance.png' if _HAS_XGB else ''))

    # 13. 三维向量表达 + 轴范围优化（更直观的尺度控制）
    # 取测试集子样画箭头（避免太密集），每类最多 200 条
    def sample_vecs(XYZ, y, per_class=200, seed=7):
        rng = np.random.default_rng(seed)
        idx0 = np.where(y==0)[0]
        idx1 = np.where(y==1)[0]
        pick0 = idx0 if len(idx0)<=per_class else rng.choice(idx0, per_class, replace=False)
        pick1 = idx1 if len(idx1)<=per_class else rng.choice(idx1, per_class, replace=False)
        return XYZ[pick0], XYZ[pick1]

    negV, posV = sample_vecs(XYZ_te, y_te, per_class=200)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Decision scores for sampled vectors
    s_neg = clf3d.decision_function(negV)
    s_pos = clf3d.decision_function(posV)
    V = np.vstack([negV, posV])
    S = np.concatenate([s_neg, s_pos])

    zeros = np.zeros((V.shape[0],))
    # Normalize directions; arrow length modulated by magnitude but capped
    norms = np.linalg.norm(V, axis=1) + 1e-9
    U = V / norms[:,None]
    lengths = np.minimum(QUIVER_LENGTH_SCALE, norms)

    for i in range(V.shape[0]):
        ax.quiver(0,0,0, U[i,0], U[i,1], U[i,2], length=lengths[i],
                  color=plt.cm.coolwarm(0.5 + 0.5*np.tanh(S[i])), alpha=0.55, arrow_length_ratio=0.06)

    # Tight limits by 2–98 percentiles
    def tight3(a):
        lo = np.percentile(a, 2); hi = np.percentile(a, 98)
        m = max(abs(lo), abs(hi)); return -m, m
    xlo,xhi = tight3(XYZ_te[:,0]); ylo,yhi = tight3(XYZ_te[:,1]); zlo,zhi = tight3(XYZ_te[:,2])
    ax.set_xlim([xlo, xhi]); ax.set_ylim([ylo, yhi]); ax.set_zlim([zlo, zhi])
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass
    ax.set_xlabel("X' (PCA residual)")
    ax.set_ylabel("Y' (PCA residual)")
    ax.set_zlabel("Z' (LDA)")
    ax.set_title("3D Quivers Colored by Decision Score")

    # Colorbar proxy (build a scalar mappable)
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(cmap='coolwarm')
    sm.set_array(S)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Decision score')

    plt.tight_layout()
    plt.savefig('bloch_only_vectors_3d.png', dpi=220)
    print('Saved: bloch_only_vectors_3d.png')

    def word_bloch_xyz_mean(word):
        if word in word2vec_model.wv:
            vec = word2vec_model.wv[word]
            key = tuple(np.round(np.asarray(vec, dtype=np.float32), ROUND_KEY))
            bloch = quantum_feature_extraction_cached(key)
            n = len(bloch)//3
            X = np.array([bloch[3*i+0] for i in range(n)], dtype=np.float32)
            Y = np.array([bloch[3*i+1] for i in range(n)], dtype=np.float32)
            Z = np.array([bloch[3*i+2] for i in range(n)], dtype=np.float32)
            return np.array([X.mean(), Y.mean(), Z.mean()], dtype=np.float32)
        return None

    def build_emotion_prototypes():
        protos = {}
        for emo, words in EMOTION_SEEDS.items():
            vecs = []
            for w in words:
                v = word_bloch_xyz_mean(w)
                if v is not None and np.isfinite(v).all():
                    vecs.append(v)
            if vecs:
                m = np.mean(np.vstack(vecs), axis=0)
                n = np.linalg.norm(m) + 1e-9
                protos[emo] = m / n
        return protos

    # 计算与情感原型的相似度（余弦）
    def emotion_scores_from_xyz(xyz, prototypes):
        scores = {}
        n = np.linalg.norm(xyz) + 1e-9
        u = xyz / n
        for emo, p in prototypes.items():
            scores[emo] = float(np.dot(u, p))  # [-1,1]
        return scores
    # -------------------------------------------------------------------------

    # 14. 更深度的情感分析：基于 Bloch 情感原型的投影与可视化
    prototypes = build_emotion_prototypes()
    if not prototypes:
        print('Warning: No emotion prototypes could be built (all OOV?). Skipping emotion projection.')
    else:
        # 14.1 原型向量展示（3D，单位球上）
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 透明球壳
        _u = _np.linspace(0, 2*pi, 48)
        _v = _np.linspace(0, pi, 24)
        _xs = _np.outer(_np.cos(_u), _np.sin(_v))
        _ys = _np.outer(_np.sin(_u), _np.sin(_v))
        _zs = _np.outer(_np.ones_like(_u), _np.cos(_v))
        ax.plot_wireframe(_xs, _ys, _zs, linewidth=0.3, alpha=0.15)
        # 用不同颜色的箭头表示各情感原型
        cmap = plt.get_cmap('tab10')
        for k, (emo, p) in enumerate(prototypes.items()):
            ax.quiver(0,0,0, p[0],p[1],p[2],
                      length=PROTOTYPE_ARROW_LENGTH, color=cmap(k%10), arrow_length_ratio=0.1)
            ax.text(p[0]*1.05, p[1]*1.05, p[2]*1.05, emo)
        ax.set_xlim([-PROTOTYPE_ARROW_LENGTH, PROTOTYPE_ARROW_LENGTH])
        ax.set_ylim([-PROTOTYPE_ARROW_LENGTH, PROTOTYPE_ARROW_LENGTH])
        ax.set_zlim([-PROTOTYPE_ARROW_LENGTH, PROTOTYPE_ARROW_LENGTH])
        try: ax.set_box_aspect((1,1,1))
        except Exception: pass
        ax.set_title('Emotion Prototypes on Bloch Sphere')
        plt.tight_layout(); plt.savefig('bloch_emotion_prototypes_3d.png', dpi=180)

        # 14.2 对测试子集进行情感投影与主导情感着色（更艺术化的 3D 向量图）
        sub_idx = np.concatenate([pos_idx, neg_idx])
        XYZ_rev = []
        dom_labels = []
        dom_scores = []
        for i in sub_idx:
            xb, yb, zb = review_bloch_xyz_means(x_test[i])
            xyz = np.array([xb, yb, zb], dtype=np.float32)
            sc = emotion_scores_from_xyz(xyz, prototypes)
            # 取主导情感
            emo = max(sc.items(), key=lambda kv: kv[1])[0]
            score = sc[emo]
            XYZ_rev.append(xyz)
            dom_labels.append(emo)
            dom_scores.append(score)
        XYZ_rev = np.vstack(XYZ_rev)
        dom_scores = np.array(dom_scores)

        # 颜色映射：按主导情感类别分配颜色，透明度按主导情感置信度（score∈[-1,1]→[0.2,1.0]）
        emo_list = list(prototypes.keys())
        emo_to_color = {emo: cmap(i%10) for i, emo in enumerate(emo_list)}
        alpha_vals = 0.2 + 0.8 * np.clip((dom_scores + 1.0)/2.0, 0, 1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 球壳
        ax.plot_wireframe(_xs, _ys, _zs, linewidth=0.3, alpha=0.08)
        # 从原点出发的向量（归一化后按长度刻画强度）
        for j, xyz in enumerate(XYZ_rev):
            n = np.linalg.norm(xyz) + 1e-9
            uvec = xyz / n
            emo = dom_labels[j]
            ax.quiver(0,0,0, uvec[0], uvec[1], uvec[2],
                      length=min(PROTOTYPE_ARROW_LENGTH, n), color=emo_to_color[emo],
                      alpha=alpha_vals[j], arrow_length_ratio=0.06)
        # 图例
        for emo in emo_list:
            ax.scatter([], [], [], color=emo_to_color[emo], label=emo)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02,1.0))
        ax.set_xlim([-PROTOTYPE_ARROW_LENGTH, PROTOTYPE_ARROW_LENGTH])
        ax.set_ylim([-PROTOTYPE_ARROW_LENGTH, PROTOTYPE_ARROW_LENGTH])
        ax.set_zlim([-PROTOTYPE_ARROW_LENGTH, PROTOTYPE_ARROW_LENGTH])
        try: ax.set_box_aspect((1,1,1))
        except Exception: pass
        ax.set_title("Review Emotion Vectors (Dominant Emotion Colored)")
        plt.tight_layout(); plt.savefig('bloch_emotion_vectors_dominant_3d.png', dpi=200)

        # 14.3 为单条影评绘制情感雷达图（示例：各情感与原型的余弦）
        if len(sub_idx) > 0:
            ridx = int(sub_idx[0])
            xb, yb, zb = review_bloch_xyz_means(x_test[ridx])
            xyz = np.array([xb, yb, zb], dtype=np.float32)
            sc = emotion_scores_from_xyz(xyz, prototypes)
            emos = list(sc.keys()); vals = [sc[e] for e in emos]
            theta = np.linspace(0, 2*np.pi, len(emos), endpoint=False)
            vals_c = np.array(vals + [vals[0]])
            theta_c = np.concatenate([theta, [theta[0]]])
            plt.figure()
            axp = plt.subplot(111, polar=True)
            axp.plot(theta_c, vals_c)
            axp.fill(theta_c, vals_c, alpha=0.2)
            axp.set_xticks(theta)
            axp.set_xticklabels(emos)
            axp.set_title('Emotion Radar (cosine w.r.t prototypes)')
            plt.tight_layout(); plt.savefig('bloch_emotion_radar_example.png', dpi=160)

        print('Saved: bloch_emotion_prototypes_3d.png, bloch_emotion_vectors_dominant_3d.png, bloch_emotion_radar_example.png')
        print('Enhanced figures: bloch_xz_density.png, bloch_scatter_xyz_3d.png (shaded sphere), bloch_only_vectors_3d.png (colored), bloch_only_decision_3d.png (colored)')