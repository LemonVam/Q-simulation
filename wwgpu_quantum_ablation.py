import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from gensim.models import Word2Vec
import numpy as np
import cirq
import matplotlib.pyplot as plt
import os

# 设置GPU
if tf.config.list_physical_devices('GPU'):
    print("Using GPU (CUDA)")
else:
    print("GPU not available, using CPU")

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# 量子特征提取
def quantum_feature_extraction(word_vector, circuit_depth=1):
    num_qubits = len(word_vector)
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    circuit = cirq.Circuit()

    for i, value in enumerate(word_vector):
        circuit.append(cirq.ry(value).on(qubits[i]))

    for d in range(circuit_depth):
        for i in range(num_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        for i in range(num_qubits):
            circuit.append(cirq.rx(np.pi / (d + 1)).on(qubits[i]))

    circuit.append(cirq.measure(*qubits, key='m'))

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=100)
    measurements = result.measurements['m']
    mean_measurements = measurements.mean(axis=0)

    return mean_measurements

def combine_classical_quantum_features(word_vector):
    quantum_features = quantum_feature_extraction(word_vector)
    combined = np.concatenate([word_vector, quantum_features])
    return combined

# 数据准备函数
def prepare_data(vocab_size=10000, embedding_dim=4, use_quantum=False):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}

    def decode_review(text_ids):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_ids])

    x_train_words = [decode_review(text_ids).split() for text_ids in x_train]
    x_test_words = [decode_review(text_ids).split() for text_ids in x_test]

    word2vec_model = Word2Vec(sentences=x_train_words, vector_size=embedding_dim, window=5, min_count=1, workers=4)

    final_dim = embedding_dim * 2 if use_quantum else embedding_dim
    embedding_matrix = np.zeros((vocab_size, final_dim))

    for word, i in word_index.items():
        if i < vocab_size:
            embedding_vector = word2vec_model.wv[word] if word in word2vec_model.wv else None
            if embedding_vector is not None:
                if use_quantum:
                    combined_vector = combine_classical_quantum_features(embedding_vector)
                    embedding_matrix[i] = combined_vector
                else:
                    embedding_matrix[i] = embedding_vector

    x_train_padded = pad_sequences(x_train, maxlen=500)
    x_test_padded = pad_sequences(x_test, maxlen=500)

    return x_train_padded, y_train, x_test_padded, y_test, embedding_matrix

# 模型构建
def build_model(vocab_size, output_dim, embedding_matrix, lstm_units=128, dropout_rate=0.2):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=500, trainable=False),
        LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.layers[0].build((None,))
    model.layers[0].set_weights([embedding_matrix])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 画对比曲线
def plot_compare(histories, titles):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for history, title in zip(histories, titles):
        plt.plot(history.history['val_accuracy'], label=f'{title} Val Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)
    for history, title in zip(histories, titles):
        plt.plot(history.history['val_loss'], label=f'{title} Val Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()

    plt.savefig("ablation_comparison.png")
    plt.show()

# 主程序
vocab_size = 10000
embedding_dim = 4

# 纯Word2Vec
x_train, y_train, x_test, y_test, emb_matrix_plain = prepare_data(use_quantum=False)
with tf.device('/GPU:0'):
    model_plain = build_model(vocab_size, embedding_dim, emb_matrix_plain)
    history_plain = model_plain.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test), verbose=2)

# Word2Vec + 量子特征
x_train, y_train, x_test, y_test, emb_matrix_quantum = prepare_data(use_quantum=True)
with tf.device('/GPU:0'):
    model_quantum = build_model(vocab_size, embedding_dim*2, emb_matrix_quantum)
    history_quantum = model_quantum.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test), verbose=2)

# 画对比图
plot_compare([history_plain, history_quantum], titles=["Word2Vec Only", "Word2Vec + Quantum"])
