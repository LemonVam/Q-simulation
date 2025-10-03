import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import os
import cirq

# 检查是否使用 GPU
if tf.config.list_physical_devices('GPU'):
    print("Using GPU (CUDA)")
else:
    print("GPU not available, using CPU")

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# 量子特征提取函数
def quantum_feature_extraction(word_vector, circuit_depth=1):
    num_qubits = len(word_vector)
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    circuit = cirq.Circuit()

    # Angle Encoding
    for i, value in enumerate(word_vector):
        circuit.append(cirq.ry(value).on(qubits[i]))

    # 纠缠层
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

# 融合 Word2Vec 向量和量子特征
def combine_classical_quantum_features(word_vector):
    quantum_features = quantum_feature_extraction(word_vector)
    combined = np.concatenate([word_vector, quantum_features])
    return combined

# 1. 加载和准备 IMDB 数据集
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(text_ids):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_ids])

x_train_words = [decode_review(text_ids).split() for text_ids in x_train]
x_test_words = [decode_review(text_ids).split() for text_ids in x_test]

# 2. Word2Vec 训练
embedding_dim = 4
word2vec_model = Word2Vec(sentences=x_train_words, vector_size=embedding_dim, window=5, min_count=1, workers=4)

# 3. 构建嵌入矩阵 (加入量子特征)
embedding_matrix = np.zeros((vocab_size, embedding_dim + embedding_dim))
for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = word2vec_model.wv[word] if word in word2vec_model.wv else None
        if embedding_vector is not None:
            combined_vector = combine_classical_quantum_features(embedding_vector)
            embedding_matrix[i] = combined_vector

# 4. 构建 LSTM 模型
with tf.device('/GPU:0'):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim + embedding_dim, input_length=500, trainable=False),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.layers[0].build((None,))
    model.layers[0].set_weights([embedding_matrix])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. 处理输入数据并训练模型
x_train_padded = pad_sequences(x_train, maxlen=500)
x_test_padded = pad_sequences(x_test, maxlen=500)
with tf.device('/GPU:0'):
    model.fit(x_train_padded, y_train, epochs=50, batch_size=128, validation_data=(x_test_padded, y_test))

# 6. 评估模型
loss, accuracy = model.evaluate(x_test_padded, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')