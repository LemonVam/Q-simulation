import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from gensim.models import Word2Vec
import numpy as np
import os

# 设置GPU
if tf.config.list_physical_devices('GPU'):
    print("Using GPU (CUDA)")
else:
    print("GPU not available, using CPU")

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# 数据处理函数（只用Word2Vec，无量子特征）
def prepare_data(vocab_size=10000, embedding_dim=4):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}

    def decode_review(text_ids):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_ids])

    x_train_words = [decode_review(text_ids).split() for text_ids in x_train]
    x_test_words = [decode_review(text_ids).split() for text_ids in x_test]

    word2vec_model = Word2Vec(sentences=x_train_words, vector_size=embedding_dim, window=5, min_count=1, workers=4)

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i < vocab_size:
            embedding_vector = word2vec_model.wv[word] if word in word2vec_model.wv else None
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    x_train_padded = pad_sequences(x_train, maxlen=500)
    x_test_padded = pad_sequences(x_test, maxlen=500)

    return x_train_padded, y_train, x_test_padded, y_test, embedding_matrix

# 建立模型
def build_model(vocab_size, embedding_dim, embedding_matrix, lstm_units, dropout_rate):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=500, trainable=False),
        LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.layers[0].build((None,))
    model.layers[0].set_weights([embedding_matrix])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 主程序 - Grid Search
x_train, y_train, x_test, y_test, embedding_matrix = prepare_data()
vocab_size = 10000
embedding_dim = 4

param_grid = {
    'lstm_units': [64, 128],
    'dropout_rate': [0.2, 0.4]
}

best_accuracy = 0
best_params = {}

for units in param_grid['lstm_units']:
    for dropout in param_grid['dropout_rate']:
        print(f"Training with LSTM units={units}, Dropout={dropout}")
        with tf.device('/GPU:0'):
            model = build_model(vocab_size, embedding_dim, embedding_matrix, units, dropout)
            history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test), verbose=2)
            loss, accuracy = model.evaluate(x_test, y_test)
            print(f"Test Accuracy: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'lstm_units': units, 'dropout_rate': dropout}

print(f"Best parameters found: {best_params}, Best accuracy: {best_accuracy}")
