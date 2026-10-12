import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("훈련 샘플: {}, 레이블: {}".format(len(train_data), len(train_labels)))

# ---- 여기가 핵심: 정수 시퀀스를 원핫(멀티-핫) 벡터로 변환 ----
# 예: [1, 14, 22] -> 길이 10000짜리 벡터에서 인덱스 1, 14, 22 위치만 1.0, 나머지는 0.0
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 레이블도 float32로 변환 (sigmoid + binary_crossentropy 쓰기 위해)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(x_train.shape)   # (25000, 10000)
print(x_train[0][:20]) # 0과 1로만 이루어진 벡터 확인

# ---- 모델: Embedding 필요 없음, 바로 Dense로 ----
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(vocab_size,)))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))  # 이진분류라 출력 1개 + sigmoid
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

results = model.evaluate(x_test, y_test, verbose=2)
print(results)

history_dict = history.history
print(history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
