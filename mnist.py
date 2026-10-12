# TensorFlow and tf.keras
import tensorflow as tf
 
# helper libraries
import numpy as np
from matplotlib import pyplot as plt
 
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
 
print(train_images.shape)
print(len(train_labels))
print(train_labels)
 
print(test_images.shape)
print(len(test_labels))
print(test_labels)
 
# 원본 이미지 확인
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
 
# 0 ~ 1 사이 값으로 정규화
train_images = train_images / 255.0
test_images = test_images / 255.0
 
# 정규화된 이미지 확인
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
 
plt.figure(figsize=(10, 10))
 
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
 
    plt.imshow(
        train_images[i],
        cmap=plt.cm.binary
    )
 
    plt.xlabel(
        class_names[train_labels[i]]
    )
 
plt.show()
 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
 
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
 
model.fit(
    train_images,
    train_labels,
    epochs=20
)
 
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()