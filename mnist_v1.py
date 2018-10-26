import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models, layers

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = train_data.reshape((60000, 28 * 28))
train_data = train_data.astype('float32') / 255
test_data = test_data.reshape((10000, 28 * 28))
test_data = test_data.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

val_data = train_data[:10000]
partial_train_data = train_data[10000:]
val_labels = train_labels[:10000]
partial_train_labels = train_labels[10000:]


model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

model.fit(
    train_data,
    train_labels,
    epochs=5,
    batch_size=128)

results = model.evaluate(test_data, test_labels)
print(results)

# hist_dict = history.history
#
# loss = hist_dict['loss']
# acc = hist_dict['acc']
# val_loss = hist_dict['val_loss']
# val_acc = hist_dict['val_acc']
# epochs = range(1, 10 + 1)
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.legend()
# plt.show()
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.legend()
# plt.show()