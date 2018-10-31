import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models, layers

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = train_data.reshape((60000, 28, 28, 1))
train_data = train_data.astype('float32') / 255

test_data = test_data.reshape((10000, 28, 28, 1))
test_data = test_data.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

model.fit(train_data, train_labels, epochs=5, batch_size=64)

results = model.evaluate(test_data, test_labels)
print(results)

# history_dict = history.history
# loss = history_dict['loss']
# acc = history_dict['acc']
# val_loss = history_dict['val_loss']
# val_acc = history_dict['val_acc']
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, loss, 'bo', label='Loss')
# plt.plot(epochs, val_loss, 'b', label='Val Loss')
# plt.legend()
# plt.figure()
#
# plt.plot(epochs, acc, 'bo', label='Acc')
# plt.plot(epochs, val_acc, 'b', label='Val Acc')
# plt.legend()
# plt.show()