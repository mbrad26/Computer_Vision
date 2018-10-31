import os
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.layers import Dense, Dropout
from keras import optimizers


def extract_features(directory, samples):
    features = np.zeros((samples, 4, 4, 512))
    labels = np.zeros(samples)
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_input = convolution_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_input
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= samples:
            break
    return features, labels


base_dir = 'C:\/Users\mbrad\Downloads\kaggle\dogs-vs-cats\cats_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


train_data, train_labels = extract_features(train_dir, 2000)
validation_data, validation_labels = extract_features(validation_dir, 1000)

train_data = np.reshape(train_data, (2000, 4 * 4 * 512))
validation_data = np.reshape(validation_data, (1000, 4 * 4 * 512))


convolution_base = VGG16(weights='imagenet',
                         include_top=False)

model = models.Sequential()
model.add(Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_data,
                    train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_data, validation_labels))


history_dict = history.history
loss = history_dict['loss']
acc = history_dict['acc']
val_loss = history_dict['val_loss']
val_acc = history_dict['val_acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Loss')
plt.plot(epochs, val_loss, 'b', label='Val Loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Acc')
plt.plot(epochs, val_acc, 'b', label='Val Acc')
plt.legend()
plt.show()


































