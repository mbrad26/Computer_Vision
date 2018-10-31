import os
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import models
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers


base_dir = 'C:\/Users\mbrad\Downloads\kaggle\dogs-vs-cats\cats_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


convolution_base = VGG16(weights='imagenet',
                         include_top=False,
                         input_shape=(150, 150, 3))

model = models.Sequential()
model.add(convolution_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


print(len(model.trainable_weights))

convolution_base.trainable = False

print(len(model.trainable_weights))


set_trainable = False
for layer in convolution_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        set_trainable = False

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(150, 150),
                                                              batch_size=20,
                                                              class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=20,
                                                  class_mode='binary')

model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print(test_loss, 'test acc: ', test_acc)

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


def smooth_curve(points, factor=0.8):
    smooth_points = []
    if smooth_points:
        previous = smooth_points[-1]
        smooth_points.append(previous * factor + points * (1 - factor))
    else:
        smooth_points.append(points)
    return smooth_points


plt.plot(epochs, smooth_curve(loss), 'bo', legend='Loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', legend='Validation Loss')
plt.legend()
plt.figure()

plt.plot(epochs, smooth_curve(acc), 'bo', label='Acc')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation Acc')
plt.legend()
plt.show()
