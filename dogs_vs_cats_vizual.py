import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import models
from keras.preprocessing import image


model = load_model('cats_dogs_small_v3.h5')
model.summary()

img_path = 'C:\/Users\mbrad\Downloads\kaggle\dogs-vs-cats\cats_dogs_small\/test\cats\cat.2944.jpg'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255

plt.imshow(img_tensor[0])

layers_output = [layer.output for layer in model.layers[:8]]
model_activations = models.Model(inputs=model.input, outputs=layers_output)

activations = model_activations.predict(img_tensor)
first_layer_output = activations[0]

plt.matshow(first_layer_output[0, :, :, 4], cmap='viridis')
plt.show()


layer_names = [layer.name for layer in model.layers[:8]]

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):

    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_rows = n_features // images_per_row

    display_grid = np.zeros((size * n_rows, images_per_row * size))

    for row in range(n_rows):
        for col in range(images_per_row):

            channel_image = layer_activation[0, :, :, row * images_per_row + col]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[row * size: (row + 1) * size, col * size: (col + 1) * size] = channel_image

        scale = 1./size

        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')