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