from tensorflow.keras.models import Model, load_model		
import numpy as np
import matplotlib.pyplot as plt

#specify the path
path = ''
generator = load_model(path +'13500_GENERATOR_weights_and_arch.hdf5')

#specify the images count
im_count = 4

images = generator.predict(np.random.normal(0, 1, (im_count, 100)))

#visualization
for img in images:
  plt.imshow(img.squeeze())
  plt.show()

