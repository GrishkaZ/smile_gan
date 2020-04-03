from tensorflow.keras.models import Model, load_model
from skimage.transform import resize
from skimage import data, color
import numpy as np
import matplotlib.pyplot as plt

import io

class ModelGeneratorService:

    def __init__(self, model = None):

        self.path = 'smiles_data/'
        self.model = model

        if (not self.model):
              self.model = '13500_GENERATOR_weights_and_arch.hdf5'

        self.generator = load_model(self.path + self.model)

    def generatePicture(self):

        images = self.generator.predict(np.random.normal(0, 1, (1, 100)))
        imgBytes = io.BytesIO()
        for image in images:
            image = image.squeeze()
            image = resize(image, (200, 200), anti_aliasing=True)
            plt.imsave(imgBytes, image, format = "png", cmap = plt.cm.gray)

        imgBytes.seek(0)

        return imgBytes

# modelGeneratorService = ModelGeneratorService()
# #visualization
# for img in modelGeneratorService.generatePicture():
#   plt.imshow(img)
#   plt.show()
