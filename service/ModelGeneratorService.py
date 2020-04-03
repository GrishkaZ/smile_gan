from tensorflow.keras.models import Model, load_model
from skimage.transform import resize
from skimage import data, color, util
import numpy as np
import matplotlib.pyplot as plt

import io

class ModelGeneratorService:

    def __init__(self, model, res):

        assert model
        assert res

        self.model = model
        self.resolution = res

        self.path = 'smiles_data/'
        self.generator = load_model(self.path + self.model)

    def generatePicture(self):

        images = self.generator.predict(np.random.normal(0, 1, (1, 100)))
        imgBytes = io.BytesIO()
        for image in images:
            image = image.squeeze()
            image = util.invert(image)
            image = resize(image, self.resolution, anti_aliasing=True)
            plt.imsave(imgBytes, image, format = "png", cmap = plt.cm.gray)

        imgBytes.seek(0)

        return imgBytes

# modelGeneratorService = ModelGeneratorService()
# #visualization
# for img in modelGeneratorService.generatePicture():
#   plt.imshow(img)
#   plt.show()
