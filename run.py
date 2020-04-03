from service.ModelGeneratorService import ModelGeneratorService
from flask import Flask, render_template, send_file, url_for, request
import matplotlib.pyplot as plt
import random

modelGeneratorService = ModelGeneratorService(
    model = '2500_GENERATOR_weights_and_arch.hdf5',
    res = (170, 170)
)
app = Flask(__name__)

@app.route("/")
def index():
    smiles = 6
    pictures = []
    for indx in range(smiles):
        pictures.append('/picture?uuid={0}'.format(indx + random.randint(100,1000000)))

    return render_template("main.html",
            pictures = pictures)

@app.route("/picture")
def picture():
    picture = modelGeneratorService.generatePicture()
    return send_file(picture, mimetype='image/jpeg')


if __name__ == "__main__":
    app.run(host='192.168.1.2', port=8080)