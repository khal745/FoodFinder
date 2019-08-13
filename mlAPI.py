import flask
import numpy as np
import keras
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import io
from PIL import Image

from flask_dropzone import Dropzone
import os

basedir = os.path.abspath(os.path.dirname(__file__))

app = flask.Flask(__name__)
model = None

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=1,
    DROPZONE_REDIRECT_VIEW='completed'
)

dropzone = Dropzone(app)

output_pred = "?"

def load_model_api():
    global model
    model = load_model('ft_model.h5')
    model._make_predict_function()


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # image = preprocess_input(image)
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the image preprocessed using the ResNet50 pre-processing function
    return image


@app.route('/')
def index():
    return flask.render_template('page.html')


@app.route("/predict", methods=["POST", "GET"])
def upload():
    data = {"success": False}
    if flask.request.method == 'POST':
        image = flask.request.files.get('file') 
        #image.save(os.path.join(app.config['UPLOADED_PATH'], image.filename))
        #filename = image.filename
    
    image = Image.open(image)

    # pre-process the image
    image = prepare_image(image, target=(224, 224))

    # classify it, give prediction to return to client
    prediction = model.predict(image)
    global output_pred
    if prediction > 0.5:
        output_pred = "not_food"
    else:
        output_pred = "food"

    # Add to data dictionary to return to client
    data["predictions"] = output_pred

    # Return the request was successful
    data["success"] = True

    return '' 

@app.route('/completed')
def completed():
    if (output_pred == 'food'):
        return '<h1>Prediction Complete</h1><p>Prediction: Its food! </p>' 
    if (output_pred == 'not_food'):
        return '<h1>Prediction Complete</h1><p>Prediction: No food here!</p>' 

    return '<h1>Redirected Page</h1><p>Oops</p>' 

if __name__ == "__main__":
    print("*Loading model and Flask starting server, please wait:")
    load_model_api()
    app.run()

