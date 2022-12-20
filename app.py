from flask import Flask
from flask import request, flash, redirect
from werkzeug.utils import secure_filename

from utils import build_processor, load_wav_16k_mono, build_yamnet_base, class_ids
from tensorflow.keras.models import load_model

import os
from pathlib import Path

import tensorflow as tf # remove later after refactoring
import numpy as np # remove later after refactoring


app = Flask(__name__)


DIRECTORY = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = f'{DIRECTORY}/user_uploads'
[f.unlink() for f in Path(UPLOAD_FOLDER).glob("*") if f.is_file()]

ALLOWED_EXTENSIONS = {'wav'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

processor = build_processor()
yamnet_base = build_yamnet_base()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
   
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))

            return redirect('/predict_yamnet')
            
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/predict_cnn', methods=['GET'])
def predict_single_cnn():
    conv_2d_model = load_model('artefacts/urban_sound_classifier_v0.h5')

    filepath = os.path.join(UPLOAD_FOLDER, os.listdir(UPLOAD_FOLDER)[0])
    waveform = load_wav_16k_mono(filepath)
    waveform = tf.expand_dims(waveform, axis=0)

    spec_mfcc = processor(waveform)

    probas = conv_2d_model.predict(spec_mfcc)
    pred = np.argmax(probas, axis=1)

    class_pred = class_ids[pred[0]]
    certainty = round(probas[:, pred][0][0], 4)

    # remove folder contents before giving prediction so new upload will be possible in same session
    [f.unlink() for f in Path(UPLOAD_FOLDER).glob("*") if f.is_file()]

    return f'{class_pred} with p of {certainty}'


@app.route('/predict_yamnet', methods=['GET'])
def predict_single_yam():
    filepath = os.path.join(UPLOAD_FOLDER, os.listdir(UPLOAD_FOLDER)[0])
    waveform = load_wav_16k_mono(filepath, expand_dim=False)
    _, embeddings, _ = yamnet_base(waveform)

    yamnet_top = load_model('artefacts/urban_sound_classifier_yam.h5')
    embeds = tf.expand_dims(embeddings, axis=0)

    probas = yamnet_top.predict(embeds)

    pred = np.argmax(probas, axis=1)
    class_pred = class_ids[pred[0]]
    certainty = round(probas[:, pred][0][0], 4)

    # remove folder contents before giving prediction so new upload will be possible
    [f.unlink() for f in Path(UPLOAD_FOLDER).glob("*") if f.is_file()]

    return f'{class_pred} with p of {certainty}'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
