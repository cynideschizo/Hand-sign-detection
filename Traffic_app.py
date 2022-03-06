from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Classes of trafic signs
classes = { 0:'A',
            1:'B',
            2:'C',
            3:'D',
            4:'E',
            5:'F',
            6:'G',
            7:'H',
            8:'I',
            9:'K',
            10:'L',
            11:'M',
            12:'N',
            13:'O',
            14:'P',
            15:'Q',
            16:'R',
            17:'S',
            18:'T',
            19:'U',
            20:'V',
            21:'W',
            22:'X',
            23:'Y',
            24:'Z' }

def image_processing(img):
    model = load_model('F:\hand sign detection\hsd.h5')
    data=[]
    image = Image.open(img)
    image = image.resize((28,28))
    data.append(np.array(image))
    X_test=np.array(data)
    predict_x=model.predict(X_test)
    Y_pred=np.argmax(predict_x,axis=1)
    #Y_pred = model.predict_classes(X_test)
    return Y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = "Predicted Hand Sign is: " +classes[a]
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)