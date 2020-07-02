from flask import Flask, render_template, request, jsonify
from keras.models import model_from_json
import cv2
import numpy as np

app = Flask(__name__, template_folder='templates')

PATH = "3-class-model/"
json_file = open(PATH + "model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights(PATH + "model.h5")
print("Loaded model from disk")

labels_name = ['Fighting', 'Normal', 'Robbery']


def predict(img):
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, 64, 64, 3)
    pred = model.predict(img)
    pred_label = labels_name[np.argmax(pred)]
    return np.argmax(pred)


def get_probability(val, val_list):
    return val_list.count(val) / len(val_list) * 100


def load_video(full_path):
    cap = cv2.VideoCapture(full_path)
    predictions = []
    i = 0
    print('Predicting ...', end='')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        pred = predict(frame)
        predictions.append(pred)
        i += 1

    unique_list = list(set(predictions))
    descriptions = []
    for val in unique_list:
        prob = round(get_probability(val, predictions), 2)
        descriptions.append(
            "Video contains {} of {}% probability.".format(labels_name[val], prob)
        )
        # print("\nVideo contains {} of {}% probability.".format(labels_name[val], prob), end='')
    return descriptions


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/uploader', methods=['POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        file_name = f.filename
        f.save(file_name)
        data = load_video(file_name)
        return jsonify(data)


if __name__ == '__main__':
    app.run(port=8000, debug=False, threaded=False)
