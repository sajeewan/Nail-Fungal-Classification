from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__, template_folder="./")

model = load_model("./my_model/model.h5", compile=False)
class_names = open("./my_model/labels.txt", "r").readlines()


@app.route('/classify', methods=['POST'])
def classify_image():
    # Get the uploaded image file from the request
    file = request.files['image']

    # Open and preprocess the image
    image = Image.open(file).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make a prediction using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Create the JSON response
    response = jsonify({
        'class': class_name,
        'confidence': float(confidence_score)
    })

    return response


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
