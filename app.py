from flask import Flask, render_template,request
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

img_size = 50
app = Flask(__name__)
model = keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template("index.html", name="Tariq")


@app.route('/prediction', methods=["POST"])
def prediction():
    img = request.files['img']
    img.save('img.jpg')

    """
    img = image.load_img("img.jpg", target_size=(50,50))
    x=image.img_to_array(img) / 255
    resized_img_np = np.expand_dims(x,axis=0)
    prediction = model.predict(resized_img_np)
    """
    img = cv2.imread("img.jpg", 0)
    #from google.colab.patches import cv2_imshow
    #cv2_imshow(img11)
    img = cv2.resize(img, (img_size, img_size))
    x=image.img_to_array(img) / 255
    resized_img_np = np.expand_dims(x,axis=0)
    prediction = model.predict(resized_img_np)
    
    p1=1-prediction
    if prediction<p1:
        pred="malignant"
    else:
        pred="benign"    


    return render_template("prediction.html", data=pred)


if __name__ =="__main__":
    app.run(debug=True)