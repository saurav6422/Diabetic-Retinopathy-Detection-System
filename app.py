from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__, static_url_path='/static')

isLoggedIn = False

global model
model = load_model('model_2.h5')

class_labels = ['Mild', 'Moderate', 'No_DR','Proliferate_DR','Severe']

def preprocess_image(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_YCrCb = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl1 = clahe.apply(img_YCrCb[:,:,0])
    
    img_YCrCb[:,:,0] = cl1         
    img_RGB_2 = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2RGB)
    
    img_blur = cv2.GaussianBlur(img_RGB_2, (5, 5), 0)
    
    return img_blur

data = "dataset/"
train_dir = data + "train/"
clases = sorted(os.listdir(train_dir))
x_train = np.array([preprocess_image(cv2.imread(os.path.join(train_dir, cl, name), cv2.IMREAD_COLOR)) for cl in clases
           for name in os.listdir(os.path.join(train_dir, cl))])

datagen = ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization = True,
    validation_split=0.2
)
datagen.fit(x_train)


def predict_class(image_path):
    try:
        img = cv2.imread(image_path,cv2.IMREAD_COLOR)
        preprocessed_image = preprocess_image(img)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        x_test = np.array(preprocessed_image)
        x_test = (x_test - datagen.mean)/(datagen.std + 0.000001)      # type: ignore  
        predictions = model.predict(x_test)                   # type: ignore                             
        predicted_class_index = predictions.argmax(axis=-1)[0]
        print(predicted_class_index)
        predicted_class = class_labels[predicted_class_index]
        return predicted_class
    except Exception as e:
        print("Error occurred during prediction:", e)
        return None

@app.route('/')
def index():
    return render_template('login.html', message="Please log in.")

@app.route('/login', methods=['POST'])
def login():
    global isLoggedIn
    username = request.form['username']
    password = request.form['password']

    if username == "Insulyser" and password == "Admin":
        print("Login successful")
        isLoggedIn = True
        return redirect(url_for('welcome'))
    else:
        print("Invalid credentials")
        return render_template('login.html', message="Invalid username or password. Please try again.")

@app.route('/welcome') 
def welcome():
    if isLoggedIn:
        return render_template('welcome.html')
    return "Unauthorized", 401
    
@app.route('/home')
def home():
    print("Rendering home.html")
    return render_template('home.html')
   

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        image.save('C:/Users/saura/Desktop/web/input/input.png') 
        return redirect(url_for('wait'))
    else:
        return render_template('error.html')

@app.route('/wait')            # type: ignore    
def wait():
    print("wait")
    #render_template('wait.html')
    image_path = 'C:/Users/saura/Desktop/web/input/input.png'
    print("Image path:", image_path)
    predicted_class= predict_class(image_path)
    print("Predicted class:", predicted_class)
    if predicted_class == "Mild":
        return render_template('output1.html')
    elif predicted_class == "Moderate":
        return render_template('output2.html')
    elif predicted_class == "No_DR":
        return render_template('output3.html')
    elif predicted_class == "Proliferate_DR":
        return render_template('output4.html')
    elif predicted_class == "Severe":
        return render_template('output5.html')
    elif predicted_class == None:
        return render_template('output6.html')
@app.route('/logout')
def logout():
    global isLoggedIn
    isLoggedIn = False
    return redirect(url_for('index'))

@app.route('/about')
def about():
    if isLoggedIn:
        return render_template('about.html')
    else:
        return render_template('about1.html')
    

@app.route('/contact')
def contact():
    if isLoggedIn:
        return render_template('contact.html')
    else:
        return render_template('contact1.html')
   


if __name__ == '__main__':
    app.run(debug=True)

