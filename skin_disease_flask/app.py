from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.filters import sobel
from PIL import Image
from scipy import ndimage as ndi
from skimage.io import imread, imshow
import random


app = Flask(__name__)
app.secret_key = 'skin_disease'

class_names = {0: 'benign', 1: 'melanoma'}

f = Path("models/model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("models/model.weights.h5")

image_name = ''


def predict_label(img_path):
    cv_img = cv2.imread(img_path, 1)
    plt.imshow(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    # plt.show()
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    ret, bw_img = cv2.threshold(cv_img, 120, 220, cv2.THRESH_BINARY)
    x = np.where(bw_img == 0, 250, bw_img)
    y = np.where(x == 255, 0, x)
    z = sobel(bw_img)
    mask_image = ndi.binary_fill_holes(z)
    mimg = Image.fromarray(np.uint8(mask_image * 255))
    plt.imshow(mimg, cmap="gray")
    plt.axis("off")
    # plt.show()
    num1 = random.randint(1,1000)
    save_path = 'static/' + str(num1) + '.png'
    mimg.save(save_path)

    img = imread(img_path, 0)
    mask = imread(save_path)
    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    extracted = cv2.bitwise_or(img, img, mask=mask)
    extracted = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
    num2 = random.randint(1000, 2000)
    save_path2 = 'static/' + str(num2) + '.png'
    cv2.imwrite(save_path2, extracted)

    img = cv2.imread(save_path2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    # plt.show()
    img = np.array(cv2.resize(img, (128, 128)))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    results = model.predict(img)
    print(results)
    model_label = np.argmax(results[0])
    predicted_class = class_names[model_label]
    accuracy = (results[0][model_label] * 100.0)
    print("predicted_class:", predicted_class)
    print("probability :", accuracy)
    return predicted_class, accuracy, save_path, save_path2

# routes


@app.route("/submit", methods=['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        global image_name
        image_name = img_path
        img.save(img_path)
        p = predict_label(img_path)
        print(p)
        mask = p[2]
        feature = p[3]
        return render_template("home.html", prediction=p[0], acc=p[1], img_path=img_path, mask=mask, feature=feature)


@app.route('/')
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        pwd = request.form["password"]
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["email"] == str(email) and row["password"] == str(pwd):
                return redirect(url_for('home'))
        else:
            mesg = 'Invalid Login Try Again'
            return render_template('login.html', msg=mesg)
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['Email']
        password = request.form['Password']
        col_list = ["name", "email", "password"]
        r1 = pd.read_excel('user.xlsx', usecols=col_list)
        new_row = {'name': name, 'email': email, 'password': password}
        r1 = r1._append(new_row, ignore_index=True)
        r1.to_excel('user.xlsx', index=False)
        print("Records created successfully")
        # msg = 'Entered Mail ID Already Existed'
        msg = 'Registration Successful !! U Can login Here !!!'
        return render_template('login.html', msg=msg)
    return render_template('login.html')


@app.route("/home", methods=['GET', 'POST'])
def home():
   return render_template("home.html")


@app.route('/password', methods=['POST', 'GET'])
def password():
    if request.method == 'POST':
        current_pass = request.form['current']
        new_pass = request.form['new']
        verify_pass = request.form['verify']
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["password"] == str(current_pass):
                if new_pass == verify_pass:
                    r1.replace(to_replace=current_pass, value=verify_pass, inplace=True)
                    r1.to_excel("user.xlsx", index=False)
                    msg1 = 'Password changed successfully'
                    return render_template('password_change.html', msg1=msg1)
                else:
                    msg2 = 'Re-entered password is not matched'
                    return render_template('password_change.html', msg2=msg2)
        else:
            msg3 = 'Incorrect password'
            return render_template('password_change.html', msg3=msg3)
    return render_template('password_change.html')


@app.route('/graphs', methods=['POST', 'GET'])
def graphs():
    return render_template('graphs.html')


@app.route('/cnn')
def cnn():
    return render_template('cnn.html')


@app.route('/logout')
def logout():
    session.clear()
    msg='You are now logged out', 'success'
    return redirect(url_for('login', msg=msg))


if __name__ == '__main__':
    app.run(debug=False)
