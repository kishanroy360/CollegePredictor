import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


def prediction(arr):
    ip = {'GRE Score': [arr[0]],
          'TOEFL Score': [arr[1]],
          'University Rating': [arr[2]],
          'SOP': [arr[3]],
          'LOR': [arr[4]],
          'CGPA': [arr[5]],
          'Research': [arr[6]]
          }
    ip_df = pd.DataFrame(data=ip)
    return model.predict(ip_df)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_input = [float(x) for x in request.form.values()]
    output = prediction(user_input)
    return render_template('index.html', prediction_text='The chances of you to get an admit are {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
