from flask import Flask, render_template, request, jsonify
#import jsonify
import pickle
import numpy as np
import sklearn

app = Flask(__name__)

# Load the pre-trained Iris classification model (replace with your model file)
model = pickle.load(open('iris_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Perform prediction using your Iris model
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

        # Map the numerical prediction to class labels
        class_names = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class = class_names[int(prediction[0])]

        return render_template('index.html', prediction=f'Iris class: {predicted_class}')
    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
