from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
from sklearn.preprocessing import scale




app = Flask(__name__)

model = joblib.load('ETC.pkl')



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def perdict():
    return render_template("predict.html")

@app.route('/pred',methods=['post'])
def predict():
        sen1 = request.form['sen1']
        sen2 = request.form['sen2']
        sen3 = request.form['sen3']
        sen4 = request.form['sen4']
        sen5 = request.form['sen5']
        sen6 = request.form['sen6']
        sen7 = request.form['sen7']
        sen8 = request.form['sen8']
        sen9 = request.form['sen9']
        sen10 = request.form['sen10']
        sample_value = [[float(sen1), float(sen2), float(sen3), float(sen4), float(sen5), float(sen6), float(sen7), float(sen8),
                 float(sen9), float(sen10)]]

        sample_value = np.array(sample_value)
        sample_value = sample_value.reshape(1, -1)

        # Scale the data
        sample_value = scale(sample_value)

        # Use the model to predict the outcome
        prediction = model.predict(sample_value)

        output = ''

        if prediction[0] == 1:
            output = 'Liver Patient'
        else:
            output = 'Healthy'
        return render_template('submit.html', prediction=output)


if __name__ == "__main__":
    app.run(debug=True)