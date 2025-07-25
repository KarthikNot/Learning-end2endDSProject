import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from src.pipeline.predictionPipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-data', methods = ['GET', 'POST'])
def predictData():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        predDf = data.getDataAsDataFrame()
        print(predDf)
        predictPipeline = PredictPipeline()
        results = predictPipeline.predict(predDf)
        return render_template('home.html', results = results[0])
    

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)