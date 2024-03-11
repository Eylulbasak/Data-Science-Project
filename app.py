from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    #load the trained model
    model = joblib.load('model.joblib')

    #load the data from the request
    data = request.json
    df = pd.DataFrame(data)

    #preprocess the data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.reset_index()
    x = df.drop(['uuid'], axis=1)

    #make predictions
    predictions = model.predict_proba(x)
    default_probabilities = predictions[:, 1]

    #create a result DataFrame
    result_df = pd.DataFrame({'uuid': df['uuid'], 'pd': default_probabilities})

    #convert the result to JSON and return
    return result_df.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
