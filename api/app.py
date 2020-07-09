#!/usr/bin/env python3

from flask import jsonify, request, Flask
import os
import sys

sys.path.extend(['/Users/josehpoladokun/PycharmProjects/pricing_model/api'])



# Load libraries
from pyspark.ml.pipeline import Estimator, Transformer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.ml.regression import GBTRegressionModel
from train import data_processing

print ("Successfully imported Spark Modules")



MODEL_PATH = "models"

app = Flask(__name__)


@app.route('/getprice', methods=['POST'])
def predict():
    spark = SparkSession.builder.appName('airbnb_price').getOrCreate()
    if request.method == 'POST':
        raw_data = spark.read.json(request.json)
    model = GBTRegressionModel.load(MODEL_PATH)
    data = data_processing(raw_data)
    gbt_predictions = model.transform(data)
    output = gbt_predictions.select('prediction')
    json_output = output.toJSON()
    return jsonify({'prediction': json_output})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
