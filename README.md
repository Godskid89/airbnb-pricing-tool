# Scalable Machine Learning with PySpark + Flask

I approached this project with the mindset of working on terabytes of data. I implement it using PySpark for the modeling, and Flask as the server.

## Approach for building the model.
I use one of the listings data provided https://recruitingupload.blob.core.windows.net/scientist202006/listings_processed.parquet.  The goal is to predict the price of the Airbnb listings using the data using machine learning.

## Data Exploration
I did a little exploration, I calculated the skewness summary statistics of our target variable (price) and that of numerical variables.

## Data Cleaning (Removing Outliers and Dealing with missing values)
97% of our data falls between the price range of 50 and 750. To get an unbiased model, I removed the outliers. Next, I convert all the boolean variable to binary response (0,1)
The missing values were filled with zero in the following columns which the assumption that it does not apply to the landlord:
- security_deposit
- extra_people
- cleaning_fee
- All the review score columns
- The Calculation for Square feet missing values: The assumption here is that When the square feet is less than or equals 100 and bedroom is 0, the square feet will be 350 and else the square feet will be 380 multiply by the number of bedrooms

## Data Preprocessing for Pipeline
Here, I apply the following methods to prepare the model for the gradient boosted tree algorithm in PySpark.
- String Indexer - This converts the categorical column to numeric data and still keeping the categorical context.
- One Hot Encoding - This is a representation of categorical variables as binary vectors. The categorical values are mapped to integer values.
- Vector Assembling - The last step in the Pipeline, this is to combine all of the columns containing our features into a single column. This has to be done before modeling can take place because every Spark modeling routine expects the data to be in this form. You can do this by storing each of the values from a column as an entry in a vector. Then, from the model's point of view, every observation is a vector that contains all of the information about it and a label that tells the modeler what value that observation corresponds to.

## The Spark Pipeline
The Spark Pipeline is a sequence of stages, and each stage is either a Transformer or an Estimator. These stages are run in order, and the input DataFrame is transformed as it passes through each stage.
The preprocessed data was passed through the pipeline is in the following sequence StringIndexer > OneHotEncoderEstimator > VectorAssember

## Train/Validation set Split
Using RandomSplit Module, I separated the dataset into 80% for training data and 20% for validation data

## Modeling with Gradient-boosted tree regression
Using the GBTRegressor module setting the features and price as featuresCol and labelCol respectively, I was able the train the model and also generate predictions on the validation data. The comparison of the output is in the notebook

## The model Explained
Gradient-Boosted Trees (GBTs) are ensembles of decision trees. GBTs iteratively train decision trees to minimize a loss function. Like decision trees, GBTs handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and can capture non-linearities and feature interactions.
When gradient boosting iteratively trains a sequence of decision trees. On each iteration, the algorithm uses the current ensemble to predict the label of each training instance and then compares the prediction with the true label. The dataset is re-labeled to put more emphasis on training instances with poor predictions. Thus, in the next iteration, the decision tree will help correct for previous mistakes.
The specific mechanism for re-labeling instances is defined by a loss function. With each iteration, GBTs further reduce this loss function on the training data.

## Assumptions of the model
The gradient boosting method assumes a real-valued y and seeks an approximation in the form of a weighted sum of functions from some class, called base (or weak) learners (poor predictions). Per the empirical risk minimization principle, the method tries to find an approximation that minimizes the average value of the loss function on the training set, i.e., minimizes the empirical risk. It does so by starting with a model, consisting of a constant function.

## Microservice approach to developing API and deployment
Pyspark was used to build the model using the gradient-boosted trees regression algorithm, Pyspark is the framework with probably the highest potential to when it comes to implementing Machine Learning in Big Data. It runs fast (up to 100x faster than traditional Hadoop MapReduce due to in-memory operation, offers robust, distributed, fault-tolerant data objects (called RDD), and integrates beautifully with the world of machine learning and graph analytics through supplementary packages like Mlib and GraphX.
Flask was used to build the API using the flask server

## What can be done better
- Replacing Flask server with a more scalable and reliable server: Using Nginx instead of Flask’s built-in server, Nginx is a lot more scalable and reliable. As a company shipping product for millions of customers, reliability is a very important factor.
- Containerization - Containerizing this app using Docker for faster and smoother deployment, Docker allows for smoother deployments and more reliability. This will also aid microservice architecture that will be implemented

## To run the this Machine Learning App
```
git clone https://github.com/Godskid89/airbnb-pricing-tool.git
cd api
pip install -r requirements.txt (or run it in anaconda environment)
python train.py #update the model
python app.py #to start the app
```
Now the server is running POST request http://0.0.0.0:5000//getprice
The request to get prediction should be in JSON format, using variables similar to the data in the [data folder](https://github.com/Godskid89/airbnb-pricing-tool/tree/master/api/data)
To check the [notebook](https://github.com/Godskid89/airbnb-pricing-tool/tree/master/notebook)
