# Scalable Machine Learning with PySpark + Flask

I approached this project with mindset of working terabytes of data. I use pyspark for the model, and flask as the server.

## Approach for building the model.
I use one of the data listings data provided. The goal is to predict the price of the airbnb listings using the data 

### Data Exploartion
I did little exploration, I calculated the skewness summary statistics of our target variable (price) and that of numnerical variables.

### Data Cleaning (Removing Outliers and Dealing with missing values)
97% of our data falls between price range of 50 and 750, in order to get unbiased model, I removed the outliers
Next, I Convert all the boolean variable to binary response (0,1)

The missing values were filled with zero in the following columns:
- security_deposit
- extra_people
- cleaning_fee
- All the review score columns: The assumption here is that: When these variable are missing, they are not applicable to the landlord.

The Calculation for Square feet missing values: The Assumption here is that When the square feet is less than or equals 100 and bedroom is 0, the square feet will be 350 and else the square feet will be 380 multiply by the number of bedrooms

### Data Preprocessing for Pipeline

Here, i apply the following methods to prepare the model for gradient boosting tree algorithm in pyspark

- String Indexer - This is used in Machine Learning algorithm to identify column as categorical variable, it converts the categorical column to numeric data and still keeping the categorical context.
- One Hot Encodeing - This is a representation of categorical variables as binary vectors. The categorical values be mapped to integer values.
- Vector Assembling - The last step in the Pipeline, this is to combine all of the columns containing our features into a single column. This has to be done before modeling can take place because every Spark modeling routine expects the data to be in this form. You can do this by storing each of the values from a column as an entry in a vector. Then, from the model's point of view, every observation is a vector that contains all of the information about it and a label that tells the modeler what value that observation corresponds to.

### The Spark Pipeline
The Spark Pipeline is a sequence of stages, and each stage is either a Transformer or an Estimator. These stages are run in order, and the input DataFrame is transformed as it passes through each stage.

The preprocessed data was passed through the pipeline is in the following sequence StringIndexer > OneHotEncoderEstimator > VectorAssember

### Train/Validation set Split
Using RandomSplit Module, i separted the dataset into 80% for training data and 20% for validation data

### Modelling with Gradient-boosted tree regression
Using GBTRegressor module setting the features and price as featuresCol and labelCol respectively, i was able the train the model and also generate predictions on the validation data. The comparison of the output is in the notebook

#### The model Explained
Gradient-Boosted Trees (GBTs) are ensembles of decision trees. GBTs iteratively train decision trees in order to minimize a loss function. Like decision trees, GBTs handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions.

When gradient boosting iteratively trains a sequence of decision trees. On each iteration, the algorithm uses the current ensemble to predict the label of each training instance and then compares the prediction with the true label. The dataset is re-labeled to put more emphasis on training instances with poor predictions. Thus, in the next iteration, the decision tree will help correct for previous mistakes.

The specific mechanism for re-labeling instances is defined by a loss function. With each iteration, GBTs further reduce this loss function on the training data.

##### Assumptions of the model
The gradient boosting method assumes a real-valued y and seeks an approximation in the form of a weighted sum of functions from some class, called base (or weak) learners (poor predictions).
In accordance with the empirical risk minimization principle, the method tries to find an approximation that minimizes the average value of the loss function on the training set, i.e., minimizes the empirical risk. It does so by starting with a model, consisting of a constant function. 


### Microservice approach to developing API and deployment

<b> Pyspark </b> was used to build the model using gradient-boosted trees regression algorithm which was explained, Pyspark is the framework with probably the highest potential to when it comes to implementing Machine Learning and Big Data. It runs fast (up to 100x faster than traditional Hadoop MapReduce due to in-memory operation, offers robust, distributed, fault-tolerant data objects (called RDD), and integrates beautifully with the world of machine learning and graph analytics through supplementary packages like Mlib and GraphX.

Flask was used to to build the API using the flask server 

## What can be done better
- Replacing Flask server with a more scalable and reliable server: Using nginx instead Flask’s built-in server, nginx is a lot more scalable and reliable. As a company shipping products for millions of customers, reliablitity is a very important factor.

- Containerization - Containerizing this app using Docker for faster and smoother deployment, Docker allows for smoother deployments, and more reliability. This will also aid microservice architecture

## To run the this Machine Learning App
```
git clone https://github.com/Godskid89/airbnb-pricing-tool.git
cd api
pip install -r requirements.txt (or run it in anaconda environment)
python train.py #update the model
python app.py #to start the app
```
Now the server is running
POST request ```http://0.0.0.0:5000//getprice```

The request to get prediction should be in JSON format, using variables similar to the data in data folder
