# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# set the experiment name
mlflow.set_experiment("salary-prediction")

# read the data from csv file
df = pd.read_csv('./Salary_Data.csv')

# split the data into X and y
x = df.drop('Salary', axis=1)
y = df['Salary']

# define random state variable
RANDOM_STATE = 112233

# split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=RANDOM_STATE )

# start tracking 
with mlflow.start_run():

    # log random_state param
    mlflow.log_param("random_state", RANDOM_STATE)

    # build model
    from sklearn.linear_model import LinearRegression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    # model evaluation
    from sklearn.metrics import (
        mean_absolute_error, 
        mean_squared_error, 
        root_mean_squared_error, 
        r2_score )
    y_pred = model_lr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # track the metrics in mlflow
    mlflow.log_metric("mean absolute error", mae)
    mlflow.log_metric("mean squared error", mse)
    mlflow.log_metric("root mean squared error", rmse)
    mlflow.log_metric("r2 score", r2)

    # track the model
    mlflow.sklearn.log_model(model_lr, 
        name="salary-prediction-model",
        registered_model_name="salary-prediction-model",
        input_example=X_train)

    # model inferencing
    salaries = model_lr.predict([[19], [18.5]])
    print(f"salaries = {salaries}")