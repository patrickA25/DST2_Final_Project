# Data Sciences Tools 2 Final Project

## Project overview
I wanted to see if adding macro-economic factors would increase the overall accuracy when predicting if a customer will
churn in the next 30 day. If we are able to better predict which customers will churn out then we can attempt to intervene 
before they fully churn out, or before they start the cancellation process entirely.

## Project Data Sources
* Data from Surveys of Consumers from University of Michigan
  * https://data.sca.isr.umich.edu/survey-info.php
* Data from Kaggle
  * https://www.kaggle.com/datasets/gsagar12/dspp1
* Data from FED
  * https://fred.stlouisfed.org/series/FEDFUNDS
* Data from BLS (Bureau of Labor Statistics)
  * https://data.bls.gov/pdq/SurveyOutputServlet

## Project Workflow
1. Reading in Both raw macro-economic data and raw customer data files.
2. Building out customer dataframe. This will remove any data that is not used in the modeling process.
3. Removing data from the macro-economic data that does not fall with in the date range of the customer data.
   * This step is located in the Data_cleaning_process.py file
4. Building out the model for predicting customer churn with no macro-economic data using the following models
    * Logistic Regression
    * Random Forest Model
    * XGboost
5. Building a model for each month trying to predict customers who will churn in the next 30 days, and adding
macro-economic data.
6. Comparing Both sets of models using accuracy and precision at the criteria.
7. Taking a small selection of months with the best accuracy and precision, and adjusting hyparameters to see if that 
improves the model.

## Final Write Up

After building and running the models here are some of nots that I found. First there was either to little data about
customers for the non-macro added data to find any meaningful prediction ie all were predicted to not churn even when 
there was more than 10% of the sample that would churn out.For the months that were able to make meaningful predictions
Here is the output table for one of the months.

|Model Type|Non-macro hyperperameters| NM Accuracy | NM precision| Macro-Data Hyperperameters|M Accuracy| M Precision|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Logist Model|Penalty: l1,Solver:liblinear|.941|.941|Penalty:l1,Solver:libinear|.943|.943|
|Random Forest|Criterion:gini,Max_features:Sqrt,n_estimators:100|.941|.941|Criterion:gini,Max_features:Sqrt,n_estimators:100|.946|.943|
|XGboost Classifier|Booster:gbtree,tree_method : approx|.941|.941|Booster:gbtree,tree_method : approx|.943|.943|

From the table above we can see for all three types of models, the ones with macro-economic data added out performed 
model that did not have macro data added. I would recommend that a company would try and add macro economic data to their 
customer churn model. I think with the addition of more customer meta data, and data provided by the CX team would help build 
a better customer churn module.

