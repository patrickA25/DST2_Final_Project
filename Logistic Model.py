import pandas as pd
import numpy as np
import datetime as datetime
import calendar
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
pd.options.mode.chained_assignment = None  # default='warn'

warnings.simplefilter(action='ignore', category=FutureWarning)


# Reading in customer data
CustomerData = pd.read_csv('prepared_data/Full_customer_data.csv')
CustomerData = CustomerData.drop(['Unnamed: 0'], axis=1)
CustomerData.info()
CustomerData['signup_date_time'] = pd.to_datetime(CustomerData['signup_date_time'])
CustomerData['cancel_date_time'] = pd.to_datetime(CustomerData['cancel_date_time'])

# Readign in macro economic Data
CPINumbers = pd.read_csv('prepared_data/CPI.csv')
fedData = pd.read_csv('prepared_data/FedData.csv')
ICSNumbers = pd.read_csv('prepared_data/ICS.csv')


# We will be only looking at people who have a monthly subscription
SubCustomerData = CustomerData[CustomerData['product'] == 'prd_2']
sns.countplot(x='has_canceled', data=SubCustomerData, palette='hls')
X_values = ['age', 'gender']


Cohort_list = list(SubCustomerData['Cohort_month'].unique())
XvaluesNoMac = ['age', 'gender']
XvaluesMac = ['age', 'gender', 'CPI_num', 'Fed_num', 'ICS_num']
MonthList = []
modelCount = 0
# building list to store values of models for basic model
logisticBaseModtrain = []
logisticBaseModtest = []
logisticBaseModPrecision = []
RanForestBaseModTrain = []
RanForestBaseModTest = []
RanForestBaseModPrecision = []
XGboostBasModTrain = []
XGboostBasModTest = []
XGboostBaseModPrecision = []
# building list to store values of models for macro added model
logisticMacModtrain = []
logisticMacModtest = []
logisticMacModPrecision = []
RanForestMacModTrain = []
RanForestMacModTest = []
RanForestMacModPrecision = []
XGboostMacModTrain = []
XGboostMacModTest = []
XGboostMacModPrecision = []

for ii in range(1, len(Cohort_list)):
    # revmoving values that are passed this date
    # print(Cohort_list[ii])
    # print(SubCustomerData['cancel_date_time'].head())

    temp_data = SubCustomerData[(SubCustomerData['cancel_date_time'] > Cohort_list[ii]) &
                                (SubCustomerData['signup_date_time'] < Cohort_list[ii])]
    # print(temp_data.shape)
    if temp_data.shape[0] < 1000:
        print('==================================>')
        print('to few samples to run model on.')
        print(f'There are only {temp_data.shape[0]} values.')
        print(f'Moving on to cohort month {Cohort_list[(ii + 1)]}')
        print('==================================>')
        continue
    # Calculating the data diff for each month.
    temp_data['active_dates'] = (pd.to_datetime(Cohort_list[ii]) - temp_data['signup_date_time']) / np.timedelta64(1,
                                                                                                                   'D')
    # checking and removing any accounts that are less then 30 day
    temp_data1 = temp_data[temp_data['active_dates'] > 30]
    if temp_data1.shape[0] < 1000:
        print('==================================>')
        print('after removing customers with less then 30 days')
        print('there are to few samples to run model on.')
        print(f'There are only {temp_data.shape[0]} values.')
        print(f'Moving on to cohort month {Cohort_list[(ii + 1)]}')
        print('==================================>')
        continue

    temp_date = pd.to_datetime(Cohort_list[ii])
    # the -1 is because the month starts at one and is counted
    NumDays = (calendar.monthrange(temp_date.year, temp_date.month)[1] - 1)
    temp_data1['has_canceled'] = (temp_data1['cancel_date_time'] >= temp_date) & (
                temp_data1['cancel_date_time'] <= (temp_date + datetime.timedelta(NumDays - 1)))

    ##################################################################
    # Adding Macro Economic Data For Last 30 Days
    ##################################################################
    # Finding the dif in CPI numbers
    CpiTM = float(CPINumbers[CPINumbers['Date'] == Cohort_list[ii]]['inflation'])
    CpiLM = float(CPINumbers[CPINumbers['Date'] == Cohort_list[(ii - 1)]]['inflation'])
    CpiDiff = CpiLM - CpiTM
    # Finding the dif in fed numbers
    FedTM = float(fedData[fedData['DATE'] == Cohort_list[(ii)]]['FEDFUNDS'])
    FedLM = float(fedData[fedData['DATE'] == Cohort_list[(ii - 1)]]['FEDFUNDS'])
    FedDiff = FedLM - FedTM
    # Finding the dif in ICS Numbers
    ICSTM = float(ICSNumbers[ICSNumbers['Date'] == Cohort_list[(ii)]]['ICS_ALL'])
    ICSLM = float(ICSNumbers[ICSNumbers['Date'] == Cohort_list[(ii - 1)]]['ICS_ALL'])
    ICSDiff = ICSLM - ICSTM

    # Adding Macro data to data frame
    temp_data1['CPI_num'] = CpiDiff
    temp_data1['Fed_num'] = FedDiff
    temp_data1['ICS_num'] = ICSDiff

    ##################################################################
    # Building the data set basic                                     #
    ##################################################################

    Y_data = temp_data1[['has_canceled']]
    X_data = temp_data1[XvaluesNoMac].copy()
    gender_dummies = pd.get_dummies(temp_data1['gender'])
    X_data['female'] = gender_dummies.loc[:, 'female'].copy()
    X_data['male'] = gender_dummies.loc[:, 'male'].copy()
    X_data = X_data.drop('gender', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_data, np.array(Y_data),
                                                        test_size=.3)
    MonthList.append(Cohort_list[ii])
    ##################################################################
    # Building out the logistc Regression section                     #
    ##################################################################

    print('==================================>')
    logisticBaseMod = LogisticRegression()
    logisticBaseMod.fit(X_train, y_train.ravel())
    logisticBaseModtrain.append(logisticBaseMod.score(X_train, y_train))
    logisticBaseModtest.append(accuracy_score(y_test, logisticBaseMod.predict(X_test)))
    logisticBaseModPrecision.append(precision_score(y_test, logisticBaseMod.predict(X_test), average='binary'))
    print(f'Base Logistic Model Completed for Cohort {Cohort_list[ii]}.')
    modelCount += 1
    print('==================================>')

    ##################################################################
    # Building out the Random Forest section                          #
    ##################################################################

    print('==================================>')
    RanForestMod = RandomForestClassifier()
    RanForestMod.fit(X_train, y_train.ravel())
    RanForestBaseModTrain.append(RanForestMod.score(X_train, y_train))
    RanForestBaseModTest.append(accuracy_score(y_test, RanForestMod.predict(X_test)))
    RanForestBaseModPrecision.append(precision_score(y_test, RanForestMod.predict(X_test), average='binary'))
    print(f'Base Random forest Model Completed for Cohort {Cohort_list[ii]}.')
    modelCount += 1
    print('==================================>')

    ####################################################################
    # Building out the XGboost base model
    ####################################################################

    print('==================================>')
    XGboostBasMod = XGBClassifier(use_label_encoder=False, verbosity=0)
    XGboostBasMod.fit(X_train, y_train * 1)
    XGboostBasModTrain.append(XGboostBasMod.score(X_train, y_train))
    XGboostBasModTest.append(accuracy_score(y_test, XGboostBasMod.predict(X_test)))
    XGboostBaseModPrecision.append(precision_score(y_test, XGboostBasMod.predict(X_test), average='binary'))
    print(f'Base XGboost Model Completed for Cohort {Cohort_list[ii]}.')
    modelCount += 1
    print('==================================>')

    ##################################################################
    # Building the data set Macro added model                         #
    ##################################################################

    YdataMacro = temp_data1[['has_canceled']]
    XdataMacro = temp_data1[XvaluesMac]
    gender_dummies = pd.get_dummies(temp_data1['gender'])
    XdataMacro['female'] = gender_dummies.loc[:, 'female'].copy()
    XdataMacro['male'] = gender_dummies.loc[:, 'male'].copy()
    XdataMacro = XdataMacro.drop('gender', axis=1)
    XMtrain, XMtest, ymtrain, ymtest = train_test_split(XdataMacro, np.array(YdataMacro),
                                                        test_size=.3)

    ##################################################################
    # Building out the logistc Regression section                     #
    ##################################################################

    print('==================================>')
    logisticMacroMod = LogisticRegression()
    logisticMacroMod.fit(XMtrain, ymtrain.ravel())
    logisticMacModtrain.append(logisticMacroMod.score(XMtrain, ymtrain))
    logisticMacModtest.append(accuracy_score(ymtest, logisticMacroMod.predict(XMtest)))
    logisticMacModPrecision.append(precision_score(ymtest, logisticMacroMod.predict(XMtest), average='binary'))
    print(f'Base Logistic Model Completed for Cohort {Cohort_list[ii]}.')
    modelCount += 1
    print('==================================>')

    ##################################################################
    # Building out the logistc Regression section                     #
    ##################################################################

    print('==================================>')
    RanForestMacroMod = RandomForestClassifier()
    RanForestMacroMod.fit(XMtrain, ymtrain.ravel())
    RanForestMacModTrain.append(RanForestMacroMod.score(XMtrain, ymtrain))
    RanForestMacModTest.append(accuracy_score(ymtest, RanForestMacroMod.predict(XMtest)))
    RanForestMacModPrecision.append(precision_score(ymtest, RanForestMacroMod.predict(XMtest), average='binary'))
    print(f'Base Random forest Model Completed for Cohort {Cohort_list[ii]}.')
    modelCount += 1
    print('==================================>')

    ####################################################################
    # Building out the XGboost base model
    ####################################################################

    print('==================================>')
    XGboostMacroMod = XGBClassifier(use_label_encoder=False, verbosity=0)
    XGboostMacroMod.fit(XMtrain, ymtrain * 1)
    XGboostMacModTrain.append(XGboostMacroMod.score(XMtrain, ymtrain))
    XGboostMacModTest.append(accuracy_score(ymtest, XGboostMacroMod.predict(XMtest)))
    XGboostMacModPrecision.append(precision_score(ymtest, XGboostMacroMod.predict(XMtest), average='binary'))
    print(f'Base XGboost Model Completed for Cohort {Cohort_list[ii]}.')
    modelCount += 1
    print('==================================>')
print('Modeling process completed.')
print(f'{modelCount} were created.')

# building the data frame for all
Modeling_results = pd.DataFrame({'Month': MonthList,
                                 'logisticBaseModtrain': logisticBaseModtrain,
                                 'logisticBaseModtest': logisticBaseModtest,
                                 'logisticBaseModPrecision': logisticBaseModPrecision,
                                 'RanForestBaseModTrain': RanForestBaseModTrain,
                                 'RanForestBaseModTest': RanForestBaseModTest,
                                 'RanForestBaseModPrecision': RanForestBaseModPrecision,
                                 'XGboostBasModTrain': XGboostBasModTrain,
                                 'XGboostBasModTest': XGboostBasModTest,
                                 'XGboostBaseModPrecision': XGboostBaseModPrecision,
                                 'logisticMacModtrain': logisticMacModtrain,
                                 'logisticMacModtest': logisticMacModtest,
                                 'logisticMacModPrecision': logisticMacModPrecision,
                                 'RanForestMacModTrain': RanForestMacModTrain,
                                 'RanForestMacModTest': RanForestMacModTest,
                                 'RanForestMacModPrecision': RanForestMacModPrecision,
                                 'XGboostMacModTrain': XGboostMacModTrain,
                                 'XGboostMacModTest': XGboostMacModTest,
                                 'XGboostMacModPrecision': XGboostMacModPrecision})


Modeling_results[['logisticBaseModtest',
                  'logisticMacModtest',
                  'logisticBaseModPrecision',
                  'logisticMacModPrecision',
                  'RanForestBaseModTest',
                  'RanForestMacModTest',
                  'RanForestBaseModPrecision',
                  'RanForestMacModPrecision',
                  'XGboostBasModTest',
                  'XGboostMacModTest',
                  'XGboostBaseModPrecision',
                  'XGboostMacModPrecision']].describe()


# Doing model optimization on one Month of data
temp_data = SubCustomerData[(SubCustomerData['cancel_date_time'] > Cohort_list[59]) &
                            (SubCustomerData['signup_date_time'] < Cohort_list[59])]
# print(temp_data.shape)
if temp_data.shape[0] < 1000:
    print('==================================>')
    print('to few samples to run model on.')
    print(f'There are only {temp_data.shape[0]} values.')
    print(f'Moving on to cohort month {Cohort_list[(59 + 1)]}')
    print('==================================>')
# Calculating the data diff for each month.
temp_data['active_dates'] = (pd.to_datetime(Cohort_list[59]) - temp_data['signup_date_time']) / np.timedelta64(1, 'D')
# checking and removing any accounts that are less then 30 day
temp_data1 = temp_data[temp_data['active_dates'] > 30]
if temp_data1.shape[0] < 1000:
    print('==================================>')
    print('after removing customers with less then 30 days')
    print('there are to few samples to run model on.')
    print(f'There are only {temp_data.shape[0]} values.')
    print(f'Moving on to cohort month {Cohort_list[(59 + 1)]}')
    print('==================================>')

temp_date = pd.to_datetime(Cohort_list[59])
# the -1 is because the month starts at one and is counted
NumDays = (calendar.monthrange(temp_date.year, temp_date.month)[1] - 1)
temp_data1['has_canceled'] = (temp_data1['cancel_date_time'] >= temp_date) & (
            temp_data1['cancel_date_time'] <= (temp_date + datetime.timedelta(NumDays - 1)))

##################################################################
# Adding Macro Economic Data For Last 30 Days
##################################################################
# Finding the dif in CPI numbers
CpiTM = float(CPINumbers[CPINumbers['Date'] == Cohort_list[59]]['inflation'])
CpiLM = float(CPINumbers[CPINumbers['Date'] == Cohort_list[(59 - 1)]]['inflation'])
CpiDiff = CpiLM - CpiTM
# Finding the dif in fed numbers
FedTM = float(fedData[fedData['DATE'] == Cohort_list[(59)]]['FEDFUNDS'])
FedLM = float(fedData[fedData['DATE'] == Cohort_list[(59 - 1)]]['FEDFUNDS'])
FedDiff = FedLM - FedTM
# Finding the dif in ICS Numbers
ICSTM = float(ICSNumbers[ICSNumbers['Date'] == Cohort_list[(59)]]['ICS_ALL'])
ICSLM = float(ICSNumbers[ICSNumbers['Date'] == Cohort_list[(59 - 1)]]['ICS_ALL'])
ICSDiff = ICSLM - ICSTM

# Adding Macro data to data frame
temp_data1['CPI_num'] = CpiDiff
temp_data1['Fed_num'] = FedDiff
temp_data1['ICS_num'] = ICSDiff

##################################################################
# Building the data set basic                                     #
##################################################################

Y_data = temp_data1[['has_canceled']]
X_data = temp_data1[XvaluesNoMac].copy()
gender_dummies = pd.get_dummies(temp_data1['gender'])
X_data['female'] = gender_dummies.loc[:, 'female'].copy()
X_data['male'] = gender_dummies.loc[:, 'male'].copy()
X_data = X_data.drop('gender', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_data, np.array(Y_data),
                                                    test_size=.3,
                                                    random_state=12345)


##################################################################
# Building the data set basic                                     #
##################################################################

Y_data = temp_data1[['has_canceled']]
X_data = temp_data1[XvaluesNoMac].copy()
gender_dummies = pd.get_dummies(temp_data1['gender'])
X_data['female'] = gender_dummies.loc[:, 'female'].copy()
X_data['male'] = gender_dummies.loc[:, 'male'].copy()
X_data = X_data.drop('gender', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_data, np.array(Y_data),
                                                    test_size=.3)
# random_state = 12345)
##################################################################
# Building out the logistc Regression section                     #
##################################################################

print('==================================>')
logisticBaseMod = LogisticRegression()
logisticBaseMod.fit(X_train, y_train.ravel())
logisticBaseMod.score(X_train, y_train)
accuracy_score(y_test, logisticBaseMod.predict(X_test))
precision_score(y_test, logisticBaseMod.predict(X_test), average='binary')
print(f'Base Logistic Model Completed for Cohort {Cohort_list[ii]}.')
modelCount += 1
print('==================================>')

logisticBaseMod.predict(X_test)


##################################################################
# Building out the Random Forest section                          #
##################################################################

print('==================================>')
RanForestMod = RandomForestClassifier()
RanForestMod.fit(X_train, y_train.ravel())
RanForestBaseModTrain.append(RanForestMod.score(X_train, y_train))
RanForestBaseModTest.append(accuracy_score(y_test, RanForestMod.predict(X_test)))
RanForestBaseModPrecision.append(precision_score(y_test, RanForestMod.predict(X_test), average='binary'))
print(f'Base Random forest Model Completed for Cohort {Cohort_list[ii]}.')
modelCount += 1
print('==================================>')

####################################################################
# Building out the XGboost base model
####################################################################

print('==================================>')
XGboostBasMod = XGBClassifier(use_label_encoder=False, verbosity=0)
XGboostBasMod.fit(X_train, y_train * 1)
XGboostBasModTrain.append(XGboostBasMod.score(X_train, y_train))
XGboostBasModTest.append(accuracy_score(y_test, XGboostBasMod.predict(X_test)))
XGboostBaseModPrecision.append(precision_score(y_test, XGboostBasMod.predict(X_test), average='binary'))
XGboostBasMod.predict(X_train)


##################################################################
# Building the data set Macro added model                         #
##################################################################

YdataMacro = temp_data1[['has_canceled']]
XdataMacro = temp_data1[XvaluesMac]
gender_dummies = pd.get_dummies(temp_data1['gender'])
XdataMacro['female'] = gender_dummies.loc[:, 'female'].copy()
XdataMacro['male'] = gender_dummies.loc[:, 'male'].copy()
XdataMacro = XdataMacro.drop('gender', axis=1)
XMtrain, XMtest, ymtrain, ymtest = train_test_split(XdataMacro, np.array(YdataMacro),
                                                    test_size=.3)
# random_state = 12345)

##################################################################
# Building out the logistc Regression section                     #
##################################################################

print('==================================>')
logisticMacroMod = LogisticRegression()
logisticMacroMod.fit(XMtrain, ymtrain.ravel())
logisticMacModtrain.append(logisticMacroMod.score(XMtrain, ymtrain))
logisticMacModtest.append(accuracy_score(ymtest, logisticMacroMod.predict(XMtest)))
logisticMacModPrecision.append(precision_score(ymtest, logisticMacroMod.predict(XMtest), average='binary'))
print(f'Base Logistic Model Completed for Cohort {Cohort_list[ii]}.')
modelCount += 1
print('==================================>')

##################################################################
# Building out the logistc Regression section                     #
##################################################################

print('==================================>')
RanForestMacroMod = RandomForestClassifier()
RanForestMacroMod.fit(XMtrain, ymtrain.ravel())
RanForestMacModTrain.append(RanForestMacroMod.score(XMtrain, ymtrain))
RanForestMacModTest.append(accuracy_score(ymtest, RanForestMacroMod.predict(XMtest)))
RanForestMacModPrecision.append(precision_score(ymtest, RanForestMacroMod.predict(XMtest), average='binary'))
print(f'Base Random forest Model Completed for Cohort {Cohort_list[ii]}.')
modelCount += 1
print('==================================>')


####################################################################
# Building out the XGboost base model
####################################################################

print('==================================>')
XGboostMacroMod = XGBClassifier(use_label_encoder=False, verbosity=0)
XGboostMacroMod.fit(XMtrain, ymtrain * 1)
XGboostMacModTrain.append(XGboostMacroMod.score(XMtrain, ymtrain))
XGboostMacModTest.append(accuracy_score(ymtest, XGboostMacroMod.predict(XMtest)))
XGboostMacModPrecision.append(precision_score(ymtest, XGboostMacroMod.predict(XMtest), average='binary'))
print(f'Base XGboost Model Completed for Cohort {Cohort_list[ii]}.')
modelCount += 1
print('==================================>')


# Non-macro data
Y_data = np.array(temp_data1[['has_canceled']])
X_data = temp_data1[XvaluesNoMac].copy()
gender_dummies = pd.get_dummies(temp_data1['gender'])
X_data['female'] = gender_dummies.loc[:, 'female'].copy()
X_data['male'] = gender_dummies.loc[:, 'male'].copy()
X_data = X_data.drop('gender', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_data, np.array(Y_data),
                                                    test_size=.3)
# Macro Data
YdataMacro = np.array(temp_data1[['has_canceled']])
XdataMacro = temp_data1[XvaluesMac]
gender_dummies = pd.get_dummies(temp_data1['gender'])
XdataMacro['female'] = gender_dummies.loc[:, 'female'].copy()
XdataMacro['male'] = gender_dummies.loc[:, 'male'].copy()
XdataMacro = XdataMacro.drop('gender', axis=1)
XMtrain, XMtest, ymtrain, ymtest = train_test_split(XdataMacro, np.array(YdataMacro),
                                                    test_size=.3)

# for RANDOMforst classifer
# criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
# max_features{“sqrt”, “log2”, None}, int or float, default=”sqrt”
# n_estimatorsint, default=100
parameters_RGC = {'criterion': ['gini', 'entropy', 'log_loss'],
                  'max_features': ['sqrt', 'log2', None],
                  'n_estimators': [100, 500, 750, 50]}
RanForestBaseMod = RandomForestClassifier()
baseMod = GridSearchCV(RanForestBaseMod, parameters_RGC)
baseMod.fit(X_data, np.array(Y_data))
RanForestMacroMod = RandomForestClassifier()
MacroMod = GridSearchCV(RanForestMacroMod, parameters_RGC)
MacroMod.fit(XdataMacro, np.array(YdataMacro))
print('best parameters for basic logistic')
print(baseMod.best_params_)
print('best parameters for macro logistic')
print(MacroMod.best_params_)


# for XGboost
# It has 3 options - gbtree, gblinear or dart.
# eta =>range : [0,1]
# tree_method => Choices: auto, exact, approx, hist, gpu_hist
parameters_boost = {'booster': ['gbtree', 'gblinear', 'dart'],
                    'tree_method': ['auto', 'exact', 'approx', 'hist']}

XGboostMacroMod = XGBClassifier(use_label_encoder=False, verbosity=0)
baseMod = GridSearchCV(XGboostMacroMod, parameters_boost)
baseMod.fit(X_data, Y_data * 1)
RanForestMacroMod = RandomForestClassifier()
MacroMod = GridSearchCV(XGboostMacroMod, parameters_boost)
MacroMod.fit(XdataMacro, YdataMacro * 1)
print('best parameters for basic logistic')
print(baseMod.best_params_)
print('best parameters for macro logistic')
print(MacroMod.best_params_)


# for logistic regression
# solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
# penalty{‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
logisticBaseMod = LogisticRegression(penalty='l1', solver='liblinear')
logisticBaseMod.fit(X_train, np.array(y_train))
print('shit for logist mod')
print(accuracy_score(y_test, logisticBaseMod.predict(X_test)))
print(precision_score(y_test, logisticBaseMod.predict(X_test), average='binary'))
logisticMacroMod = LogisticRegression(penalty='l1', solver='liblinear')
logisticMacroMod.fit(XMtrain, np.array(ymtrain))
print('more stuff for logist mod with macro')
print(accuracy_score(ymtest, logisticMacroMod.predict(XMtest)))
print(precision_score(ymtest, logisticMacroMod.predict(XMtest), average='binary'))


logisticBaseMod = LogisticRegression(penalty='l1', solver='liblinear')
logisticBaseMod.fit(X_train, np.array(y_train))
print('shit for logist mod')
print(accuracy_score(y_test, logisticBaseMod.predict(X_test)))
print(precision_score(y_test, logisticBaseMod.predict(X_test), average='binary'))
logisticMacroMod = LogisticRegression(penalty='l1', solver='liblinear')
logisticMacroMod.fit(XMtrain, np.array(ymtrain))
print('more stuff for logist mod with macro')
print(accuracy_score(ymtest, logisticMacroMod.predict(XMtest)))
print(precision_score(ymtest, logisticMacroMod.predict(XMtest), average='binary'))


RanForestBaseMod = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=100)
RanForestBaseMod.fit(X_train, np.array(y_train))
print('shit for random forest mod')
print(accuracy_score(y_test, RanForestBaseMod.predict(X_test)))
print(precision_score(y_test, RanForestBaseMod.predict(X_test), average='binary'))

RanForestMacroMod = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=100)
RanForestMacroMod.fit(XMtrain, np.array(ymtrain))
print('more stuff for random forest mod with macro')
print(accuracy_score(ymtest, RanForestMacroMod.predict(XMtest)))
print(precision_score(ymtest, RanForestMacroMod.predict(XMtest), average='binary'))


XGboostBaseMod = XGBClassifier(use_label_encoder=False, verbosity=0, booster='gbtree', tree_method='approx')
XGboostBaseMod.fit(X_train, y_train * 1)
print('shit for XGBoost mod')
print(accuracy_score(y_test, XGboostBaseMod.predict(X_test)))
print(precision_score(y_test, XGboostBaseMod.predict(X_test), average='binary'))
XGboostMacroMod = XGBClassifier(use_label_encoder=False, verbosity=0, booster='gbtree', tree_method='approx')
XGboostMacroMod.fit(XMtrain, ymtrain * 1)
print('more stuff for XGboost mod with macro')
print(accuracy_score(ymtest, XGboostMacroMod.predict(XMtest)))
print(precision_score(ymtest, XGboostMacroMod.predict(XMtest), average='binary'))