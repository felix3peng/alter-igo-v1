global cc_dict, cm_dict

'''
CODE PAIRINGS
'''
# load data, no args
s1 = '''import pandas as pd
df = pd.read_csv('TIF_FIX_v0.1.csv', delimiter='|', header=0, low_memory=False)
print(df.head())'''
# summarize data, no args
s2 = '''print(df.describe())'''
# print feature names, no args
s3 = '''print(list(df.columns))'''
# set feature as target, 1 feature args
s4 = '''X = df.copy()
target_name = '{0}'
y = X.pop('{0}')
y.columns = [target_name]'''
# calculate corr between 2 features, 2 features args
s5 = '''corr = df['{0}'].corr(df['{1}'])
print('Correlation between {0} and {1}: ', corr)'''
# plot correlation heatmap, no args
s6 = '''import seaborn as sns
import numpy as np
mask = np.triu(np.ones_like(df.corr()))
sns.heatmap(df.corr(), xticklabels=df.columns, yticklabels=df.columns, mask=mask)'''
# plot histogram of feature, 1 feature args
s7 = '''df.hist(column='{0}')'''
# plot scatterplot of 2 features, 2 feature args
s8 = '''df.plot.scatter(x='{0}', y='{1}')'''
# print target name
s9 = '''print(list(y.columns))'''
# train test split data, 2 numbers args
s10 = '''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size={0})'''
# train xgboost model, no args
s11 = '''from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)
print('XGBRegressor parameters:')
print(model.get_xgb_params())'''
# train random forest model default, no args
s12 = '''from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
print('Random Forest Regressor parameters:')
print(model.get_params())'''
# train linear regression model, no args
s13 = '''from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print('Linear Regression parameters:')
print(model.get_params())'''
# get train and test r2 score, no args
s14 = '''from sklearn.metrics import r2_score
y_pred_train = model.predict(X_train)
model_r2_train = r2_score(y_train, y_pred_train)
y_pred_test = model.predict(X_test)
model_r2_test = r2_score(y_test, y_pred_test)
print("Train R2: ", model_r2_train)
print("Test R2: ", model_r2_test)'''
# get train and test MAE score, no args
s15 = '''from sklearn.metrics import mean_absolute_error
y_pred_train = model.predict(X_train)
model_mae_train = mean_absolute_error(y_train, y_pred_train)
y_pred_test = model.predict(X_test)
model_mae_test = mean_absolute_error(y_test, y_pred_test)
print("Train MAE: ", model_mae_train)
print("Test MAE: ", model_mae_test)'''
# get feature importance, no args
s16 = '''import matplotlib.pyplot as plt
import numpy as np
importances = model.feature_importances_
indices = np.argsort(importances)
plt.barh(range(X_train.shape[1]), importances[indices])
_ = plt.title('Feature Importances')
_ = plt.yticks(ticks=range(X_train.shape[1]), labels=X_train.columns[indices])'''
# plot shap feature importance, no args
s17 = '''import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)'''
# get shap feature interaction between 2 features, 2 feature args
s18 = '''import shap
explainer = shap.TreeExplainer(model)
interx_vals = explainer.shap_interaction_values(X_train)
shap.dependence_plot(('{0}', '{1}'), interx_vals, X_train, display_features=X_train)'''
# print r2, MAE, RMSE score for train and test, no args
s19 = '''from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
model_r2_train = r2_score(y_train, y_pred_train)
model_r2_test = r2_score(y_test, y_pred_test)
model_mae_train = mean_absolute_error(y_train, y_pred_train)
model_mae_test = mean_absolute_error(y_test, y_pred_test)
model_rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
model_rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
print("Train R2: ", model_r2_train)
print("Test R2: ", model_r2_test)
print("Train MAE: ", model_mae_train)
print("Test MAE: ", model_mae_test)
print("Train RMSE: ", model_rmse_train)
print("Test RMSE: ", model_rmse_test)'''
# train random forest model with given number of trees, 1 number arg
s20 = '''from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators={0})
model.fit(X_train, y_train)
print('Random Forest Regressor parameters:')
print(model.get_params())'''
# calculate train r2 score, no args
s21 = '''from sklearn.metrics import r2_score
y_pred = model.predict(X_train)
model_r2 = r2_score(y_train, y_pred)
print("Train R2: ", model_r2)'''
# calculate test r2 score, no args
s22 = '''from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
model_r2 = r2_score(y_test, y_pred)
print("Test R2: ", model_r2)'''
# calculate train MAE score, no args
s23 = '''from sklearn.metrics import mean_absolute_error
y_pred = model.predict(X_train)
model_mae = mean_absolute_error(y_train, y_pred)
print("Train MAE: ", model_mae)'''
# calculate test MAE score, no args
s24 = '''from sklearn.metrics import mean_absolute_error
y_pred = model.predict(X_test)
model_mae = mean_absolute_error(y_test, y_pred)
print("Test MAE: ", model_mae)'''
# calculate train and test RMSE, no args
s25 = '''from sklearn.metrics import mean_squared_error
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
model_rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
model_rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
print("Train RMSE: ", model_rmse_train)
print("Test RMSE: ", model_rmse_test)'''
# calculate train RMSE, no args
s26 = '''from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_train)
model_rmse = mean_squared_error(y_train, y_pred, squared=False)
print("Train RMSE: ", model_rmse)'''
# calculate test RMSE, no args
s27 = '''from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
model_rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Test RMSE: ", model_rmse)'''
# get shape of data, no args
s28 = '''print(df.shape)'''
# get number of rows, no args
s29 = '''print(len(df))'''
# get number of columns, no args
s30 = '''print(df.shape[1])'''
# remove a feature, 1 feature args
s31 = '''X.drop(columns=['{0}'], inplace=True)'''
# remove features, 2 feature args
s32 = '''X.drop(columns=['{0}', '{1}'], inplace=True)'''
# add a log of a feature, 1 feature args
s33 = '''import numpy as np
X['{1}'] = np.log(X['{0}'])'''
# show X
s34 = '''print(X.head())'''
# show X_train
s35 = '''print(X_train.head())'''
# show X_test
s36 = '''print(X_test.head())'''
# show y
s37 = '''print(y.head())'''
# show y_train
s38 = '''print(y_train.head())'''
# show y_test
s39 = '''print(y_test.head())'''
# add a log feature of every feature
s40 = '''import numpy as np
Xlog = X.apply(np.log)
Xlog.columns = ['LOG_' + x for x in X.columns]
X = pd.concat([X, Xlog], axis=1)'''
# add a multiplicative interaction column of two features
s41 = '''X['{2}'] = X['{0}'] * X['{1}']'''
# show columns of X
s42 = '''print(list(X.columns))'''
# create decile of feature
s43 = '''temp = X['{0}'].replace(0, np.nan)
X['{0}_DECILE'] = pd.qcut(temp, 10, labels=False)
X['{0}_DECILE'] += 1
X['{0}_DECILE'].replace(np.nan, 0, inplace=True)
print(X['{0}_DECILE'].value_counts().sort_index())'''
# create deciles for all trx columns
s44 = '''for col in X.columns:
    if 'trx' in col.lower():
        temp = X[col].replace(0, np.nan)
        X[col + '_DECILE'] = pd.qcut(temp, 10, labels=False)
        X[col + '_DECILE'] += 1
        X[col + '_DECILE'].replace(np.nan, 0, inplace=True)
        print(X[col + '_DECILE'].value_counts().sort_index())'''
# create deciles for all nbrx columns
s45 = '''for col in X.columns:
    if 'nbrx' in col.lower():
        temp = X[col].replace(0, np.nan)
        X[col + '_DECILE'] = pd.qcut(temp, 10, labels=False)
        X[col + '_DECILE'] += 1
        X[col + '_DECILE'].replace(np.nan, 0, inplace=True)
        print(X[col + '_DECILE'].value_counts().sort_index())'''
# generate a count for each product


'''
COMMAND-CODE DICTIONARY (BASE COMMANDS)
'''
cc_dict = {'load data': s1,
           'summarize data': s2,
           'get feature names': s3,
           'set feature as target': s4,
           'get correlation between f1 and f2': s5,
           'show correlation heatmap': s6,
           'plot histogram of feature': s7,
           'show scatter plot of f1 and f2': s8,
           'get target name': s9,
           'train test split of given ratio': s10,
           'train an xgboost model': s11,
           'train a random forest model': s12,
           'train a linear regression model': s13,
           'calculate r2 score': s14,
           'calculate mae score': s15,
           'show the feature importance': s16,
           'show shap feature importances': s17,
           'show shap interaction between f1 and f2': s18,
           'calculate model performance': s19,
           'train a random forest model with x trees': s20,
           'calculate r2 score on train': s21,
           'calculate r2 score on test': s22,
           'calculate mae score on train': s23,
           'calculate mae score on test': s24,
           'calculate rmse score': s25,
           'calculate rmse score on train': s26,
           'calculate rmse score on test': s27,
           'get shape of data': s28,
           'how many rows are there': s29,
           'how many columns are there': s30,
           'drop feature': s31,
           'drop feature and feature': s32,
           'create new feature, log of feat and name it new_feat': s33,
           'print x': s34,
           'show x_train': s35,
           'show x_test': s36,
           'show me y': s37,
           'show y_train': s38,
           'show y_test': s39,
           'create new features, log of every feature': s40,
           'create new feature, product of feat and feat and call it new_feat': s41,
           'what are the columns of x': s42}

'''
COMMAND-COMMAND DICTIONARY
'''
cm_dict = {'load the data': 'load data',
           'load HCP dataset': 'load data',
           'load data from file': 'load data',
           'load data from file named x': 'load data',
           'show summary of the data': 'summarize data',
           'summarize the data': 'summarize data',
           'describe the data': 'summarize data',
           'what are the features': 'get feature names',
           'what are the columns': 'get feature names',
           'get column names': 'get feature names',
           'correlation of f1 and f2': 'get correlation between f1 and f2',
           'what is the correlation between f1 and f2': 'get correlation between f1 and f2',
           'what is the correlation of f1 and f2': 'get correlation between f1 and f2',
           'calculate correlation between f1 and f2': 'get correlation between f1 and f2',
           'calculate correlation of f1 and f2': 'get correlation between f1 and f2',
           'heatmap of correlations': 'show correlation heatmap',
           'calculate all pairwise correlations': 'show correlation heatmap',
           'show density of feature': 'plot histogram of feature',
           'show distribution of feature': 'plot histogram of feature',
           'distribution of f1 and f2': 'show scatterplot of f1 and f2',
           'what is the target': 'get target name',
           'do a ratio ratio train test split': 'train test split of given ratio',
           'xgboost model': 'train an xgboost model',
           'train a rf model': 'train a random forest model',
           'random forest': 'train a random forest model',
           'train a linear model': 'train a linear regression model',
           'calculate r2 on train and test': 'calculate r2 score',
           'show model r2': 'calculate r2 score',
           'what is the r2 score': 'calculate r2 score',
           'what is the mae score': 'calculate mae score',
           'calculate mae on train and test': 'calculate mae score',
           'show model mae': 'calculate mae score',
           'what is the feature importance': 'show the feature importance',
           'what is the shap feature importance': 'show shap feature importances',
           'shap interaction plot between f1 and f2': 'show shap interaction between f1 and f2',
           'what is the shap interaction of f1 and f2': 'show shap interaction between f1 and f2',
           'caclulate performance of model on train and test': 'calculate model performance',
           'what is the performance of rf': 'calculate model performance',
           'what is the performance of linear reg': 'calculate model performance',
           'what is the performance of xgboost': 'calculate model performance',
           'how good is the model': 'calculate model performance',
           'what is the model accuracy': 'calculate model performance',
           'random forest with x trees': 'train a random forest model with x trees',
           'r2 train': 'calculate r2 score on train',
           'r2 test': 'calculate r2 score on test',
           'mae train': 'calculate mae score on train',
           'mae test': 'calculate mae score on test',
           'rmse': 'calculate rmse score',
           'what is the rmse': 'calculate rmse score',
           'get rmse score': 'calculate rmse score',
           'show model rmse': 'calculate rmse score',
           'calculate rmse on train and test': 'calculate rmse score',
           'rmse train': 'calculate rmse score on train',
           'what is rmse on train': 'calculate rmse score on train',
           'rmse test': 'calculate rmse score on test',
           'what is rmse on test': 'calculate rmse score on test',
           'what is the shape of the data': 'get shape of data',
           'number of rows': 'how many rows are there',
           'number of records': 'how many rows are there',
           'how many records are there': 'how many rows are there',
           'how many features are there': 'how many columns are there',
           'number of columns': 'how many columns are there',
           'number of features': 'how many columns are there',
           'delete feature': 'drop feature',
           'remove feature': 'drop feature',
           'remove feature and feature': 'drop feature and feature',
           'delete feature and feature': 'drop feature and feature',
           'take log of feat and call it new_feat': 'create new feature, log of feat and name it new_feat',
           'show me x': 'print x',
           'show x': 'print x',
           'show y': 'show me y',
           'print y': 'show me y',
           'take the log of every feature': 'create new features, log of every feature',
           'take product of feat and feat and call it new_feat': 'create new feature, product of feat and feat and call it new_feat',
           'multiply feat and feat and call it new_feat': 'create new feature, product of feat and feat and call it new_feat',
           'column names of x': 'what are the columns of x',
           'features of x': 'what are the columns of x'}
