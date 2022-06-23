global cc_dict, cm_dict

'''
CODE PAIRINGS
'''
# load boston data
s0 = '''import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target
print(df.head())'''
# load data, no args
s1 = '''import pandas as pd
import os
from os import path
cwd = os.getcwd()
df = pd.read_csv(path.join(cwd, 'TIF_FIX_v0.1.txt'), delimiter='|', header=0, dtype=str)
numerical_cols = ['NBRX', 'TRX', 'DECILE', 'APPROVED', 'RATING', 'PERC', 'CALLS', 'PDES']
for col in df.columns:
    if any(x in col for x in numerical_cols):
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
    if 'PROPORTION' in col:
        df[col] = df[col].str.replace('%', '').astype(float)/100
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
s31 = '''df.drop(columns=['{0}'], inplace=True)'''
# remove features, 2 feature args
s32 = '''df.drop(columns=['{0}', '{1}'], inplace=True)'''
# add a log of a feature, 1 feature args
s33 = '''import numpy as np
df['{1}'] = np.log(df['{0}'])'''
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
dflog = df.apply(np.log)
dflog.columns = ['LOG_' + x for x in df.columns]
df = pd.concat([df, dflog], axis=1)'''
# add a multiplicative interaction column of two features
s41 = '''df['{2}'] = df['{0}'] * df['{1}']'''
# show columns of X
s42 = '''print(list(X.columns))'''
# create decile of LOLO NBRX
s43 = '''import numpy as np
sums = df['LOLO_NBRX_52WK'].replace(0, np.nan).sort_values(ascending=True).cumsum()
sums_dec = np.floor(sums/(df['LOLO_NBRX_52WK'].sum()/10))
sums_dec[sums_dec == 10] = 9
sums_dec += 1
sums_dec.replace(np.nan, 0, inplace=True)
df['LOLO_NBRX_52WK_DECILE'] = sums_dec.sort_index(ascending=True)
for i in range(0, 11):
    print('Decile', i, ':', df['LOLO_NBRX_52WK_DECILE'].value_counts()[i])'''
# create decile of ORIAHNN NBRX
s44 = '''import numpy as np
sums = df['ORIAHNN_NBRX_LTD'].replace(0, np.nan).sort_values(ascending=True).cumsum()
sums_dec = np.floor(sums/(df['ORIAHNN_NBRX_LTD'].sum()/10))
sums_dec[sums_dec == 10] = 9
sums_dec += 1
sums_dec.replace(np.nan, 0, inplace=True)
df['ORIAHNN_NBRX_LTD_DECILE'] = sums_dec.sort_index(ascending=True)
for i in range(0, 11):
    print('Decile', i, ':', df['LOLO_NBRX_52WK_DECILE'].value_counts()[i])'''
# generate a count for each product
s45 = '''product_types = ['LOLO', 'ORIAHNN', 'ORILISSA']
for p in product_types:
    print(df[df['APPROVED_' + p] == 1].shape[0], 'approved for', p)'''
# check how many null entries are in IMS_NUMBER
s46 = '''print(df['IMS_NUMBER'].isnull().sum(), 'null values in IMS_NUMBER')'''
# remove records with null IMS_NUMBER
s47 = '''df = df[df['IMS_NUMBER'].notnull()]'''
# check how many null entries are in ZIP
s48 = '''print(df['ZIP'].isnull().sum(), 'null values in ZIP')'''
# remove records with null ZIP
s49 = '''df = df[df['ZIP'].notnull()]'''
# check how many null entries are in STANDARD_FIRST_NAME or STANDARD_LAST_NAME
s50 = '''mask = df['STANDARD_FIRST_NAME'].isnull() | df['STANDARD_LAST_NAME'].isnull()
print(df[mask].shape[0], 'records with null STANDARD_FIRST_NAME or STANDARD_LAST_NAME')'''
# remove records with null STANDARD_FIRST_NAME or STANDARD_LAST_NAME
s51 = '''mask = df['STANDARD_FIRST_NAME'].isnull() | df['STANDARD_LAST_NAME'].isnull()
df = df[~mask]'''
# how many records have a null specialty
s52 = '''print(df['ABBOTT_BEST_SPECIALTY_CODE'].isnull().sum(), 'null entries for ABBOTT_BEST_SPECIALTY_CODE')'''
# remove records with null specialty
s53 = '''df = df[df['ABBOTT_BEST_SPECIALTY_CODE'].notnull()]'''
# check if the length of IMS_NUMBER is the same for every record
s54 = '''if df[df['IMS_NUMBER'].notnull()]['IMS_NUMBER'].astype(str).str.len().nunique() == 1:
    print('IMS_NUMBER is the same length for every record')
else:
    print('IMS_NUMBER is not the same length for every record')'''
# check if the length of ZIP is the same for every record
s55 = '''if df[df['ZIP'].notnull()]['ZIP'].astype(str).str.len().nunique() == 1:
    print('ZIP is the same length for every record')
else:
    print('ZIP is not the same length for every record')'''
# maximum number of addresses
s56 = '''dupes = df.duplicated(subset=['ABBOTT_CUSTOMER_ID'])
nondupe_addr = df[~dupes][['ABBOTT_CUSTOMER_ID', 'ADDR1', 'ADDR2']]
dupe_addr = df[dupes][['ABBOTT_CUSTOMER_ID', 'ADDR1', 'ADDR2']]
merged = nondupe_addr.merge(dupe_addr, on='ABBOTT_CUSTOMER_ID', how='outer')
print('Maximum number of addresses for an HCP is:', (merged.count(axis=1) - 1).max())'''
# top 10 prescribers of LOLO by state
s57 = '''top10bystate = pd.DataFrame(index=np.arange(0, 10), columns=df['ST'].unique())
for s in df['ST'].unique():
    flname = df[df['ST'] == s].sort_values(by='LOLO_NBRX_52WK', ascending=False).drop_duplicates(subset=['ABBOTT_CUSTOMER_ID']).reset_index(drop=True)[:10]
    top10bystate[s] = (flname['STANDARD_FIRST_NAME'] + ' ' + flname['STANDARD_LAST_NAME'])
print(top10bystate)'''
# top 10 prescribers of LOLO by territory
s58 = '''top10byterritory = pd.DataFrame(index=np.arange(0, 10), columns=df['TERRITORY_NUMBER_WH3'].unique())
for t in df['TERRITORY_NUMBER_WH3'].unique():
    flname = df[df['TERRITORY_NUMBER_WH3'] == t].sort_values(by='LOLO_NBRX_52WK', ascending=False).drop_duplicates(subset=['ABBOTT_CUSTOMER_ID']).reset_index(drop=True)[:10]
    top10byterritory[t] = (flname['STANDARD_FIRST_NAME'] + ' ' + flname['STANDARD_LAST_NAME'])
print(top10byterritory)'''
# Histogram by product by priority
s59 = '''import matplotlib.pyplot as plt
priority_cols = ['LOLO_PRIORITY', 'ORIAHNN_PRIORITY', 'ORIL_PRIORITY']
fig, ax = plt.subplots(1, len(priority_cols), figsize=(20, 5))
for i, feature in enumerate(df[priority_cols]):
    df[feature].value_counts()[['H', 'M', 'L', 'VL']].plot(kind="bar", ax=ax[i]).set_title(feature)'''
# Venn diagram of HCP approvals by product
s60 = '''import matplotlib_venn as v
lolo_hcp = set(df[df['FINAL_PRIORIITY'].notnull() & df['APPROVED_LOLO'] == 1.0]['ABBOTT_CUSTOMER_ID'])
oriahnn_hcp = set(df[df['FINAL_PRIORIITY'].notnull() & df['APPROVED_ORIAHNN'] == 1.0]['ABBOTT_CUSTOMER_ID'])
orilissa_hcp = set(df[df['FINAL_PRIORIITY'].notnull() & df['APPROVED_ORILISSA'] == 1.0]['ABBOTT_CUSTOMER_ID'])
v.venn3([lolo_hcp, oriahnn_hcp, orilissa_hcp], set_labels=('LOLO', 'ORIAHNN', 'ORILISSA'))'''
# Co-relation of writers vs priority
s61 = '''priority_coded_lolo = df['LOLO_PRIORITY'].replace(['H', 'M', 'L', 'VL'], [4, 3, 2, 1])
print('Correlation between LOLO writers and priority:', df['APPROVED_LOLO'].corr(priority_coded_lolo))
priority_coded_oriahnn = df['ORIAHNN_PRIORITY'].replace(['H', 'M', 'L', 'VL'], [4, 3, 2, 1])
print('Correlation between ORIAHNN writers and priority:', df['APPROVED_ORIAHNN'].corr(priority_coded_oriahnn))
priority_coded_orilissa = df['ORIL_PRIORITY'].replace(['H', 'M', 'L', 'VL'], [4, 3, 2, 1])
print('Correlation between ORILISSA writers and priority:', df['APPROVED_ORILISSA'].corr(priority_coded_orilissa))'''
# Which metric is a better indicator of writers (Medically treated over Diagnosed)?
s62 = '''med_trx_corr = df[df['LOLO_TRX_52WK'] != 0]['LOLO_TRX_52WK'].corr(df[df['LOLO_TRX_52WK'] != 0]['PROPORTION_OF_MED_TREATED'])
diag_trx_corr = df[df['LOLO_TRX_52WK'] != 0]['LOLO_TRX_52WK'].corr(df[df['LOLO_TRX_52WK'] != 0]['UF_IN_OFFICE_DIAGNOSED_PAT_DECILE'])
print('Medication treatment correlation:', med_trx_corr)
print('Diagnosis treatment correlation:', diag_trx_corr)'''
# find Kaiser HCPs
s63 = '''kaiser_hcps = df[df['IN_KAISER'] == 1]
print('Number of Kaiser HCPs:', kaiser_hcps.shape[0])'''
# find DEA revoked HCPs
s64 = '''dea_revoked_hcps = df[df['DEA_REVOKED'] == 1]
print('Number of DEA revoked HCPs:', dea_revoked_hcps.shape[0])'''
#Access by state (highly accessible state vs least accessible)
s65 = '''access = df.groupby('ST')['AM_NO_SEE_RATING'].mean().sort_values(ascending=False)
print(access)'''
# Average call activity, access monitor at Priority level
s66 = '''call_activity = df.groupby('LOLO_PRIORITY')['ANNUAL_CALL_FREQ_PERC_50'].mean().sort_values(ascending=False)
print(call_activity)'''
# Distribution of HCPs by TAP feedback (current/prior)
s67 = '''df[df['ACCESSIBILITY_FEEDBACK'] != '-']['ACCESSIBILITY_FEEDBACK'].value_counts().plot(kind="bar").set_title('ACCESSIBLITY_FEEDBACK')'''
# HCPs with HR compliance issue
s68 = '''print('Number of HCPs with HR compliance issue:', df[df['HR_COMPLIANCE_RMV'].notnull()].shape[0])'''
# Ability to create cross-tabs on any 2 metrics
s69 = '''print(pd.crosstab(df['LOLO_NBRX_DECILE'], df['ORILISSA_NBRX_DECILE']))'''
# Correlation b/w Oriahnn and Orilissa writers
s70 = '''print('Correlation between ORIAHNN writers and ORILISSA writers:', df['APPROVED_ORIAHNN'].corr(df['APPROVED_ORILISSA']))'''
# Average LOLO TRx at Priority level
s71 = '''lolo_priority_trx = df.groupby('LOLO_PRIORITY')['LOLO_TRX_52WK'].mean()
print(lolo_priority_trx)'''
#Range of TRx, NBRx at Priority level
s72 = '''priorities = ['H', 'M', 'L', 'VL']
maxvals_trx = df.groupby('LOLO_PRIORITY')['LOLO_TRX_52WK'].max()[priorities].values
minvals_trx = df.groupby('LOLO_PRIORITY')['LOLO_TRX_52WK'].min()[priorities].values
for i in range(len(priorities)):
    print('Range of TRx for', priorities[i],'priority:', minvals_trx[i], '-', maxvals_trx[i])
maxvals_nbrx = df.groupby('LOLO_PRIORITY')['LOLO_NBRX_52WK'].max()[priorities].values
minvals_nbrx = df.groupby('LOLO_PRIORITY')['LOLO_NBRX_52WK'].min()[priorities].values
for i in range(len(priorities)):
    print('Range of NBRx for', priorities[i],'priority:', minvals_nbrx[i], '-', maxvals_nbrx[i])'''
# Average call activity, access monitor at Orilissa priority level
s73 = '''call_activity = df.groupby('ORIL_PRIORITY')['ANNUAL_CALL_FREQ_PERC_50'].mean().sort_values(ascending=False)
print(call_activity)'''
# Average call activity, access monitor at Oriahnn priority level
s74 = '''call_activity = df.groupby('ORIAHNN_PRIORITY')['ANNUAL_CALL_FREQ_PERC_50'].mean().sort_values(ascending=False)
print(call_activity)'''


'''
COMMAND-CODE DICTIONARY (BASE COMMANDS)
'''
cc_dict = {'load boston data': s0,
           'load HCP data': s1,
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
           'what are the columns of x': s42,
           'create a decile column for LOLO NBRx': s43,
           'create a decile column for ORIAHNN NBRx': s44,
           'generate a count for each product': s45,
           'check how many null IMS_NUMBER exist': s46,
           'remove null IMS_NUMBER': s47,
           'check how many null ZIP exist': s48,
           'remove null ZIP': s49,
           'check how many null first or last name exist': s50,
           'remove records with null first or last name': s51,
           'check how many null specialty exist': s52,
           'remove records with null specialty': s53,
           'check if length of IMS_NUMBER is the same for all records': s54,
           'check if length of ZIP is the same for all records': s55,
           'maximum number of addresses': s56,
           'top 10 prescribers of LOLO by state': s57,
           'top 10 prescribers of LOLO by territory': s58,
           'create histogram by product by priority': s59,
           'create venn diagram of HCPs by product': s60,
           'correlation of writers by priority': s61,
           'is medically treated or diagnosed a better indicator of writers': s62,
           'find kaiser affiliated HCPs': s63,
           'find DEA revoked HCPs': s64,
           'accessibility by state': s65,
           'average call activity by priority level for lolo': s66,
           'distribution of HCPs by TAP feedback': s67,
           'HCPs with HR compliance issue': s68,
           'ability to create cross-tabs on any 2 metrics': s69,
           'correlation between Oriahnn and Orilissa writers': s70,
           'average LOLO TRx at Priority level': s71,
           'range of LOLO TRx, NBRx at Priority level': s72,
           'average call activity by priority level for orilissa': s73,
           'average call activity by priority level for oriahnn': s74}

'''
COMMAND-COMMAND DICTIONARY
'''
cm_dict = {'load boston data': 'load boston data',
           'load housing data': 'load boston data',
           'load boston housing dataset': 'load boston data',
           'load data': 'load HCP data',
           'load hcp dataset': 'load HCP data',
           'load data from file': 'load HCP data',
           'load data from file named x': 'load HCP data',
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
           'features of x': 'what are the columns of x',
           'add a decile for lolo nbrx': 'create a decile column for LOLO NBRx',
           'add a decile for oriahnn nbrx': 'create a decile column for ORIAHNN NBRx',
           'count hcps by product': 'generate a count for each product',
           'hcps distributed by product': 'generate a count for each product',
           'null entries for ims number': 'check how many null IMS_NUMBER exist',
           'records with empty ims number': 'check how many null IMS_NUMBER exist',
           'delete entries with empty ims number': 'remove null IMS_NUMBER',
           'null entries for zip': 'check how many null ZIP exist',
           'records with empty zip': 'check how many null ZIP exist',
           'delete entries with empty zip': 'remove null ZIP',
           'null entries for first name or last name': 'check how many null first or last name exist',
           'records with empty first name or last name': 'check how many null first or last name exist',
           'delete entries with empty first name or last name': 'remove records with null first or last name',
           'null entries for specialty': 'check how many null specialty exist',
           'records with empty specialty': 'check how many null specialty exist',
           'delete entries with empty specialty': 'remove records with null specialty',
           'is ims number the same length for all records?': 'check if length of IMS_NUMBER is the same for all records',
           'is ims number length consistent': 'check if length of IMS_NUMBER is the same for all records',
           'is zip the same length for all records?': 'check if length of ZIP is the same for all records',
           'is zip length consistent': 'check if length of ZIP is the same for all records',
           'what is the maximum number of addresses an hcp has on file': 'maximum number of addresses',
           'do any hcps have multiple addresses on file': 'maximum number of addresses',
           'show the top 10 prescribers of lolo by state': 'top 10 prescribers of LOLO by state',
           'show the top 10 prescribers of lolo by territory': 'top 10 prescribers of LOLO by territory',
           'plot a histogram for each product priority': 'create histogram by product by priority',
           'plot histogram by product by priority': 'create histogram by product by priority',
           'plot venn diagram of hcps by product': 'create venn diagram of HCPs by product',
           'venn diagram by product': 'create venn diagram of HCPs by product',
           'correlation of writers by priority': 'correlation of writers by priority',
           'correlation between writers and product priority': 'correlation of writers by priority',
           'medically treated vs diagnosed to indicate writers': 'is medically treated or diagnosed a better indicator of writers',
           'is medical treatment or diagnoses a better indicator of writers': 'is medically treated or diagnosed a better indicator of writers',
           'how many hcps are affiliated with kaiser': 'find kaiser affiliated HCPs',
           'find kaiser affiliated hcps': 'find kaiser affiliated HCPs',
           'how many hcps are dea revoked': 'find DEA revoked HCPs',
           'find dea revoked hcps': 'find DEA revoked HCPs',
           'show accessibility by state': 'accessibility by state',
           'access by state': 'accessibility by state',
           'average call activity by lolo priority': 'average call activity by priority level for lolo',
           'what is the average call activity by lolo priority': 'average call activity by priority level for lolo',
           'distribution of hcps by tap feedback': 'distribution of HCPs by TAP feedback',
           'create a histogram of hcps by tap feedback': 'distribution of HCPs by TAP feedback',
           'how many hcps have hr compliance issues': 'HCPs with HR compliance issue',
           'hcps with hr compliance issues': 'HCPs with HR compliance issue',
           'cross tabs of lolo and orilissa nbrx': 'ability to create cross-tabs on any 2 metrics',
           'show me the cross tabs of lolo and orilissa nbrx': 'ability to create cross-tabs on any 2 metrics',
           'correlation between oriahnn and orilissa writers': 'correlation between Oriahnn and Orilissa writers',
           'what is the correlation between oriahnn and orilissa writers': 'correlation between Oriahnn and Orilissa writers',
           'average lolo trx by priority': 'average LOLO TRx at Priority level',
           'what is the average lolo trx by priority': 'average LOLO TRx at Priority level',
           'range of lolo trx and nbrx by priority': 'range of LOLO TRx, NBRx at Priority level',
           'what is the range of lolo trx and nbrx by priority': 'range of LOLO TRx, NBRx at Priority level',
           'average call activity by orilissa priority': 'average call activity by priority level for orilissa',
           'what is the average call activity by orilissa priority': 'average call activity by priority level for orilissa',
           'average call activity by oriahnn priority': 'average call activity by priority level for oriahnn',
           'what is the average call activity by oriahnn priority': 'average call activity by priority level for oriahnn'}
