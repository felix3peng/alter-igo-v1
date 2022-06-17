# import libraries and helper files
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)
from flask import Flask, Blueprint, flash, g, redirect, render_template
from flask import request, session, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
import sqlite3
import openai
import inspect
from itertools import groupby
from subprocess import Popen, PIPE
from io import StringIO, BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import base64
import sys
import re
from datetime import datetime
from PIL import Image
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')
from openai.embeddings_utils import get_embeddings, distances_from_embeddings
from openai.embeddings_utils import get_embedding, cosine_similarity
import pickle
import shap


# global declarations
global cc_dict, numtables, numplots

'''
COMMAND-CODE DICTIONARY
'''
# load data, no args
s1 = '''import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
df_X = pd.DataFrame(boston.data)
df_X.columns = boston.feature_names
df_y = pd.DataFrame(boston.target)
df_y.columns = ['MEDV']
df = pd.concat([df_X, df_y], axis=1)
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
# train logistic regression model, no args
s13 = '''from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print('Logistic Regression parameters:')
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

cc_dict = {'load data': s1,
           'summarize data': s2,
           'describe data': s2,
           'get feature names': s3,
           'what are the features': s3,
           'what are the columns': s3,
           'get column names': s3,
           'set feature as target': s4,
           'get correlation between f1 and f2': s5,
           'correlation of f1 and f2': s5,
           'what is the correlation between f1 and f2': s5,
           'what is the correlation of f1 and f2': s5,
           'calculate correlation between f1 and f2': s5,
           'calculate correlation of f1 and f2': s5,
           'show correlation heatmap': s6,
           'heatmap of correlations': s6,
           'plot histogram of feature': s7,
           'show density of feature': s7,
           'show distribution of feature': s7,
           'show scatter plot of f1 and f2': s8,
           'distribution of f1 and f2': s8,
           'get target name': s9,
           'what is the target': s9,
           'train test split of given ratio': s10,
           'do a ratio ratio train test split': s10,
           'train an xgboost model': s11,
           'xgboost model': s11,
           'train a random forest model': s12,
           'train a rf model': s12,
           'random forest': s12,
           'train a logistic regression model': s13,
           'train a log model': s13,
           'calculate r2 score': s14,
           'calculate r2 on train and test': s14,
           'show model r2': s14,
           'what is the r2 score': s14,
           'calculate mae score': s15,
           'what is the mae score': s15,
           'calculate mae on train and test': s15,
           'show model mae': s15,
           'show the feature importance': s16,
           'what is the feature importance': s16,
           'show shap feature importances': s17,
           'what is the shap feature importance': s17,
           'show shap interaction between f1 and f2': s18,
           'shap interaction plot between f1 and f2': s18,
           'what is the shap interaction of f1 and f2': s18,
           'calculate model performance': s19,
           'caclulate performance of model on train and test': s19,
           'what is the performance of rf': s19,
           'what is the performance of log reg': s19,
           'what is the performance of xgboost': s19,
           'how good is the model': s19,
           'what is the model accuracy': s19,
           'train a random forest model with x trees': s20,
           'random forest with x trees': s20,
           'calculate r2 score on train': s21,
           'r2 train': s21,
           'calculate r2 score on test': s22,
           'r2 test': s22,
           'calculate mae score on train': s23,
           'mae train': s23,
           'calculate mae score on test': s24,
           'mae test': s24,
           'calculate rmse score': s25,
           'what is the rmse': s25,
           'get rmse score': s25,
           'show model rmse': s25,
           'calculate rmse on train and test': s25,
           'calculate rmse score on train': s26,
           'rmse train': s26,
           'what is rmse on train': s26,
           'calculate rmse score on test': s27,
           'rmse test': s27,
           'what is rmse on test': s27,
           'get shape of data': s28,
           'what is the shape of the data': s28,
           'how many rows are there': s29,
           'number of rows': s29,
           'how many records are there': s29,
           'number of records': s29,
           'how many columns are there': s30,
           'how many features are there': s30,
           'number of columns': s30,
           'number of features': s30,
           'drop feature': s31,
           'delete feature': s31,
           'remove feature': s31,
           'drop feature and feature': s32,
           'remove feature and feature': s32,
           'delete feature and feature': s32,
           'create new feature, log of feat and name it new_feat': s33,
           'take log of feat and call it new_feat': s33,
           'print x': s34,
           'show me x': s34,
           'show x': s34,
           'show x_train': s35,
           'show x_test': s36,
           'show me y': s37,
           'show y': s37,
           'print y': s37,
           'show y_train': s38,
           'show y_test': s39,
           'create new features, log of every feature': s40,
           'take the log of every feature': s40,
           'create new feature, product of feat and feat and call it new_feat': s41,
           'take product of feat and feat and call it new_feat': s41,
           'multiple feat and feat and call it new_feat': s41,
           'what are the columns of x': s42,
           'column names of x': s42,
           'features of x': s42}

'''
EMBEDDINGS
'''
cache_path = 'embeddings_cache.pkl'
try:
    embedding_cache = pd.read_pickle(cache_path)
    print('cache file located, reading...')
    if len(embedding_cache) != len(cc_dict):
        print('outdated cache file, re-calculating embeddings...')
        # if cache doesn't have the right number of embeddings, re-run
        embedding_cache = get_embeddings(list(cc_dict.keys()),
                                         engine="text-similarity-davinci-001")
        with open(cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
            print('successfully dumped embeddings to cache')
    else:
        print('successfully loaded cached embeddings')
except FileNotFoundError:
    print('cache file not found, creating new cache')
    embedding_cache = get_embeddings(list(cc_dict.keys()),
                                     engine="text-similarity-davinci-001")
    with open(cache_path, "wb") as embedding_cache_file:
        pickle.dump(embedding_cache, embedding_cache_file)
        print('successfully dumped embeddings to cache')


'''
HELPER FUNCTIONS
'''
# store normal stdout in variable for reference
old_stdout = sys.stdout

# initialize dictionary for storing variables generated by code
ldict = {}
numtables = 0
numplots = 0


# helper function for running code stored in dictionary
# passing on KeyErrors when re-running due to column drop errors
def runcode(text, args=None):
    global numtables, numplots
    # turn off plotting and run function, try to grab fig and save in buffer
    tldict = ldict.copy()
    plt.ioff()
    if args is None:
        try:
            exec(cc_dict[text], tldict)
        except:
            print('something went wrong. ensure target & train-test split set')
    elif len(args) == 1:
        try:
            exec(cc_dict[text].format(args[0]), tldict)
        except:
            print('something went wrong. ensure target & train-test split set')
    else:
        try:
            exec(cc_dict[text].format(*args), tldict)
        except:
            print('something went wrong. ensure target & train-test split set')
    fig = plt.gcf()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close()
    p = Image.open(buf)
    x = np.array(p.getdata(), dtype=np.uint8).reshape(p.size[1], p.size[0], -1)
    # if min and max colors are the same, it wasn't a plot - re-run as string
    if np.min(x) == np.max(x):
        new_stdout = StringIO()
        sys.stdout = new_stdout
        if args is None:
            try:
                exec(cc_dict[text], ldict)
            except:
                print('something went wrong. ensure target & train-test split set')
        elif len(args) == 1:
            try:
                exec(cc_dict[text].format(args[0]), ldict)
            except KeyError:
                pass
            except:
                print('something went wrong. ensure target & train-test split set')
        else:
            try:
                exec(cc_dict[text].format(*args), ldict)
            except KeyError:
                pass
            except:
                print('something went wrong. ensure target & train-test split set')
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        # further parsing to determine if plain string or dataframe
        if bool(re.search(r'[\s]{3,}', output)):
            outputtype = 'dataframe'
            temp_df = pd.read_csv(StringIO(output), delim_whitespace=True)
            if '[' in str(temp_df.index[-1]):
                temp_df.drop(temp_df.tail(1).index, inplace=True)
            output = temp_df.to_html(classes='table', table_id='table'+str(numtables), max_cols=500)
            numtables += 1
        else:
            outputtype = 'string'
        return [outputtype, output]
    # if it was a plot, then output as HTML image from buffer
    else:
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        output = "<img id='image{0}' src='data:image/png;base64,{1}'/>".format(numplots, data)
        outputtype = 'image'
        numplots += 1
        ldict.update(tldict)
        return [outputtype, output]


# log results to db
def log_commands(outputs):
    # unpack outputs into variables
    _, cmd, code, _ = outputs
    feedback = 'none'
    dt = str(datetime.now())
    record = Log(dt, cmd, code, feedback)
    db.session.add(record)
    db.session.commit()
    return record.id


'''
FLASK APPLICATION CODE & ROUTES
'''
# set up flask application
app = Flask(__name__)
app.config.update(
    TESTING=True,
    SECRET_KEY='its-a-secret'
)

# set up database connection for log
db_name = 'log.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_name
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

# create a class for the table in db
class Log(db.Model):
    __tablename__ = 'log'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(100))
    command = db.Column(db.String(1000))
    codeblock = db.Column(db.String(1000))
    feedback = db.Column(db.String(1000))

    def __init__(self, timestamp, command, codeblock, feedback):
        self.timestamp = timestamp
        self.command = command
        self.codeblock = codeblock
        self.feedback = feedback

# base route to display main html body
@app.route('/', methods=["GET", "POST"])
def home():
    try:
        db.session.test_connection()
        pass
    except:
        flash('database connection failed')
    return render_template('icoder.html')


# create a function to read form inputs and process a set of outputs in json
@app.route('/process')
def process():
    command = request.args.get('command')
    extra_args = []

    # check for any feature names
    feat_params = [a.strip() for a in command.split() if a.isupper()]
    # strip commas and quotes from feature names
    feat_params = [a.replace(',', '') for a in feat_params]
    feat_params = [a.replace('"', '') for a in feat_params]
    feat_params = [a.replace("'", '') for a in feat_params]
    # remove X if it is in the param list
    try:
        feat_params.remove('X')
    except ValueError:
        pass
    # remove MAE if it is present
    try:
        feat_params.remove('MAE')
    except ValueError:
        pass
    # remove RMSE if it is present
    try:
        feat_params.remove('RMSE')
    except ValueError:
        pass
    if len(feat_params) > 0:
        extra_args.extend(feat_params)

    # turn to lowercase for uniformity
    lcommand = command.lower()

    # parse command for any numbers
    num_params = re.findall(r'[\s-]*(\d+)[\s-]*', lcommand)
    # parse train-test ratio
    if len(num_params) > 1:
        nums = [float(n) for n in num_params]
        nums = [n/100 for n in nums if n > 1.0]
        nums = [str(round(n, 2)) for n in nums]
        extra_args.extend(nums)
    # parse if given number of trees or other single number args
    elif len(num_params) == 1:
        extra_args.extend(num_params)

    # check if command is in the dictionary keys; if not, match via embedding
    cmd_match = True
    if lcommand not in list(cc_dict.keys()):
        cmd_embed = get_embedding(lcommand)
        sims = [cosine_similarity(cmd_embed, x) for x in embedding_cache]
        ind = np.argmax(sims)
        # for debugging; print out command matching schema
        print('\n\nEntered: ', command)
        print('Best match similarity: ', np.max(sims))
        print('Best match command: ', list(cc_dict.keys())[ind])
        # set cmd_match flag to False if best similarity is 0.80 or less
        if np.max(sims) <= 0.80:
            cmd_match = False
            print('Best match rejected\n')
        else:
            cmd = list(cc_dict.keys())[ind]
            code = list(cc_dict.values())[ind]
            print('Best match accepted\n')
    else:
        cmd = command.lower()
        code = cc_dict[cmd]

    # supplement cmd with parameters (if applicable) and pass to runcode
    if cmd_match == True:
        argtuple = tuple(extra_args)
        if len(argtuple) == 1:
            codeblock = code.format(argtuple[0])
        else:
            codeblock = code.format(*argtuple)
        print(codeblock, '\n')
        if len(argtuple) > 0:
            [outputtype, output] = runcode(cmd, argtuple)
        else:
            [outputtype, output] = runcode(cmd)
        outputs = [outputtype, command, codeblock, output]
    elif cmd_match == False:
        outputs = ['string', command, '', 'No matching command found']
    
    # commit results to db and get id of corresponding entry
    newest_id = log_commands(outputs)
    # append id to outputs
    outputs.append(newest_id)

    return jsonify(outputs=outputs)

# create a function to process positive feedback
@app.route('/positive_feedback')
def positive_feedback():
    id = request.args.get('db_id')
    record = Log.query.filter_by(id=id).first()
    # update feedback; none if already positive, positive otherwise
    if record.feedback == 'positive':
        record.feedback = 'none'
        print('Canceled positive feedback on entry', id)
    else:
        record.feedback = 'positive'
        print('Positive feedback on entry', id)
    db.session.commit()
    return jsonify(id=id)

# create a function to process negative feedback
@app.route('/negative_feedback')
def negative_feedback():
    id = request.args.get('db_id')
    record = Log.query.filter_by(id=id).first()
    # update feedback; none if already negative, negative otherwise
    if record.feedback == 'negative':
        record.feedback = 'none'
        print('Canceled negative feedback on entry', id)
    else:
        record.feedback = 'negative'
        print('Negative feedback on entry', id)
    db.session.commit()
    return jsonify(id=id)

if __name__ == '__main__':
    app.run(debug=True)