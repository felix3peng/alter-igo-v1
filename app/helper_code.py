global df, X, y, X_valid, y_valid, lm, score, path


def load_data():
    global df
    import pandas as pd
    import os
    path = os.path.dirname(__file__)
    df = pd.read_csv(path+'./data.csv')
    print(df.head())


def summarize_data():
    global df
    print(df.describe())


def correlation_heatmap():
    global df
    import seaborn as sns
    sns.heatmap(df.corr(), xticklabels=df.columns, yticklabels=df.columns)


def train_model():
    global df, lm, X, y, X_train, y_train, X_valid, y_valid, lm
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    X = df.copy()
    y = X.pop('y')
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    train_score = lm.score(X_train, y_train)
    print('Training score: ', train_score)


def score_model():
    global lm, X_valid, y_valid, score
    score = lm.score(X_valid, y_valid)
    print('Validation score: ', score)


text_code_dict = {
    'load data': load_data,
    'summarize data': summarize_data,
    'correlation heatmap': correlation_heatmap,
    'train model': train_model,
    'score model': score_model}
