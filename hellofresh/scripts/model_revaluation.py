"""
1. Read in ingredients.csv and recipe_ratings.csv
2. Process the data to generate feature vectors
3. Train the model
4. K-fold cross-validation. Generate avg metric score.
5. If metric is better, save the new model and metric score to db
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import math
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from database import Database


def extract_main_ingredient(ing_str):
    """Extracts the main ingredient from the given string.
    If the ing_str is not in expected format, return NaN
    """
    nan = float('nan')
    ing_str = str(ing_str)

    # check for blank ingrdient strings
    if (ing_str == 'nan') or (ing_str == ''):
        return nan

    # check for invalid characters in Ingredients
    if not re.match('^[\"\'\w\d\s,/()-]+$', ing_str):
        return nan

    main_ingredient = ing_str.split(",")
    main_ingredient = main_ingredient[0].strip()
    main_ingredient = main_ingredient.split("-")[0]
    return main_ingredient.strip()


def process_ingredient_data(df):

    df.Ingredient = df.Ingredient.map(extract_main_ingredient)
    df = df.dropna()

    ing_usage = df.groupby('Ingredient').size()
    one_hot = pd.get_dummies(df['Ingredient'])

    grouped_ing = pd.concat(
        [df.Recipe_code, one_hot],
        axis=1
    ).drop_duplicates()
    grouped_ing = grouped_ing.groupby('Recipe_code').sum()

    grouped_ing['num_ing'] = grouped_ing.loc[
        :, grouped_ing.columns != 'Recipe_code'
    ].sum(axis=1)

    return grouped_ing


def process_ratings_data(ratings):

    # bin the ratings into 4 categories
    ratings['label'] = pd.cut(
        ratings.score,
        # note: we are merging [0, 1) & [1, 2) and [4, 5) & [5, 6) bins
        [0, 2, 3, 4, 6],
        labels=[*range(1, 5)]
    )

    ratings = ratings.loc[ratings.new == 0] \
        .set_index('Recipe_code')

    return ratings


def xgb_model(train_data, train_label):
    """
    define the classifier model and fit it to training data
    """
    clf = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=30,
        objective='multi:softprob',
        colsample_bytree=0.9,
        verbose=True
    )
    clf.fit(train_data, train_label)
    return clf


def compute_score(df):
    """
    uses k-fold cross validation to compute
    the average metric score
    """
    X = df.values[:, :-1]
    y = df.values[:, -1]

    kf = KFold(n_splits=10)
    cum_score = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = xgb_model(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy_measure = np.linalg.norm(
            y_pred-y_test, ord=1) / X_test.shape[0]
        cum_score += accuracy_measure
        print(accuracy_measure)

    print("Average Score: {}".format(cum_score / 10.0))
    return cum_score / 10.0


if __name__ == '__main__':
    # just a database simulator
    db = Database()

    # read ingredients
    ing = db.read_ingr_table()
    df = process_ingredient_data(ing)
    unique_ingredients = [
        ingr for ingr in df.columns if ingr not in ['num_ing']
    ]

    # read ratings
    ratings = db.read_ratings_table()
    ratings = process_ratings_data(ratings)

    df = pd.merge(df, ratings['label'], on='Recipe_code')
    # compute model score
    new_model_score = compute_score(df)

    model_info = db.read_model()
    if model_info:
        old_model_score = model_info['score']
    else:
        old_model_score = 0

    if new_model_score >= old_model_score:
        # if new model is better, store it.
        model = dict(
            clf=xgb_model(df.values[:, :-1], df.values[:, -1]),
            score=new_model_score,
            unique_ingredients=unique_ingredients,
            timestamp=pd.Timestamp.now()
        )
        db.write_model(model)
