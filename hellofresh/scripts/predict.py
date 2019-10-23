"""
1. Load the current prediction model
2. Load data (ingredients of new recipes)
3. Use the model to generate predictions.
4. Write the results to db
"""
import pandas as pd
import xgboost as xgb
from database import Database
import re
import sys


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


def prepare_feature_set(df, model_info):
    """
    One hot encode the ingredients and reindex the dataframe
    so that it has the same features as used in training
    """
    # df.Ingredient = df.Ingredient.map(lambda x: x.split(",")[0].strip())
    df.Ingredient = df.Ingredient.map(extract_main_ingredient)

    if df.isnull().values.any():
        # the ingredient format of new recipes might be different from what we have seen till now
        print("Skipping some ingredients because of new format. Following recipes affected:\n {}".format(
            df.loc[df.Ingredient.isnull()])
        )
    one_hot = pd.get_dummies(df['Ingredient'])

    grouped_ing = pd.concat(
        [df.Recipe_code, one_hot],
        axis=1
    ).drop_duplicates()
    grouped_ing = grouped_ing.groupby('Recipe_code').sum()
    # Important step
    grouped_ing = grouped_ing.reindex(
        columns=model_info['unique_ingredients']
    ).fillna(0)

    grouped_ing['num_ing'] = df.groupby('Recipe_code').size()
    return grouped_ing


db = Database()

model_info = db.read_model()
if model_info is None:
    print('***No model in database. Run model_revaluation.py first***')
    sys.exit()
else:
    clf = model_info['clf']

df = db.get_test_data()
df = prepare_feature_set(df, model_info)
y_pred = clf.predict(df.values)
# save predictions as series
db.save_predictions(
    pd.Series(
        data=y_pred,
        index=df.index
    )
)
