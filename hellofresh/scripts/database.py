import pandas as pd
import pickle
import os


class Database:

    def __init__(self):
        self.model_loc = "db/model.pkl"

    def read_ingr_table(self):
        return pd.read_csv('db/ingredients.csv')

    def read_ratings_table(self):
        return pd.read_csv('db/recipe_ratings.csv')

    def read_model(self):
        """
        reads in the trained model
        """
        if os.path.isfile(self.model_loc):
            model = pickle.load(
                open(self.model_loc, "rb")
            )
        else:
            model = None
        return model

    def write_model(self, model):
        """
        writes the trained model to db
        """
        pickle.dump(
            model,
            open(self.model_loc, "wb")
        )

    def save_predictions(self, pred_series):
        """Save predictions for new recipes
        """
        pred_series.to_csv('db/predictions.csv')

    def get_test_data(self):
        ingr = self.read_ingr_table()
        ratings = self.read_ratings_table()
        new_recipes = ratings.loc[ratings.new == 1, 'Recipe_code']

        df = ingr.loc[ingr.Recipe_code.isin(new_recipes)]
        return df
