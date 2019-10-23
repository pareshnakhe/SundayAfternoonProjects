"""
In this script, we do some basic exploratory data analysis
and generate plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.database import Database

db = Database()

ing = db.read_ingr_table()
ing = ing.dropna()

# Plotting histogram of # of ingredients in a recipe
fig, ax = plt.subplots(1, 1)
ing.groupby('Recipe_code').size().plot.hist(ax=ax)
ax.set_xlabel('Number of ingredients')
ax.set_title('Histogram: # of Ingredients')

plt.savefig('data_exploration/ingredient_histogram')


# Scatter plot of # of ingredients-vs-ratings score

ratings = db.read_ratings_table()
rtgs = ratings.set_index('Recipe_code')
ing_size = ing.groupby('Recipe_code').size()
ing_size.name = 'ing_size'
df = pd.merge(ing_size, rtgs, on='Recipe_code')
df = df.dropna()

fig, ax = plt.subplots(1, 1)
ax.scatter(df.ing_size, df.score, s=30, alpha=0.5)

ax.plot(
    [df.ing_size.min(), df.ing_size.max()],
    [df.score.max(), df.score.min()],
    marker='o', color='green'
)
ax.set_xlabel('Number of ingredients')
ax.set_ylabel('Rating score')
ax.set_title('Scatterplot: Score vs recipe complexity')
plt.savefig('data_exploration/ingredient_sscore_scatterplot')

# Distribution of ratings scores

fig, ax = plt.subplots(1, 1)

ratings['label'] = pd.cut(
    ratings.score,
    [0, 2, 3, 4, 6],  # note: we are merging [0, 1) and [1, 2) bins
    labels=[*range(1, 5)]
)


score_dist = ratings.dropna().groupby('label').size()
score_dist.plot.bar(ax=ax)
ax.set_xlabel('Rating level')
ax.set_ylabel('# of recipes')
ax.set_title('Distribution of scores')
plt.savefig('data_exploration/ratings_distribution_grouped')

# Distribution of ratings scores (without binning)

fig, ax = plt.subplots(1, 1)

ratings['label'] = pd.cut(
    ratings.score,
    [0, 1, 2, 3, 4, 5, 6],  # note: we are merging [0, 1) and [1, 2) bins
    labels=[*range(1, 7)]
)


score_dist = ratings.dropna().groupby('label').size()
score_dist.plot.bar(ax=ax)
ax.set_xlabel('Rating level')
ax.set_ylabel('# of recipes')
ax.set_title('Distribution of scores')
plt.savefig('data_exploration/ratings_distribution_overall')

# cost compensation factors due to class imbalance
score_dist.apply(lambda x: score_dist.sum()/x)
