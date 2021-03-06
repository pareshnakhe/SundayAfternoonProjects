# Predict Consumer Purchases

This a toy problem I came across on a site (don't want to mention the source). It belongs to a class of problems I don't have much experience with. Although the solution is not very sophisticated, I do think there are some nice ideas in it. This mini-project is meant to document it.

> The data set attached to this task contains the purchase histories (i.e., shopping baskets) of 2,000 consumers over 49 weeks across 5 categories (​train.csv​). In simulating the basket data we assumed that consumers only buy one unit of a product in a given week. The data set also contains the price consumers paid for one unit of product j in week t and a boolean variable that indicates whether the purchased product was advertised (1) or not (0). We also provide the week 50 promotion schedule (discounts and advertising) for all products (​promotion_schedule.csv​).

> **The Task:**
> Use the data to build a ML model for consumer purchases. With the trained model, predict week 50 purchases for all 80,000 possible consumer-product combinations (40 products x 2,000 consumers) in the data. Provide your predictions as a ​.csv file (cf. ​prediction_example.csv​) that contains the columns i (consumer), j (product), and prediction. Model performance will be evaluated using AUC score​.

## General Approach

There are two sub-problems that I solve to reach my solution. First, is capturing consumer preferences for individual products, for example, we want to know, what is the probability that a given consumer i would buy a product j in week t. Second is modelling the impact of advertisement on the total sale of for each product.

### Capturing Consumer Preferences

I model the probability that consumer i buys product j as a bernoulli random variable. I have taken a bayesian approach, in that I use the Beta distribution as prior. This prior distribution uses as its mean the average sale per person per week and  is shared by all consumers. We use the fact that Beta distribution is a conjugate prior to analytically compute the parameters of the posterior distribution.

### Modelling Impact of Advertisement

Across all products over all weeks, I measure the percentage bump in sales for the given discount on that product. The relationship is roughly linear. (Use the dicount_impact plot in main.py). This linear relationship is computed using least squares method. In the test phase, we compute the percent discount available on a given product and use the percent bump in sales as returned by our computed linear model to adjust buying probabilities.
