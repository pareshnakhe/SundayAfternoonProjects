import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)


def get_promotions_schedule():
    """
    read and process promotions schedule for direct use
    """
    prom_schedule = pd.read_csv('promotion_schedule.csv')
    missing_entry = pd.DataFrame(
        data=[[20.0, 0.0, 0.0]], 
        columns=['j', 'discount', 'advertised']
    )

    prom_schedule = pd.concat(
        [prom_schedule, missing_entry], 
        axis=0, 
        ignore_index=True
    ).sort_values('j').set_index('j')

    normal_price_series = pd.DataFrame(
        data=no_discount_price_list, 
        index=range(40),
        columns=['normal_price']
    )

    prom_schedule = pd.concat([prom_schedule, normal_price_series], axis=1)
    prom_schedule['discount_percent'] =  prom_schedule.apply(
        lambda x: x.discount / x.normal_price, 
        axis=1
    )

    prom_schedule.index = prom_schedule.index.map(int)
    prom_schedule.advertised = prom_schedule.advertised.map(int)
    return prom_schedule

def get_discount_to_sale_line(df):
    # Assumption: Impact of advertisements and discounts is independent of product
    # tracking the percent change in sales with percent change in price
    # across all customer-product combinations
    price_change_list = list()
    sale_change_list = list()

    for t in range(n_weeks):
        tmp_df = df.loc[df.t == t]
        for j in tmp_df.loc[tmp_df.advertised == 1].j.unique():
            avg_sale_no_ads = avg_sale_no_ads_list[j]
            discount_sales_percent = (tmp_df.loc[tmp_df.j == j].shape[0] - avg_sale_no_ads) / avg_sale_no_ads
            sale_change_list.append(discount_sales_percent)

            discount_prices = tmp_df.loc[tmp_df.j == j].price.mean()
            no_discount_price = no_discount_price_list[j]

            price_change = (no_discount_price - discount_prices) / no_discount_price
            price_change_list.append(price_change)


    slope, intercept, _, _, _ = linregress(
        price_change_list,
        sale_change_list
    )

    discount_line = np.poly1d([slope, intercept])
    # x = np.linspace(min(price_change_list), max(price_change_list), 100)
    # y = discount_line(x)
    # plt.plot(x, y, linestyle='--', color='r')
    # plt.scatter(price_change_list, sale_change_list)
    # plt.xlabel('percent change in price')
    # plt.ylabel('percent change in sale')
    # plt.title('Impact of discounts')
    # plt.show()
    # plt.savefig('discount_impact')
    return discount_line

df = pd.read_csv('train.csv')

n_weeks = df.t.nunique()
n_customers = df.i.nunique()
n_products = df.j.nunique()

total_sale = df.groupby(['i', 'j']) \
    .size() \
    .unstack() \
    .fillna(0)


product_sale_mean = total_sale.mean(axis=0) / n_weeks

# this is the prior for all consumers
prior_alpha = product_sale_mean.apply(lambda x: x*19)
prior_beta = product_sale_mean.apply(lambda x: (1.0-x)*19)

posterior_alpha = total_sale + prior_alpha
posterior_beta = (n_weeks-total_sale) + prior_beta

# model the impact of discounts on increase in sales
avg_sale_no_ads_list = [df.loc[(df.advertised == 0) & (df.j == j)].groupby('t').size().mean() for j in range(n_products)]
no_discount_price_list = [df.loc[(df.advertised == 0) & (df.j == j)].price.mean() for j in range(n_products)]
discount_line = get_discount_to_sale_line(df)

# Generate buying probabilities from posterior distribution
all_draws = list()
prom_schedule = get_promotions_schedule()

for i in range(n_customers):
    for j in range(n_products):
        ij_draw = np.random.beta(
            posterior_alpha.iloc[i, j],
            posterior_beta.iloc[i, j]
        )
        if prom_schedule.loc[j, 'advertised']:
            discount_percent = prom_schedule.loc[j, 'discount_percent']
            exp_sale_jump = discount_line(discount_percent)
            ij_draw = (1+exp_sale_jump) * ij_draw

        all_draws.append([i, j, ij_draw])

output = pd.DataFrame(
    data=np.array(all_draws),
    columns=['i', 'j', 'prediction']
)

output.i = output.i.map(int)
output.j = output.j.map(int)
output = output.set_index('i')
output.to_csv('output.csv')