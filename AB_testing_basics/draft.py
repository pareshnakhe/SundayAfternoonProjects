# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats

# COMMAND ----------

# MAGIC %md
# MAGIC ### A. Null and Alternative Hypothesis

# COMMAND ----------

# Define the two distributions under the null and alternative hypotheses
x = np.linspace(-4, 7, 1000)
mean1, mean2 = 0, 3
std = 1
dist1 = norm(loc=mean1, scale=std)
dist2 = norm(loc=mean2, scale=std)
y1 = dist1.pdf(x)
y2 = dist2.pdf(x)

# Plot the two  distributions
plt.figure(figsize=(12, 5))
plt.plot(x, y1, label="Null")
plt.plot(x, y2, label="Alternative", linestyle="--")
plt.axvline(x=mean1, color="lightblue", linestyle="--", linewidth=1)
plt.axvline(x=mean2, color="orange", linestyle="--", linewidth=1)

# Fill the area under the curves
quantile_975_dist1 = dist1.ppf(0.975)
plt.fill_between(
    x, y1, where=(x > quantile_975_dist1), color="blue", alpha=0.3, label="Type-I Error"
)
plt.fill_between(
    x, y2, where=(x > quantile_975_dist1), color="green", alpha=0.3, label="Power"
)

# Add labels and legend
plt.title("Hypothesis Testing")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### B. Power Curves and MDEs

# COMMAND ----------

x = np.linspace(0, 5, 1000)
threshold = dist1.ppf(0.975)
power = norm.sf(threshold, loc=x, scale=1)

plt.figure(figsize=(12, 5))
plt.plot(x, power, label="Power as a function of x", color="red")

# Draw a horizontal line at 0.8
plt.axhline(y=0.8, linestyle="--", color="black", label="Power = 0.8")

# Find the intersection point (where power crosses 0.8)
mde_index = np.abs(power - 0.8).argmin()
mde_x_value = x[mde_index]
mde_y_value = power[mde_index]

# Draw a vertical line at the intersection point and label it "MDE"
plt.axvline(
    x=mde_x_value, color="blue", linestyle="--", label=f"MDE = {mde_x_value:.2f}"
)
plt.text(mde_x_value + 0.05, 0.5, f"MDE: {mde_x_value:.2f}", color="blue", fontsize=12)

# Add labels and legend
plt.title("Power Curve with MDE")
plt.xlabel("x")
plt.ylabel("Power")
plt.legend()
plt.grid()

# Show the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### C. Simulation Under the Alternative

# COMMAND ----------

seed = 2024
np.random.seed(seed)

alpha = 0.05
true_mean = 0
mde = 2.8
observations = np.random.normal(true_mean, 1, 100)

z_value = stats.norm.ppf(1 - alpha / 2)
lwr = observations - z_value
upr = observations + z_value

plt.figure(figsize=(10, 6))
for i in range(100):
    if lwr[i] * upr[i] > 0:
        plt.errorbar(
            i,
            observations[i],
            yerr=[[observations[i] - lwr[i]], [upr[i] - observations[i]]],
            fmt="o",
            color="mediumaquamarine",
            ecolor="mediumaquamarine",
            elinewidth=2,
            capsize=3,
        )
    else:
        plt.errorbar(
            i,
            observations[i],
            yerr=[[observations[i] - lwr[i]], [upr[i] - observations[i]]],
            fmt="o",
            color="coral",
            ecolor="coral",
            elinewidth=2,
            capsize=3,
        )

plt.axhline(y=mde, color="black", linestyle="--", linewidth=0.6)
plt.text(0, mde + 0.2, f"MDE: {mde:.2f}", color="black", fontsize=10)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.6)

plt.title(f"MDE = {mde} and True Mean = {true_mean}", fontsize=14)
plt.xlabel("Index", fontsize=12)
plt.ylabel("Confidence Intervals", fontsize=12)
plt.grid()
plt.show()


# COMMAND ----------

array1 = observations < mde
array2 = lwr * upr > 0

# Create a 2x2 matrix of the counts
true_true = np.sum((array1 == True) & (array2 == True))
true_false = np.sum((array1 == True) & (array2 == False))
false_true = np.sum((array1 == False) & (array2 == True))
false_false = np.sum((array1 == False) & (array2 == False))

summary_matrix = np.array([[true_true, true_false], [false_true, false_false]])

row_totals = np.sum(summary_matrix, axis=0)
col_totals = np.sum(summary_matrix, axis=1)
overall_total = np.sum(summary_matrix)

summary_matrix_with_totals = np.vstack([summary_matrix, row_totals])
summary_matrix_with_totals = np.hstack(
    [
        summary_matrix_with_totals,
        np.reshape(np.append(col_totals, overall_total), (-1, 1)),
    ]
)

# Plot the table
fig, ax = plt.subplots(figsize=(4, 4))
ax.axis("off")
row_labels = ["Observation < MDE", "Observation >= MDE", "Row Total"]
col_labels = ["Significant", "Not Significant", "Col Total"]
table = ax.table(
    cellText=summary_matrix_with_totals,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.5, 1.5)
plt.title("Summary", fontsize=14)
plt.show()
