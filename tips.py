"""Import the dataset using pandas.
2. Answer the following questions using summary statistics and grouped analyses (no linear modeling
yet)
a. Examine the dataset by looking at the distribution of all variables (frequency tables for
categorical variables and mean/quantiles for continuous)
b. Examine the distribution of total tip using a histogram. Is this distribution symmetric? How
would you describe it?
c. Do smokers tip more than non-smokers on average?
d. What about females versus males? Who is the better tipper?
e. What is the trend in tip average tip size across day of week? In other words, starting from
Monday, does tips get larger or smaller as you approach Sunday?
f. Are tips typically higher over lunch or dinner?
g. Is the size of the party correlated with the total tip?
h. Perhaps it is more accurate to consider the tip percentage rather than the total tip. Create a
new variable which is the tip percentage, then repeat b-c above for tip percentage.
i. Is there any correlation between bill size and tip percentage?
3. Construct a linear model with tip percentage as the outcome.
4. Based on your analyses above, construct what you think is the best multivariate linear model for tip
percentage.
5. Describe, in your opinion, the attributes of the “best” customer for this waiter in terms of a tip."""

print("Tips")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
tips = pd.read_csv(r"data/tips.csv")

#data = pd.read_csv('data.csv', sep=',',header=None, index_col =0)

tips.plot(kind='bar')
print(tips.describe())


# b. Examine the distribution of total tip using a histogram. 
# Is this distribution symmetric? How would you describe it?
# plot a histogram
tips.tip.plot(kind="hist")
print(tips.tip.mean())
print(tips.tip.std())


# c. Do smokers tip more than non-smokers on average?
tips.tip.mean()
tips_only = tips.tip
tips.query('smoker == "Yes"').tip.mean()
tips.query('smoker == "No"').tip.mean()
tips.groupby(["smoker"]).mean()
tips.groupby(["sex", "smoker"]).mean()

# d. What about females versus males?
tips.groupby(["sex"]).mean()

# e. What is the trend in tip average tip size across day of week?
tips.groupby(["day"]).mean()

# f. Are tips typically higher over lunch or dinner?
tips.groupby(["time"]).mean()

# g. Is the size of the party correlated with the total tip?
tips.groupby(["size"]).mean()

# h. Perhaps it is more accurate to consider the tip percentage rather than 
# the total tip. Create a new variable which is the tip percentage, 
# then repeat c-g above for tip percentage.
tips['tip_pcn'] =  100 * (tips['tip'] / tips['total_bill'])
tips.query('smoker == "Yes"').tip_pcn.mean()
tips.query('smoker == "No"').tip_pcn.mean()
tips.groupby(["smoker"]).mean()
tips.groupby(["sex", "smoker"]).mean()

    
# females versus males?
print(tips.groupby(["sex"]).mean())

# trend in tip average tip size across day of week?
print(tips.groupby(["day"]).mean())

# Are tips typically higher over lunch or dinner?
print(tips.groupby(["time"]).mean())

# Is the size of the party correlated with the total tip?
print(tips.groupby(["size"]).mean())

# i. Is there any correlation between bill size and tip percentage?
sns.jointplot(x="total_bill", y="tip_pcn", data=tips, kind="reg")
sns.lmplot(x="total_bill", y="tip_pcn", data=tips)

# 3. Construct a linear model with tip percentage as the outcome. Based on 
# your analyses above, construct what you think is the best multivariate 
# linear model for tip percentage.
# use ordinary least squares model
#Specify C for Categorical
tips_lm1=smf.ols('tip_pcn~total_bill+C(smoker)', data=tips).fit() 
print(tips_lm1.summary())
tips_lm1=smf.ols('tip_pcn~C(sex)+total_bill+C(smoker)+size+C(time)+C(day)', data=tips).fit()
print(tips_lm1.summary())

# Looking at the summary
# coef - the various coefficients for each predictor
# std err - the estimate of standard deviation of the coefficient
# t - t statistic
# P>|t| - p-value
# last is confidence interval