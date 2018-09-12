# -*- coding: utf-8 -*-
"""
Created on Tue sep 11 18:43:29 2018

@author: Shanthi
"""

# slide 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# slide 3
## Setting Working directory
import os
## get working directory
os.getcwd()
## set working directory
#path = r'D:\xDevelopment\Courseware\Python\IntroPython\Datasets'
#os.chdir(path)

mtcars = pd.read_csv(r"data/mtcars.csv")
mtcars.describe()

# slide 4
## get datasets
longley = sm.datasets.longley.load_pandas()
longley.data.plot()
## Rand Health Insurance Data
rand_hi = sm.datasets.randhie.load_pandas().data
## Affairs data
affairs = sm.datasets.fair.load_pandas().data
## Election Studies Dataset
election_96 = sm.datasets.anes96.load_pandas().data

# slide 5
## Comparing Two Distributions
mtcars = pd.read_csv(r"data/mtcars.csv")
mtcars.describe()

## Do mpgs differ between automatic and manual transmission?
mtcars.mpg.hist()
mtcars.mpg.plot(kind="kde")
mtcars.mpg.mean()
mtcars.mpg.std()
mtcars.mpg.var()

# slide 6
## Sufficient Statistics for the normal distribution
mtcars.mpg.mean()
mtcars.mpg.std()

# slide 10
## Compare MPGs across groups
sns.boxplot(x="am", y="mpg",data=mtcars)
mtcars.boxplot("mpg",by="am")
mtcars_group = mtcars.groupby(["am"])
mtcars_group.mpg.mean()
mtcars_group.mpg.hist()
mtcars_group.mpg.plot(kind="kde",legend=True)
mtcars_group.mpg.agg([np.mean,np.std])
mtcars_group['mpg'].agg([np.mean,np.std])

# slide 12
## Perform t-test
# Returns test statistic, p-value, degrees of freedom
# usevar='unequal' use Welch’s t-test, otherwise use standard t-test
# standard t-test
sm.stats.ttest_ind(mtcars.mpg[mtcars.am==0], mtcars.mpg[mtcars.am==1])

## Welch's t-test with unequal variance
sm.stats.ttest_ind(mtcars.mpg[mtcars.am==0],
                   mtcars.mpg[mtcars.am==1],
                   usevar="unequal")

# slide 14
## mpg across cylinders
mtcars.boxplot("mpg",by="cyl")
mtcars_group = mtcars.groupby(["cyl"])
mtcars_group.mpg.plot(kind="kde",legend=True)
mtcars_group.mpg.agg([np.mean,np.std])
mtcars_group['mpg'].agg([np.mean,np.std])

# slide 16
## perform anova
mtcars_lm1=smf.ols('mpg~C(cyl)', data=mtcars) #Specify C for Categorical
sm.stats.anova_lm(mtcars_lm1.fit(), typ=2)
## get f statistic and p-value
# use the f-statistic to test the model fit
# the f-statistic will tell you if the means between two
# populations are statistically significant
# use in combination with the p-value and F-Table
sm.stats.anova_lm(mtcars_lm1.fit(), typ=2).ix[0,2:4]

# slide 18
## Correlation
## seaborn image 
# mpg vs displacement
sns.jointplot(x="mpg", y="disp", data=mtcars, kind="reg")
# mpg vs. hp
sns.jointplot(x="mpg", y="hp", data=mtcars, kind="reg")

# slide 20
## Examine the linear correlation between mpg and other values
# get help
#?pd.DataFrame.corr
# try different correlations
# default is pearson
mtcars.loc[:,["mpg","disp","hp"]].corr()
mtcars.loc[:,["mpg","disp","hp"]].corr(method="kendall")
mtcars.loc[:,["mpg","disp","hp"]].corr(method="spearman")

## Calculate the covariance
mtcars.loc[:,["mpg","disp","hp"]].cov()
mtcars.loc[:,["mpg"]].var()

# Correlation is scaled to be between -1 and +1 depending on whether there 
# is positive or negative correlation, and is dimensionless. 
# The covariance however, ranges from zero, in the case of two independent 
# variables, to Var(X), in the case where the two sets of data are equal.

# Variance is the value down the middle of the matrix

# slide 28
## Linear Regression with One predictor
## Fit regression model
mtcars["constant"] = 1
# create an artificial value to add a dimension/independent variable
# we create a constant term so that we fit the intercept of our linear model
# we can then use the intercept-only model as our null hypothesis
X = mtcars.loc[:,["constant","am"]]
Y = mtcars.mpg
# create the model
mod1res = sm.OLS(Y, X).fit()

# slide 29
## Inspect the results
print(mod1res.summary())
print(mod1res.summary2()) ## different format

# slide 30
# what do we see
# Dep variable: Which variable is the response
# Df Residuals: Number of observations
# Df Model - number of parameters, not including constant term
# R-squared - the coefficient of determination.  How well the model
# approximates the real data
# Adj r-squared - based upon the degrees of freedom 
# and number of observations
# Log-likelihood - log of the likelihood function, used to describe possible
# future outcomes given a fixed value for the parameter
# AIC is a measure of relative quality of a model. it adjusts the 
# log-likelihood based on the number of observations 
# and complexity of the model - lower is better
# BIC - stands for Bayesian Information Criterion, like AIC but
# carries a higher penality for models with more parameters - lower is better
# F-statistic - measure of how significant the fit is, it is the mean
# squared error of the model divided by the mean squared error of residuals
# higher F-statistic means lower p-value means model is better
# residuals
# prob F-statistic, the probability you would get the above statistic
# given the hypothesis the squared error of the model and the residuals
# are unrelated

# Coef - estimated value for the coefficient
# std err - Basic standard error estimate - a measure of the accuracy 
# of predictions
# t - t-statistic value, a measure of how statistically significant
# the coefficient is
# p - the p-value that the null hypothesis that the coefficient = 0 is true
# smaller p-value means there is a statistically significant relationship
# between term and the response
# last bracket is the confidence interval

# Omnibus - combined statistical test for the presence of 
# skewness and kurtosis
# prob Omnibus - statistic turned into a probability
# skewness - measure of symmetry of the data about the mean
# Kurtosis - Compares the amount of data close to the mean with data
# in the tails
# Jarque-Bera - different test of skewness and kurtosis
# Prob (JB) statistic turned into probability
# Durbin-Watson - test for the presence of autocorrelation (that the 
# errors are not independent), useful in time-series analysis
# Condition No. - A test for multicollinearity (that parameters are related
# to each other)

# slide 31
## Get AIC for model - stands for Akaike Information Criterion
mod1res.aic

## Get r-squared
mod1res.rsquared
## r-squared penalized for the number of parameters
mod1res.rsquared_adj

## Mean squared error...?
# subtract ratio of mean squared error of residuals to total error
1 - mod1res.mse_resid/mod1res.mse_total
# mean sqaured error of the model
mod1res.mse_model

# slide 32
## View residuals
# for each value
mod1res.resid
# using pearson correlation
mod1res.resid_pearson
## View Predicted Values
mod1res.predict()

## Get the influence of each value? DBETAS?
mod1res.get_influence()

## View available model attributes
dir(mod1res)

# slide 33
## Goodness of fit
## Find outlier observations
mod1res.outlier_test()
# studentized residuals, unadjusted p-value, 
# corrected p-value (Bonferroni), in which p-values are multipled
# by the number of comparisons

## Goodness of fit plots
sm.graphics.plot_regress_exog(mod1res, "am")
# plot regression results against one regressor
# in this case
# endog(mpg) vs. exog (am) - straight comparison
# residuals vs. exog
# fitted vs. exog
# fitted plus residual vs exog

# slide 34
## Linear Regression with Categorical predictors
## mpg and cylinder
## Fit regression model
mtcars["constant"] = 1
X = mtcars.loc[:,["constant","cyl"]]
Y = mtcars.mpg
mod2res = sm.OLS(Y, X).fit()
print(mod2res.summary())
## Alternatively:
X = sm.add_constant(mtcars.cyl)

# slide 35
## We need to dummy code the categorical variable
# remember we are talking regression here
mtcars['cyl_6'] = 0
mtcars.loc[mtcars.cyl == 6,'cyl_6'] = 1
mtcars['cyl_8'] = 0
mtcars.loc[mtcars.cyl == 8,'cyl_8'] = 1
## Rerun the model with more parameters
X = mtcars.loc[:,["constant","cyl_6","cyl_8"]]
Y = mtcars.mpg
mod3res = sm.OLS(Y, X).fit()
print(mod3res.summary())

# slide 36
## Is there an easier way to set up these matrices?!?
## We can use the formula interface, patsy package
## Formula interface included with statsmodels
## Fit same model
mod4res = smf.ols('mpg ~ C(cyl)', data=mtcars).fit()
mod4res.summary() 
# note the C allows us to automatically assign values to 
# categorical variables

# slide 37
## Linear regression with multiple predictors
mod5res = smf.ols('mpg ~ C(cyl) + disp', data=mtcars).fit()
mod5res.summary() 

mod6res = smf.ols('mpg ~ C(cyl) + disp + am', data=mtcars).fit()
mod6res.summary() 

## Does AM add anything?
sm.stats.anova_lm(mod6res, typ=2)
# doesn't look like it

# slide 38
## AIC - stands for Akaike Information Criterion
# AIC is a measure of relative quality of a model. it adjusts the 
# log-likelihood based on the number of observations 
# and complexity of the model 

## comparing model fits - AIC
print("model-6 AIC: ", mod6res.aic)
print("model-5 AIC: ", mod5res.aic)
print("model-4 AIC: ", mod4res.aic)

# generally smaller AICs are better

# slide 39
## Compare Model Fits - r-squared
# r-squared shows how much of variation is explained by model
print("mod6res.rsquared: ", mod6res.rsquared)
print("mod5res.rsquared: ", mod5res.rsquared)
print("mod4res.rsquared: ", mod4res.rsquared)

# generally higher is better

## Compare Model Fits - r-squared adjusted
print("mod6res.rsquared_adj: ", mod6res.rsquared_adj)
print("mod5res.rsquared_adj: ", mod5res.rsquared_adj)
print("mod4res.rsquared_adj: ", mod4res.rsquared_adj)

# slide 40
# higher is better, more useless predictors will lower score
## Compare Model fits - likelihood
# you want to use this measure combined with others
# because it is a function of sample size
print("model-6 LLF: ", mod6res.llf)
print("model-5 LLF: ", mod5res.llf)

# smaller (more negative) is generally better

## Likelihood ratio test
# in this case model 6 is a subset of model 5
# compare the goodness of fit between them
sm.stats.anova_lm(mod5res,mod6res)
# you get residuals, sum of squares and differences between
# them, plus F-statistic and p-value

# slide 41
## Seaborn regression diagnostics: non-linear relationships
# note the lowess plot creates a smooth curve to see trends
sns.lmplot(x="disp", y="mpg", data=mtcars, lowess=True)
# this is as opposed to a typical regression line
sns.lmplot(x="disp", y="mpg", data=mtcars)
# or a simple scatter plot
mtcars.plot.scatter(x='mpg',y='disp')

# slide 42
## Examine the goodness of fit plots 
# this first plot creates a scatterplot of observed values to fitted values
sm.graphics.plot_fit(mod5res,exog_idx=3)
# four-graph plot of regression results against one regressor
sm.graphics.plot_regress_exog(mod5res,exog_idx=3)
# plot the influence of points, size is related to Cook's distance
sm.graphics.influence_plot(mod5res)
# plot leverage vs. residuals squared
sm.graphics.plot_leverage_resid2(mod5res)

# slide 43
## Scoring Data
## create sample dataframe
mtcars.cyl.unique()
mtcars.disp.describe()
# cylindars and some displ values
new_dat = pd.DataFrame({'cyl':mtcars.cyl.unique(), 'disp':(80,100,120)})
## Predict the new values
mod5res.predict(new_dat)
# the relationship does not appear to be linear one
# maybe a quadratic would fit better

# slide 44
## Linear regression with polynomials
## add disp squared
mtcars['disp_2'] = mtcars.disp **2
# here we test original displacement
mod7res = smf.ols('mpg ~ disp', data=mtcars).fit()
mod7res.summary() 
# now we can add the squared value to create a quadratic relationship
mod8res = smf.ols('mpg ~ disp + disp_2', data=mtcars).fit()
mod8res.summary() 
## Did this improve the model fit?
sm.stats.anova_lm(mod8res, typ=2)
sm.stats.anova_lm(mod7res, typ=2)
# sure looks like it

# slide 45
## We can rewrite this with formulas...
smf.ols('mpg ~ disp + disp_2', data=mtcars).fit().summary()
smf.ols('mpg ~ disp + np.square(disp)', data=mtcars).fit().summary()
# another version of the same thing
smf.ols('mpg ~ disp + np.power(disp,2)', data=mtcars).fit().summary()

# slide 46
## Linear regression with other functions (log)
mod10res = smf.ols('mpg ~ np.log(disp)', data=mtcars).fit()
mod10res.summary()

# slide 47
## Using Patsy alone to generate matrices
from patsy import dmatrices
from patsy import dmatrix
f = 'mpg ~ disp + C(cyl)'
y,X = dmatrices(f, mtcars)
y
X
y,X = dmatrices(f, mtcars,return_type='dataframe')

# slide 48
## notice we used OLS instead of ols...
## for OLS we don't need to pass the formula
mod11res = smf.OLS(y,X).fit()
mod11res.summary()

### Other useful patsy functions
## Center and standardize
dmatrix("disp + center(disp) + standardize(disp)", mtcars)

# slide 49
## Linear regression with Spline terms
## basis splines
dmatrix("bs(disp,df=4)", mtcars)

## cubic splines
dmatrix("cr(disp,df=4)", mtcars)

## cyclic cubic splines
dmatrix("cc(disp,df=4)", mtcars)

# slide 50
## Linear regression with interactions
## make interaction plot seaborn cyl, mpg, hp
sns.lmplot(x="hp", y="mpg", hue="cyl", data=mtcars);
mod12 = smf.ols('mpg ~ hp + C(cyl)', data=mtcars).fit()
sm.stats.anova_lm(mod12, typ=2)

# slide 52
## Test for an interaction here
# hp with cyl
mod13 = smf.ols('mpg ~ hp + C(cyl) + hp:C(cyl)', data=mtcars).fit()
sm.stats.anova_lm(mod13, typ=2)
