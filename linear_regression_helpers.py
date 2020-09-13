########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.stattools import durbin_watson


########################################################################################################################
# FUNCTIONS ############################################################################################################
########################################################################################################################

# See https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/


# Tests the linearity of the model
def test_linearity(lin_reg_results, X, y, data_type):
    pred = lin_reg_results.predict(sm.add_constant(X, has_constant='add'))
    df = pd.DataFrame({'Actual': y, 'Predicted': pred})

    sns.lmplot(x='Actual', y='Predicted', data=df, fit_reg=False, size=7)

    line_coords = np.arange(df.min().min(), df.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()

    plt.savefig('out/linear-regression-{}-linearity-test.png'.format(data_type), transparent=False)


# Tests for autocorrelation in the error terms
def test_autocorrelation_of_errors(lin_reg_results, X, y, data_type):
    pred = lin_reg_results.predict(sm.add_constant(X, has_constant='add'))
    df = pd.DataFrame({'Actual': y, 'Predicted': pred})
    df['Residuals'] = abs(df['Actual']) - abs(df['Predicted'])

    print('\nPerforming Durbin-Watson Test on '.format(data_type))
    print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
    print('0 to 2< is positive autocorrelation')
    print('>2 to 4 is negative autocorrelation')
    durbinWatson = durbin_watson(df['Residuals'])
    print('Durbin-Watson:', durbinWatson)


# Tests for homoscedasticity in the error terms
def test_homoscedasticity(lin_reg_results, X, y, data_type):
    pred = lin_reg_results.predict(sm.add_constant(X, has_constant='add'))
    df = pd.DataFrame({'Actual': y, 'Predicted': pred})
    df['Residuals'] = abs(df['Actual']) - abs(df['Predicted'])

    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=df.index, y=df.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()

    plt.savefig('out/linear-regression-{}-homoscedasticity-test.png'.format(data_type), transparent=False)


# Tests for normality of the error terms
def test_error_normality(lin_reg_results, X, y, data_type):
    pred = lin_reg_results.predict(sm.add_constant(X, has_constant='add'))
    df = pd.DataFrame({'Actual': y, 'Predicted': pred})
    df['Residuals'] = abs(df['Actual']) - abs(df['Predicted'])

    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df['Residuals'])
    plt.show()

    plt.savefig('out/linear-regression-{}-error-normality-test.png'.format(data_type), transparent=False)

    p_value = normal_ad(df['Residuals'])[1]
    print('p-value of the Anderson-Darling test for normality = {} (below 0.05 generally means non-normal)'.format(
        p_value))
