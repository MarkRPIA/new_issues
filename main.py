########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import multiprocessing
import pandas as pd
# import pandas_profiling

import preprocessing
import feature_engineering
import models

########################################################################################################################
# GLOBALS ##############################################################################################################
########################################################################################################################

verbose = True

num_days_performance = 0  # used to get the similarity columns
field = 'Performance{}'.format(num_days_performance)

use_X_addl = True  # whether or not we want to include the columns in the DataFrame X_addl

train_size = 0.8
test_size = 0.2

pd.set_option('display.max_columns', None)  # show all columns!

########################################################################################################################
# MAIN #################################################################################################################
########################################################################################################################

# Uses pandas_profiling to perform EDA.
# def profile_data(df, filename):
#    profile_report = pandas_profiling.ProfileReport(df)
#    profile_report.to_file(output_file=filename)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # needed for pandas_profiling
    print()

    # Load the data
    df = preprocessing.load_data(verbose)

    # Profile the data
    #if verbose:
    #    profile_data(df, 'new_issues_before_preprocessing.html')

    # Drop rows and columns
    df = preprocessing.drop_rows(df, verbose)
    df = preprocessing.drop_cols(df, verbose)

    # Create the DataFrame
    X = pd.DataFrame()

    # Build X
    X, df = preprocessing.add_non_feat_engineered_cols(X, df, verbose)

    # One-hot encode the categorical columns
    X, df = preprocessing.one_hot_encode(X, df, verbose)

    # Create additional columns
    X, df = preprocessing.add_use_of_proceeds_cols(X, df, verbose)
    X, df = preprocessing.add_distribution_cols(X, df, verbose)
    X, df = preprocessing.add_dealer_role_cols(X, df, verbose)
    X, df = preprocessing.add_years(X, df, verbose)
    X, df = preprocessing.add_ratings(X, df, verbose)
    X, df = preprocessing.add_seniority(X, df, verbose)
    X, df = preprocessing.add_deal_info(X, df, verbose)

    # Create X_addl (which holds the columns that will be used in some models but not in others)
    X_addl, df = preprocessing.create_X_addl(X, df, verbose)

    # Add the engineered columns
    X = feature_engineering.add_similarity_cols(X, field, verbose)

    # Profile the data (again)
    # if verbose:
    #    profile_data(X, 'new_issues_after_preprocessing.html')

    # MODELS:

    # Linear regression
    #print('Linear regression:\n')
    #models.run_linear_regression(X, X_addl, use_X_addl, train_size, test_size)

    # Lasso regression
    #print('Ridge / Lasso regression:\n')
    #models.run_ridge_lasso_regression(X, X_addl, use_X_addl, train_size, test_size)

    # Random forest regression
    #print('Random forest regression:\n')
    #models.run_random_forest_regression(X, X_addl, use_X_addl, train_size, test_size)

    # SVR
    print('SVR:\n')
    models.run_svr(X, X_addl, use_X_addl, train_size, test_size)
