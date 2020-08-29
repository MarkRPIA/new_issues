########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import multiprocessing
import pandas as pd

import new_issues_preprocessing

########################################################################################################################
# GLOBALS ##############################################################################################################
########################################################################################################################

verbose = True

pd.set_option('display.max_columns', None)  # show all columns!

########################################################################################################################
# MAIN #################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    multiprocessing.freeze_support()  # needed for pandas_profiling
    print()

    # Load the data
    df = new_issues_preprocessing.load_data(verbose)

    # Profile the data
    #if verbose:
    #    helper_fns.profile_data(df, 'new_issues_before_preprocessing.html')

    # Drop rows and columns
    df = new_issues_preprocessing.drop_rows(df, verbose)
    df = new_issues_preprocessing.drop_cols(df, verbose)

    # Create the DataFrame
    X = pd.DataFrame()

    # Build X
    X, df = new_issues_preprocessing.add_non_feat_engineered_cols(X, df, verbose)

    # One-hot encode the categorical columns
    X, df = new_issues_preprocessing.one_hot_encode(X, df, verbose)

    # Create new columns
    X, df = new_issues_preprocessing.add_use_of_proceeds_cols(X, df, verbose)

