########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import model_helpers

import numpy as np

import statsmodels.api as sm


########################################################################################################################
# FUNCTIONS ############################################################################################################
########################################################################################################################


# Optimizes and runs the linear regression models.
def run_linear_regression(X, X_addl, use_X_addl, train_size, test_size):
    # Prepare the data
    X_train, y_train, X_test, y_test = model_helpers.prepare_training_and_test_data(X, X_addl, use_X_addl, train_size,
                                                                                    test_size)

    # Fit the model on training data
    lin_reg_model = sm.OLS(y_train, sm.add_constant(X_train, has_constant='add'))
    lin_reg_results = lin_reg_model.fit()

    # Print the results
    print(lin_reg_results.summary())
    print()

    model_helpers.show_linear_regression_stats(lin_reg_results, X_train, y_train, X_test, y_test, None)

    # Only keep the statistically significant features
    cols_to_keep = [
        'NumBs',
        'CouponRate',
        'IptSpread',
        'Guidance',
        'Area',
        'Concession',
        'NumActiveBookrunners',
        'HasCouponSteps',
        'AddOn',
        'IsYield',
        'Vix',
        'Vix5d',
        'Srvix',
        'Usgg10y',
        'CouponType_Fixed to FRN',
        'CouponType_Floating',
        'Industry_Gaming',
        'IsLowTierB&D',
        'YearsToMaturity',
        'FitchRating',
        'Rank',
        'IptToGuidance',
        'HistRatingPerformance',
        'OverSubscription'
    ]

    X_train = X_train[cols_to_keep]
    X_test = X_test[cols_to_keep]

    # Fit the model on training data
    lin_reg_model = sm.OLS(y_train, sm.add_constant(X_train, has_constant='add'))
    lin_reg_results = lin_reg_model.fit()

    # Print the results
    print(lin_reg_results.summary())
    print()

    model_helpers.show_linear_regression_stats(lin_reg_results, X_train, y_train, X_test, y_test, "optimal-features")


# Optimizes and runs the ridge / lasso regression models.
def run_ridge_lasso_regression(X, X_addl, use_X_addl, train_size, test_size):
    # Prepare the data
    X_train, y_train, X_test, y_test = model_helpers.prepare_training_and_test_data(X, X_addl, use_X_addl, train_size,
                                                                                    test_size)

    # Fit the model on training data
    ridge_lasso_rg_model = sm.OLS(y_train, sm.add_constant(X_train, has_constant='add'))
    ridge_lasso_rg = ridge_lasso_rg_model.fit_regularized(alpha=0.1)

    model_helpers.show_model_stats(ridge_lasso_rg, sm.add_constant(X_train, has_constant='add'), y_train,
                                   sm.add_constant(X_test, has_constant='add'), y_test, 'ridge-lasso')

    # Do univariate feature selection
    support = model_helpers.do_univariate_feature_selection(X_train, y_train)
    print('The optimal features are: ')
    print(X_train.columns[support])

    # Get the optimal set of features
    X_train_optimal_features = X_train.loc[:, support]
    X_test_optimal_features = X_test.loc[:, support]

    print('Shape after only including the optimal features:')
    print('X train: {}'.format(X_train_optimal_features.shape))
    print()

    # Run the regression again on the optimal set of features
    ridge_lasso_rg_model = sm.OLS(y_train, sm.add_constant(X_train_optimal_features, has_constant='add'))
    ridge_lasso_rg = ridge_lasso_rg_model.fit_regularized(alpha=0.1)

    model_helpers.show_model_stats(ridge_lasso_rg, sm.add_constant(X_train_optimal_features, has_constant='add'),
                                   y_train, sm.add_constant(X_test_optimal_features, has_constant='add'), y_test,
                                   'ridge-lasso-optimal-features')

    # Do hyperparameter grid search
    alphas = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    L1_wts = [0, 0.25, 0.5, 0.75, 1]

    maes = {}
    min_mae = np.inf
    min_mae_alpha = None
    min_mae_L1_wt = None
    for alpha in alphas:
        maes[alpha] = {}
        for L1_wt in L1_wts:
            ridge_lasso_rg = ridge_lasso_rg_model.fit_regularized(alpha=alpha, L1_wt=L1_wt)

            pred_test = ridge_lasso_rg.predict(sm.add_constant(X_test_optimal_features, has_constant='add'))
            diff_test = pred_test - y_test
            mae_test = sum(abs(i) for i in diff_test) / len(diff_test)
            maes[alpha][L1_wt] = mae_test

            if mae_test < min_mae:
                min_mae = mae_test
                min_mae_alpha = alpha
                min_mae_L1_wt = L1_wt

    print('The optimal hyperparameters are alpha = {} and L1_wt = {}.'.format(min_mae_alpha, min_mae_L1_wt))
    print('They result in an MAE of {}.'.format(min_mae))

    # Run the ridge / lasso regression again on the optimal set of hyperparameters
    ridge_lasso_rg_model = sm.OLS(y_train, sm.add_constant(X_train_optimal_features, has_constant='add'))
    ridge_lasso_rg = ridge_lasso_rg_model.fit_regularized(alpha=min_mae_alpha, L1_wt=min_mae_L1_wt)

    model_helpers.show_model_stats(ridge_lasso_rg, sm.add_constant(X_train_optimal_features, has_constant='add'),
                                   y_train, sm.add_constant(X_test_optimal_features, has_constant='add'), y_test,
                                   'ridge-lasso-optimal-hyperparameters')
