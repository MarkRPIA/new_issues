########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import model_helpers

import numpy as np

import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


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


# Optimizes and runs the random forest models.
def run_random_forest_regression(X, X_addl, use_X_addl, train_size, test_size):
    # Prepare the data
    X_train, y_train, X_test, y_test = model_helpers.prepare_training_and_test_data(X, X_addl, use_X_addl, train_size,
                                                                                    test_size)

    # Run the model
    rf_rg = RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=20, bootstrap=True, random_state=7)
    rf_rg.fit(X_train, y_train)

    model_helpers.show_model_stats(rf_rg, X_train, y_train, X_test, y_test, "random-forest")

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

    # Run the Random Forest again on the optimal set of features
    rf_rg = RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=20, bootstrap=True, random_state=7)
    rf_rg.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(rf_rg, X_train_optimal_features, y_train, X_test_optimal_features, y_test,
                                   "random-forest-optimal-features")

    # Do hyperparameter grid search
    rf_hyperparams = {
        'n_estimators': [10, 25, 50, 100],  # number of trees in the forest
        'max_features': (None, 'sqrt', 'log2'),
        'max_depth': [5, 10, 20, 30, 40, 50],  # the maximum depth of a tree
        'criterion': ('mse', 'mae'),  # function to measure the quality of a split
        'max_leaf_nodes': [None, 25, 50, 100, 200]
    }

    rf_rg = RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=20, bootstrap=True, random_state=7)

    initial_train_size = 700
    val_size = 200
    cv_splits = model_helpers.get_cv_splits(X_train, initial_train_size, val_size)

    results = model_helpers.do_hyperparameter_grid_search(rf_rg, rf_hyperparams, X_train_optimal_features, y_train,
                                                          cv_splits, 'neg_mean_absolute_error', 'random-forest')

    print('The results of the random forest hyperparameter grid search are as follows:')
    print(results)

    # Run the Random Forest again on the optimal set of hyperparameters
    rf_rg = RandomForestRegressor(criterion='mae', max_depth=40, max_features=None, max_leaf_nodes=200, n_estimators=25,
                                  bootstrap=True, random_state=7)
    rf_rg.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(rf_rg, X_train_optimal_features, y_train, X_test_optimal_features, y_test,
                                   "random-forest-optimal-hyperparameters")


# Optimizes and runs the SVR models.
def run_svr(X, X_addl, use_X_addl, train_size, test_size):
    # Prepare the data
    X_train, y_train, X_test, y_test = model_helpers.prepare_training_and_test_data(X, X_addl, use_X_addl, train_size,
                                                                                    test_size)

    # Run the model
    svr_rg = SVR(kernel='poly', C=1.0)  # linear, poly, rbf, sigmoid
    svr_rg.fit(X_train, y_train)

    model_helpers.show_model_stats(svr_rg, X_train, y_train, X_test, y_test, 'svr')

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

    # Run the SVR again on the optimal set of features
    svr_rg = SVR(kernel='linear', C=1.0)
    svr_rg.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(svr_rg, X_train_optimal_features, y_train, X_test_optimal_features, y_test,
                                   "svr-optimal-features")

    # Do hyperparameter grid search
    svr_hyperparams = {
        'C': [0.10, 1.0, 10.0, 25.0, 50.0, 100.0],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    svr_rg = SVR(kernel='linear', C=1.0)

    initial_train_size = 700
    val_size = 200
    cv_splits = model_helpers.get_cv_splits(X_train, initial_train_size, val_size)

    results = model_helpers.do_hyperparameter_grid_search(svr_rg, svr_hyperparams, X_train_optimal_features, y_train,
                                                          cv_splits, 'neg_mean_absolute_error', 'svr')

    print('The results of the SVR hyperparameter grid search are as follows:')
    print(results)

    # Run the SVR again on the optimal set of hyperparameters
    svr_rg = SVR(kernel='linear', C=10.0)
    svr_rg.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(svr_rg, X_train_optimal_features, y_train, X_test_optimal_features, y_test,
                                   "svr-optimal-hyperparameters")


# Optimizes and runs the KNN models.
def run_knn_regression(X, X_addl, use_X_addl, train_size, test_size):
    # Prepare the data
    X_train, y_train, X_test, y_test = model_helpers.prepare_training_and_test_data(X, X_addl, use_X_addl, train_size,
                                                                                    test_size)

    # Run the model
    knn_rg = KNeighborsRegressor(n_neighbors=5)
    knn_rg.fit(X_train, y_train)

    model_helpers.show_model_stats(knn_rg, X_train, y_train, X_test, y_test, 'knn')

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

    # Run the KNN again on the optimal set of features
    knn_rg = KNeighborsRegressor(n_neighbors=5)
    knn_rg.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(knn_rg, X_train_optimal_features, y_train, X_test_optimal_features, y_test,
                                   "knn-optimal-features")

    # Do hyperparameter grid search
    knn_hyperparams = {
        'n_neighbors': [1, 2, 5, 10, 15, 20, 30, 40, 50]
    }

    knn_rg = KNeighborsRegressor(n_neighbors=5)

    initial_train_size = 700
    val_size = 200
    cv_splits = model_helpers.get_cv_splits(X_train, initial_train_size, val_size)

    results = model_helpers.do_hyperparameter_grid_search(knn_rg, knn_hyperparams, X_train_optimal_features, y_train,
                                                          cv_splits, 'neg_mean_absolute_error', 'knn')

    print('The results of the KNN hyperparameter grid search are as follows:')
    print(results)

    # Run the KNN again on the optimal set of hyperparameters
    knn_rg = KNeighborsRegressor(n_neighbors=5)
    knn_rg.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(knn_rg, X_train_optimal_features, y_train, X_test_optimal_features, y_test,
                                   "knn-optimal-hyperparameters")
