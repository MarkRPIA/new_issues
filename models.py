########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import model_helpers

from sklearn.ensemble import RandomForestClassifier

########################################################################################################################
# FUNCTIONS ############################################################################################################
########################################################################################################################


# Optimizes and runs a random forest model.
def run_random_forest(X, X_addl, use_X_addl, num_days_performance, lower_threshold, upper_threshold,
                      train_size, test_size, n_estimators, max_features, max_depth):
    # Prepare the data
    X_train, y_train, X_test, y_test, labels = model_helpers.prepare_training_and_test_data(X, X_addl, use_X_addl,
                                                                                            num_days_performance,
                                                                                            lower_threshold,
                                                                                            upper_threshold, train_size,
                                                                                            test_size)

    # Run the model
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                    bootstrap=True, random_state=7)
    rf_clf.fit(X_train, y_train)

    model_helpers.show_model_stats(rf_clf, X_train, y_train, X_test, y_test, labels, "random-forest")

    # Do recursive feature elimination
    initial_train_size = 500
    val_size = 100

    cv_splits = model_helpers.get_cv_splits(X_train, initial_train_size, val_size)
    rf_rfecv = model_helpers.do_recursive_feature_elimination(rf_clf, X_train, y_train, cv_splits, 'roc_auc_ovo',
                                                              'random-forest')

    # Get the optimal set of features
    X_train_optimal_features = X_train[X_train.columns[rf_rfecv.get_support()]]
    X_test_optimal_features = X_test[X_train.columns[rf_rfecv.get_support()]]

    print('Shape after only including the optimal features:')
    print('X train: {}'.format(X_train_optimal_features.shape))
    print()

    # Run the Random Forest again on the opimal set of features
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                    bootstrap=True, random_state=7)
    rf_clf.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(rf_clf, X_train_optimal_features, y_train, X_test_optimal_features, y_test, labels,
                                   "random-forest-optimal-features")

    # Do hyperparameter grid search
    rf_hyperparams = {
        'n_estimators': [10, 25, 50, 100],  # number of trees in the forest
        'max_features': (None, 'sqrt', 'log2'),
        'max_depth': [5, 10, 20, 30, 40, 50],  # the maximum depth of a tree
        'criterion': ('gini', 'entropy'),  # function to measure the quality of a split
        'max_leaf_nodes': [None, 25, 50, 100, 200]
    }

    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                    bootstrap=True, random_state=7)

    cv_splits = model_helpers.get_cv_splits(X_train, initial_train_size, val_size)
    results = model_helpers.do_hyperparameter_grid_search(rf_clf, rf_hyperparams, X_train_optimal_features, y_train,
                                                          cv_splits, 'roc_auc_ovo', 'random-forest-grid-search')

    print('The results of the random forest hyperparameter grid search are as follows:')
    print(results)

    # Run the Random Forest again on the optimal set of hyperparameters
    rf_clf = RandomForestClassifier(criterion='gini', max_depth=20, max_features='sqrt', max_leaf_nodes=None,
                                    n_estimators=25, bootstrap=True, random_state=7)
    rf_clf.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(rf_clf, X_train_optimal_features, y_train, X_test_optimal_features, y_test, labels,
                                   "random-forest-optimal-hyperparameters")
