########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import model_helpers

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


########################################################################################################################
# FUNCTIONS ############################################################################################################
########################################################################################################################


# Optimizes and runs the random forest models.
def run_random_forest(X, X_addl, use_X_addl, num_days_performance, lower_threshold, upper_threshold, train_size,
                      test_size):
    # Prepare the data
    X_train, y_train, X_test, y_test, labels = model_helpers.prepare_training_and_test_data(X, X_addl, use_X_addl,
                                                                                            num_days_performance,
                                                                                            lower_threshold,
                                                                                            upper_threshold, train_size,
                                                                                            test_size)

    # Run the model
    rf_clf = RandomForestClassifier(n_estimators=100, max_features='auto', max_depth=20, bootstrap=True, random_state=7)
    rf_clf.fit(X_train, y_train)

    model_helpers.show_model_stats(rf_clf, X_train, y_train, X_test, y_test, labels, "random-forest")

    # Do recursive feature elimination
    initial_train_size = 500
    val_size = 100

    cv_splits = model_helpers.get_cv_splits(X_train, initial_train_size, val_size)
    rf_rfecv = model_helpers.do_recursive_feature_elimination(rf_clf, X_train, y_train, cv_splits, 'roc_auc',
                                                              'random-forest')

    # Get the optimal set of features
    X_train_optimal_features = X_train[X_train.columns[rf_rfecv.get_support()]]
    X_test_optimal_features = X_test[X_train.columns[rf_rfecv.get_support()]]

    print('Shape after only including the optimal features:')
    print('X train: {}'.format(X_train_optimal_features.shape))
    print()

    # Run the Random Forest again on the optimal set of features
    rf_clf = RandomForestClassifier(n_estimators=100, max_features='auto', max_depth=20, bootstrap=True, random_state=7)
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

    rf_clf = RandomForestClassifier(n_estimators=100, max_features='auto', max_depth=20, bootstrap=True, random_state=7)

    cv_splits = model_helpers.get_cv_splits(X_train, initial_train_size, val_size)
    results = model_helpers.do_hyperparameter_grid_search(rf_clf, rf_hyperparams, X_train_optimal_features, y_train,
                                                          cv_splits, 'roc_auc', 'random-forest')

    print('The results of the random forest hyperparameter grid search are as follows:')
    print(results)

    # Run the Random Forest again on the optimal set of hyperparameters
    rf_clf = RandomForestClassifier(criterion='entropy', max_depth=30, max_features='sqrt', max_leaf_nodes=25,
                                    n_estimators=10, bootstrap=True, random_state=7)
    rf_clf.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(rf_clf, X_train_optimal_features, y_train, X_test_optimal_features, y_test, labels,
                                   "random-forest-optimal-hyperparameters")


# Optimizes and runs the SVM models.
def run_svm(X, X_addl, use_X_addl, num_days_performance, lower_threshold, upper_threshold, train_size, test_size):
    # Prepare the data
    X_train, y_train, X_test, y_test, labels = model_helpers.prepare_training_and_test_data(X, X_addl, use_X_addl,
                                                                                            num_days_performance,
                                                                                            lower_threshold,
                                                                                            upper_threshold, train_size,
                                                                                            test_size)

    # Run the model
    svm_clf = SVC(kernel='poly', C=10.0, random_state=7)  # linear, poly, rbf, sigmoid
    svm_clf.fit(X_train, y_train)

    model_helpers.show_model_stats(svm_clf, X_train, y_train, X_test, y_test, labels, 'svm')

    # Do variance threshold feature selection
    support = model_helpers.do_variance_threshold_feature_selection(X_train, y_train, 1.0)
    print('The optimal features are: ')
    print(X_train.columns[support])

    # Get the optimal set of features
    X_train_optimal_features = X_train.loc[:, support]
    X_test_optimal_features = X_test.loc[:, support]

    print('Shape after only including the optimal features:')
    print('X train: {}'.format(X_train_optimal_features.shape))
    print()

    # Run the SVM again on the optimal set of features
    svm_clf = SVC(kernel='poly', C=10.0, random_state=7)
    svm_clf.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(svm_clf, X_train_optimal_features, y_train, X_test_optimal_features, y_test, labels,
                                   "svm-optimal-features")

    # Do hyperparameter grid search
    svm_hyperparams = {
        'C': [0.001, 0.01, 0.10, 0.50, 1.0, 10.0],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    svm_clf = SVC(kernel='poly', C=10.0, random_state=7)

    initial_train_size = 500
    val_size = 100
    cv_splits = model_helpers.get_cv_splits(X_train, initial_train_size, val_size)

    results = model_helpers.do_hyperparameter_grid_search(svm_clf, svm_hyperparams, X_train_optimal_features, y_train,
                                                          cv_splits, 'roc_auc', 'svm')

    print('The results of the SVM hyperparameter grid search are as follows:')
    print(results)

    # Run the SVM again on the optimal set of hyperparameters
    svm_clf = SVC(kernel='rbf', C=0.1, random_state=7)
    svm_clf.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(svm_clf, X_train_optimal_features, y_train, X_test_optimal_features, y_test, labels,
                                   "svm-optimal-hyperparameters")


# Optimizes and runs the logistic regression models.
def run_logistic_regression(X, X_addl, use_X_addl, num_days_performance, lower_threshold, upper_threshold, train_size,
                            test_size):
    # Prepare the data
    X_train, y_train, X_test, y_test, labels = model_helpers.prepare_training_and_test_data(X, X_addl, use_X_addl,
                                                                                            num_days_performance,
                                                                                            lower_threshold,
                                                                                            upper_threshold, train_size,
                                                                                            test_size)

    # Run the model
    lr_clf = LogisticRegression(penalty='l2', C=0.07, class_weight='balanced', max_iter=10000, random_state=7)
    lr_clf.fit(X_train, y_train)

    model_helpers.show_model_stats(lr_clf, X_train, y_train, X_test, y_test, labels, 'logistic-regression')

    # Do variance threshold feature selection
    support = model_helpers.do_variance_threshold_feature_selection(X_train, y_train, 1.0)
    print('The optimal features are: ')
    print(X_train.columns[support])

    # Get the optimal set of features
    X_train_optimal_features = X_train.loc[:, support]
    X_test_optimal_features = X_test.loc[:, support]

    print('Shape after only including the optimal features:')
    print('X train: {}'.format(X_train_optimal_features.shape))
    print()

    # Run the logistic regression again on the optimal set of features
    lr_clf = LogisticRegression(penalty='l2', C=0.07, class_weight='balanced', max_iter=10000, random_state=7)
    lr_clf.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(lr_clf, X_train_optimal_features, y_train, X_test_optimal_features, y_test, labels,
                                   "logistic-regression-optimal-features")

    # Do hyperparameter grid search
    lr_hyperparams = {
        'C': [0.001, 0.01, 0.10, 0.50, 1.0, 10.0]
    }

    lr_clf = LogisticRegression(penalty='l2', C=0.07, class_weight='balanced', max_iter=10000, random_state=7)

    initial_train_size = 500
    val_size = 100
    cv_splits = model_helpers.get_cv_splits(X_train, initial_train_size, val_size)

    results = model_helpers.do_hyperparameter_grid_search(lr_clf, lr_hyperparams, X_train_optimal_features, y_train,
                                                          cv_splits, 'roc_auc', 'logistic-regression')

    print('The results of the logistic regression hyperparameter grid search are as follows:')
    print(results)

    # Run the logistic regression again on the optimal set of hyperparameters
    lr_clf = LogisticRegression(penalty='l2', C=10.0, class_weight='balanced', max_iter=10000, random_state=7)
    lr_clf.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(lr_clf, X_train_optimal_features, y_train, X_test_optimal_features, y_test, labels,
                                   "logistic-regression-optimal-hyperparameters")


# Optimizes and runs the naive Bayes models.
def run_naive_bayes(X, X_addl, use_X_addl, num_days_performance, lower_threshold, upper_threshold, train_size,
                    test_size):
    # Prepare the data
    X_train, y_train, X_test, y_test, labels = model_helpers.prepare_training_and_test_data(X, X_addl, use_X_addl,
                                                                                            num_days_performance,
                                                                                            lower_threshold,
                                                                                            upper_threshold, train_size,
                                                                                            test_size)

    # Run the model
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)

    model_helpers.show_model_stats(nb_clf, X_train, y_train, X_test, y_test, labels, 'naive-bayes')

    # Do variance threshold feature selection
    support = model_helpers.do_variance_threshold_feature_selection(X_train, y_train, 1.0)
    print('The optimal features are: ')
    print(X_train.columns[support])

    # Get the optimal set of features
    X_train_optimal_features = X_train.loc[:, support]
    X_test_optimal_features = X_test.loc[:, support]

    print('Shape after only including the optimal features:')
    print('X train: {}'.format(X_train_optimal_features.shape))
    print()

    # Run the naive Bayes again on the optimal set of features
    nb_clf = GaussianNB()
    nb_clf.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(nb_clf, X_train_optimal_features, y_train, X_test_optimal_features, y_test, labels,
                                   "naive-bayes-optimal-features")


# Optimizes and runs the KNN models.
def run_knn(X, X_addl, use_X_addl, num_days_performance, lower_threshold, upper_threshold, train_size, test_size):
    # Prepare the data
    X_train, y_train, X_test, y_test, labels = model_helpers.prepare_training_and_test_data(X, X_addl, use_X_addl,
                                                                                            num_days_performance,
                                                                                            lower_threshold,
                                                                                            upper_threshold, train_size,
                                                                                            test_size)

    # Run the model
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)

    model_helpers.show_model_stats(knn_clf, X_train, y_train, X_test, y_test, labels, 'knn')

    # Do variance threshold feature selection
    support = model_helpers.do_variance_threshold_feature_selection(X_train, y_train, 1.0)
    print('The optimal features are: ')
    print(X_train.columns[support])

    # Get the optimal set of features
    X_train_optimal_features = X_train.loc[:, support]
    X_test_optimal_features = X_test.loc[:, support]

    print('Shape after only including the optimal features:')
    print('X train: {}'.format(X_train_optimal_features.shape))
    print()

    # Run the KNN again on the optimal set of features
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(knn_clf, X_train_optimal_features, y_train, X_test_optimal_features, y_test, labels,
                                   "knn_clf-optimal-features")

    # Do hyperparameter grid search
    knn_hyperparams = {
        'n_neighbors': [1, 2, 5, 10, 15, 20, 30, 40, 50]
    }

    knn_clf = KNeighborsClassifier(n_neighbors=5)

    initial_train_size = 500
    val_size = 100
    cv_splits = model_helpers.get_cv_splits(X_train, initial_train_size, val_size)

    results = model_helpers.do_hyperparameter_grid_search(knn_clf, knn_hyperparams, X_train_optimal_features, y_train,
                                                          cv_splits, 'roc_auc', 'knn')

    print('The results of the KNN hyperparameter grid search are as follows:')
    print(results)

    # Run the KNN again on the optimal set of hyperparameters
    knn_clf = KNeighborsClassifier(n_neighbors=40)
    knn_clf.fit(X_train_optimal_features, y_train)

    model_helpers.show_model_stats(knn_clf, X_train_optimal_features, y_train, X_test_optimal_features, y_test, labels,
                                   "knn-optimal-hyperparameters")
