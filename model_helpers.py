########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import linear_regression_helpers

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV

from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ROCAUC


########################################################################################################################
# FUNCTIONS ############################################################################################################
########################################################################################################################


# Sets up the training and test data.
def prepare_training_and_test_data(X, X_addl, use_X_addl, train_size, test_size):
    # Add the additional columns if requested
    if use_X_addl:
        X = pd.concat([X, X_addl], axis=1)

    # Drop performance columns
    for i in range(11):
        X = X.drop(['Performance{}'.format(i)], axis=1)

    # Drop rows where there is no Guidance
    X = X.drop(X[X['Guidance'] == 0].index)

    # Set the target column
    target_col = 'IssueSpread'

    # Drop rows that have no target
    X = X.drop(X[X[target_col] == '#VALUE!'].index)

    # Convert the target to float
    X[target_col] = X[target_col].astype(float)

    # Get the target
    y = X[target_col]
    X = X.drop([target_col], axis=1)

    # Scale (note that I leave binary columns alone)
    standard_scaler_cols = ['NumBs',
                            'NumTranches',
                            'CouponRate',
                            'IptSpread',
                            'Guidance',
                            'Area',
                            'Concession',
                            'NumBookrunners',
                            'NumActiveBookrunners',
                            'NumPassiveBookrunners',
                            'ParAmt',
                            'Vix',
                            'Vix5d',
                            'Srvix',
                            'Srvix5d',
                            'CdxIg',
                            'CdxIg5d',
                            'Usgg2y',
                            'Usgg2y5d',
                            'Usgg3y',
                            'Usgg3y5d',
                            'Usgg5y',
                            'Usgg5y5d',
                            'Usgg10y',
                            'Usgg10y5d',
                            'Usgg30y',
                            'Usgg30y5d',
                            'EnterpriseValue',
                            'YearsToCall',
                            'YearsToMaturity',
                            'MoodyRating',
                            'SpRating',
                            'FitchRating',
                            'AvgRating',
                            'Rank',
                            'OrderbookSize',
                            'IptToGuidance']

    scaler = StandardScaler()
    X[standard_scaler_cols] = scaler.fit_transform(X[standard_scaler_cols])

    # Split into training and test data
    n = len(X)
    train_end_ind = int(n * train_size)
    test_end_ind = train_end_ind + int(n * test_size)

    # Note that this splitting is inclusive of the first index and exclusive of the second index
    X_train = X[0:train_end_ind]
    y_train = y[0:train_end_ind]

    X_test = X[train_end_ind:test_end_ind]
    y_test = y[train_end_ind:test_end_ind]

    return X_train, y_train, X_test, y_test


# Helper function for showing model statistics.
def show_model_stats(clf, X_train, y_train, X_test, y_test, labels, name):
    i = 0
    print('In the training data, there are...')
    for label in labels:
        print(
            '{} with performance {} ({:.0f}%)'.format(sum(y_train == i), label, sum(y_train == i) / len(y_train) * 100))
        i = i + 1
    print()

    i = 0
    print('In the test data, there are...')
    for label in labels:
        print('{} with performance {} ({:.0f}%)'.format(sum(y_test == i), label, sum(y_test == i) / len(y_test) * 100))
        i = i + 1
    print()

    # Create the confusion matrices
    confusion_matrix = plot_confusion_matrix(clf, X_train, y_train, display_labels=labels, cmap=plt.cm.Blues)
    confusion_matrix.ax_.set_title('Confusion Matrix, Training')

    fig = confusion_matrix.ax_.get_figure()
    fig.savefig('out/confusion-matrix-training-{}.png'.format(name), transparent=False)

    confusion_matrix = plot_confusion_matrix(clf, X_test, y_test, display_labels=labels, cmap=plt.cm.Blues)
    confusion_matrix.ax_.set_title('Confusion Matrix, Test')

    fig = confusion_matrix.ax_.get_figure()
    fig.savefig('out/confusion-matrix-test-{}.png'.format(name), transparent=False)

    plt.show()

    # Predict on training and test data
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Classification report
    print("Classification report: Training data")
    print(classification_report(y_train, y_train_pred))
    print()

    print("Classification report: Test data")
    print(classification_report(y_test, y_test_pred))
    print()

    # Class prediction error
    visualizer1 = ClassPredictionError(clf, classes=labels)
    visualizer1.fit(X_train, y_train)
    visualizer1.score(X_test, y_test)
    visualizer1.show()

    fig = visualizer1.ax.get_figure()
    fig.savefig('out/class-prediction-error-{}.png'.format(name), transparent=False)

    # ROC curve
    if clf.__class__.__name__ == 'SVC':  # if SVM
        visualizer2 = ROCAUC(clf, micro=False, macro=False, per_class=False, classes=labels)
    else:
        visualizer2 = ROCAUC(clf, classes=labels)

    visualizer2.fit(X_train, y_train)  # fits the training data to the visualizer
    visualizer2.score(X_test, y_test)  # evaluate the model on test data
    visualizer2.show()

    fig = visualizer2.ax.get_figure()
    fig.savefig('out/roc-curve-{}.png'.format(name), transparent=False)

    # Feature importance
    if hasattr(clf, 'feature_importances_'):
        features = X_train.columns
        importances = clf.feature_importances_
        indices = np.argsort(importances)

        fig = plt.figure(figsize=(10, 20))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')

        fig.savefig('out/feature-importances-{}.png'.format(name), transparent=False)


# Shows relevant info for the linear regression model
def show_linear_regression_info(X_train, y_train, X_test, y_test, lin_reg_results, name):
    # Train
    if name is None:
        data_type = "training-data"
    else:
        data_type = "training-data-{}".format(name)

    linear_regression_helpers.test_linearity(lin_reg_results, X_train, y_train, data_type)
    linear_regression_helpers.test_autocorrelation_of_errors(lin_reg_results, X_train, y_train, data_type)
    linear_regression_helpers.test_homoscedasticity(lin_reg_results, X_train, y_train, data_type)
    linear_regression_helpers.test_error_normality(lin_reg_results, X_train, y_train, data_type)

    print()
    pred_train = lin_reg_results.predict(sm.add_constant(X_train, has_constant='add'))
    print('R-squared on training data: {}'.format(r2_score(y_train, pred_train)))

    diff_train = pred_train - y_train
    mae_train = sum(abs(i) for i in diff_train) / len(diff_train)
    print('MAE on training data: {}'.format(mae_train))

    # Test
    if name is None:
        data_type = "test-data"
    else:
        data_type = "test-data-{}".format(name)

    linear_regression_helpers.test_linearity(lin_reg_results, X_test, y_test, data_type)
    linear_regression_helpers.test_autocorrelation_of_errors(lin_reg_results, X_test, y_test, data_type)
    linear_regression_helpers.test_homoscedasticity(lin_reg_results, X_test, y_test, data_type)
    linear_regression_helpers.test_error_normality(lin_reg_results, X_test, y_test, data_type)

    print()
    pred_test = lin_reg_results.predict(sm.add_constant(X_test, has_constant='add'))
    print('R-squared on test data: {}'.format(r2_score(y_test, pred_test)))

    diff_test = pred_test - y_test
    mae_test = sum(abs(i) for i in diff_test) / len(diff_test)
    print('MAE on test data: {}'.format(mae_test))


# Helper function for getting the cross-validation splits in a way that eliminates look-ahead bias.
def get_cv_splits(X_train, initial_train_size, val_size):
    n = len(X_train)
    inds = np.arange(n)

    cv_splits = [(inds[:i], inds[i:i + val_size]) for i in range(initial_train_size, n, val_size)]
    if len(cv_splits[-1][1]) < val_size * 0.75:  # the last fold must have at least 75% of the data we are asking for
        cv_splits = cv_splits[:-1]

    return cv_splits


# Helper function to do variance threshold feature selection.
# This is for models that don't expose "coef_" or "feature_importances_".
def do_variance_threshold_feature_selection(X_train, y_train, variance_threshold):
    sel = VarianceThreshold(threshold=variance_threshold)
    sel.fit(X_train, y_train)

    print('\nVariances:')
    print(sel.variances_)
    print('\nSupport:')
    print(sel.get_support())
    print('\n# of selected features = {}'.format(sum(sel.get_support())))

    return sel.get_support()


# Helper function to do hyperparameter grid search
def do_hyperparameter_grid_search(clf, hyperparams, X_train, y_train, cv_splits, scoring_fn, name):
    gridsearch = GridSearchCV(clf, hyperparams, cv=cv_splits, scoring=scoring_fn, return_train_score=True, verbose=1)
    gridsearch.fit(X_train, y_train)

    cv_results = gridsearch.cv_results_

    results = pd.DataFrame(list(cv_results['params']))
    results['mean_fit_time'] = cv_results['mean_fit_time']
    results['mean_score_time'] = cv_results['mean_score_time']
    results['mean_train_score'] = cv_results['mean_train_score']
    results['std_train_score'] = cv_results['std_train_score']
    results['mean_test_score'] = cv_results['mean_test_score']
    results['std_test_score'] = cv_results['std_test_score']
    results['rank_test_score'] = cv_results['rank_test_score']

    results = results.sort_values(['mean_test_score'], ascending=False)
    results.to_csv(results.to_csv('out/gridsearch-{}.csv'.format(name)))

    return results
