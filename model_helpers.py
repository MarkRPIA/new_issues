########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV

from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ROCAUC

from imblearn.over_sampling import RandomOverSampler


########################################################################################################################
# FUNCTIONS ############################################################################################################
########################################################################################################################


# Sets up the training and test data.
# Note that this splits the target from -inf to lower_threshold, lower_threshold to upper_threshold,
# and upper_threshold to inf. If lower_threshold == upper_threshold, then it splits the target from -inf
# to lower_threshold == upper_threshold and lower_threshold == upper_threshold to inf.
def prepare_training_and_test_data(X, X_addl, use_X_addl, num_days_performance, lower_threshold, upper_threshold,
                                   train_size, test_size):
    # Add the additional columns if requested
    if use_X_addl:
        X = pd.concat([X, X_addl], axis=1)

    # Drop all Performance columns other than the one for the day we care about (this will become the target)
    for i in range(11):
        if i != num_days_performance:
            X = X.drop(['Performance{}'.format(i)], axis=1)

    target_col = 'Performance{}'.format(num_days_performance)

    # Drop rows that have no target
    X = X.drop(X[X[target_col] == '#VALUE!'].index)

    # Convert the target to float
    X[target_col] = X[target_col].astype(float)

    # Get the target
    y = X[target_col]
    X = X.drop([target_col], axis=1)

    # Label encode y (it is a multiclass problem since each sample must belong to exactly one class)
    if lower_threshold == upper_threshold:
        y = y.mask(y <= lower_threshold, '<={}'.format(lower_threshold)).mask(y > lower_threshold,
                                                                              '>{}'.format(lower_threshold))
    else:
        y = y.mask(y <= lower_threshold, '<={}'.format(lower_threshold)).mask(
            (y > lower_threshold) & (y < upper_threshold), '{}<x<{}'.format(lower_threshold, upper_threshold)).mask(
            y >= upper_threshold, '>={}'.format(upper_threshold))

    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split into training and test data
    n = len(X)
    train_end_ind = int(n * train_size)
    test_end_ind = train_end_ind + int(n * test_size)

    # Note that this splitting is inclusive of the first index and exclusive of the second index
    X_train = X[0:train_end_ind]
    y_train = y[0:train_end_ind]

    # Oversample the minority class
    rs = RandomOverSampler(sampling_strategy=1, random_state=7)
    X_train, y_train = rs.fit_resample(X_train, y_train)

    # Re-sort so we don't lose the time series ordering
    X_train['temp'] = rs.sample_indices_
    X_train = X_train.sort_values(by=['temp'])
    X_train = X_train.drop(['temp'], axis=1)

    sort_inds = np.argsort(rs.sample_indices_)
    y_train = y_train[sort_inds]

    X_test = X[train_end_ind:test_end_ind]
    y_test = y[train_end_ind:test_end_ind]

    return X_train, y_train, X_test, y_test, le.classes_


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


# Helper function for getting the cross-validation splits in a way that eliminates look-ahead bias.
def get_cv_splits(X_train, initial_train_size, val_size):
    n = len(X_train)
    inds = np.arange(n)

    cv_splits = [(inds[:i], inds[i:i + val_size]) for i in range(initial_train_size, n, val_size)]
    if len(cv_splits[-1][1]) < val_size * 0.75:  # the last fold must have at least 75% of the data we are asking for
        cv_splits = cv_splits[:-1]

    return cv_splits


# Helper function to do recursive feature elimination.
def do_recursive_feature_elimination(clf, X_train, y_train, cv_splits, scoring_fn, name):
    rfecv = RFECV(estimator=clf, step=1, cv=cv_splits, scoring=scoring_fn, verbose=1)
    rfecv.fit(X_train, y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)
    print('\nOptimal columns:')
    print(X_train.columns[rfecv.get_support()])

    # Plot number of features vs cross-validation scores
    plt.figure(figsize=(16, 10))
    plt.grid(True)
    plt.title('New Issues, {}'.format(name), fontsize=22)
    plt.xlabel("Number of features", fontsize=22)
    plt.ylabel("Cross validation score ({})".format(scoring_fn), fontsize=22)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('out/recursive-feature-elimination-{}.png'.format(name))

    return rfecv


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
