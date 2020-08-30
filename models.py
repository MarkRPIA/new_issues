########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import model_helpers

from sklearn.ensemble import RandomForestClassifier

########################################################################################################################
# FUNCTIONS ############################################################################################################
########################################################################################################################


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
