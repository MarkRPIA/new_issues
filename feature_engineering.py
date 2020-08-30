########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import math
import numpy as np
from datetime import timedelta

########################################################################################################################
# FUNCTIONS ############################################################################################################
########################################################################################################################


# Helper function to get the rating bucket (ex: A-, A, and A+ are all grouped together).
def get_rating_bucket(rating_value):
    # Note that this relies on the fact that our ratings scale starts at 0, then goes up in groups of 3
    return math.ceil(rating_value / 3)


# Gets the performance of similar new issues. Note that "field" must be "Performance0", ..., "Performance10".
def get_similar_performance(row, X, field, num_lookback_days, same_sector, same_major_dealers,
                            same_seniority, same_rating, same_tenor):
    # Get the subset in the correct date range
    to_date = row['PricingDate']
    from_date = to_date - timedelta(days=num_lookback_days)
    X_lookback = X[(X['PricingDate'] < to_date) & (X['PricingDate'] >= from_date)]

    # Drop rows that have no value for "field"
    X_lookback = X_lookback.drop(X_lookback[X_lookback[field] == '#VALUE!'].index)

    # Convert the "field" value to float
    X_lookback[field] = X_lookback[field].astype(float)

    # Get the inds we want to average over
    inds = X_lookback['PricingDate'] < to_date  # all true

    # Same sector
    if same_sector:
        sector_inds = X_lookback['PricingDate'] < to_date  # all true

        if row['Industry_Consumer & Retail'] > 0:
            sector_inds = sector_inds & (X_lookback['Industry_Consumer & Retail'] > 0)
        if row['Industry_Energy & Power'] > 0:
            sector_inds = sector_inds & (X_lookback['Industry_Energy & Power'] > 0)
        if row['Industry_Financial Institutions'] > 0:
            sector_inds = sector_inds & (X_lookback['Industry_Financial Institutions'] > 0)
        if row['Industry_Gaming'] > 0:
            sector_inds = sector_inds & (X_lookback['Industry_Gaming'] > 0)
        if row['Industry_Healthcare'] > 0:
            sector_inds = sector_inds & (X_lookback['Industry_Healthcare'] > 0)
        if row['Industry_Industrials'] > 0:
            sector_inds = sector_inds & (X_lookback['Industry_Industrials'] > 0)
        if row['Industry_Media & Telecom'] > 0:
            sector_inds = sector_inds & (X_lookback['Industry_Media & Telecom'] > 0)
        if row['Industry_Real Estate'] > 0:
            sector_inds = sector_inds & (X_lookback['Industry_Real Estate'] > 0)
        if row['Industry_Technology'] > 0:
            sector_inds = sector_inds & (X_lookback['Industry_Technology'] > 0)
        inds = inds & sector_inds

    # Same major dealers
    if same_major_dealers:
        dealer_inds = X_lookback['PricingDate'] > to_date  # all false

        if row['BARC'] > 0:
            dealer_inds = dealer_inds | (X_lookback['BARC'] > 0)
        if row['BNP'] > 0:
            dealer_inds = dealer_inds | (X_lookback['BNP'] > 0)
        if row['BOA'] > 0:
            dealer_inds = dealer_inds | (X_lookback['BOA'] > 0)
        if row['C'] > 0:
            dealer_inds = dealer_inds | (X_lookback['C'] > 0)
        if row['CS'] > 0:
            dealer_inds = dealer_inds | (X_lookback['CS'] > 0)
        if row['DB'] > 0:
            dealer_inds = dealer_inds | (X_lookback['DB'] > 0)
        if row['GS'] > 0:
            dealer_inds = dealer_inds | (X_lookback['GS'] > 0)
        if row['HSBC'] > 0:
            dealer_inds = dealer_inds | (X_lookback['HSBC'] > 0)
        if row['JPM'] > 0:
            dealer_inds = dealer_inds | (X_lookback['JPM'] > 0)
        if row['Mitsubishi'] > 0:
            dealer_inds = dealer_inds | (X_lookback['Mitsubishi'] > 0)
        if row['Mizuho'] > 0:
            dealer_inds = dealer_inds | (X_lookback['Mizuho'] > 0)
        if row['MS'] > 0:
            dealer_inds = dealer_inds | (X_lookback['MS'] > 0)
        if row['RBC'] > 0:
            dealer_inds = dealer_inds | (X_lookback['RBC'] > 0)
        if row['Wells'] > 0:
            dealer_inds = dealer_inds | (X_lookback['Wells'] > 0)
        inds = inds & dealer_inds

    # Same seniority
    if same_seniority:
        seniority_inds = X_lookback['Rank'] == row['Rank']
        inds = inds & seniority_inds

    # Same rating
    if same_rating:
        rating_bucket = get_rating_bucket(row['AvgRating'])
        rating_inds = X['AvgRating'].apply(get_rating_bucket) == rating_bucket
        inds = inds & rating_inds

    # Same tenor
    if same_tenor:
        tenor = row['YearsToMaturity']
        tenor_inds = (X['YearsToMaturity'] >= (tenor - 1.5)) & (X['YearsToMaturity'] <= (tenor + 1.5))  # within 3 years
        inds = inds & tenor_inds

    return X_lookback[inds][field].mean()


# Add the "similarity" columns.
def add_similarity_cols(X, field, verbose):
    X['HistSectorPerformance'] = X.apply(get_similar_performance, X=X, field=field, num_lookback_days=7,
                                         same_sector=True, same_major_dealers=False, same_seniority=False,
                                         same_rating=False, same_tenor=False, axis=1)
    X['HistSectorPerformance'] = X['HistSectorPerformance'].replace(np.nan, 0).astype(float)

    X['HistDealerPerformance'] = X.apply(get_similar_performance, X=X, field=field, num_lookback_days=7,
                                         same_sector=False, same_major_dealers=True, same_seniority=False,
                                         same_rating=False, same_tenor=False, axis=1)
    X['HistDealerPerformance'] = X['HistDealerPerformance'].replace(np.nan, 0).astype(float)

    X['HistSeniorityPerformance'] = X.apply(get_similar_performance, X=X, field=field, num_lookback_days=7,
                                            same_sector=False, same_major_dealers=False, same_seniority=True,
                                            same_rating=False, same_tenor=False, axis=1)
    X['HistSeniorityPerformance'] = X['HistSeniorityPerformance'].replace(np.nan, 0).astype(float)

    X['HistRatingPerformance'] = X.apply(get_similar_performance, X=X, field=field, num_lookback_days=7,
                                         same_sector=False, same_major_dealers=False, same_seniority=False,
                                         same_rating=True, same_tenor=False, axis=1)
    X['HistRatingPerformance'] = X['HistRatingPerformance'].replace(np.nan, 0).astype(float)

    X['HistTenorPerformance'] = X.apply(get_similar_performance, X=X, field=field, num_lookback_days=7,
                                        same_sector=False, same_major_dealers=False, same_seniority=False,
                                        same_rating=False, same_tenor=True, axis=1)
    X['HistTenorPerformance'] = X['HistTenorPerformance'].replace(np.nan, 0).astype(float)

    X['HistAllPerformance'] = X.apply(get_similar_performance, X=X, field=field, num_lookback_days=30,
                                      same_sector=True, same_major_dealers=True, same_seniority=True,
                                      same_rating=True, same_tenor=True, axis=1)
    X['HistAllPerformance'] = X['HistAllPerformance'].replace(np.nan, 0).astype(float)

    X = X.drop(['PricingDate'], axis=1)

    if verbose:
        print('Shape after adding the similarity columns:')
        print('X: {}'.format(X.shape))
        print()

    return X
