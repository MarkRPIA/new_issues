########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import pandas as pd
# import pandas_profiling
import numpy as np


########################################################################################################################
# FUNCTIONS ############################################################################################################
########################################################################################################################

# Loads new issues data from file.
def load_data(verbose):
    if verbose:
        print('Loading data..')

    df = pd.read_csv('new_issue_data.csv', encoding='utf-8-sig')

    if verbose:
        print('Columns: {}'.format(list(df)))
        print('Shape: {}'.format(df.shape))

        print('Successfully loaded data!')
        print()

    return df


# Uses pandas_profiling to perform EDA.
# def profile_data(df, filename):
#    profile_report = pandas_profiling.ProfileReport(df)
#    profile_report.to_file(output_file=filename)


# Drops rows we don't want.
def drop_rows(df, verbose):
    # Drop rows that are not corps or financials
    df = df.drop(df[(df['Issuer Type'] == 'SOV') | (df['Issuer Type'] == '-')].index)

    # Drop rows that do not have a CUSIP
    df = df.drop(df[df['CUSIP'] == '-'].index)

    # Drop rows that have no Spread
    df = df.drop(df[df['Spread '] == '-'].index)

    if verbose:
        print('Shape after dropping rows: {}'.format(df.shape))
        print()

    return df


# Drops columns we don't want.
def drop_cols(df, verbose):
    df = df.drop(['Settlement Date'], axis=1)  # not useful
    df = df.drop(['Bond Ticker', 'Issuer'], axis=1)  # could in theory one-hot encode, but then the input vectors would
    # be way too big
    df = df.drop(['Moody\'s/S&P/Fitch Rating'], axis=1)  # we have each of these individually
    df = df.drop(['Asset Class'], axis=1)  # this is almost all empty, plus 'Rank' is very similar
    df = df.drop(['Deal Size (Lcl Ccy)'], axis=1)  # it's all in USD, so 'Total Deal Size (USD)' is the same
    df = df.drop(['Tranche Size (USD)'], axis=1)  # do not know this beforehand
    df = df.drop(['Tenor'], axis=1)  # we get this information from 'Maturity Date' already
    df = df.drop(['Structure'], axis=1)  # we get this information from 'Call Date' already
    df = df.drop(['Benchmark'], axis=1)  # not useful
    df = df.drop(['Spread to Gov Benchmark'], axis=1)  # all zeroes
    df = df.drop(['Free r'], axis=1)  # this is the performance after it freed (i.e., after the greys) --> this is
    # basically the target, but I will pull from Bloomberg to make sure the data is
    # good
    df = df.drop(['3mL Equivalent (bps)', 'M/S Equivalent', 'Spread vs. Midswaps', 'MS'], axis=1)  # basically all
    # zeroes
    df = df.drop(['Movement from IPTS'], axis=1)  # will derive this to make sure it is accurate
    df = df.drop(['Orderbook Size (mm)'], axis=1)  # will derive this to make sure it is accurate
    df = df.drop(['Yield', 'Price'], axis=1)  # spread captures this information
    df = df.drop(['League Table Eligible'], axis=1)  # not useful
    df = df.drop(['Product Type '], axis=1)  # this is all 'Investment Grade'
    df = df.drop(['BofA ML Role'], axis=1)  # this is captured in the list of dealers running the deal
    df = df.drop(['Listing'], axis=1)  # this is not important and is almost all empty
    df = df.drop(['Phy. Bks'], axis=1)  # all empty
    df = df.drop(['Self Led'], axis=1)  # not useful and almost all 'No'
    df = df.drop(['Tapped Bond'], axis=1)  # not useful
    df = df.drop(['Sub industry'], axis=1)  # too many options, will just use 'Industry' instead
    df = df.drop(['CUSIP', 'ISIN '], axis=1)  # identifiers that are not useful
    df = df.drop(['Week Start', 'Quarter'], axis=1)  # not useful
    df = df.drop(['Country ', 'Sub-Region'], axis=1)  # too granular, will use 'Region' instead
    df = df.drop(['Tranche Size (Local)'], axis=1)  # everything in USD, so same as 'Tranche Size (USD)'
    df = df.drop(['Tranche Currency', 'FX Rate per USD'], axis=1)  # everything in USD
    df = df.drop(['Call Structure'], axis=1)  # same as 'Structure' and we get this information from 'Call Date' already
    df = df.drop(['Special Call Features'], axis=1)  # text and almost all empty
    df = df.drop(['Par Call Period'], axis=1)  # the issuer can redeem at par within this many months of maturity,
    # not useful
    df = df.drop(['Up/Downsized'], axis=1)  # do not know this beforehand
    df = df.drop(['Price Execution'], axis=1)  # all empty
    df = df.drop(['Tranche Comments', 'Coupon Rate Comments', 'General Comments (deal level)'], axis=1)  # text and
    # almost all
    # empty
    df = df.drop(['Bond Ticker.1'], axis=1)  # could in theory one-hot encode, but then the input vectors would be
    # way too big
    df = df.drop(['Desk'], axis=1)  # not useful
    df = df.drop(['Gross Spread '], axis=1)  # how much the syndicate is taking, not useful
    df = df.drop(['Total Fees'], axis=1)  # total fees including gross spread, lawyer fees, etc., not useful
    df = df.drop(['Co1', 'Co 2', 'Co 3', 'Co 4', 'Co 5'], axis=1)  # Cos don't even get books, they have nothing to do
    # with the actual deal (they only get involved for
    # relationship reasons amongst the dealers)
    df = df.drop(['IsDM'], axis=1)  # same as floating coupon type
    df = df.drop(['IsSpread'], axis=1)  # get this from coupon type and IsYield
    df = df.drop(['IssueBenchmark'], axis=1)  # not useful
    df = df.drop(['Price0', 'Yield0', 'Spread0',
                  'Price1', 'Yield1', 'Spread1',
                  'Price2', 'Yield2', 'Spread2',
                  'Price3', 'Yield3', 'Spread3',
                  'Price4', 'Yield4', 'Spread4',
                  'Price5', 'Yield5', 'Spread5',
                  'Price6', 'Yield6', 'Spread6',
                  'Price7', 'Yield7', 'Spread7',
                  'Price8', 'Yield8', 'Spread8',
                  'Price9', 'Yield9', 'Spread9',
                  'Price10', 'Yield10', 'Spread10'],
                 axis=1)  # these columns are only needed in order to compute the Performance columns, which I use later
    df = df.drop(['Ticker'], axis=1)  # not useful

    if verbose:
        print('Shape after dropping columns: {}'.format(df.shape))
        print()

    return df


# Adds columns that will be used directly into X.
# i.e., columns that don't require any feature engineering.
def add_non_feat_engineered_cols(X, df, verbose):
    X['NumBs'] = df['No. of B\'s'].replace('-', 0).astype(float)
    X['IsBailIn'] = df['Bail-in'].replace('Yes', 1).mask(df['Bail-in'] != 'Yes', 0)
    X['NumTranches'] = df['No of Tranches'].astype(float)
    X['CouponRate'] = df['Coupon Rate (Fixed %; FRN if Float)'].replace('-', 0).astype(float)
    X['IptSpread'] = df['IPT Spread'].astype(float)
    X['Guidance'] = df['Guidance'].astype(float)
    X['Area'] = df['Area'].astype(float)
    X['Concession'] = df['New Issue Premium'].replace('-', 0).replace(np.nan, 0).astype(float)
    X['NumBookrunners'] = df['# of Bookrunners'].replace('-', 0).astype(float)
    X['NumActiveBookrunners'] = df['# of Active'].replace('-', 0).astype(float)
    X['NumPassiveBookrunners'] = df['# of Passive'].replace('-', 0).astype(float)
    X['HasCoC'] = df['CoC'].replace('Yes', 1).mask(df['CoC'] != 'Yes', 0)
    X['HasCouponSteps'] = df['Coupon Steps'].replace('Yes', 1).mask(df['Coupon Steps'] != 'Yes', 0)
    X['HasParCall'] = df['Par Call'].replace('Yes', 1).mask(df['Par Call'] != 'Yes', 0)
    X['HasMakeWhole'] = df['Make Whole'].replace('Yes', 1).mask(df['Make Whole'] != 'Yes', 0)
    X['Marketing'] = df['Marketing'].replace('Yes', 1).mask(df['Marketing'] != 'Yes', 0)
    X['HasSpecialMandatoryRedemption'] = df['Special Mandatory Redemption'].replace('Yes', 1).mask(
        df['Special Mandatory Redemption'] != 'Yes', 0)
    X['ParAmt'] = df['Denom'].replace('-', 0).astype(float)
    X['AddOn'] = df['Add-On'].replace('Yes', 1).mask(df['Add-On'] != 'Yes', 0)
    X['Deal'] = df['Deal']
    X['NoGrow'] = df['No-Grow'].replace('Yes', 1).mask(df['No-Grow'] != 'Yes', 0)
    X['FirstTimeIssuer'] = df['First Time Bond Issuer'].replace('Yes', 1).mask(df['First Time Bond Issuer'] != 'Yes', 0)
    X['UnequalEconomics'] = df['Unequal Economics'].replace('Yes', 1).mask(df['Unequal Economics'] != 'Yes', 0)
    X['IsYield'] = df['IsYield'].replace('Y', 1).mask(df['IsYield'] != 'Y', 0)
    X['Performance0'] = df['Performance0']
    X['Performance1'] = df['Performance1']
    X['Performance2'] = df['Performance2']
    X['Performance3'] = df['Performance3']
    X['Performance4'] = df['Performance4']
    X['Performance5'] = df['Performance5']
    X['Performance6'] = df['Performance6']
    X['Performance7'] = df['Performance7']
    X['Performance8'] = df['Performance8']
    X['Performance9'] = df['Performance9']
    X['Performance10'] = df['Performance10']
    X['Vix'] = df['Vix'].astype(float)
    X['Vix5d'] = df['Vix5d'].astype(float)
    X['Srvix'] = df['Srvix'].astype(float)
    X['Srvix5d'] = df['Srvix5d'].astype(float)
    X['CdxIg'] = df['CdxIg'].astype(float)
    X['CdxIg5d'] = df['CdxIg5d'].astype(float)
    X['Usgg2y'] = df['Usgg2y'].astype(float)
    X['Usgg2y5d'] = df['Usgg2y5d'].astype(float)
    X['Usgg3y'] = df['Usgg3y'].astype(float)
    X['Usgg3y5d'] = df['Usgg3y5d'].astype(float)
    X['Usgg5y'] = df['Usgg5y'].astype(float)
    X['Usgg5y5d'] = df['Usgg5y5d'].astype(float)
    X['Usgg10y'] = df['Usgg10y'].astype(float)
    X['Usgg10y5d'] = df['Usgg10y5d'].astype(float)
    X['Usgg30y'] = df['Usgg30y'].astype(float)
    X['Usgg30y5d'] = df['Usgg30y5d'].astype(float)
    X['EnterpriseValue'] = df['EnterpriseValue'].replace(np.nan, 0).replace('#N/A Invalid Security').astype(float)

    # Fill in NaNs with 0
    X = X.fillna(0)

    # Drop the columns from df
    df = df.drop(['No. of B\'s',
                  'Bail-in',
                  'No of Tranches',
                  'Coupon Rate (Fixed %; FRN if Float)',
                  'IPT Spread',
                  'Guidance',
                  'Area',
                  'New Issue Premium',
                  '# of Bookrunners',
                  '# of Active',
                  '# of Passive',
                  'CoC',
                  'Coupon Steps',
                  'Par Call',
                  'Make Whole',
                  'Marketing',
                  'Special Mandatory Redemption',
                  'Denom',
                  'Add-On',
                  'Deal',
                  'No-Grow',
                  'First Time Bond Issuer',
                  'Unequal Economics',
                  'IsYield',
                  'Performance0',
                  'Performance1',
                  'Performance2',
                  'Performance3',
                  'Performance4',
                  'Performance5',
                  'Performance6',
                  'Performance7',
                  'Performance8',
                  'Performance9',
                  'Performance10',
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
                  ], axis=1)

    if verbose:
        print('Shape after adding non-feature engineered columns:')
        print('X: {}'.format(X.shape))
        print('df: {}'.format(df.shape))
        print()

    return X, df


# One-hot encode the categorical variables
def one_hot_encode(X, df, verbose):
    df['Rate Type Selection'] = df['Rate Type Selection'].replace('3 month LIBOR', 'LIBOR')  # these are the same
    df = df.rename(columns={
                            'Coupon Type': 'CouponType',
                            'Rate Type Selection': 'RateTypeSelection',
                            'Issuer Type': 'IssuerType',
                            'Industry ': 'Industry'})  # rename some columns

    categorical_cols = ['CouponType', 'RateTypeSelection', 'IssuerType', 'Industry', 'Region']
    X_ohe = pd.get_dummies(df[categorical_cols], drop_first=True)

    # Add to the DataFrame

    X = pd.concat([X, X_ohe], axis=1)

    # Drop the columns from df

    df = df.drop(categorical_cols, axis=1)

    if verbose:
        print('Shape after one-hot encoding:')
        print('X: {}'.format(X.shape))
        print('df: {}'.format(df.shape))
        print()

    return X, df


# Helper function for creating the use of proceeds columns
def get_use_of_proceeds_col(row, field, check_equality):
    if check_equality:
        if field == row['UOP 1'].lower():
            return 2
        elif field == row['UOP 2'].lower():
            return 1
        else:
            return 0
    else:
        if field in row['UOP 1'].lower():
            return 2
        elif field in row['UOP 2'].lower():
            return 1
        else:
            return 0


# Add the use of proceeds columns to X
def add_use_of_proceeds_cols(X, df, verbose):
    X['UseOfProceedsAcquisition'] = df.apply(get_use_of_proceeds_col, field="acquisition", check_equality=False,
                                             axis=1)
    X['UseOfProceedsCap'] = df.apply(get_use_of_proceeds_col, field="cap", check_equality=True,
                                     axis=1)  # working capital
    X['UseOfProceedsCapex'] = df.apply(get_use_of_proceeds_col, field="capital expenditures", check_equality=True,
                                       axis=1)
    X['UseOfProceedsDividend'] = df.apply(get_use_of_proceeds_col, field="dividend", check_equality=True,
                                          axis=1)
    X['UseOfProceedsGcp'] = df.apply(get_use_of_proceeds_col, field="gcp", check_equality=True,
                                     axis=1)  # general corporate purposes
    X['UseOfProceedsGreen'] = df.apply(get_use_of_proceeds_col, field="green", check_equality=True,
                                       axis=1)  # green bond
    X['UseOfProceedsLiabilityManagement'] = df.apply(get_use_of_proceeds_col, field="liability management",
                                                     check_equality=True, axis=1)
    X['UseOfProceedsPensionContributions'] = df.apply(get_use_of_proceeds_col, field="pension contributions",
                                                      check_equality=True, axis=1)
    X['UseOfProceedsRefi'] = df.apply(get_use_of_proceeds_col, field="refi", check_equality=False,
                                      axis=1)
    X['UseOfProceedsBridgeLoan'] = df.apply(get_use_of_proceeds_col, field="rolled bridge loan",
                                            check_equality=True, axis=1)
    X['UseOfProceedsShareRepurchase'] = df.apply(get_use_of_proceeds_col, field="share repurchase",
                                                 check_equality=True, axis=1)

    # Drop the columns from df

    df = df.drop(['UOP 1', 'UOP 2'], axis=1)

    if verbose:
        print('Shape after adding the use of proceeds columns:')
        print('X: {}'.format(X.shape))
        print('df: {}'.format(df.shape))
        print()

    return X, df


# Helper function for creating the distribution columns
def get_distribution_type_col(row, field, check_equality):
    if check_equality:
        if field == row['Distribution'].lower():
            return 1
        else:
            return 0
    else:
        if field in row['Distribution'].lower():
            return 1
        else:
            return 0


# Add the distribution columns to X
def add_distribution_cols(X, df, verbose):
    X['Is144a'] = df.apply(get_distribution_type_col, field="144a", check_equality=False, axis=1)
    X['IsRegS'] = df.apply(get_distribution_type_col, field="reg s", check_equality=False, axis=1)
    X['HasRegRights'] = df.apply(get_distribution_type_col, field="reg right", check_equality=False, axis=1)
    X['Is3a2'] = df.apply(get_distribution_type_col, field="3(a)(2)", check_equality=True, axis=1)
    X['IsPublicOffering'] = df.apply(get_distribution_type_col, field="public offering", check_equality=True, axis=1)
    X['IsCD'] = df.apply(get_distribution_type_col, field="cd", check_equality=True, axis=1)

    # Drop the column from df

    df = df.drop('Distribution', axis=1)

    if verbose:
        print('Shape after adding the distribution columns:')
        print('X: {}'.format(X.shape))
        print('df: {}'.format(df.shape))
        print()

    return X, df







