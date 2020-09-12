########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import pandas as pd
import numpy as np


########################################################################################################################
# GLOBALS ##############################################################################################################
########################################################################################################################

# Dealers

# Note that some dealers are listed multiple times under slightly different names

dealer_to_aliases = {'ABN': ['ABN', 'ABN AMRO Bank N.V.'],
                     'Academy': ['Academy'], 'ANZ': ['ANZ'],
                     'Axis': ['Axis Bank'],
                     'BARC': ['Barc'],
                     'BBT': ['BB&T'],
                     'BBVA': ['BBVA'],
                     'BMO': ['BMO'],
                     'BNP': ['BNP'],
                     'BNS': ['Scotia', 'The Bank Of Nova Scotia GB', 'Scotiabank Europe Plc.'],
                     'BNY': ['BNY', 'BNY Mellon Capital Markets, LLC'],
                     'BOA': ['BofA'],
                     'BOC': ['BOC International Holdings Limited.', 'Bank of China (Hong Kong) Limited'],
                     'BoCom': ['Bank of Communications Co., Ltd. Hong Kong Branch'],
                     'Bradesco': ['Banco Bradesco BBI S.A.'],
                     'BTG': ['BTG Pactual US'],
                     'C': ['Citi'],
                     'CA': ['CA'],
                     'CCB': ['CCB International Capital Limited'],
                     'CG': ['Credit Agricole Corporate and Investment Bank (New York)'],
                     'CIBC': ['CIBC'],
                     'CIC': ['CIC'],
                     'Citizens': ['Citizens'],
                     'CJSC': ['CJSC Sberbank CIB'],
                     'CMB': ['CMB International Capital Limited'],
                     'CoBank': ['CoBank'],
                     'Commerzbank': ['Commerz'],
                     'COSC': ['COSC'],
                     'CS': ['CS'],
                     'Daiwa': ['Daiwa'],
                     'DB': ['DB'],
                     'DBS': ['DBS Bank Ltd'],
                     'DNB': ['DnB', 'DNB Bank ASA'],
                     'Emirates': ['Emirates NBD PJSC'],
                     'Fifth Third': ['Fifth Third Securiti'],
                     'Gazprombank': ['Gazprombank'],
                     'GS': ['GS', 'Goldman Sachs UK'],
                     'HSBC': ['HSBC'],
                     'Huntington': ['Huntington'],
                     'ICBC': ['ICBC'],
                     'IMI': ['Banca IMI'],
                     'ING': ['ING'],
                     'Itau': ['Itau BBA'],
                     'Jefferies': ['Jeff'],
                     'JPM': ['JPM'],
                     'KBC': ['KBC'],
                     'Keefe': ['Keefe, Bruyette & Woods, Inc', 'Keefe'],
                     'KeyBank': ['Key'],
                     'Lloyds': ['Lloyds'],
                     'Loop': ['Loop'],
                     'Macquarie': ['Macquarie Bank Limited'],
                     'Mandiri': ['Mandiri Securities Pte. Ltd.'],
                     'Mitsubishi': ['MUFJ', 'MUFG'],
                     'Mizuho': ['Mizuho', 'Mizuho Securities USA Inc'],
                     'MS': ['MS'],
                     'NAB': ['NAB'],
                     'Natixis': ['Natixis'],
                     'NatWest': ['NATWEST'],
                     'NBAD': ['National Bank of Abu Dhabi'],
                     'NBC': ['NBC'],
                     'Nomura': ['Nomura'],
                     'NW': ['NW Farm'],
                     'Oversea Chinese': ['Oversea-Chinese Bank'],
                     'Piper': ['Piper'],
                     'PNC': ['PNC'],
                     'Rabo': ['Rabo'],
                     'Raiffeisen': ['Raiffeisen'],
                     'RBC': ['RBC'],
                     'RBS': ['RBS'],
                     'Regions': ['Regions'],
                     'RJ': ['RJ'],
                     'Santander': ['Santander', 'Banco Santander SA'],
                     'SEB': ['SEB'],
                     'Seelaus': ['R. Seelaus'],
                     'SG': ['SG'],
                     'SMBC': ['SMBC', 'SMBC Nikko Securities', 'SMBC Nikko Capital Markets Limited'],
                     'Standard': ['Standard Bank'],
                     'Standard Chartered': ['Std Chrtrd'],
                     'Stephens': ['Stephens'],
                     'STRH': ['STRH'],
                     'Sumitomo': ['SumiMit'],
                     'SunTrust': ['SunTrust Robinson', 'SunTrust Robinson Humphrey, Inc.',
                                  'SunTrust Robinson Humphrey'],
                     'TD': ['TD', 'TD Securities', 'TD Securities (USA), LLC'],
                     'UBS': ['UBS'],
                     'UMB': ['UMB'],
                     'Unicredit': ['Unicredit'],
                     'US Bancorp': ['US Bancorp', 'U.S. Bancorp Investments, Inc'],
                     'US Bank': ['US Bank N.A', 'US Bk'],
                     'VTB': ['VTB Capital'],
                     'Westpac': ['Westpac Banking Corporation'],
                     'Wells': ['WFS', 'Wells Fargo Securities International Limited',
                               'Wells Fargo Brokerage Services, LLC']}

major_dealers = ['BARC', 'BNP', 'BOA', 'C', 'CS', 'DB', 'GS', 'HSBC', 'JPM', 'Mitsubishi', 'Mizuho', 'MS', 'RBC', 'Wells']
major_dealer_aliases = []
for dealer in major_dealers:
    major_dealer_aliases = major_dealer_aliases + dealer_to_aliases[dealer]

mid_tier_dealers = ['BBVA', 'BMO', 'BNS', 'BNY', 'CG', 'CIBC', 'PNC', 'Santander', 'SG', 'SMBC', 'STRH', 'SunTrust', 'TD', 'UBS', 'US Bancorp']
mid_tier_dealer_aliases = []
for dealer in mid_tier_dealers:
    mid_tier_dealer_aliases = mid_tier_dealer_aliases + dealer_to_aliases[dealer]

# All other dealers are considered low tier


# Ratings

rating_to_value = {
                    # Moody's

                    'NR': 0.0,
                    'B3': 1.0,
                    'B2': 2.0,
                    'B1': 3.0,
                    'Ba3': 4.0,
                    'Ba2': 5.0,
                    'Ba1': 6.0,
                    'Baa3': 7.0,
                    'Baa2': 8.0,
                    'Baa1': 9.0,
                    'A3': 10.0,
                    'A2': 11.0,
                    'A1': 12.0,
                    'Aa3': 13.0,
                    'Aa2': 14.0,
                    'Aa1': 15.0,
                    'Aaa': 16.0,

                    # S&P and Fitch

                    'B-': 1.0,
                    'B': 2.0,
                    'B+': 3.0,
                    'BB-': 4.0,
                    'BB': 5.0,
                    'BB+': 6.0,
                    'BBB-': 7.0,
                    'BBB': 8.0,
                    'BBB+': 9.0,
                    'A-': 10.0,
                    'A': 11.0,
                    'A+': 12.0,
                    'AA-': 13.0,
                    'AA': 14.0,
                    'AA+': 15.0,
                    'AAA': 16.0
                }

# Outlooks

outlook_to_offset = {
                         'Stable': 0.0,
                         'Negative': -0.1,          # 10% of the time this ends up meaning a downgrade
                         'Positive': 0.1,           # 10% of the time this ends up meaning an upgrade
                         'Negative Watch': -0.5,    # 50% of the time this ends up meaning a downgrade
                         'Positive Watch': 0.5,     # 50% of the time this ends up meaning an upgrade
                         'Developing': 0.0
                     }
# Note that "Developing" could be positive or negative (ex: announced a merger but we don't yet know if it's being
# financed with debt or equity)


# Seniority

seniority_to_value = {
                        'Preferred': 0.0,
                        'Jr Sub': 1.0,
                        'Sub': 2.0,
                        'Sr Sub': 3.0,
                        'Sr': 4.0,
                        'Secured': 5.0,
                        'FMB': 5.0,  # first mortgage bond
                        'FA Backed': 5.0  # funding-agreement (backed by a senior claim on the insurer's balance sheet)
                    }


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
# Note that 144A offerings can only be held by Qualified Institutional Buyers. RegS offerings can be held by any non-US
# holder. An unregistered security is one that has not been registered with the SEC and therefore cannot be sold
# publicly. Reg Rights means the issuer can register it, which improves liquidity. 3(a)(2) securities are bank notes
# that are issued on a regular or continuous basis. Money market funds can buy them.
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


# Helper function for determining the role of a given dealer in the new issue
# Note that I decided to combine the B&D Agent, the Bookrunners, and the Joint Leads columns since they all have very
# similar meaning and including them all separately would have added far too many variables.
def get_dealer_role(row, dealer):
    dealer_aliases = dealer_to_aliases[dealer]

    if row['B&D Agent'] in dealer_aliases: # most important dealer
        return 2

    # Bookrunners have same impact as joint leads

    elif row['Bk1'] in dealer_aliases:
        return 1
    elif row['Bk2'] in dealer_aliases:
        return 1
    elif row['Bk3'] in dealer_aliases:
        return 1
    elif row['Bk4'] in dealer_aliases:
        return 1
    elif row['Bk5'] in dealer_aliases:
        return 1
    elif row['Bk6'] in dealer_aliases:
        return 1
    elif row['Bk7'] in dealer_aliases:
        return 1
    elif row['Bk8'] in dealer_aliases:
        return 1
    elif row['Bk9'] in dealer_aliases:
        return 1
    elif row['Bk10'] in dealer_aliases:
        return 1

    # Joint leads have same impact as bookrunners

    elif row['Jt Ld 1'] in dealer_aliases:
        return 1
    elif row['Jt Ld 2'] in dealer_aliases:
        return 1
    elif row['Jt Ld 3'] in dealer_aliases:
        return 1
    elif row['Jt Ld 4'] in dealer_aliases:
        return 1
    elif row['Jt Ld 5'] in dealer_aliases:
        return 1
    else:
        return 0


# Helper function to get the B&D Agent tier for the given dealer tier
def get_bd_agent_tier_col(row, dealer_tier):
    if dealer_tier == 'mid':
        if row['B&D Agent'] in mid_tier_dealer_aliases:
            return 1
        else:
            return 0
    elif dealer_tier == 'low':
        if (row['B&D Agent'] not in major_dealer_aliases) and (row['B&D Agent'] not in mid_tier_dealer_aliases):
            return 1
        else:
            return 0


# Helper function used in get_count_dealer_tier_col
def increment_counts(row, field, count_major, count_mid, count_low):
    if row[field] in major_dealer_aliases:
        count_major += 1
    elif row[field] in mid_tier_dealer_aliases:
        count_mid += 1
    elif row[field] != '-':
        count_low += 1

    return count_major, count_mid, count_low


# Helper function used to get the count of each dealer tier
def get_count_dealer_tier_col(row, dealer_tier):
    count_major = 0
    count_mid = 0
    count_low = 0

    count_major, count_mid, count_low = increment_counts(row, 'B&D Agent', count_major, count_mid, count_low)

    count_major, count_mid, count_low = increment_counts(row, 'Bk1', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Bk2', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Bk3', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Bk4', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Bk5', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Bk6', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Bk7', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Bk8', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Bk9', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Bk10', count_major, count_mid, count_low)

    count_major, count_mid, count_low = increment_counts(row, 'Jt Ld 1', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Jt Ld 2', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Jt Ld 3', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Jt Ld 4', count_major, count_mid, count_low)
    count_major, count_mid, count_low = increment_counts(row, 'Jt Ld 5', count_major, count_mid, count_low)

    if dealer_tier == 'major':
        return count_major
    elif dealer_tier == 'mid':
        return count_mid
    elif dealer_tier == 'low':
        return count_low


# Add the dealers' roles to X
# Note that the final vector includes a separate column for each of the major dealers as well as booleans to represent
# whether or not the B&D Agent was mid or low tier. It also includes a count of the number of major, mid tier, and low
# tier dealers. Dealers were tiered according to business knowledge.
def add_dealer_role_cols(X, df, verbose):
    # Major dealers

    X['BARC'] = df.apply(get_dealer_role, dealer='BARC', axis=1)
    X['BNP'] = df.apply(get_dealer_role, dealer='BNP', axis=1)
    X['BOA'] = df.apply(get_dealer_role, dealer='BOA', axis=1)
    X['C'] = df.apply(get_dealer_role, dealer='C', axis=1)
    X['CS'] = df.apply(get_dealer_role, dealer='CS', axis=1)
    X['DB'] = df.apply(get_dealer_role, dealer='DB', axis=1)
    X['GS'] = df.apply(get_dealer_role, dealer='GS', axis=1)
    X['HSBC'] = df.apply(get_dealer_role, dealer='HSBC', axis=1)
    X['JPM'] = df.apply(get_dealer_role, dealer='JPM', axis=1)
    X['Mitsubishi'] = df.apply(get_dealer_role, dealer='Mitsubishi', axis=1)
    X['Mizuho'] = df.apply(get_dealer_role, dealer='Mizuho', axis=1)
    X['MS'] = df.apply(get_dealer_role, dealer='MS', axis=1)
    X['RBC'] = df.apply(get_dealer_role, dealer='RBC', axis=1)
    X['Wells'] = df.apply(get_dealer_role, dealer='Wells', axis=1)

    # B&D for mid tier and low tier dealers
    X['IsMidTierB&D'] = df.apply(get_bd_agent_tier_col, dealer_tier='mid', axis=1)
    X['IsLowTierB&D'] = df.apply(get_bd_agent_tier_col, dealer_tier='low', axis=1)

    # Count of each dealer tier

    X['CountMajor'] = df.apply(get_count_dealer_tier_col, dealer_tier='major', axis=1)
    X['CountMidTier'] = df.apply(get_count_dealer_tier_col, dealer_tier='mid', axis=1)
    X['CountLowTier'] = df.apply(get_count_dealer_tier_col, dealer_tier='low', axis=1)

    # Drop the columns from df

    df = df.drop(['B&D Agent',
                  'Bk1',
                  'Bk2',
                  'Bk3',
                  'Bk4',
                  'Bk5',
                  'Bk6',
                  'Bk7',
                  'Bk8',
                  'Bk9',
                  'Bk10',
                  'Jt Ld 1',
                  'Jt Ld 2',
                  'Jt Ld 3',
                  'Jt Ld 4',
                  'Jt Ld 5',
                  ], axis=1)

    if verbose:
        print('Shape after adding the dealer role columns:')
        print('X: {}'.format(X.shape))
        print('df: {}'.format(df.shape))
        print()

    return X, df


# Helper function to get years to call
def get_years_to_call(row):
    pricing_date = row['Pricing Date']
    call_date = row['Call Date']

    return (call_date - pricing_date).days / 365.25


# Helper function to get years to maturity
def get_years_to_maturity(row):
    pricing_date = row['Pricing Date']
    maturity_date = row['Maturity']

    return (maturity_date - pricing_date).days / 365.25


# Add years to call/maturity to X
# Note that "YearsToCall" serves as "IsCallable" as well.
def add_years(X, df, verbose):
    # Set 'Call Date' to 'Pricing Date' so that 'YearsToCall' will be 0 for non-callable securities
    df['Call Date'] = df['Call Date'].mask(df['Call Date'] == '-', df['Pricing Date'])

    # Set 'Maturity' to 'Pricing Date' so that 'YearsToMaturity' will be 0 for securities without a maturity date
    # (ex: perps)
    df['Maturity'] = df['Maturity'].mask(df['Maturity'] == '-', df['Pricing Date'])

    # Convert 'Pricing Date', 'Call Date', and 'Maturity' to date times
    # Note that 'Pricing Date' is the true issue date (when it starts trading)
    df['Pricing Date'] = pd.to_datetime(df['Pricing Date'])
    X['PricingDate'] = df['Pricing Date']  # save PricingDate for later use
    df['Call Date'] = pd.to_datetime(df['Call Date'])
    df['Maturity'] = pd.to_datetime(df['Maturity'])

    # Add the columns to X

    X['YearsToCall'] = df.apply(get_years_to_call, axis=1)
    X['YearsToMaturity'] = df.apply(get_years_to_maturity, axis=1)

    # Drop the columns from df

    df = df.drop(['Pricing Date', 'Call Date', 'Maturity'], axis=1)

    if verbose:
        print('Shape after adding the years to call/maturity columns:')
        print('X: {}'.format(X.shape))
        print('df: {}'.format(df.shape))
        print()

    return X, df


# Add ratings to X
def add_ratings(X, df, verbose):
    # Convert ratings to values
    df['Moody Rating Value'] = df['Moody’s Rating'].map(rating_to_value)
    df['Moody Outlook Value'] = df['Moody\'s Outlook'].map(outlook_to_offset)

    df['S&P Rating Value'] = df['S&P Rating'].map(rating_to_value)
    df['S&P Outlook Value'] = df['S&P Outlook'].map(outlook_to_offset)

    df['Fitch Rating Value'] = df['Fitch Rating'].map(rating_to_value)
    df['Fitch Outlook Value'] = df['Fitch Outlook'].map(outlook_to_offset)

    # Create 'MoodyRating', 'SpRating', and 'FitchRating'
    X['MoodyRating'] = df['Moody Rating Value'].add(df['Moody Outlook Value'], fill_value=0.0)
    X['SpRating'] = df['S&P Rating Value'].add(df['S&P Outlook Value'], fill_value=0.0)
    X['FitchRating'] = df['Fitch Rating Value'].add(df['Fitch Outlook Value'], fill_value=0.0)

    # Add 'AvgRating'
    ratings = pd.concat([X['MoodyRating'], X['SpRating'], X['FitchRating']], axis=1)
    ratings = ratings.replace(0, np.NaN)  # so the mean() will work

    X['AvgRating'] = ratings.mean(axis=1)
    X['AvgRating'] = X['AvgRating'].replace(np.nan, 0)  # replace null with 0 when there are no ratings at all

    # Drop the columns from df

    df = df.drop(['Moody’s Rating', 'Moody\'s Outlook', 'Moody Rating Value', 'Moody Outlook Value'], axis=1)
    df = df.drop(['S&P Rating', 'S&P Outlook', 'S&P Rating Value', 'S&P Outlook Value'], axis=1)
    df = df.drop(['Fitch Rating', 'Fitch Outlook', 'Fitch Rating Value', 'Fitch Outlook Value'], axis=1)

    if verbose:
        print('Shape after adding ratings columns:')
        print('X: {}'.format(X.shape))
        print('df: {}'.format(df.shape))
        print()

    return X, df


# Add seniority to X
def add_seniority(X, df, verbose):
    X['Rank'] = df['Rank '].map(seniority_to_value)

    # Drop the column from df

    df = df.drop(['Rank '], axis=1)

    if verbose:
        print('Shape after adding seniority column:')
        print('X: {}'.format(X.shape))
        print('df: {}'.format(df.shape))
        print()

    return X, df


# Add deal information to X
def add_deal_info(X, df, verbose):
    # Compute 'OrderbookSize' from 'Total Deal Size (USD)' and 'Over Subscription'.
    # Note that we know the orderbook size but not the deal size prior to launch.

    df['Total Deal Size (USD)'] = df['Total Deal Size (USD)'].astype(float)
    df['Over Subscription'] = df['Over Subscription'].replace('-', 0).replace(np.nan, 0).astype(float)

    X['OrderbookSize'] = df['Total Deal Size (USD)'].multiply(df['Over Subscription'])

    # Compute 'IptToGuidance'

    X['Guidance'] = X['Guidance'].mask(X['Guidance'] == 0, X['IptSpread'])  # if 'Guidance' is 0, set it to 'IptSpread'
    X['IptToGuidance'] = X['Guidance'].subtract(X['IptSpread'])  # create the column

    if verbose:
        print('Shape after adding the deal info columns:')
        print('X: {}'.format(X.shape))
        print('df: {}'.format(df.shape))
        print()

    return X, df


# Creates the DataFrame x_addl.
# X_addl holds columns that will be used in some models but not in others.
def create_X_addl(X, df, verbose):
    X_addl = pd.DataFrame()

    # 'TotalDealSize' and 'OverSubscription'

    X_addl['TotalDealSize'] = df['Total Deal Size (USD)']
    X_addl['OverSubscription'] = df['Over Subscription']

    # 'IssueSpread', 'IptToIssueSpread', 'GuidanceToIssueSpread'

    X_addl['IssueSpread'] = df['Spread '].astype(float)
    X_addl['IptToIssueSpread'] = X_addl['IssueSpread'].subtract(X['IptSpread'])
    X_addl['IptToIssueSpread'] = X_addl['IptToIssueSpread'].mask(X['IptSpread'] == 0, 0)  # use 0 when 'IptSpread' is 0
    X_addl['GuidanceToIssueSpread'] = X_addl['IssueSpread'].subtract(X['Guidance'])
    X_addl['GuidanceToIssueSpread'] = X_addl['GuidanceToIssueSpread'].mask(X['Guidance'] == 0, 0)

    # Drop the columns from df

    df = df.drop(['Total Deal Size (USD)',
                  'Over Subscription',
                  'Spread '
                  ], axis=1)

    if verbose:
        print('Shape after creating X_addl:')
        print('X_addl: {}'.format(X_addl.shape))
        print('df: {}'.format(df.shape))
        print()

    return X_addl, df


# Helper function to categorize the given numerical column based on the desired categories.
# If "categories" is a list of length n, it creates n+1 categories. For example, if "categories" is [-10, 0, 10], it
# would create categories corresponding to (-inf, -10], (-10, 0], (0, 10], (10, inf). If "categories" is a number n>=2,
# it creates n equally sized buckets as categories.
def categorize_numerical_col(X, col_name, categories):
    if not (isinstance(categories, list)):
        categories_num = categories

        categories = []
        for i in range(categories_num - 1):
            categories.append(X[col_name].quantile((i + 1) / categories_num))

    vec = X[col_name].copy()
    X[col_name] = np.nan

    # First bucket
    partial_col = vec.mask(vec <= categories[0], '<={}'.format(categories[0]))
    X[col_name][vec <= categories[0]] = partial_col

    # Middle buckets
    if len(categories) > 1:
        for i in range(len(categories) - 1):
            inds = (vec > categories[i]) & (vec <= categories[i + 1])
            category = '{}<x<={}'.format(categories[i], categories[i + 1])

            partial_col = vec.mask(inds, category)
            X[col_name][inds] = partial_col

    # Last bucket
    partial_col = vec.mask(vec > categories[-1], '>{}'.format(categories[-1]))
    X[col_name][vec > categories[-1]] = partial_col

    return X


# Categorizes numerical columns of X.
# Note that I chose the categories below based on my domain expertise. For example, I chose to use [15, 30, 45] for the
# categories of VIX since <= 15 represents a low VIX, 15 to 30 is medium, 30 to 45 is high, and above 45 is very high.
def categorize_numerical_cols(X):
    # Categorize the numerical columns
    X = categorize_numerical_col(X, 'CouponRate', 3)
    X = categorize_numerical_col(X, 'IptSpread', 4)
    X = categorize_numerical_col(X, 'Guidance', 4)
    X = categorize_numerical_col(X, 'Area', 3)
    X = categorize_numerical_col(X, 'Concession', 4)
    X = categorize_numerical_col(X, 'NumBookrunners', 2)
    X = categorize_numerical_col(X, 'NumActiveBookrunners', 2)
    X = categorize_numerical_col(X, 'NumPassiveBookrunners', 2)
    X = categorize_numerical_col(X, 'ParAmt', 2)
    X = categorize_numerical_col(X, 'Vix', [15, 30, 45])
    X = categorize_numerical_col(X, 'Vix5d', [-10, 0, 10])
    X = categorize_numerical_col(X, 'Srvix', 3)
    X = categorize_numerical_col(X, 'Srvix5d', [-10, 0, 10])
    X = categorize_numerical_col(X, 'CdxIg', [50, 70, 100])
    X = categorize_numerical_col(X, 'CdxIg5d', [-10, 0, 10])
    X = categorize_numerical_col(X, 'Usgg2y', 3)
    X = categorize_numerical_col(X, 'Usgg2y5d', [-10, 0, 10])
    X = categorize_numerical_col(X, 'Usgg3y', 3)
    X = categorize_numerical_col(X, 'Usgg3y5d', [-10, 0, 10])
    X = categorize_numerical_col(X, 'Usgg5y', 3)
    X = categorize_numerical_col(X, 'Usgg5y5d', [-10, 0, 10])
    X = categorize_numerical_col(X, 'Usgg10y', 3)
    X = categorize_numerical_col(X, 'Usgg10y5d', [-10, 0, 10])
    X = categorize_numerical_col(X, 'Usgg30y', 3)
    X = categorize_numerical_col(X, 'Usgg30y5d', [-10, 0, 10])
    X = categorize_numerical_col(X, 'EnterpriseValue', 3)
    X = categorize_numerical_col(X, 'YearsToCall', [2, 4, 6, 9, 15])
    X = categorize_numerical_col(X, 'YearsToMaturity', [2, 4, 6, 9, 15])
    X = categorize_numerical_col(X, 'AvgRating', 4)
    X = categorize_numerical_col(X, 'OrderbookSize', 3)
    X = categorize_numerical_col(X, 'IptToGuidance', 3)
    X = categorize_numerical_col(X, 'HistSectorPerformance', [-10, 0, 10])
    X = categorize_numerical_col(X, 'HistDealerPerformance', [-10, 0, 10])
    X = categorize_numerical_col(X, 'HistSeniorityPerformance', [-10, 0, 10])
    X = categorize_numerical_col(X, 'HistRatingPerformance', [-10, 0, 10])
    X = categorize_numerical_col(X, 'HistTenorPerformance', [-10, 0, 10])
    X = categorize_numerical_col(X, 'HistAllPerformance', [-10, 0, 10])

    # One-hot encode the numerical columns we just categorized
    categorical_cols = ['CouponRate',
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
                        'AvgRating',
                        'OrderbookSize',
                        'IptToGuidance',
                        'HistSectorPerformance',
                        'HistDealerPerformance',
                        'HistSeniorityPerformance',
                        'HistRatingPerformance',
                        'HistTenorPerformance',
                        'HistAllPerformance']
    X_ohe = pd.get_dummies(X[categorical_cols], drop_first=True)

    # Add to the DataFrame
    X = pd.concat([X, X_ohe], axis=1)

    # Drop the columns from X
    X = X.drop(categorical_cols, axis=1)

    return X
