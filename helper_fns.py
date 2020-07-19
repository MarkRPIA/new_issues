import pandas as pd


def load_data(verbose):
    if verbose:
        print()
        print('Loading data..')
        print()

    df = pd.read_csv('new_issue_data.csv', encoding='utf-8-sig')

    if verbose:
        print('Columns:')
        print(list(df))
        print()
        print('Successfully loaded data!')
        print()

    return df

