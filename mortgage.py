"""
functions for determining allowable mortgage
-
"""
from typing import TYPE_CHECKING, Union
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns


# create class? or just startup_functions?

def init_interp1d_mortgage_rates_from_table(input_path: str,
                                            interpolate_kind: str = 'previous'):
    # returns dict of extrapolate-able functions, with the keys as floor values for the mortgage interest rate
    # use extracted mortgage rates from NIBUD
    with open(input_path,'rb') as readfile:
        mortgage_df = pickle.load(readfile)
    assert type(mortgage_df) == pd.DataFrame, f'{input_path} must be a pandas DataFrame with index of gross income, and columns of interest rates'
    mortgage_rates_dict = {}
    for col in mortgage_df.columns:
        # get columns with data
        column_data = mortgage_df[col]
        # insert data into interpolate
        mortgage_rates_dict[column_data.name] = interp1d(x=column_data.index.tolist(),y=column_data.values.tolist(),kind=interpolate_kind, fill_value='extrapolate')
    return mortgage_rates_dict

def test_and_plot_interpolation(functions_to_test: dict, inputs=np.linspace(22000,150000, 128)):
    # get the type stuff
    types = []
    xs = []
    ys = []

    for key, func in functions_to_test.items():
        types.extend([key for _ in range(len(inputs))])
        xs.extend(inputs)
        ys.extend(func(inputs))

    # plot all
    sns.relplot(x=xs, y=ys, hue=types, kind='line', legend='full', palette='plasma')
    plt.xlabel('Gross Income')
    plt.ylabel('Income Burden Proportion for Mortgage')
    plt.title('Income Burden %-ages per Mortgage Interest', **{'wrap':True})
    plt.show()
    # return


def extract_interp1d_funcs_from_pandas_tabular(input_df: Union[pd.DataFrame, pd.Series], interpol_kind: str='previous'):
    # WIP: need to write for future usage!
    # extracts interpolation data from pandas DF, and creates a 1d interpolation function per interest rate category
    # CAVEAT: need to write a caveat such that the column headings of the dataframe must be representative

    return