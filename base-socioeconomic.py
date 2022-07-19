from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def interp1d_income_lookup(input_gross_income, input_disposable_income,
                           interpolate_kind: str = 'quadratic'):
    # use scipy interp1d method to create interpolating function between gross income to disposable income.
    return interp1d(x=input_gross_income, y=input_disposable_income, kind=interpolate_kind)
