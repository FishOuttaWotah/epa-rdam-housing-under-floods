from typing import TYPE_CHECKING, Union, Callable, Sequence
import numpy as np
import numpy.random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numba import jit

"""
_____________________________________________________

Generating Percentiles from Area and National Socio-economic brackets
_____________________________________________________
"""
"""
Overview:
This function generates a population of agents' percentiles with a specific area-based socio-economic distribution. This is based on Rotterdam CBS data (area: neighbourhood or district level) and Dutch national-level data. The former describes the socio-economic composition of the area, with relation to the national socio-economic distribution.

 What these functions do:
 These functions takes one area's socio-economic distribution (in relation to national-level distribution), and returns an N-sized population of agents' percentiles fitting the distribution. By design, it ranges between 0 to 1, but the lower bound could be truncated with the lowest_percentile_bound.    

* bracket_distribution are the size of the 
* national_brackets_cutoffs are the cumulative upper bounds of the brackets. They should be the same size as the bracket_distribution argument. If the area's socioeconomic distribution is the same as the national distribution, then their brackets should be the same
* ensure that the brackets must sum up to be 1. This is not checked in this code.
"""


def generate_local_socioecons_percentiles(n_agents_to_draw: int,
                                          area_composition: list[float],
                                          national_composition: [list[float]],
                                          national_is_cumulative: bool = True,
                                          lowest_percentile_bound: float = 0.,
                                          to_shuffle: bool = True) -> np.ndarray:
    """
    Generates an N-sized 1-D population array of a local area's (buurt/wijk) socio-economic distribution. Values are in percentiles of the agent's national socio-economic standing.
    :param n_agents_to_draw: number of agents to generate.
    :param area_composition: composition of socio-economic brackets in local area, e.g. low/mid/high. Must sum to 1, and categories must be monotonically increasing.
    :param national_composition: composition of socio-economic brackets in national/reference area, e.g. low/mid/high. Must sum to 1, and categories must be monotonically increasing.
    :param national_is_cumulative: if False, the national_composition array is converted to cumulative form
    :param lowest_percentile_bound: lowest socioeconomic percentile allowed to be generated.
    :param to_shuffle: if False, the output is not shuffled, so in order of low to high brackets. However, the values within the brackets are randomly generated.
    :return:
    """
    # Set of sanity checks
    if not bool(national_composition):  # if national composition is not provided
        raise ValueError('No national_composition array provided')
    if not bool(area_composition):
        raise ValueError('No area_composition array provided')
    if len(area_composition) != len(national_composition):
        raise ValueError('area_composition and national_composition should be equal in number of values')

    # process input data
    if not national_is_cumulative:  # check if need to convert to cumulative
        # convert to cumulative, and inserts a lower bound of 0 (defaurt
        national_composition = [sum(national_composition[:idx + 1]) for idx, _ in enumerate(national_composition)]

    # add zero bound to national composition bounds, used later as a lower bound for generation
    national_composition = [lowest_percentile_bound] + national_composition

    # generate the groups discretely... and use the size param
    # change implementation because of large number restriction in Numpy
    # now another maximum allowed dimension exceeding
    pop_per_bracket = np.multiply(np.array(area_composition), n_agents_to_draw)
    pop_per_bracket = [int(round(item, 0)) for item in
                       pop_per_bracket]  # using python lists because they can handle arbitrary large floats
    # the following are indices, must be int
    population = np.empty(sum(pop_per_bracket), dtype=np.half)  #
    counter = 0
    for idx, size in enumerate(pop_per_bracket):
        population[counter:counter + size] = np.random.uniform(low=national_composition[idx],
                                                               high=national_composition[idx + 1], size=size)
        counter += size
    if to_shuffle:
        np.random.shuffle(population)  # shuffles in-place
        # this is very slow for very large arrays
    return population


"""
_____________________________________________________

Interpolation Functions for Income (socioeconomic percentile to disposable/gross income)
_____________________________________________________
"""
"""
Overview:
The following methods are interpolation function generators, creating callable functions that take the input of agents' socioeconomic percentiles and returning their expected income level (disposable/gross). By default, the functions have these characteristics:
    * they are quadratically-fitted, meaning that the points are (somewhat) smoothly connected
    * the input values beyond the range of the x-values are truncated to the smallest or largest known values
        * therefore, the interpolation input should optimally include the maximal input bounds (ie. between the 0th percentile till the 100th percentile). Truncating the lower bound is somewhat reasonable, but truncating the higher bound (the very rich) is less reasonable.
    * the output is a callable function, and can take a single or arrayed input (ie. input of array of x values)
    
Usage Notes:
    * the x_values and income_values arguments must be the same in array length.
    * the interpolate_kind argument can be switched to others, like 'linear' or 'cubic'. More documentation is available at the scipy interp1d page.
    * default_low and default_high refers to the default values if the output function encounters an input beyond its known interpolation. 
    * to_extrapolate, when True, turns on the extrapolation option. It assumes a constant slope beyond the edge points
        * test the extrapolated function via plotting, you may get illogical values
    * the output function will not examine whether the input values are expected (ie. <0th percentile or >100th percentile) 
"""


def create_interpolate_income_func(x_values:Sequence,
                                   income_values:Sequence,
                                   interpolate_kind: str = 'quadratic',
                                   default_low: Union[float, str] = 'fix',
                                   default_high: Union[float, str] = 'fix',
                                   to_extrapolate: bool = False) -> Callable:
    """
    Convenience function for generating interpolation functions, with Scipy's interp1d. The length of x_values and incomes_values arrays should be equal.
    :param x_values: Percentile points corresponding to the income values' points. Must be equal in length to the income_values argument.
    :param income_values: income values (disposable or gross) according to population percentile.
    :param interpolate_kind: Kind of interpolation scheme to be used on interp1d
    :param default_low: a default low bound if the interpolation input is out of bounds. 'fix' means to have the same value as the lowest available value.
    :param default_high: a default high bound if the interpolation input is out of bounds. 'fix' means to have the same value as the highest available value.
    :param to_extrapolate: if the 'extrapolate' option is to be used. This assumes an equal slope beyond the edge points. Not set by default because the low-bound extrapolation tends to 0 (not realistic). If this option is on, default_low and default_high arguments will be overridden.
    :return: interpolate function that can take arrayed inputs (list of agents' socioeconomic percentiles)
    """
    # include test for lower than 0 bound?
    if to_extrapolate:  # if extrapolate, override
        return interp1d(x=x_values, y=income_values, kind=interpolate_kind, fill_value='extrapolate')
    else:
        bounds_arr = [np.nan, np.nan]
        if type(default_low) is str:
            bounds_arr[0] = income_values[0]  # to fix to the lowest available point
        else:
            bounds_arr[0] = default_low

        if type(default_high) is str:
            bounds_arr[1] = income_values[-1]  # fix to the highest available point
        else:
            bounds_arr[1] = default_high
        return interp1d(x=x_values, y=income_values, kind=interpolate_kind, bounds_error=False, fill_value=bounds_arr)


"""
_____________________________________________________

Generating population with income functions (disposable and gross) and percentiles 
_____________________________________________________

"""
"""
blarg goes here
"""

# note: maybe this function could be placed in a more general area
def create_population(population:Sequence,
                      mapping:Callable) -> Sequence:
    # given agents' percentiles, do the mapping
    return mapping(population)


# verification function to plot data
def test_generation_distribution(input_population):
    sns.displot(x=input_population)
    plt.show()


def test_generation_relation(x, y):
    sns.relplot(x=x, y=y)
    plt.show()


if __name__ == '__main__':
    ## Test block for population generation
    # national = [0.33, 0.33, 0.33]
    national = [1 / 3] * 3
    # local = [0.33, 0.33, 0.33]
    local = [1 / 3] * 3
    n_agents = 10000  # beware of running very large numbers (eg. 1e9) here, very slow
    pop = generate_local_socioecons_percentiles(n_agents_to_draw=n_agents,
                                                area_composition=local,
                                                national_composition=national,
                                                national_is_cumulative=False)
    test_generation_distribution(pop)
    print('generated pop')

    ## Test block for interpolation scheme
    incomes = pd.read_pickle('data_model_inputs/income_gross_to_disposable.pickletable')
    incomes_gross = incomes.bruto.to_list()
    incomes_disposable = incomes.besteedbaar.to_list()
    x_range = np.linspace(0.1, 1.0, len(incomes_gross)) - 0.05  # should represent every 10% percentile at their median

    # modification for hack (including the lowest_possible bounds)
    x_range = np.insert(x_range, obj=0, values=[0.])
    incomes_gross = [incomes_gross[0]] + incomes_gross
    incomes_disposable = [incomes_disposable[0]] + incomes_disposable

    # generate output functions and plot the output functions
    gen_disposable = interp1d(x=x_range, y=incomes_disposable, kind='quadratic', fill_value='extrapolate')
    gen_gross = interp1d(x=x_range, y=incomes_gross, kind='quadratic', fill_value='extrapolate')
    test_agents = np.linspace(0, 1.1, 100000)
    pop_disposable = gen_disposable(test_agents)
    pop_gross = gen_gross(test_agents)
    test_generation_relation(test_agents, pop_disposable)
    test_generation_relation(test_agents, pop_gross)
