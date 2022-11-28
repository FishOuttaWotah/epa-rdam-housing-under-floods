import decimal
from typing import TYPE_CHECKING, Union, Callable, Sequence
import numpy as np
import numpy.random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy
from decimal import localcontext, Decimal, ROUND_HALF_UP
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
that the brackets must sum up to be 1. This is not checked in this code.
"""


def generate_local_socioecons_percentiles(area_composition: Sequence[int],
                                          national_composition: Sequence[float],
                                          pop_truncation: tuple[float, float] = (0., 1.),
                                          SE_resolution: float = .01):
    """
       Generates an N-sized 1-D population array of a local area's (buurt/wijk) socio-economic distribution. Values are in percentiles of the agent's national socio-economic standing.
       :param area_composition: composition of socio-economic brackets in local area, e.g. low/mid/high, as number of houses per bracket
       :param national_composition: composition of socio-economic brackets in national/reference area, e.g. low/mid/high. Must sum to 1, and must be the same length as area_composition
       :param pop_truncation: tuple with percentages saying which aspect of the population would be truncated. The pop size for the area would remain unchanged, the truncated population would be distributed to the population socio-economic range
       :param SE_resolution: the resolution used for the socio-economic generation, ie. the distribution used would be in discretised steps of N percent.
       :return: 1d numpy array
       """

    # WARNING: very messy code ahead
    # SHERMAN: I had an issue with floating point arithmetic leading to incorrect rounding
    # and I tried to use Decimal to improve rounding precision
    # which improved the rounding but the problem still persists
    # current workaround: let the incorrect rounding go and update the model agent number, usually only
    # a +/- 1 agent difference.

    # TODO: insert IF block if pop_truncation is None?

    with decimal.localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP  # set up accurate decimal creation
        # process input data
        # convert to cumulative, and inserts a lower bound of 0 (defaurt
        # NB: prefix N = national, prefix L = Local
        # create Decimal instance for float-based inputs
        national_composition = tuple(Decimal(str(n)) for n in national_composition)
        pop_truncation = tuple(Decimal(str(n)) for n in pop_truncation)
        SE_resolution = Decimal(str(SE_resolution))

        n_cumulative = [sum(national_composition[:idx + 1]) for idx, _ in enumerate(national_composition)]
        # add zero bound to national composition bounds, used later as a lower bound for generation
        n_cumulative = [Decimal(0)] + n_cumulative
        area_composition_ori = copy(area_composition)  # mostly for debug

        # TODO: describe the truncation and correction algorithm we care about

        """
        the following bit of code does the following:
        - 
        - the output population socio-economic percentiles shall be in the determined socio-economic bounds 
        """
        # PART 1: identifying truncation points
        to_preserve = [False] * (len(n_cumulative) - 1)
        truncations_l = [Decimal(1)] * (len(n_cumulative) - 1)
        truncations_n = [Decimal(0)] * (len(n_cumulative) - 1)
        # get the ratios again
        for idx, bracket in enumerate(n_cumulative[:-1]):
            if pop_truncation[0] <= bracket <= pop_truncation[1]:
                to_preserve[idx] = True
                truncations_n[idx] = national_composition[idx]
            for bound in pop_truncation:
                # check if bound lies between national cumulative bounds
                if n_cumulative[idx] <= bound <= n_cumulative[idx + 1]:
                    to_preserve[idx] = True  # if national bound lies within defined truncation bound, preserve it
                    # get the truncated fraction of LOCAL population
                    truncations_l[idx] = abs((n_cumulative[idx+1] - bound) / national_composition[idx])  # added round for extra precision
                    # get the truncated fraction of NATIONAL brackets
                    truncations_n[idx] = abs(n_cumulative[idx+1]-bound)


        # PART 2: recreating new national SE bounds and local composition
        # create new national composition bounds wrt. truncated bounds
        national_comp_bounds = [pop_truncation[0]] + [comp for idx, comp in enumerate(truncations_n) if to_preserve[idx]]
        # rebuilding new national cumulated composition
        n_cumulative = [sum(national_comp_bounds[:idx + 1]) for idx, _ in enumerate(national_comp_bounds)]

        # distribute agents from truncated brackets into valid population, to preserve agent numbers
        total_agents_ori = sum(area_composition)
        pop_preserved = [round(comp * truncations_l[idx]) for idx, comp in enumerate(area_composition)]
        pop_truncated = total_agents_ori - sum(pop_preserved)
        # update area_composition
        area_composition_inter = [pop_preserved[idx] + pop_truncated * pop_preserved[idx] / sum(pop_preserved) for idx, is_true in enumerate(to_preserve) if is_true]
        area_composition = [int(round(inter)) for inter in area_composition_inter]
        total_agents_new = sum(area_composition)

        # PART 3: generating a population with discretised SE values
        # some depreciated parts:
        # gravitate = np.linspace(0., 1., int(round(1./SE_resolution))+1)   # steps of 5% normally
        # bins = [0.] + [gravitate[idx] + round(SE_resolution/2, 3) for idx, locus in enumerate(gravitate[:-1])] + [1.]

        population = np.empty(total_agents_new, dtype=np.object)  #
        choices2 = [Decimal(f"{n_cumulative[0] + SE_resolution * x}") for x in range(int(round((n_cumulative[-1] - n_cumulative[0])/SE_resolution)+1))]


        counter = 0
        for idx, size in enumerate(area_composition):
            bottom_bound = choices2.index(n_cumulative[idx])
            upper_bound = choices2.index(n_cumulative[idx + 1])
            choices = choices2[bottom_bound:upper_bound+1]

            # choices = np.linspace(
            #     start = float(n_cumulative[idx]),
            #     stop = float(n_cumulative[idx +1]),
            #     num= int(round((n_cumulative[idx+1] - n_cumulative[idx])/SE_resolution)) +1,
            #     dtype=np.half
            # )

            # choices = [Decimal(str(c)) for c in choices]
            # depreciated in cold storage:
            # population[counter:counter + size] = np.random.uniform(low=national_cum[idx],
            #                                                        high=national_cum[idx + 1], size=size)
            population[counter:counter+size] = np.random.choice(
                a=choices,
                replace=True,
                size=size
            )
            counter += size

        # DEPRECIATED post-process to discretise/digitise
        # bins_idxs = np.digitize(population,bins)
        # population1 = [gravitate[idx-1] for idx in bins_idxs]
    population = [float(pop_SE) for pop_SE in population]

    return population, area_composition


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


def create_interpolate_income_func(x_values: Sequence,
                                   income_values: Sequence,
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
Basic function for house value generation
"""


def derive_home_value_simple(agent_gross_income: float,
                             income_to_value_factor: float = 5.) -> float:
    # simple linear relation between agent
    return agent_gross_income * income_to_value_factor


"""
_____________________________________________________

Generating population with income functions (disposable and gross) and percentiles 
_____________________________________________________

"""
"""
blarg goes here
"""


# verification function to plot data
def test_generation_distribution(input_population):
    sns.displot(x=input_population)
    plt.show()


def test_generation_relation(x, y):
    sns.relplot(x=x, y=y)
    plt.show()


if __name__ == '__main__':
    # Test block for population generation
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

    # Test block for interpolation scheme
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
