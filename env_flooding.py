""""
flood generation
"""
from typing import Any, Optional
from scipy.interpolate import interp1d
from math import isnan
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import copy

def get_this_bread(table_paths: list[str] = ['data_model_inputs/flood_depth_damage_functions.tsv', 'data_model_inputs/flood_depth_damage_functions2.tsv']):
    """
    Convenience function for extracting the default depth-damage functions from the tables. Uses the retrieve_depth_damage_functions method.
    :param table_paths:
    :return: dictionary of depth-damage functions keyed with the name of dataframe indices
    """
    # for the default depth-damage curves
    # retrieve the tables for depth-damage functions
    fx_dicts = {}
    for path in table_paths:
        fx_dicts.update(retrieve_depth_damage_functions(path))

    return fx_dicts


def retrieve_depth_damage_functions(table_path :str = 'data_model_inputs/flood_depth_damage_functions.tsv'):
    # NB: some things are hardcoded here
    # read TSV file
    flood_dd_df = pd.read_csv(table_path,sep='\t', comment='#', na_values='N', index_col=0)

    # convert table from str to float (since it cannot parse percentages)
    for col in flood_dd_df.columns:
        flood_dd_df[col] = flood_dd_df[col].str.rstrip('%').astype(float)
    # convert table columns to numeric float
    flood_dd_df.columns = flood_dd_df.columns.astype('float')

    # separate this section onwards for further processing?
    # create the functions per row
    # need to truncate the x for the y
    xs = flood_dd_df.columns.to_list()

    fx_dicts = {}
    for row in flood_dd_df.itertuples(name=None):
        idx = row[0].lower() # the index name, converted to lowercase
        # truncate the x and y inputs to the interpolation function (because some functions reach 100% early)
        y_inputs = [y for y in row[1:] if not isnan(y)]
        x_inputs = xs[:len(y_inputs)]
        assert len(y_inputs) == len(x_inputs), "y_inputs array different sized to x_inputs array"
        fx_dicts[idx] = interp1d(x_inputs, y_inputs, bounds_error=False, fill_value=(0, y_inputs[-1]))

    return fx_dicts


def extract_scenarios_per_area(input_df: pd.DataFrame,
                               spatial_label: str = 'wijk',
                               scenario_label: str = 'scenario',
                               specific_scenario: dict[int: list[str]] = None,
                               general_scenario: int = None,
                               default_scenario: int = 0) -> pd.DataFrame:
    """
    Returns a subset containing the specific scenario rows from the input dataframe. Input can be either general inputs (all areas get the same scenario), specific (dict of lists with the location names, keyed with scenario number) and mixed (specific dict plus general scenario). The specific scenario input is prioritised first.
    :param input_df: pandas dataframe containing all the flood scenario data (i.e. locations, scenarios, flood distributions)
    :param spatial_label: pandas column label to retrieve the location data from
    :param scenario_label: pandas column label to retrieve the scenario data from
    :param general_scenario: integer specifying the scenarios for all locations (if specific_scenario is not provided). If specific_scenario is provided, the remainder of locations will follow the general_scenario.
    :param specific_scenario: dict specifying the scenarios as key, and list of locations as value
    :param default_scenario: if no scenario input (general or specific) is provided, return this default (0 = no flooding)
    :return:
    """

    # spatial_resolution is the label for the df
    # function to filter specific scenario sects from the big flood distribution dataframe
    # specific scenarios should be overriding

    # get unique entries from flood table
    # gateway check: no given scenario or valid dict (None or empty dict returns False)
    if general_scenario is None and not bool(specific_scenario):
        raise ValueError('either general_scenario or specific_scenario should be provided')

    unique_scenarios = input_df[scenario_label].unique()
    unique_areas = input_df[spatial_label].unique()  # only valid for

    specific_locations: list = []  # if specific_scenario is submitted, this would be non-empty
    specific_indices: list = []
    general_indices: list = []
    # check if a valid specific scenario is given:
    if bool(specific_scenario):
        # extract pandas indices and locations from the specific dict
        specific_indices, specific_locations = extract_indices_from_specific_scenario(unique_scenarios=unique_scenarios,
                                                                                      unique_areas=unique_areas,
                                                                                      specific_scenario=specific_scenario)

    # get the other locations that are not included in the areas, if specific_scenario is empty, other areas = unique areas
    other_areas = list(set(unique_areas) - set(specific_locations))

    # handle general locations
    if bool(other_areas):  # if the other areas are not empty
        if general_scenario is None:
            general_scenario = default_scenario  # default to normal
        general_indices = [f'{general_scenario}.{area}' for area in other_areas]

    # if not all scenarios are given, return the 0 rows for associated
    return input_df.loc[general_indices + specific_indices, :]


def extract_indices_from_specific_scenario(unique_scenarios: list,
                                           unique_areas: list,
                                           specific_scenario: dict) -> tuple[list[str], list]:
    # used as a subfunction
    # check keys of scenario numbers
    # check if scenario keys are correct
    scen_check = [item in unique_scenarios for item in specific_scenario.keys()]
    if not all(scen_check):
        raise ValueError(
            f"Base dataframe only contains scenarios ({unique_scenarios}), not present in specific scenarios({specific_scenario.keys()})")

    ## get all locations within dict for sanity checks
    input_locations = [loc for sublist in specific_scenario.values() for loc in
                       sublist]  # think of this as two nested for loops sequentially
    if len(input_locations) != len(set(input_locations)):  # simple check for duplicates
        raise ValueError(f"There exists duplicate inputs in the specific_scenario arg")

    # check if location keys are correct
    loc_check = [item in unique_areas for item in input_locations]
    if not all(loc_check):
        faulty = [input_locations[idx] for idx, boolean in enumerate(loc_check) if boolean]
        raise ValueError(f"Specific_scenarios contains invalid location names: {faulty} ")

    # if all input checks pass, extract from pandas
    indices = [f'{k}.{v}' for k, lst in specific_scenario.items() for v in lst]

    return indices, input_locations

def test_interpolation():
    # generate x points and plot
    x_inputs = np.linspace(0, 8, 8*5)
    functions = get_this_bread(table_paths=['data_model_inputs/flood_depth_damage_functions.tsv', 'data_model_inputs/flood_depth_damage_functions2.tsv'])

    print(functions)
    outputs = {}
    fig, ax = plt.subplots()

    for key, fx in functions.items():
        y = np.vectorize(fx)(x_inputs)
        plt.plot(x_inputs, y, label=key)

    plt.xlabel('flood height [m]')
    plt.ylabel('damage factor [-]')
    plt.title('Test plot of DD Curves')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # item = get_this_bread()
    test_interpolation()
    # # test mode
    # random.seed(1111)  # for testing consistency
    # test_df = pd.read_pickle('data_model_inputs/flood_scenarios_per_wijk.pickletable')
    # locations = test_df.wijk.unique()
    # scenarios = test_df.scenario.unique()
    # test_dict1 = {1: ['rotterdam-centrum', 'feijenoord'], 0: ['pernis', 'noord']}
    #
    # # test dict 2: a randomly-generated bunch assignment of flood scenarios
    # test_sample2 = list(zip(random.choices(scenarios, k=len(locations)), locations))
    # test_dict2 = dict([(scenario, []) for scenario in scenarios])
    # for s, l in test_sample2:  # s = scenario, l = location
    #     test_dict2[s].append(l)
    #
    # # test dict 3: mix
    #
    # # test dict 4: with errors (duplicates or out of bounds)
    # test_dict4 = copy.deepcopy(test_dict2)
    # for s, l in test_sample2:
    #     test_dict4[s].append(l)
