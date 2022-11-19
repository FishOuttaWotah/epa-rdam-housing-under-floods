from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Union, Sequence, Callable, Tuple, Type, Dict, Mapping
import collections
import pandas as pd
import numpy as np
import rasterio

import agent_base
import agent_firm_basic

if TYPE_CHECKING:
    import model_init

import agent_household
import env_socioeconomic

"""
_____________________________________________________

Agent generation functions
_____________________________________________________
"""


## NEED to add:
# generate per area (input per area)
# generate according to distribution -> assign socioeconomic attributes
# generate household mesa entities
# generate house objects and create house values as well


def spawn_household_agent(model: model_init.RHuCC_Model, agent_id: str, agent_SE_percentile: float,
                          agent_location: str, income_disposable: float, income_gross: float, home_value: float,
                          flood_exp: tuple[float],
                          flood_dd_func: Callable):
    # create agent entity
    agent = agent_household.HouseholdAgent(unique_id=agent_id,
                                           model=model,
                                           SE_percentile=agent_SE_percentile)

    # agent_home = agent_household.House(location=agent_location,
    #                                    house_value=home_value,
    #                                    household=agent)

    agent.location = agent_location
    # agent.income_disposable = income_disposable // model.STEPS_PER_YEAR
    # agent.income_gross = income_gross // model.STEPS_PER_YEAR
    # agent.income_d_ratio = income_disposable / model.h_income_median_d
    # agent.income_g_ratio = income_gross / model.h_income_median_g
    agent.flood_exp = flood_exp
    agent.flood_dmg = tuple(np.round(flood_dd_func(flood_exp),decimals=4))
    # agent.house = agent_home
    agent.value = home_value


    # insert into schedule and ledgers (alert, depreciate!)
    model.schedule.add(agent)

    # maybe ledger should extract what they have instead of reporting...
    # model.ledger.h_wage_disposable.append()

    return agent  #, agent_home


def generate_household_agents_bulk(households_df: pd.DataFrame, scenarios_df: pd.DataFrame,
                                   scenarios_order: list[str],
                                   model: model_init.RHuCC_Model, func_i_disp: Callable, func_i_gross: Callable,
                                   func_h_gross: Callable, flood_dd_func: Callable,
                                   agent_ratio: Union[int, float] = None, target_population: int = 3000,
                                   national_SE_brackets: Sequence = (0.4, 0.4, 0.2),
                                   SE_truncation: Tuple[float, float] = (0., 1.), scenarios_dfkey: str = 'scenario',
                                   area_names_dfkey: str = 'wijk', land_tiles_dfkey: str = 'flatten'):
    # generate household agents in bulk for model initiation

    # reduce number of agents for simulation
    reduced_df = reduce_agents_count(
        agents_df=households_df,
        agent_ratio=agent_ratio,
        target_population=target_population)

    # update model attributes

    # socio-economic percentiles plus truncation (!)
    hh_percentiles_dict, hh_distr_dict = generate_household_SE_distribution(reduced_df,
                                                             pop_truncation=SE_truncation,
                                                             national_SE_brackets=national_SE_brackets)

    # updating values
    # total_hhs_per_area = pd.Series([sum(brackets) for brackets in hh_percentiles_dict.values()],
    #                                index=hh_percentiles_dict.keys())
    reduced_df = pd.DataFrame.from_dict(hh_distr_dict, orient='index', columns=reduced_df.columns)
    total_hhs_per_area = reduced_df.sum(axis=1)
    model.h_SE_percentile_range = SE_truncation
    model.h_SE_regional_distr = reduced_df

    # flood-vulnerability distribution

    hh_flood_dict = generate_flood_exposure_distribution(scenarios_df, agents_to_spawn=total_hhs_per_area,
                                                         scenarios_order=scenarios_order, scenarios_key=scenarios_dfkey,
                                                         areas_key=area_names_dfkey, distribution_key=land_tiles_dfkey)

    # sanity check for area name mismatch between percentiles and flood dicts
    match_keys = [district in hh_percentiles_dict.keys() for district in hh_flood_dict.keys()]
    if not all(match_keys):
        # NB: not really tested
        non_match_keys = [list(hh_flood_dict.keys())[idx] for idx, found in enumerate(match_keys) if not found]
        raise KeyError(f'mismatching keys in household DF and scenarios DF:{non_match_keys}')

    # compile percentiles and flood risk to agents
    district_counter = 0
    for district, SE_distr in hh_percentiles_dict.items():
        flood_distr = hh_flood_dict[district]
        if len(SE_distr) != len(flood_distr):  # sanity check, both must be the same length
            raise ValueError(f'socio-economic distr. different to flood exposure distr., location {district} ')

        # create agent_ids for district
        agent_ids: list[str] = generate_agent_id_per_area(agent_area=str(district),  # ignore typecheck warning
                                                          agent_type=agent_household.HouseholdAgent.__name__,
                                                          agent_count=len(SE_distr))

        # generate incomes in bulk
        inc_disp = np.rint(func_i_disp(SE_distr)).astype(int)
        inc_gross = np.rint(func_i_gross(SE_distr)).astype(int)  # interp1d of socioeconomic brackets and gross income
        # home_values = np.rint(func_h_gross(inc_gross)).astype(int)  # env_socioeconomic.derive_home_value_simple

        # compile stuff! TODO cannot multi-compile anymore...
        for idx, pct in enumerate(SE_distr):
            flood_exp = hh_flood_dict[district][idx]
            agent = spawn_household_agent(model=model, agent_id=agent_ids[idx], agent_SE_percentile=pct,
                                                agent_location=str(district), income_disposable=inc_disp[idx],
                                                income_gross=inc_gross[idx], home_value=1.0, #home_values[idx],
                                                flood_exp=flood_exp, flood_dd_func=flood_dd_func)

        district_counter += 1

    return


def generate_household_SE_distribution(hh_agents_df: pd.DataFrame,
                                       pop_truncation: Tuple[float, float],
                                       national_SE_brackets: Sequence = (0.4, 0.4, 0.2)):
    # generate dummy households, expand till model requirements
    # adds stuff to the schedule, so returns none
    # assumes that the dataframe has no NA values (that will give errors)

    # reduces number of households to number of agents for simulation, per district/area
    # this function returns the same format as the hh_df, but reduced in number
    # NB: this line may be hard to understand, documentation needed.

    # generate population per area, based on composition
    gen_SE_percentiles = {}
    truncated_SE_distribution = {}
    for area_row in hh_agents_df.iterrows():  # NB for expansion: limitations of iterrows()
        area_name, area_brackets = area_row  # extract from tuple
        # NB: area_name is str, the district name. Used as a common key for all data sources
        # NB: area_brackets are pd.Series, default containing 3 socioeconomic brackets, and number of households agents per bracket.
        # generate population from local
        gen_SE_percentiles[area_name], truncated_SE_distribution[area_name] = env_socioeconomic.generate_local_socioecons_percentiles(
            area_composition=area_brackets.tolist(),
            national_composition=national_SE_brackets,
            pop_truncation=pop_truncation  # NB: repacked into hacky method
            # lowest_percentile_bound=0.1
        )

    return gen_SE_percentiles, truncated_SE_distribution


# _____________________________

# With given agent count,
# _____________________________

def generate_flood_exposure_distribution(scenarios_applied: pd.DataFrame,
                                         agents_to_spawn: pd.Series,
                                         scenarios_order: list[str],
                                         scenarios_key: str = 'scenario',
                                         areas_key: str = 'wijk',
                                         distribution_key: str = 'flatten',
                                         size_key: str = 'flatten_size',
                                         DEFAULT_SAFE: float = -1) -> Dict:
    """
    // something about commensurability of data
    :param scenarios_order:
    :param scenarios_applied: dataframe containing the areas flooded in a scenario. Must have a column
    for the distribution of flooded and non-flooded land parcels (only for flooded districts).
    Rows/indices are areas affected and associated scenarios
    :param agents_to_spawn: pd.Series containing total agents to spawn per area (district)
    :param scenarios_key: str key to retrieve the scenario names from the scenarios_applied DF
    :param areas_key: str key to retrieve the area names from the scenarios DF and agents_to_spawn DF
    :param distribution_key: str key to retrieve the distribution of land parcels (flooded and non-flooded)
    for areas affected by flood (scenarios_applied)
    :param size_key:  str key to retrieve the size of the distribution of land parcels. Mostly for
    verification
   :param DEFAULT_SAFE: default value to signify a no-flood tile, as may be mistaken to be a flood.
    :return:
    """
    # assumes that the land parcel distribution is already verified to be the same size

    # return only items with flooding
    wijks: Sequence = agents_to_spawn.index
    affected_wijks: Sequence = scenarios_applied[areas_key].unique()
    # scenarios: Sequence = scenarios_applied[scenarios_key].unique()
    output_distr = dict.fromkeys(affected_wijks)  # values are None by default

    # generate the distribution first before doing values
    for wijk in wijks:
        exposures_lst = []  # preparation for later
        n_agents = agents_to_spawn[wijk]  # number of agents to spawn

        if wijk in affected_wijks:
            # get the subset of dataframe for only districtcon
            subsubset: pd.DataFrame = scenarios_applied.loc[scenarios_applied[areas_key] == wijk, :]
            subsubset.set_index(scenarios_key, inplace=True, verify_integrity=True)  # set index to scenario name

            # check if the land tiles are consistent
            land_tiles = subsubset[size_key].unique()
            if land_tiles.size != 1:  # error check
                raise ValueError(
                    "Mismatch in scenarios dataframe, size of land distribution per district/area is not consistent. The "
                    "spatial resolution of the flood scenario files may be different, or the bounds of the flood scenario "
                    "files may be different.")

            ordinals = np.random.randint(low=land_tiles, size=n_agents)  # ordinal polling

            # retrieve flood values from land parcel map
            for scenario in scenarios_order:  # follow the given ordering
                if scenario in subsubset.index:  # TODO: think here?
                    # extract specific land parcel distribution in affected wijk
                    parcel_distr: Sequence[float] = subsubset.loc[scenario, distribution_key]

                    # modify the parcel distr for non-flooded tiles only
                    parcel_distr = [val if val > 0 else DEFAULT_SAFE for val in parcel_distr]
                    # extract the land parcel flood heights using ordinal indexing
                    exposures_lst.append([parcel_distr[idx] for idx in ordinals])
                else:
                    exposures_lst.append([DEFAULT_SAFE] * ordinals.size)  # non affected get 0m flood
            output_distr[wijk] = list(zip(*exposures_lst))

        else:  # if the wijk is not exposed to flooding, just create zeroes
            output_distr[wijk] = list([tuple([DEFAULT_SAFE] * len(scenarios_order)) for _ in range(n_agents)])

        # per wijk, have a list of tuples, where the tuples are the flood heights per agent for simulation
    return output_distr


# ---------------------------------
# Generic functions, usable for both Household and Firms
# ---------------------------------

def reduce_agents_count(agents_df: Union[pd.DataFrame, pd.Series],
                        agent_ratio: Union[int, float] = None,
                        target_population: int = None) -> Union[pd.DataFrame, pd.Series]:
    """
    Generates number of representative households to be created as agents (1 agent per N households). The user can either specify the target population to be achieved (and the function would find the required agent-household ratio, or specify the agent-household_ratio themselves. All numbers are rounded to integers.
    :param agents_df: pandas dataframe containing brackets of households, columns for bracket, row for area
    :param agent_ratio: numerical input stating 1 agent per N households
    :param target_population: numerical input stating the desired agent population size
    :return: reduced pandas dataframe describing the number of agents per area per bracket.
    """

    # identify whether is company or not
    if type(agents_df) == pd.DataFrame:  # not perfect catch but sufficient
        total_agents = agents_df.sum().sum()  # get total households for entire model
    elif type(agents_df) == pd.Series:
        total_agents = agents_df.sum()
    else:
        raise TypeError('agents_df arg should either be a pandas dataframe or series')

    if agent_ratio is not None and target_population:
        raise ValueError('Only provide either agent_ratio or target_population')
    elif agent_ratio is None and target_population is None:
        raise ValueError('Provid either agent_ratio or target_population')

    if agent_ratio is None:
        agent_ratio = total_agents / target_population  # is this premature rounding?
    else:
        target_population = total_agents / agent_ratio

    # print(f'Derived agent ratio (unrounded): {agent_ratio}')
    # print(f'Target population (unrounded): {target_population}')

    # apply to households_df
    return (agents_df / agent_ratio).round(decimals=0).astype(int)


def generate_agent_id_per_area(agent_area: Union[str, Sequence[str]],
                               agent_type: Union[str, Sequence[str]],
                               agent_count: Union[int, None] = None):
    # either agent area or agent_type could be sequences... need to iterate through either
    # if both agent_area and agent_type are iterable, just unpack
    # if either are iterable, unpack and generate
    # neither are, use normal generation

    # dumb tree method
    if isinstance(agent_area, Sequence) and not isinstance(agent_area, str):
        if isinstance(agent_type, str):
            return [f'{agent_type}_{area}_{count}' for count, area in enumerate(agent_area)]
        elif isinstance(agent_type, Sequence) and len(agent_type) == len(agent_area):
            return [f'{agent_type[count]}_{area}_{count}' for count, area in enumerate(agent_area)]
        else:
            raise ValueError("agent_type object not the same length/type as agent_area")
    elif isinstance(agent_area, str):
        if isinstance(agent_type, str):
            return [f'{agent_type}_{agent_area}_{count}' for count in range(agent_count)]
        elif isinstance(agent_type, Sequence):
            return [f'{type}_{agent_area}_{count}' for count, type in agent_type]


# ------------------------------
# Firms functions
# ------------------------------
def spawn_firm_agent_bulk(firms_to_spawn: Mapping[Type[agent_firm_basic.Firm], dict],
                          model: model_init.RHuCC_Model) -> None:
    for firm_class, attributes in firms_to_spawn.items():
        for attr_set in attributes:
            spawn_firm_agent(model=model,
                             firm_type=firm_class,
                             agent_attributes=attr_set)


def spawn_firm_agent(model: model_init.RHuCC_Model,
                     firm_type: Type[agent_firm_basic.Firm],
                     agent_attributes: dict) -> None:
    # NB: no typechecking here unfortunately
    agent = firm_type(**agent_attributes)  # unpack into dict

    # NB: agents will be picked up by Ledger from schedule
    model.schedule.add(agent)


def generate_firms_bulk(firms_per_area: pd.Series,
                        scenarios_df: pd.DataFrame,
                        scenarios_order: list[str],
                        model: model_init.RHuCC_Model,
                        firms_classes: list[Type[agent_firm_basic.Firm]],
                        firms_args_per_class: Dict, firms_composition: Sequence[float],
                        firms_flood_dd_funcs: Mapping[Type[agent_base.CustomAgent], Callable],
                        agent_ratio: Union[int, float] = None, target_population: int = 1000,
                        scenarios_dfkey: str = 'scenario', area_names_dfkey: str = 'wijk',
                        land_tiles_dfkey: str = 'flatten'):
    # NB: the firms_id_lists functions is taken over by Ledger object
    # NB: added firm_agent_labels for id_idenification
    # NB: workaround for the location bits (need to set to None)

    # SANITY CHECKS
    # basic sanity check
    length_check = [len(firms_classes),
                    len(firms_args_per_class),
                    len(firms_composition)]
    if len(set(length_check)) != 1:  # meaning that the list inputs are of different length
        raise ValueError('firm_agent_functions, firm_composition, and firm_agent_labels must have '
                         'the same length')

    # sanity check 2: check if the inputs for the firms_args_per_class match the required args from the class
    for f_class in firms_classes:
        check_class_args(firm_class=f_class,
                         input_args=firms_args_per_class[f_class],
                         ignore_args=model.f_ignored_args)
        # will raise an error if the input args are not found or incorrect.

    # REDUCE number of actual agents for simulation
    reduced_df = reduce_agents_count(
        agents_df=firms_per_area,
        agent_ratio=agent_ratio,
        target_population=target_population)

    # GENERATE firm distribution (cap, con, serv) per area in Rdam
    firm_distribution: pd.DataFrame = generate_firm_distribution(reduced_df,
                                                                 firms_classes,
                                                                 firms_composition)
    # (DEPRECIATED) collect total number of firms, per area and per sector (columnar summation)
    total_firms_per_area: pd.Series = firm_distribution.sum(axis=1)
    total_firms_per_sector: pd.Series = firm_distribution.sum(axis=0)

    # GENERATE flood and location attributes
    f_fnl_attributes: list[list[tuple[str, float]]] = []  # FNL = flood and location
    # iterate through columns (of firms sectors)
    for _, sector in firm_distribution.items():  # 1st item is the column label, not used currently
        # the following creates a dict with areas as labels. needs a list.
        f_flood_dict: dict = generate_flood_exposure_distribution(scenarios_df, agents_to_spawn=sector,
                                                                  scenarios_order=scenarios_order,
                                                                  scenarios_key=scenarios_dfkey,
                                                                  areas_key=area_names_dfkey,
                                                                  distribution_key=land_tiles_dfkey)
        # issue with reduction of firm numbers
        gen = [(loc, exp) for loc, flood_distr in f_flood_dict.items() for exp in flood_distr]
        f_fnl_attributes.append(gen)

    # GENERATE bulk attributes per firm class, and spawn firm agents into model
    output_dict: dict = dict.fromkeys(firms_classes)
    for idx, f_class in enumerate(firms_classes):
        args: dict = firms_args_per_class[f_class]
        distr_size = firm_distribution[f_class.firm_type].sum()  # from pd.Series to int
        bulk_attrs: dict = generate_agent_attributes_bulk(pop_size=distr_size, args_dict=args)
        # generates the attributes that were set as initially IntrinsicNone, overwriting the dict entry
        # use black magic to extract the locations attributes and flood exposure attributes
        bulk_attrs['location'], bulk_attrs['flood_exp'] = zip(*f_fnl_attributes[idx])
        # convert flood from depth to damage TODO: (watch out for tuples)
        # (retrieve functions and apply the list of flood exposures)
        bulk_attrs['flood_dmg'] = [tuple(firms_flood_dd_funcs[f_class](fld_tup)) for fld_tup in bulk_attrs['flood_exp']]
        # generate the price via list comprehension black magic
        bulk_attrs['price'] = [wage / prod for wage, prod in zip(bulk_attrs['wage'], bulk_attrs['productivity'])]
        bulk_attrs['model'] = [model] * distr_size
        bulk_attrs['unique_id'] = generate_agent_id_per_area(
            agent_area=bulk_attrs['location'],
            agent_type=f_class.__name__)

        # COLLECT individual attributes from bulk attributes, and spawn agents
        agent_attrs = []
        for idx2, _ in enumerate(bulk_attrs['unique_id']):
            agent_attrs.append(dict([(attr, bulk_attrs[attr][idx2]) for attr in bulk_attrs.keys()]))
            # save into dict for spawn later

            # # spawn firm agent individually
            # spawn_firm_agent(model=model,
            #                  firm_type=f_class,
            #                  agent_attributes=attrs)
        output_dict[f_class] = agent_attrs
    return output_dict


def generate_agent_attributes_bulk(pop_size: int,
                                   args_dict: dict) -> dict:
    """
    Bulk create a dict of agents' attributes, intended to be incorporated into a class init function. Attributes
    comes as a large list of pop_size length per agent. Input args_dict could either be a singular constant
    or a tuple with a function in the first position, and the function's args in the 2nd position as a dict.
    :param pop_size: number of values to generate per attribute
    :param args_dict: dict of attributes to create, with values to assign/generate via constant or
    functions with arguments.
    :return: dict with attributes as key, and values the size of pop_size.
    """
    generate = {}
    for arg, value in args_dict.items():

        # check if item is functional
        try:
            if isinstance(value, tuple) and isinstance(value[0], Callable):
                # item 1 is the function, item 2 is the input arg dict
                func: Callable = value[0]  # func must be able to cast to multiple items...?
                func_args: dict = value[1]
                func_gen = func(**func_args)  # unpack the dict into function
                # catch: if function only generates singular items, do dumb iterate
                if not hasattr(func_gen, '__len__'):
                    func_gen = [func(**func_args) for _ in range(pop_size)]
                generate[arg] = func_gen
            # check if item is singular: make constant
            else:  # hacky way of allowing things
                generate[arg] = [value] * pop_size  # all items have the constant attribute
        except TypeError:  # hacky way
            generate[arg] = [value] * pop_size

    check = [len(v) for v in generate.values()]  # see if generated attributes are all the same size
    if len(set(check)) != 1:
        # might as well print the full output
        print('Firm attributes are not the same size!')
        for line in [(k, len(v)) for k, v in generate.items()]:
            print(line)
        raise ValueError('Firm attributes are not the same size, see diagnostic print above')

    return generate


def generate_firm_distribution(firms_per_area: pd.Series,
                               firm_types: list[Type[agent_firm_basic.Firm]],
                               firm_composition_weights: Sequence[float]) -> pd.DataFrame:
    # NB: Assumption applied: that the firm_composition weights are constant for all areas
    # NB: the firm_composition weights must add up to equal 1 (restriction)
    if len(firm_types) != len(firm_composition_weights):
        raise ValueError('number of firm_types should equal number of weights in firm_composition_weights')

    # create new DF
    temp = {}
    firm_labels = [firm.firm_type for firm in firm_types]
    for idx, weight in enumerate(firm_composition_weights):
        temp[firm_labels[idx]] = (firms_per_area * weight).round().astype(int)

    return pd.DataFrame.from_dict(temp, orient='columns')


# could be generic?
def check_class_args(firm_class: Type[agent_base.CustomAgent],
                     input_args: Sequence[str],
                     ignore_args: Sequence[str]) -> None:
    firm_args = inspect.signature(firm_class).parameters.values()

    # filter according to mandatory args and default/optional args
    mandatory_args: list = [arg.name for arg in firm_args if arg.default is inspect.Parameter.empty]
    optional_args: dict = dict((arg.name, arg.default) for arg in firm_args
                               if arg.default is not inspect.Parameter.empty)

    # check for mandatory arguments
    check_mandatory = [arg in input_args for arg in mandatory_args if
                       arg not in ignore_args]  # checks if the mandatory arguments are present
    if not all(check_mandatory):  # if all mandatory args are not fully present, raise error
        diag = [mandatory_args[idx] for idx, condition in enumerate(check_mandatory) if not condition]
        raise ValueError(f'input args for {firm_class} missing: {diag}')

    # check for optional arguments
    # so for arg in optional_arg_keys(), if present in
    check_optional = [arg for arg in input_args if arg not in mandatory_args]
    check_optional2 = [arg in optional_args.keys() for arg in check_optional]
    # all optional args in input must exist in class
    if not all(check_optional2):
        diag = [optional_args[idx] for idx, condition in enumerate(check_optional) if not condition]
        raise ValueError(f'input args for {firm_class} missing: {diag}')


if __name__ == "__main__":
    flood_scenarios_df_test = pd.read_pickle('data_model_inputs/flood_scenarios_v2.pickletable')
    flood_scenarios = np.random.choice(flood_scenarios_df_test.scenario.unique(), size=3)
    hhs_df = pd.read_pickle('data_model_inputs/households_brackets_per_wijk_city_only.pickletable')
    # agents_to_spawn_test = pd.Series(data=np.random.randint(5, 10, size=hhs_df.size),
    #                                  index=hhs_df)
    # output_dayo = generate_flood_exposure_distribution(flood_scenarios_df_test,
    #                                                    flood_scenarios,
    #                                                    agents_to_spawn_test)
    hhs_to_spawn = reduce_agents_count(hhs_df, target_population=500)
    hh_pcts = generate_household_SE_distribution(hhs_to_spawn, (0., 1.))
    # toy_bulk = generate_household_agents_bulk(households_df=hhs_df,
    #                                           scenarios_df=flood_scenarios_df_test,
    #                                           model.
    #                                           scenarios_applied=flood_scenarios,
    #                                           target_population=500)
