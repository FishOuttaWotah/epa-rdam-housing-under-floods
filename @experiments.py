from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import scipy.interpolate
import random
import pandas as pd
import numpy as np
import copy
from multiprocessing import Pool
import time

import agent_firm_basic
import agent_household
import env_damage
import env_socioeconomic
import model_init
import model_experiment_utils as MEU
from agent_base import DelayAssigned
import env_flooding
from env_housing_market import HousingMarket, Buyer


# dummy reporting for Pool
def report_callback(result):
    print(result)


def report_error(error):
    raise error


# Variables
if __name__ == '__main__':

    # CONSTANTS SETUP FOR RUN, MODEL SETTINGS ARE FURTHER BELOW
    # set up dataframe for households and floods
    hh_df = pd.read_pickle('data_model_inputs/households_brackets_per_wijk_city_only.pickletable')
    fld_df = pd.read_pickle('data_model_inputs/flood_scenarios_v2.pickletable')
    hh_movers_df = pd.read_pickle('data_model_inputs/households_migration.pickletable')
    buyer_args = {}  # {'bid_bounds':(0.9, 1.1, 0.01)}
    fld_events = fld_df.scenario.unique()
    fld_shortnames = dict(zip(fld_events, MEU.get_trimmed_names(fld_events)))  # depreciated

    ## Set up interpolation for socio-economics (based off env_socieoconomic.py)
    rdam_income_df = pd.read_pickle('data_model_inputs/income_gross_to_disposable.pickletable')
    income_gross = rdam_income_df.bruto.to_list()
    income_gross = [income_gross[0]] + income_gross
    income_disposable = rdam_income_df.besteedbaar.to_list()
    income_disposable = [income_disposable[0]] + income_disposable
    x_range = np.linspace(0.1, 1.0,
                          len(income_gross) - 1) - 0.05  # should represent every 10% percentile at their median
    x_range = np.insert(x_range, obj=0, values=[0.])

    # set up inputs for flooding
    fld_funcs_base: dict = env_flooding.get_slager_huizinga_DDfuncs()
    fld_funcs: dict = {agent_household.HouseholdAgent: fld_funcs_base['2-house_structural'],
                       agent_firm_basic.CapitalFirm: fld_funcs_base['2-industry'],
                       agent_firm_basic.ConsumptionGoodFirm: fld_funcs_base['2-shops'],
                       agent_firm_basic.ConsumptionServiceFirm: fld_funcs_base['2-offices']}

    # MODEL KEY FUNCTIONS
    func_SE_to_disposable = scipy.interpolate.interp1d(x=x_range,
                                                       y=income_disposable,
                                                       kind='quadratic',
                                                       fill_value='extrapolate')
    func_SE_to_gross = scipy.interpolate.interp1d(x=x_range,
                                                  y=income_gross,
                                                  kind='quadratic',
                                                  fill_value='extrapolate')
    func_gross_to_home_value = env_socioeconomic.derive_home_value_simple  # function
    func_discount = env_damage.generate_asli_discount_curve(ratio_mode=True, bias_t=1)

    agent_priority_A = [(HousingMarket, HousingMarket.stagelist1),
                        (Buyer, Buyer.stagelist_A),
                        (HousingMarket, HousingMarket.stagelist2)]
    agent_priority_B = [(HousingMarket, HousingMarket.stagelist1),
                        (Buyer, Buyer.stagelist_B),
                        (HousingMarket, HousingMarket.stagelist2)]

    # --- MODEL INPUTS ---
    attract_max_pen = 100  ## note this could be set to 100 for B runs with full devastation
    buyer_seller_ratio = 7
    FLD_REPAIR_RATE = 1.  # essentially repair when done
    DMG_DISCOUNT_RATE = -1  # assumption
    n_scenarios_per_category = np.inf  # 10 or np.inf for all permutations
    n_variants = None  # depreciated
    STEPS_PER_YEAR = 4


    household_target_pop = 5000  # target number of houses for model. Default 3000
    t_next_flood_upper_bound = 10 * STEPS_PER_YEAR  # for interval-based upper limit
    # t_last_trailing = 16 * STEPS_PER_YEAR  # original
    t_last_trailing = 16 * STEPS_PER_YEAR
    timing_generation_mode = 2

    run_sim = True  # True to run all sims.
    test_sim = False

    run_pt = 1  # todo: ensure this is 1 or 2
    if run_pt == 1:  # run with 2-floods for 1 rep each
        flood_event_categories = [2]
        seeds = list(range(1))  # number of iterations with consistent seeds.
        overwrite_scen_df = True # set true to restart overwrite
    elif run_pt == 2:  # run control and 1-flood for 40 reps
        flood_event_categories = [0,1]
        seeds = list(range(40))
        overwrite_scen_df = False
    else:
        raise ValueError("currently only accepts run_pt 1 and 2")

    label_addition = 'v5'  # suffix for future retrieval
    seed = 0  # for testing
    random.seed(seed)  # random draw seeds for scenarios
    n_processes = 7

    # DEFINING INPUTS FOR SCENARIO GENERATION
    # flood scenarios (from LIWO)
    in_flood_scenarios = {
        'fld_scenarios': fld_df.scenario.unique(),
        'n_fld_events': flood_event_categories,
        'l_scenarios_per_cat': n_scenarios_per_category,
    }
    # number and timing of floods
    in_flood_timings = {
        'n_fld_events': flood_event_categories,
        't_last_trailing': t_last_trailing,
        't_next_flood_lim': t_next_flood_upper_bound,  # steps
        'n_variants': n_variants,
        't_first': 0,
        'mode': timing_generation_mode,
        # NB: mode 2 is incremental, mode 1 is random, but mode 2 is limited in capability, see docs
        't_intervals': np.array([1, 2, 4, 6, 8, 10]) * STEPS_PER_YEAR
    }
    # constant variables to be set in model
    # todo !!! NB: CHECK agent priority is in mode A/B
    in_constants = {
        'func_SE_to_disposable': func_SE_to_disposable,
        'func_SE_to_gross': func_SE_to_gross,
        'func_gross_to_home_value': func_gross_to_home_value,
        'funcs_depth_damage': fld_funcs,
        'A_func_discount': func_discount,
        'AB_agent_priority': agent_priority_B,
        'AB_run_config': "C",
        'B_repair_rate': FLD_REPAIR_RATE,
        'B_damage_discount_rate': DMG_DISCOUNT_RATE,
        'B_dev_dmg_only': False,
        "attract_max_pen": attract_max_pen,
        'buyer_args': buyer_args,
        'buyer_seller_ratio': 7,
        'households_df': hh_df,
        'households_seller_df': hh_movers_df,
        'households_target_pop': household_target_pop
    }
    """
    modify:
    "flood_scenario_df"
    "flood_scenarios"
    "buyer_seller_ratio"
    
    
    """
    # constant variables to be set in model_single_run object
    model_params = {
        'm_config_params': {},  # can accept other arguments, this run doesn't use them
        'set_label': 'modeC',
        'm_constants': in_constants  # constants for model
    }

    # for diagnostic
    # f_scenarios = MEU.generate_flood_scenarios(**in_flood_scenarios)
    # f_timing = MEU.generate_flood_timings(**in_flood_timings)

    # generate all scenarios from scenario choices, flood timings, model settings, and seeds
    m_sets, m_scenarios, m_names = MEU.prepare_model_runs(fld_scenarios_df=fld_df,
                                                          fld_names_mapping=fld_shortnames,  # depreciated
                                                          params_fldS=in_flood_scenarios,
                                                          params_mSets=model_params,
                                                          params_fldT=in_flood_timings,
                                                          params_seeds=seeds,
                                                          label_add=label_addition)

    # combine model filenames and scenarios to make reference for post-processing
    m_ref = {}
    # find the number of floods, timing, and scenarios applied, used for recording
    for scenarios, name, s_set in zip(m_scenarios, m_names, m_sets):
        affected = []
        for idx, sc in enumerate(scenarios.values()):
            affected.append(fld_df[fld_df['scenario'].isin([sc])]['wijk'].unique())
        entry = {
            'n_flds': len(scenarios),
            'timings': tuple(scenarios.keys()),
            'sim_time': s_set['sim_horizon'],
            'events': tuple(scenarios.values()),
            'affected': tuple(affected)
        }

        m_ref[name] = entry

    # create dataframe of experiments. this will be saved to disk in the sim mode
    m_ref_df = pd.DataFrame.from_dict(m_ref, orient='index')

    # t_set = m_sets[-1]
    #
    #
    # t_sets = m_sets[:10]
    # t_model = MEU.Model_Single_Run(t_set['model_params'], sim_horizon=t_set['sim_horizon'],
    #                                run_label=t_set['run_name'], seed=seed)

    # save the scenarios to file when run complete


    if test_sim:
        model_test = MEU.Model_Single_Run(m_sets[-1], suffix='_trial')

    if run_sim:
        # initialise multiprocessing stuff
        print(f'Entering Multiprocess with {len(m_sets)} scenarios')
        start_time = time.time()

        pool = Pool(processes=n_processes)
        experiments = pool.map_async(func=MEU.Model_Single_Run,
                                     iterable=m_sets,
                                     callback=report_callback,
                                     error_callback=report_error)
        # experiments.get()  # return values, but also exceptions
        pool.close()  # close entry to the pool
        # while True:
        #     time.sleep(5)
        #     try:
        #         ready=[task.ready() for task in experiments]
        #         successful = [task.successful() for task in experiments]
        pool.join()

        if not overwrite_scen_df:  # not overwrite, just append
            m_ref_df_ori = pd.read_pickle(f'data_model_outputs/experiment_scenarios_ref_{label_addition}.pkl.xz')
            m_ref_df = pd.concat([m_ref_df_ori, m_ref_df])

        m_ref_df.to_pickle(f'data_model_outputs/experiment_scenarios_ref_{label_addition}.pkl.xz')

        print(f"*** Experiments completed at elapsed {round((time.time() - start_time) / 60, 2)} mins")
    else:
        print('@experiments.py: multiprocess not run')
