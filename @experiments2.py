from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import scipy.interpolate
import random
import pandas as pd
import numpy as np
import copy
from multiprocessing import Pool

import agent_firm_basic
import agent_household
import env_damage
import env_socioeconomic
import model_init
import model_experiment_utils
from agent_base import DelayAssigned
import env_flooding
from env_housing_market import HousingMarket, Buyer

# treat these as toy experiments sandbox for now

seed = 0  # for testing
random.seed(seed)  # random draw seeds for scenarios

# Variables
fld_attraction_penalty = np.arange(start=0, stop=101, step=20, dtype=int)
run_configs = {
    "A": "agent_priority_A",
    "B": "agent_priority_B"  # todo update this!
}

a_bias_e = [-1, 0, 1]  # value bias for Asli's regression
a_bias_t = [0.5, 0.75, 1.0, 1.25]  # time-bias, corresponding to price forgetting in 8, 12, 16, 20 years

b_repair = np.arange(start=5, stop=21, step=5, dtype=int)  # repair per turn post flood
b_depreciation_to_dmg = [-0.5, -1.0, -1.5, -2.0]  # percentage per damage level percentage

# sensitivity
s_buyer_seller_ratio = 7 # np.arange(start=4, stop=8, dtype=int)
s_pop_size = 5000 #np.arange(start=2000, stop=10000, step=1000, dtype=int)

fld_timing_min = 0
fld_timing_last_flood_interval = 26 * 4  # simulate 24 years after last flood
fld_timing_min_interval = 4  # next flood cannot be within a year (IRRATIONAL)
fld_events_max = 3 # number of events, maximum
scenarios_per_cat = 10  # 10 scenarios per 1/2/3-flood category (for now)
timings_per_scenario = 5  # could also affect timing?


# set up dataframe for households and floods
hh_df = pd.read_pickle('data_model_inputs/households_brackets_per_wijk_city_only.pickletable')
fld_df = pd.read_pickle('data_model_inputs/flood_scenarios_v2.pickletable')
fld_choices = np.random.choice(fld_df.scenario.unique(), size=3)  # with replace
fld_timings = [2, 40, 70]
fld_sub_df = fld_df[fld_df['scenario'].isin(fld_choices)]
hh_movers_df = pd.read_pickle('data_model_inputs/households_migration.pickletable')
buyer_args = {}  # {'bid_bounds':(0.9, 1.1, 0.01)}

## Set up interpolation for socio-economics (based off env_socieoconomic.py)
rdam_income_df = pd.read_pickle('data_model_inputs/income_gross_to_disposable.pickletable')
income_gross = rdam_income_df.bruto.to_list()
income_gross = [income_gross[0]] + income_gross
income_disposable = rdam_income_df.besteedbaar.to_list()
income_disposable = [income_disposable[0]] + income_disposable
x_range = np.linspace(0.1, 1.0, len(income_gross) - 1) - 0.05  # should represent every 10% percentile at their median
x_range = np.insert(x_range, obj=0, values=[0.])

# set up inputs for firms
# f_df = pd.read_pickle('data_model_inputs/companies_per_wijk.pickletable')
# f_types = [agent_firm_basic.CapitalFirm,
#            agent_firm_basic.ConsumptionGoodFirm,
#            agent_firm_basic.ConsumptionServiceFirm]
# f_ratios = [0.2, 0.4, 0.4]
# f_args_assigned = ['unique_id', 'model', 'location', 'price', 'flood_exp', 'flood_dmg']  # args assigned later
# f_args = {agent_firm_basic.CapitalFirm:
#               {'productivity':
#                    (np.random.uniform, {'low': 0.9, 'high': 1.1}),  # from CRAB
#                'wage':
#                    (np.random.uniform, {'low': 0.9, 'high': 1.1}),  # from CRAB
#                'initial_net_worth': 2,  # from CRAB
#                'transport_cost': 0.03},  # from CRAB
#           agent_firm_basic.ConsumptionGoodFirm:
#               {'productivity':
#                    (np.random.uniform, {'low': 0.9, 'high': 1.1}),  # from CRAB
#                'wage':
#                    (np.random.uniform, {'low': 0.9, 'high': 1.1}),  # from CRAB
#                'initial_net_worth': 2,  # from CRAB
#                'initial_capital_output_ratio': 0.7,  # from CRAB
#                'initial_machines': 5,  # from CRAB
#                'initial_amount_capital': 3,  # from CRAB
#                },
#           agent_firm_basic.ConsumptionServiceFirm:
#               {'productivity':
#                    (np.random.uniform, {'low': 0.9, 'high': 1.1}),  # from CRAB
#                'wage':
#                    (np.random.uniform, {'low': 0.9, 'high': 1.1}),  # from CRAB
#                'initial_net_worth': 2,  # from CRAB
#                'initial_capital_output_ratio': 1.3,  # from CRAB
#                'initial_machines': 5,  # from CRAB
#                'initial_amount_capital': 3,  # from CRAB
#                },
#           }  # long definitiion

# set up inputs for flooding
fld_funcs_base: dict = env_flooding.get_slager_huizinga_DDfuncs()
fld_funcs: dict = {agent_household.HouseholdAgent: fld_funcs_base['2-house_structural'],
                   agent_firm_basic.CapitalFirm: fld_funcs_base['2-industry'],
                   agent_firm_basic.ConsumptionGoodFirm: fld_funcs_base['2-shops'],
                   agent_firm_basic.ConsumptionServiceFirm: fld_funcs_base['2-offices']}

# inputs for model
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
steps = 30 * 4
attract_max_pen = 70  ## note this could be set to 100 for B runs with full devastation
buyer_seller_ratio = 7
FLD_REPAIR_RATE = 1.   # essentially repair when done
DMG_DISCOUNT_RATE = -1

# create model instance plus inputs
model = model_init.RHuCC_Model(func_SE_to_disposable=func_SE_to_disposable, func_SE_to_gross=func_SE_to_gross,
                               func_gross_to_home_value=func_gross_to_home_value, funcs_depth_damage=fld_funcs,
                               flood_scenario_df=fld_sub_df, flood_scenarios=dict(zip(fld_timings, fld_choices)),
                               AB_agent_priority=agent_priority_B, AB_run_config='C', A_func_discount=func_discount,
                               B_repair_rate=FLD_REPAIR_RATE, B_damage_discount_rate=DMG_DISCOUNT_RATE,
                               B_dev_dmg_only=False, attract_max_pen=attract_max_pen,
                               buyer_seller_ratio=buyer_seller_ratio, buyer_args=buyer_args, households_df=hh_df,
                               households_seller_df=hh_movers_df, households_target_pop=3000)
#
for step in range(steps):
    model.step()
#
output: Sequence[pd.DataFrame] = model.datacollector.model_vars['hh_transactions']
output2 = pd.concat(output, keys=list(range(1, len(output)+1)))  # .set_index(keys=['step','index'])
output2 = output2.loc[:, ['district','win_bid','price_delta']]
output2.to_pickle('data_model_outputs/trial_CBv2_3k.pickletable')

# name = "trial_CBv2_3k"
# output1_1: Sequence[pd.DataFrame] = model.datacollector.model_vars['hh_transactions']
# output1_2 = pd.concat(output1_1, keys=list(range(1, len(output1_1) + 1)))  # .set_index(keys=['step','index'])
# output1_2 = output1_2.loc[:, ['district', 'win_bid', 'price_delta']]
# output1_2.to_pickle(f'data_model_outputs/hh_transactions_{name}.pickletable')
#
# output2_1 : Sequence[pd.DataFrame] = model.datacollector.model_vars['hh_agents']
# output2_2 = pd.concat(output2_1, keys= list(range(1, len(output2_1) + 1)))
# output2_2 = output2_2.loc[:, ['district','h_value','fld_dsc']]
# output2_2.to_pickle(f'data_model_outputs/hh_agents_{name}.pickletable')