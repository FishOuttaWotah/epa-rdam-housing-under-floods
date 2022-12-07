from __future__ import annotations
from typing import Union, TYPE_CHECKING, Sequence, Callable, Type, Mapping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mesa
import mesa.datacollection
import random

# imports from local stuff
import agent_base
import agent_household
import env_damage

if TYPE_CHECKING:
    pass
import model_scheduler
import model_ledger
import agent_generator
# import agent_firm_basic
import agent_housing_market
import model_datacollection_h as dc_h

## NB!!! : be very careful with renaming the arguments in the model init. These changes may not be reflected in some strings (due to the kwargs input method used in this model) and would raise errors.
class RHuCC_Model(mesa.Model):
    def __init__(self, func_SE_to_disposable: Callable, func_SE_to_gross: Callable,
                    func_gross_to_home_value: Callable,
                 funcs_depth_damage: Mapping[Type[agent_base.CustomAgent], Callable], flood_scenario_df: pd.DataFrame,
                 flood_scenarios: dict[int, str], AB_agent_priority: Sequence[tuple[Callable, tuple[str]]],
                 AB_run_config: str, A_func_discount: Callable = None, B_repair_rate: float = 0.1,
                 B_damage_discount_rate: float = -1, B_dev_dmg_only: bool = False, attract_max_pen: int = 50,
                 buyer_seller_ratio=3, bid_min=96, bid_increment=1, buyer_args: dict = None, households_df=None,
                 households_seller_df: pd.Series = None, households_target_pop: int = 1000, seed=None):
        super().__init__()
        self._seed = seed  # seed attribution MESA

        # Miscellaneous items
        # self.f_ignored_args = firm_args_ignore  # for ignoring arguments during a check in
        self.STEPS_PER_YEAR = 4  # year in 4 quarters
        self.DEVA_CONS_DMG = B_dev_dmg_only
        self.RUN_CONFIG = AB_run_config

        # SELF: set up self calls for model inputs for future retrieval

        # Interpolation functions used in this model
        self.func_SE_to_disposable = func_SE_to_disposable
        self.func_SE_to_gross = func_SE_to_gross
        self.func_gross_to_home_value = func_gross_to_home_value
        self.funcs_depth_damage = funcs_depth_damage
        self.func_h_discount = A_func_discount

        self.num_agents = None

        # FLOOD attributes
        self.is_flood_now = False
        self.fld_df = flood_scenario_df.set_index(keys=['scenario', 'wijk'])
        self.fld_scenario = flood_scenarios  # str
        self.fld_scenario_events = list(flood_scenarios.values())
        self.fld_scenario_timings = list(flood_scenarios.keys())  # timesteps in which flood happens
        self.fld_scenario_idx = -1  # 0 used to access location in list, will be incremented
        self.fld_affected_districts = [[],[]]
        for idx, scenario in enumerate(self.fld_scenario_events):
            self.fld_affected_districts[idx] = self.fld_df.loc[(scenario, slice(None))].index.get_level_values('wijk')

        # DISTRICT-level attributes
        self.d_attraction = pd.Series([100 for _ in households_df.index],
                                      index=households_df.index,
                                      name='d_attraction')
        self.districts = households_df.index
        self.D_MAX_PENALTY = attract_max_pen
        self.d_discounts = pd.DataFrame(columns=['discount', 't_elapsed']).astype(
            {'discount': 'float', 't_elapsed': 'int'})  # currently empty
        self.d_fld_affected = []
        self.d_devastation = pd.DataFrame(0, index=self.districts, columns=['total', 'start','current','ratio']).astype(
            {'total': 'int','start':'int','current':'int','ratio':'float'})

        ## NEW: households-based recording
        self.h_df = None  # will be a dataframe
        self.h_df_by_district = None
        self.h_sellers = None
        self.h_sellers_weights = None
        # self.h_sellers_by_price = None
        # self.h_sellers_by_price_counts = None
        # self.h_sellers_by_district = None
        # self.h_sellers_by_district_counts = None
        self.h_sellers_distr_year = households_seller_df.loc[households_df.index, :]  # appended in env_housing_market.
        self.h_sellers_distr_step = self.h_sellers_distr_year / self.STEPS_PER_YEAR  # appended in env_housing_market.
        self.h_buyers = None
        self.h_income_median_d = self.func_SE_to_disposable(0.5)  # NB: maybe depreciated
        self.h_income_median_g = self.func_SE_to_gross(0.5)
        self.h_SE_percentile_range = None  # appended in generate_households_bulk
        self.h_SE_regional_distr = None  # appended in generate_households_bulk, a df
        self.h_value_bins_labels = None
        self.h_value_bins = None
        self.h_flooded = pd.DataFrame(columns=['h_obj','fld_dmg'])
        self.h_discounted = pd.DataFrame(columns=['h_obj','discount','discount_r','elapsed','repaired']).astype(
            {'discount': float,'discount_r': float,'elapsed':int})
        self.h_fld_categories = None  # converted into flood exposure categories

        # Recovery attributes
        self.REPAIR_RATE = B_repair_rate
        self.DAMAGE_DISCOUNT_RATE = B_damage_discount_rate

        # initialise objects
        self.schedule = model_scheduler.Scheduler(model=self,
                                                  timestep=3,  # months
                                                  agent_priority=AB_agent_priority,
                                                  shuffle=True)

        # HOUSEHOLD generation section
        self.num_households = None  # updated by ledger object
        self.households = None
        agent_generator.generate_household_agents_bulk(households_df=households_df, scenarios_df=flood_scenario_df,
                                                       scenarios_order=self.fld_scenario_events, model=self,
                                                       func_i_disp=self.func_SE_to_gross,
                                                       func_i_gross=self.func_SE_to_disposable,
                                                       func_h_gross=self.func_gross_to_home_value,
                                                       flood_dd_func=funcs_depth_damage[
                                                           agent_household.HouseholdAgent],
                                                       target_population=households_target_pop,
                                                       SE_truncation=(
                                                           0.1, 0.9))  # NB: other optional arguments not included
        model_ledger.initialise_households(self)
        # generate number of sellers per district, guaranteeing at least 1 seller (via round)
        self.h_sellers_step = (
                self.h_sellers_distr_step.mul(self.h_SE_regional_distr.sum(axis=1), axis=0) + 0.51).round(
            decimals=0).astype('int').squeeze()

        self.housing_market = env_housing_market.HousingMarket(model=self, buyer_seller_ratio=buyer_seller_ratio,
                                                               sellers_per_area=self.h_sellers_step,
                                                               buyer_args=buyer_args, bid_min=bid_min,
                                                               bid_increment=bid_increment)
        self.datacollector = mesa.datacollection.DataCollector()
            # model_reporters={'hh_agents': 'h_df'})
        # hack
        self.datacollector.model_vars['hh_agents'] = []
        self.datacollector.model_vars['hh_transactions'] = []


    def step(self):
        if self.schedule.steps in self.fld_scenario_timings:
            self.is_flood_now = True
            self.fld_scenario_idx += 1  # incrementing counter
            self.fld_scenario = self.fld_scenario_events[self.fld_scenario_idx]
            self.d_fld_affected = self.fld_df.xs(self.fld_scenario, level=0).index.to_list()
            # print(f"is flood now at {self.schedule.steps}")

            # damage handler? -> ledger?
            env_damage.collect_flooded_houses(model=self)
            env_damage.set_discount_C(model=self)
        else:
            self.is_flood_now = False



        if self.RUN_CONFIG == 'A':
            env_damage.update_attraction_A(model=self, is_flood_now=self.is_flood_now)
            env_damage.update_discount(model=self,
                                       is_flood_now=self.is_flood_now,
                                       discount_curve=self.func_h_discount)
        elif self.RUN_CONFIG == 'B' or self.RUN_CONFIG == 'C':
            env_damage.update_devastation(model=self, is_flood_now=self.is_flood_now,
                                          consider_only_damaged=self.DEVA_CONS_DMG)
            env_damage.update_attraction_B(model=self)
        # elif self.RUN_CONFIG == "C":
        #     pass  # mode C does not modify attraction
        else:
            raise ValueError("run_config arg out of viable scope, current run modes are only A, B, or C")

        self.schedule.step()

        # Post-step organisation
        model_ledger.update_house_prices(model=self,
                                         to_update=self.housing_market.sold)
        # todo inspect the model ledger if Mode C can work

        env_damage.repair_flooded_houses(model=self)
        env_damage.update_discount_C(model=self)
        # self.datacollector.collect(self)
        self.datacollector.model_vars['hh_transactions'].append(self.housing_market.sold.copy(deep=True))
        self.datacollector.model_vars['hh_agents'].append(self.h_df.copy(deep=True))
        # pass


if __name__ == "__main__":
    # model = RHuCC_Model()
    pass
