from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Type
import numpy as np

# if TYPE_CHECKING:
import env_flooding
import pandas as pd

# import agent_firm_basic
import model_init
import agent_household

"""
_____________________________________

Model Ledger class
_____________________________________
"""
"""
Serves as the accountant for records in the model. 
Has an overlap with the Government agent TBH, but Government serves as a background agent, while the Ledger class serves as an environmental object for other agents to retrieve stuff

Also serves as a datacollection point. Might need to integrate a bit with the Mesa datacollector object (or Datacollector would retrieve from this)

"""


# TODO: update model with number of firms for core functions after firm generation
class ModelLedger:

    def __init__(self, model: model_init.RHuCC_Model):
        # BASIC RECORDING: carried over from original model
        self.model = model
        self.scheduler = model.schedule  # assumes the scheduler is attached to model



def initialise_households(model: model_init.RHuCC_Model):
    # between agent population and model start, initialise
    # retrieve households from scheduler
    households: dict[str, agent_household.HouseholdAgent] = model.schedule.agents_by_type[agent_household.HouseholdAgent.__name__]
    # think of preserving which entries for columns
    h_ids = households.keys()
    i_disp = [h.income_disposable for h in households.values()]
    i_gross = [h.income_gross for h in households.values()]
    h_obj = [h for h in households.values()]
    h_values = [h.value for h in households.values()]
    h_locs = [h.location for h in households.values()]
    h_fld_exposure = [sum(h.flood_dmg)>0 for h in households.values()]
    h_fld_dmg = [h.fld_dmg_current for h in households.values()]
    h_fld_dsc = [h.fld_discount for h in households.values()]
    h_SE = [h.SE_percentile for h in households.values()]

    # prepare items in dict form to generate DF
    to_df = {
        'unique_id' : list(h_ids),
        'district': h_locs,
        'h_obj': h_obj,
        'h_value': h_values,
        # 'h_SE' : h_SE,
        # 'fld_dmg': h_fld_dmg,
        'fld_exp': h_fld_exposure,
        'fld_dsc': h_fld_dsc,
        # 'i_disp': i_disp,
        # 'i_gross': i_gross
    }

    model.h_df = pd.DataFrame.from_dict(to_df,orient='columns').set_index('unique_id')
    model.h_df.district = model.h_df.district.astype('category')  # set to categorical
    model.num_households = len(households)
    model.households = households
    model.h_df_by_district = model.h_df.set_index(keys=['district'],append=True) # create view via district
    model.d_devastation['total'] = model.h_df.district.value_counts()

    # all agents' flood exposures in the simulation
    model.h_fld_categories = pd.DataFrame(index=h_ids, columns=[0,1,'district'])

    extract_exp = pd.DataFrame.from_dict(
        dict([(a_id, dict([(idx, exp) for idx, exp in enumerate(a_obj.flood_exp)])) for a_id, a_obj in
              model.households.items()]), orient='index')
    model.h_fld_categories.loc[:,extract_exp.columns] = extract_exp
    # for col in extract_exp.columns:
        # model.h_fld_categories[col] = model.
    model.h_fld_categories.index = model.h_fld_categories.index.astype('category')
    # if 2 not in model.h_fld_categories.columns:  # if the model is only one flood,
    #     model.h_fld_categories[1] = -1  # create new blank column
    # add new column for location
    model.h_fld_categories['district'] = [h.location for h in model.households.values()]
    # sort according to category
    model.h_fld_categories = env_flooding.sort_flood_categories(model.h_fld_categories, model.fld_affected_districts)
    model.h_df['category'] = model.h_fld_categories['category']

    model.h_df['was_discounted'] = False  # addition to track previously discounted, for sales purposes
    pass
    # SUPPLEMENT COLUMNS in h_df
    # add bins to sort for house value
    # BIN_SIZE = 20  # todo may need to parametrise
    # label_percentiles = 100
    # model.h_value_bins_labels = range(0, label_percentiles, label_percentiles // BIN_SIZE)
    # discriminate into 20 brackets
    # model.h_df['h_value_pct'], model.h_value_bins = pd.qcut(model.h_df.h_value,
    #                                                         q=BIN_SIZE,  #bins=BIN_SIZE,
    #                                                         labels=model.h_value_bins_labels,
    #                                                         retbins=True)


def update_house_prices(model: model_init.RHuCC_Model,
                        to_update: pd.DataFrame):
    # # first_
    # to_update['h_value_pct'] = pd.cut(
    #     to_update.h_value,
    #     bins=model.h_value_bins,
    #     labels=model.h_value_bins_labels,
    #     retbins=False
    # )

    # to_update['h_value'] = to_update['h_value'].multiply(to_update['win_bid'], axis=0)
    # update ledgers with object items
    model.h_df.update(to_update)

    # update agent objects?
    for row in to_update.itertuples():
        agent: agent_household.HouseholdAgent = row.h_obj
        agent.value = row.h_value


def something_flood(self):
    # update special for flood
    return
