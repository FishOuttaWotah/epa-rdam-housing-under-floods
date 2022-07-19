
from typing import Union, TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mesa
# Todo
# parametrise the number of number of household agents

# create function to generate agents of different socio-economic standing
# determine number of starting companies in the model
# input for areas with
# create function to generate agents in different mortgage situation (for start only, with different
# need geographic entity for company terrain (
# documentation: perhaps some extensibility for future gentrification or housing stock changes
# TODO: decide on datatype of model inputs

class CRAB_H_Model(mesa.Model):
    def __init__(self,
                 flood_scenario_df: pd.DataFrame,
                 flood_frequency: int,
                 household_wealth_gen_df,
                 areas_and_demographics_df,
                 mortgage_rates,
                 household_agent_ratio: int=1,
                 spatial_res: str = 'wijk'
                 ):
        super().__init__()
        # include some variable describing the starting migration flow for R'dam
        #

        return

    def generate_areas(self, label='wijk'):
        # extract number of areas to be represented in model

        # extract number of household types per area

        # extract number of companies per area

        # compare with flood vulnerability list: vulnerability list should be smaller or equal to the area list
        return

    def generate_households(self):
        # generate distribution of household incomes from the household wealth function and socioeconomic data from areas
        # generate households' socioeconomic status (low/middle/high) income
        # assign area as name to household (perhaps generate an ENUM object?)
        return

    def generate_firms(self):
        # need to look at Alessandro's stuff
        return

    def gen_houses_assign_flood_risk(self):
        # assign a flood risk to household's house (not the agent)
        # this flood risk must be consistent throughout the run, representing persistent risk
        return

    def remove_household_agent(self):
        # gathers one agent and removes it from the simulation
        # agent's end-of-life removal should be recorded by the model

        return

