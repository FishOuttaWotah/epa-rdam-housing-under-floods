from typing import TYPE_CHECKING

# ---------------------------------------------------------------------
## Environment Generation Functions
# Areas (Wijken or Buurten)
# Houses
# ---------------------------------------------------------------------
from typing import TYPE_CHECKING
import pandas as pd
import numpy as np

## Function for generating Areas (input into Model instance)
# assume that input is given as a dict or pandas df
def model_generate_city(input_df: pd.DataFrame,
                        agent_ratio: int,
                        location_label:str):
    # read the df and generate number of areas into ledgers
    # get the number of houses per socio-economic bracket too
    # think of converting the socio-economic brackets into one unified function/relation?
    # as weights

    # generate

    return

## Function for generating houses (input into Model instance)
# input: agent-actual (1 Agent per N households)
def model_generate_houses():

    return


# ---------------------------------------------------------------------
## Agent generation functions
# Households
# Firms (Capital goods firms, Consumption goods firms, Consumption services firms)
# ---------------------------------------------------------------------

# function for Household agent generation: import stuff from inputs side of things
# input: agent-actual (1 Agent per N households)
# input into model level
#
def generate_Households_in_areas():
    # generate households in areas, return as array or set of arrays (which will be
    # includes Mesa inclusion process?
    # draw with 2-phase choice:
    # 1st with the bracket choice with weights (local weights)
    # 2nd with the ranged draw (national income degree)
    return

# function for firms generation
# input: agent-actual ratio (1 Agent per N firms)
# number of types of firms (ie. capital, goods and services)
# need to make an assumption and set as input
# number of firms

if __name__ == '__main__':
    households_df = pd.read_pickle('data_model_inputs/households_brackets_per_wijk.pickletable')
    companies_df = pd.read_pickle('data_model_inputs/companies_per_wijk.pickletable')
    agent_ratio = 200
    # need the agent socio-economic functions here (where was it again?)
    pass


