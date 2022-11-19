import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

import os, re

def plot_transactions():
    # modify pandas df

    return

def plot_price_indices():

    return

def generate_bulk_plots():
    # generate intermediate options and probably save them (later)

    # save paths (later)

    # paths required
    save_dir = 'data_model_outputs'
    prefix_A = 'hh_A'
    prefix_T = 'hh_T'
    prefix_exp = 'hh_exp'
    ending = '_.pkl.xz'

    scenarios_dict = {}

    path_scenarios = f'{save_dir}/experiment_scenarios_ref.pkl.xz'

    for file_name in os.listdir(save_dir):
        # target = None
        if prefix_A in file_name:
            # find scenario name
            scenario = "_".join(file_name.split("_")[2:-1])
            scenarios_dict[scenario] = {
                "A": file_name,
                "T": f"{prefix_T}_{scenario}{ending}",
                "E": f"{prefix_exp}_{scenario}{ending}"
            }

    # get rest of data

    # should link together?

    return

def something():

    return