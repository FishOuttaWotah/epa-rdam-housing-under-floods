from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union, Any

if TYPE_CHECKING:
    pass
import model_init

import numpy as np
import itertools, random, time
import pandas as pd
import functools
import operator


# someplace to collect the experiment parameters to vary

def generate_flood_scenarios(fld_scenarios,
                             n_fld_events: Sequence[int],
                             l_scenarios_per_cat: int):

    # generate permutations of scenarios based on number of flood events within simulation
    generate = []
    for n in n_fld_events:
        choicelist = list(itertools.permutations(fld_scenarios, r=n))
        if l_scenarios_per_cat < len(choicelist): # choose subset of choices
            select = random.sample(choicelist, k=l_scenarios_per_cat)
        else:  # if number of scenarios is larger than choices, just put whatever is available
            select = choicelist
        generate.append(select)
    return generate


def generate_flood_timings(n_fld_events: Sequence[int], t_last_trailing: int, t_next_flood_lim: int, n_variants: int, t_first: int, mode=2, t_intervals:Sequence[int] = None):
    """
    Generates a set of flood timings, defined by number of floods events to generate, the upper limit of the next flood event, and the number of sequences to create.
    :param n_fld_events: range of number of floods to create
    :param t_last_trailing: how many steps to simulate after the last flood
    :param t_next_flood_lim: upper limit to when the next flood should be generated
    :param n_variants: number of combinations to generate
    :param t_first: timing of which the first flood event should start.
    :param mode: choose mode 1 for random choice (but exclusive) and mode 2 for defined intervals. Mode 2 can only handle n_fld_events of <= 2, and must have n_variants > 1.
    :param t_intervals: a custom list of intervals for flood timing, only for mode 2
    :return: dict of list of tuples: keys are number of flood events, tuples are the timings for flood events, with the last element in the tuple being the stop time of the sim
    """

    generate = []
    for n in n_fld_events:
        if n == 1:  # just create first and last
            t_end = t_last_trailing
            generate.append([((t_first,), t_end)])
            # first position is always known
        elif n == 0:
            generate.append([((-1,), t_next_flood_lim + t_last_trailing)])
        else:
            sublist = None  # dummy placeholder for python type hint
            # sublist = [[t_first] * n_variants]  # generate first flood (always 0)
            if mode == 1:
                raise NotImplementedError('mode 1 for generate_flood_timings() is depreciated. Please either update the mode 1 logic or use mode 2.')
                # choice_range = range(1, t_next_flood_lim + 1)  # 1 = immediate next step another flood
                # # generate the next steps until next flood (1st flood timing is constant)
                # n_to_draw = n - 1
                # while n_to_draw > 0:
                #     sublist.append(random.sample(choice_range,
                #                                  k=n_variants))  # sample ensure some degree of uniqueness/mutual exclusivity
                #     n_to_draw -= 1

            elif mode == 2:
                sublist = [[t_first] * len(t_intervals)]
                if n > 2:
                    raise NotImplementedError('Warning <model_experiment_utils.generate_flood_timings>: mode 2 is not designed for more than 2 floods')
                if n_variants < 2:
                    raise ValueError('Warning <model_experiment_utils.generate_flood_timings>: mode 2 requires n_variants > 1')
                if t_intervals is None:
                    sublist.append(list(np.linspace(start=1,
                                             stop=t_next_flood_lim+1,
                                             num=n_variants,
                                             dtype=int)))
                else:
                    # modify to use custom intervals (in steps, not years)
                    sublist.append(t_intervals)

            # sum cumulative to get the flood timings
            times = []
            for elem in zip(*sublist):
                cumulative = [sum(elem[:idx + 1]) for idx, _ in enumerate(elem)]
                times.append(
                    (tuple(cumulative), elem[-1] + t_last_trailing))  # also add the last point of simulation
            generate.append(times)

    return generate


def generate_model_sets(m_config_params: dict[str, Union[Sequence, Any]], set_label: str,
                        m_constants: dict[str, Any]):
   # generate permutations of model configs (not scenarios)
    combos = list(itertools.product(*m_config_params.values()))
   # generate dict of params for model input via kwargs
    combos_dict = [dict(tuple(zip(m_config_params.keys(), scen))) for scen in combos]
    # insert constant entries into dict of params
    _ = [a.update(m_constants) for a in combos_dict]

   # set name for model config. Currently model config is not used, so may require an update
   # todo: check if the naming mode is correct when the model config is modified.
    if len(combos) > 1:
        labels = ["+".join(str(elem) for elem in tup) for tup in combos]
        labels = [f"{set_label}({elem})" for elem in labels]
    else:
        labels = [f"{set_label}"]
   # set labels as dict keys, and the dict of model params as value (dict of dicts)
    combos_dict = dict(zip(labels, combos_dict))
    return combos_dict


def prepare_model_runs(fld_scenarios_df,
                       fld_names_mapping,
                       params_mSets,
                       params_fldT,
                       params_fldS,
                       params_seeds,
                       label_add):
    # todo write docstring for this function
    # generate sets of model configs, flood timings and flood scenarios respectively
    mSets = generate_model_sets(**params_mSets)
    fldT = generate_flood_timings(**params_fldT)
    fldS = generate_flood_scenarios(**params_fldS)

    # combine all 3 sets into model setups, finding all possible permutations
    combine = []
    # per flood frequency in sim (1, 2, 3 floods), generate permutations
    for idx, _ in enumerate(fldT):
        combine.extend(itertools.product(*[mSets.keys(), fldT[idx], fldS[idx], params_seeds]))
        # combine would have len(3) elements, one for the model config key, the flood scenario timing, and flood scenario label

    beeg_list =[]  # list collecting all model setups
    scen_list= []  # list collecting scenarios per setup
    name_list = []  # list collecting identifying labels per setp

    # convert expanded model setups into model input form
    for raw_tup in combine:
        # extract model config key, flood scenario timing, and scenarios
        mset_key, timings, scenarios, seed = raw_tup
        mset = mSets[mset_key].copy()  # get all model params from key
        fld_time, sim_end = timings  # extract flood event timing and simulation horizon
        # extract necessary flood scenario information via flood scenario labels
        fld_sub_df = fld_scenarios_df[fld_scenarios_df['scenario'].isin(scenarios)]
        # append flood scenarios arguments for model input (kwargs)
        # NB: very sensitive to naming!
        mset['flood_scenarios'] = dict(zip(fld_time, scenarios))
        mset['flood_scenario_df'] = fld_sub_df

        # make label that will be used for filing/saving
        name = f"{mset_key}_" \
               f"{label_add}"\
               f"[{'+'.join([fld_names_mapping[name] for name in scenarios])}]_" \
               f"({'+'.join([str(t) for t in fld_time])})_" \
               f"iter{seed}"

        name_list.append(name)

        # wrap everything into another overarching dict, for the single-run kwargs input
        beeg_list.append(dict(zip(
            ["run_label",'model_params','sim_horizon','seed'],
            [name, mset, sim_end, seed])))
        scen_list.append(dict(zip(fld_time,scenarios)))

    return beeg_list, scen_list, name_list

def get_trimmed_names(names_list: Sequence[str]):
    blah = []
    for name in names_list:
        splitted = name.split('_')
        # preserve only first 3 chars and all of last
        blah.append(".".join([splitted[0][:4], splitted[-1]]))

    return blah

# NB!!!: be very careful with changing the arg names of the init level, the rename may not update strings, which is crucial for the kwargs method for model input.
class Model_Single_Run:
    def __init__(self, input_batch,
                 save_dir: str = 'data_model_outputs/',
                 suffix:str= ''):
        model_params = input_batch['model_params']
        sim_horizon = input_batch['sim_horizon']
        run_label = input_batch['run_label']
        seed = input_batch['seed']
        print(f'Simulating {run_label}')

        random.seed(seed)  # double set seed just in case
        self.model: model_init.RHuCC_Model = model_init.RHuCC_Model(seed=seed, **model_params)

        while self.model.schedule.steps < sim_horizon:
            self.model.step()

        # ----- Postprocessing -------
        # drop_cols = ['h_obj', 'h_SE', 'i_disp', 'i_gross', 'h_value_pct']
        model_preserve = ['district','h_value','fld_exp', 'fld_dsc', 'category']
        agent_preserve = ['district','win_bid','price_delta','category','h_value']

        # all household agents
        hh_all_1: list[pd.DataFrame] = self.model.datacollector.model_vars['hh_agents']
        keys_range = list(range(1, len(hh_all_1) + 1))
        self.hh_all_2 = pd.concat(hh_all_1, keys=keys_range, names=['step']).loc[:,model_preserve]
        # re-assert as boolean due to limitation of pandas update() with changing the dtypes
        # self.hh_all_2['fld_dmg'] = self.hh_all_2['fld_exp'].astype(bool)
        # self.hh_all_2['h_value_r'] = 0.  # new column to be edited later

        # all transactions done in the model
        hh_transactions_1: list[pd.DataFrame] = self.model.datacollector.model_vars['hh_transactions']
        self.hh_transactions_2 = pd.concat(hh_transactions_1, keys=keys_range,
                                           names=['step']).loc[:,agent_preserve]
        # conversion of dfs to categorical
        for df in [self.hh_all_2, self.hh_transactions_2]:
            idx = df.index
            # change the index to categorical
            df.index = df.index.set_levels([idx.levels[0].astype('category'), idx.levels[1].astype('category')])
            df['district'] = df['district'].astype('category')
            df['category'] = df['category'].astype('category')


        # save as pickles
        self.hh_all_2.to_pickle(f'{save_dir}hh_A_{run_label}_{suffix}.pkl.xz')
        self.hh_transactions_2.to_pickle(f'{save_dir}hh_T_{run_label}_{suffix}.pkl.xz')
        self.model.h_fld_categories.to_pickle(f'{save_dir}hh_expo_{run_label}_{suffix}.pkl.xz')
