from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

import agent_household
import model_init


def generate_asli_discount_curve(
        bias_e: float = None, bias_t: float = None, steps_per_year: int = 4,
        ratio_mode: bool = False, path = 'data/recovery_prices_asli22.tsv', curve_kind='quadratic'):
    extract = pd.read_csv(path,
                          sep='\t', index_col=0, comment='#')

    # ignore first line
    prices = extract.loc[:, 'discount']
    error = extract.loc[:, 'error']
    prices_r = None
    if ratio_mode:
        prices_r = np.insert(prices[1:].to_numpy() / prices.to_numpy()[0], 0, 1.)
        prices.index = extract['time_alt(yr)']  # shift discount to

    # correct time index to model level (4 steps per year)
    prices.index = prices.index * steps_per_year

    # bias curve up/down based on error bars (Big caveat: error bars are just the SD)
    if bias_e is not None:
        if abs(bias_e) <= 1:
            prices = prices - bias_e * error
        else:
            print('Warning: generate_asli_discount_curve can only handle -1 <= bias_e <= 1 ')

    # bias curve length-wise
    if bias_t is not None:
        prices.index = prices.index * bias_t

    # generate interpolation curve
    if ratio_mode:
        print("A_discount_curve: using ratio-based curve")
        bounds = (prices_r[0], 0.)
        curve = interp1d(x=prices.index, y=prices_r, kind=curve_kind,
                         fill_value=bounds, bounds_error=False)
    else:
        print("A_discount_curve: using '93/'95 Limburg discount regression")
        bounds = (prices.values.min(), np.NAN)
        curve = interp1d(x=prices.index, y=prices.values, kind=curve_kind,
                         fill_value=bounds, bounds_error=False)

    return curve


def update_discount(model: model_init.RHuCC_Model,
                    is_flood_now: bool,
                    discount_curve: Callable):
    # take only the items that are flooded
    if is_flood_now:
        # sets the damages for areas
        data = dict([(name, [-99, -1]) for idx, name in enumerate(model.d_fld_affected)])
        # -99 : discount, but as a null placeholder
        # -1 : elapsed time, but will be 0 as modified under Normal Operations block
        df = pd.DataFrame.from_dict(
            data=data, orient='index',
            columns=model.d_discounts.columns)
        model.d_discounts = pd.concat([model.d_discounts, df])  # attach to the model

        # drop duplicate indices (essentially overwrite)
        model.d_discounts = model.d_discounts[~model.d_discounts.index.duplicated(keep='last')]

    # normal operations
    if not model.d_discounts.empty:  # if discounts contains anything
        model.d_discounts.t_elapsed = model.d_discounts.t_elapsed + 1  # increment time
        model.d_discounts.discount = discount_curve(
            model.d_discounts.t_elapsed / model.STEPS_PER_YEAR)  # calculate damage

        # ended = model.d_discounts.t_elapsed.isnull()  # should be a boolean mask
        model.d_discounts.dropna(subset=['discount'], inplace=True)
        model.d_fld_affected = model.d_discounts.index.to_list()  # updates if a region is free of flood


def update_attraction_A(model: model_init.RHuCC_Model,
                        is_flood_now: bool):
    if is_flood_now:
        model.d_attraction.update(model.d_attraction.loc[model.d_fld_affected] - model.D_MAX_PENALTY)  # is Series
    else:
        # increment damaged ones to plus 1
        # NB: assumption that subset would be in district!
        if model.d_fld_affected:  # empty check
            subset = model.d_attraction[model.d_attraction < 100]
            denom = abs(model.func_h_discount.y.max() - model.func_h_discount.y.min())
            for district in model.d_fld_affected:
                # get district current discount rate
                discount = model.d_discounts.loc[district, 'discount']
                subset[district] = 100 - int(round(model.D_MAX_PENALTY * (abs(discount) / denom)))
            model.d_attraction.update(subset)


def update_attraction_B(model: model_init.RHuCC_Model):
    # get non-NA values from the devastation model attribute
    # and just update according to devastation
    disc_distr = model.d_devastation.ratio.notna()  # find districts that are still discounted

    if not disc_distr.empty:  #
        affected = model.d_devastation.loc[disc_distr, 'ratio']
        disattract = model.MAX_ATTRACTION - (model.D_MAX_PENALTY * affected).round().astype(int)
        model.d_attraction.update(disattract.rename('d_attraction'))

    # force other districts to be back to normal attraction
    model.d_attraction[~disc_distr] = model.MAX_ATTRACTION
    return


def collect_flooded_houses(model: model_init.RHuCC_Model,
                           keyword: str = agent_household.HouseholdAgent.__name__):
    house_agents: dict[str, agent_household.HouseholdAgent] = model.schedule.agents_by_type[keyword]

    flooded = dict([(u_id, {'h_obj': house, 'fld_dmg': house.flood_dmg[model.fld_scenario_idx]})
                    for u_id, house in house_agents.items()
                    if house.flood_dmg[model.fld_scenario_idx] > 0.])

    # update the current value of the agent's current experienced flood damage.
    for row in flooded.values():
        agent = row['h_obj']
        damage = row['fld_dmg']
        if damage > agent.fld_dmg_current:  # assumption: overlapping damages; only pick the more severe
            agent.fld_dmg_current = damage

        # mode C: define starting discount
        agent.fld_discount = agent.fld_dmg_current * model.DAMAGE_DISCOUNT_RATE
        agent.fld_discount_rate = 1.
        agent.fld_discount_elapsed = 0  # reset to 0

    df = pd.DataFrame.from_dict(flooded, orient='index')
    model.h_flooded = pd.concat([model.h_flooded, df])
    model.h_flooded = model.h_flooded[~model.h_flooded.index.duplicated(keep='last')]
    # model.h_flooded.update(df.fld_dmg)
    model.h_df.update(df.fld_dmg)  # update entries with flood history


def repair_flooded_houses(model: model_init.RHuCC_Model):
    # eliminate repaired houses from DF
    if not model.h_flooded.empty:
        model.h_flooded.fld_dmg = model.h_flooded.fld_dmg - model.REPAIR_RATE
        h_repaired = model.h_flooded.fld_dmg <= 0.  # boolean array
        model.h_flooded.loc[h_repaired, 'fld_dmg'] = 0.

        for row in model.h_flooded.itertuples():
            home = row.h_obj
            if row.fld_dmg <= 0:
                home.fld_dmg_current = 0
            else:
                home.fld_dmg_current = row.fld_dmg

        # preserve only still-damaged houses
        model.h_df.update(model.h_flooded)  # update big df first for values
        model.h_flooded = model.h_flooded[~h_repaired]  # then eliminate


def update_devastation(model: model_init.RHuCC_Model,
                       is_flood_now: bool,
                       consider_only_damaged: bool):
    # get ratio of devastation (current damaged/total damaged or current damaged/total households)
    # get full data of flooded homes
    to_check = False
    if model.RUN_CONFIG == 'C':
        if not model.h_discounted.empty:  # check if discounted pool is empty
            to_check = True
            affected = model.h_df.loc[model.h_discounted.index, :]['district'].value_counts().astype('int')
        else:
            to_check = False
    elif model.RUN_CONFIG == "B":
        if not model.h_flooded.empty:
            to_check = True
            affected = model.h_df.loc[model.h_flooded.index, :]['district'].value_counts().astype('int')
        else:
            to_check = False

    if to_check:
        if is_flood_now:
            model.d_devastation.loc[affected.index, ['start', 'current']] = affected

        else:  # not flooded
            model.d_devastation.loc[affected.index, 'current'] = affected

        if consider_only_damaged:
            model.d_devastation['ratio'] = model.d_devastation['current'] / model.d_devastation['start']
        else:
            model.d_devastation['ratio'] = model.d_devastation['current'] / model.d_devastation['total']
            model.d_devastation.loc[model.d_devastation['current'] == 0, 'ratio'] = np.nan
    else:  # force to zero
        model.d_devastation.loc[:, ['start', 'current']] = 0
        model.d_devastation['ratio'] = np.nan


def set_discount_C(model: model_init.RHuCC_Model):
    # maybe track household agents' duration flooded (via counter)
    # define
    houses = model.h_flooded['h_obj']
    # todo warning potential overlap, use dict probably

    # model.h_discounted['h_obj'] = houses
    discounts = [h.fld_discount for h in houses.values]
    discount_r = [h.fld_discount_rate for h in houses.values]
    elapsed = [h.fld_discount_elapsed for h in houses.values]
    # how to discriminate between houses with differeent

    # repaired = [False] * houses.size
    temp = pd.DataFrame(data={'h_obj': houses.values,
                              'discount': discounts,
                              'discount_start': discounts,
                              'discount_r': discount_r,
                              'elapsed': elapsed}, index=houses.index)

    model.h_discounted = pd.concat([model.h_discounted, temp])
    model.h_discounted = model.h_discounted[~model.h_discounted.index.duplicated(keep='last')]

    model.h_df.update(model.h_discounted['discount'].rename("fld_dsc"))
    model.h_df.loc[model.h_discounted.index, 'was_discounted'] = True
    return


def update_discount_C(model: model_init.RHuCC_Model):
    # calculate and update discount (independent of repair)
    if not model.h_discounted.empty:  # only run if there are values
        # check for repaired elements to start counting down
        repaired_idx = model.h_discounted.index.difference(model.h_flooded.index)

        if not repaired_idx.empty:
            repaired = model.h_discounted.loc[repaired_idx]
            repaired['elapsed'] = repaired['elapsed'] + 1
            repaired['discount_r'] = model.func_h_discount(repaired['elapsed'])
            repaired['discount'] = repaired['discount_start'].multiply(repaired['discount_r'], axis=0,)#fill_value=0)
            # check for NANs or negligible discounts in DF (indicating discount has passed)
            is_zero = repaired.loc[(repaired['discount'] >= -0.001) | repaired['discount'].isnull(),:]
            # update
            repaired.loc[is_zero.index, 'discount'] = 0.
            # todo repair not encapsulated in agent level
            # update house's own values
            for row in repaired.itertuples():
                h_obj: agent_household.HouseholdAgent = row.h_obj
                h_obj.fld_discount = row.discount
                if row.discount == 0:
                    h_obj.fld_discount_elapsed = 0  # reset again
                else:
                    h_obj.fld_discount_elapsed = row.elapsed

            # update main model ledger
            # todo: why is the update after the itertuples again?
            model.h_discounted.update(repaired)
            model.h_df.update(model.h_discounted['discount'].rename('fld_dsc'))
            model.h_discounted.drop(index=is_zero.index, inplace=True)

    return
