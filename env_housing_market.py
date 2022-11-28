"""
bunch of functions here again
- determines which household agent would relocate
-
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import random
import copy

import agent_base
import agent_household
import model_init
from agent_household import HouseholdAgent


class HousingMarket(agent_base.CustomAgent):
    stagelist1 = ('generate_sellers', 'generate_buyers')
    stagelist2 = ('clear_sales', 'collect_price_indices')

    def __init__(self, model: model_init.RHuCC_Model, buyer_seller_ratio: float,
                 sellers_per_area: pd.Series, buyer_args: dict,
                 bid_min: int = 93, bid_increment: int = 1):
        super(HousingMarket, self).__init__(model=model, unique_id=self.__class__.__name__)

        # self.model is already present from super call

        self.seller_size = sellers_per_area.sum()
        self.buyer_seller_ratio = buyer_seller_ratio
        self.buyer_size = int(round(buyer_seller_ratio * self.seller_size))
        self.bids = {}
        self.bids_interest = {}
        self.sellers_per_area = sellers_per_area
        self.buyer_ids = [f"buyer_{num}" for num in range(self.buyer_size)]
        self.buyer_args = buyer_args
        self.win_bidders = {}
        self.win_bids = {}
        self.unsold_ids = []
        self.sold_ids = []
        self.sold = None
        self.model.schedule.add(self)
        self.buyers_generated = False
        self.BID_MIN = bid_min  ## int percentage #
        self.BID_INCREMENT = bid_increment  ## int percentage

        # should buyers just look elsewhere? and distribute themselves across

    def generate_sellers(self):
        # select population of households to sell
        # perhaps create age of sale to see if agents
        # check if vacancies are still present, and fill to max
        # generate per location
        seller_ids = []
        # need to extract from location-based view
        for district, n_movers in self.sellers_per_area.items():
            # get view
            view = self.model.h_df_by_district.xs(key=district, level='district')
            seller_ids.extend(list(np.random.choice(view.index, size=n_movers, replace=False)))

        # set up mode C
        # if self.model.RUN_CONFIG == "C":  # eliminate sellers that are currently flooded
        #     seller_ids = pd.Index(data=seller_ids).difference(self.model.h_flooded)
        self.model.h_sellers = self.model.h_df.loc[seller_ids, :]

        # self.model.h_sellers_by_price = self.model.h_sellers.set_index(keys=['h_value_pct'], append=True)
        # self.model.h_sellers_by_price_counts = self.model.h_sellers_by_price.groupby(level=[1]).size()
        # self.model.h_sellers_by_district = self.model.h_sellers.set_index(keys=
        #                                                                   ['district'], append=True)
        # self.model.h_sellers_by_district_counts = self.model.h_sellers_by_district.groupby(
        #     level=[1]).size()
        # ^ collects number of seller vacancies per step
        # create dictionary for bidding process
        self.bids = dict([(seller_id, []) for seller_id in self.model.h_sellers.index])
        self.bids_interest = copy.deepcopy(self.bids)
        self.model.h_sellers_weights = [self.model.d_attraction[district] for
                                        district in self.model.h_sellers.district]

    def generate_buyers(self):
        # generate_buyers for schedule, initialise once, use buyers again
        # TODO: current assumption that buyer demand is constant over time
        # if not self.buyers_generated:
        self.model.h_buyers = [Buyer(unique_id=u_id, model=self.model, housing_market=self,
                                     **self.buyer_args) for idx, u_id in enumerate(self.buyer_ids)]

        for buyer in self.model.h_buyers:
            self.model.schedule.add(buyer, overwrite=True)

    def clear_sales(self):
        # take down vacancies that were sold
        self.win_bidders = {}  # flush all from last step
        self.win_bids = {}
        self.sold_ids = []
        self.unsold_ids = []
        discounted = {}
        n_bidders = {}
        for listing, all_bids in self.bids.items():
            if all_bids:  # if there are bids
                bid_values, bidders = list(zip(*all_bids))
                max_bid = max(bid_values)
                max_bidder: Buyer = bidders[bid_values.index(max_bid)]
                self.win_bidders[listing] = max_bidder
                self.win_bids[listing] = max_bid
                self.sold_ids.append(listing)
                # record if the listing was is currently  discounted
                discounted[listing] = max_bidder.choice_discounted
                n_bidders[listing] = len(bidders)
            else:  # no bids for listing
                self.unsold_ids.append(listing)

        # convert sold_ids into DF
        self.sold = self.model.h_sellers.loc[self.sold_ids, :]
        self.sold['is_discounted'] = pd.Series(discounted, dtype=bool, name='is_discounted').astype('category')
        self.sold['n_bidders'] = pd.Series(n_bidders, dtype=int)

    def collect_price_indices(self):
        # self.sold = self.sold.h_value  # get series, but preserve in DF?
        self.sold['was'] = self.sold['h_value'].copy(deep=True)
        self.sold['win_bid'] = pd.Series(self.win_bids)
        self.sold['price_delta'] = self.sold.win_bid - 1.  # get the pct increase
        # differential calculation for price: is_discounted only gets overridden, non-is_discounted gets multiplied by the percentage increase.
        for dmg_category in self.sold['is_discounted'].cat.categories:
            sliced = self.sold[self.sold['is_discounted'] == dmg_category]
            # if the home is damaged or the last value is still under the min bid threshold
            if dmg_category:  # True
                # check for prior discount
                self.sold.loc[sliced.index, 'h_value'] = sliced.win_bid
            else: # False
                # also include items that were discounted but
                sliced2_bool = sliced['was_discounted'] == True  # bool mask
                sliced2 = sliced.loc[sliced2_bool]
                if not sliced2.empty: # if ex-discounted not empty
                    self.sold.loc[sliced2.index, 'h_value'] = sliced2['win_bid']
                    self.sold.loc[sliced2.index, 'was_discounted'] = False  # reset discounted memory
                    show = self.sold.loc[sliced2.index, :]  # for debug
                sliced2 = sliced.loc[~sliced2_bool]
                self.sold.loc[sliced2.index, 'h_value'] = sliced2['win_bid'] * sliced2['h_value']
                # TODO: add update to original!
        pass

class Buyer(agent_base.CustomAgent):
    stagelist_A = ('shortlist_and_pick', 'appraise_house', 'set_bid')
    stagelist_B = ('choose_any_from_all', 'appraise_house', 'set_bid')

    def __init__(self, unique_id: str, model: model_init.RHuCC_Model,
                 housing_market: HousingMarket,
                 shortlist_length=5):
        super(Buyer, self).__init__(model=model,
                                    unique_id=unique_id)

        # self.bid_bound = bid_bounds
        self.housing_market = housing_market
        # self.bid_range = np.linspace(bid_bounds[0],
        #                              bid_bounds[1],
        #                              int(round((bid_bounds[1] - bid_bounds[0]) / bid_bounds[2])) + 1)
        self.price_range = None
        self.shortlist = None
        self.shortlist_length = shortlist_length
        self.choice = None
        self.discount = 0
        self.choice_discounted = False  # todo consider floodplain and flooded separation
        # self.district = house
        # self.choose_price_range()

    # def choose_price_range(self):
    #     # chooses house price bracket to start with and start searching. Only for this step
    #     self.price_range = random.choice(self.model.h_sellers_by_price_counts[
    #                                          self.model.h_sellers_by_price_counts > 0].index)

    # def make_shortlist(self):
    #     self.shortlist =  random.choices(
    #         population=self.model.h_sellers.index,
    #         weights=self.model.h_sellers_weights,
    #         k=self.shortlist_length)
    #
    def shortlist_and_pick(self):
        # chooses the houses with the least number of bidders now
        self.shortlist = random.choices(
            population=self.model.h_sellers.index,
            weights=self.model.h_sellers_weights,
            k=self.shortlist_length)
        bids_n = [len(self.housing_market.bids_interest[option]) for option in self.shortlist]
        self.choice = self.shortlist[bids_n.index(min(bids_n))]
        self.housing_market.bids_interest[self.choice].append(self)

    # def choose_cheapest_from_shortlist(self):
    #     # choose house that with the lowest bid RN
    #     # see current highest bid in the current shortlist
    #     bids_n = [len(self.housing_market.bids[option]) for option in self.shortlist]
    #
    #     # bids_n = [len(self.housing_market.bids)]
    #     # bids_max = [max(self.housing_market.bids[option]) if bids_n[idx] > 0 else None
    #     #           for idx, option in self.shortlist]
    #     # pick option with lowest bid
    #     # NB: currently not very slick method, only valid for incremental method
    #     self.choice = self.shortlist[bids_n.index(min(bids_n))]
        return

    def choose_any_from_all(self):
        self.choice = random.choices(population=self.model.h_sellers.index,
                                     weights=self.model.h_sellers_weights,
                                     k=1)[0]

    # def select_house_to_bid(self):
    #     # appraise house, decide on buy or not (if latter, move down priority list)
    #
    #     # access houses for given price range
    #     # self.choices = self.model.h_sellers_by_price.xs(key=self.price_range, level=1)
    #     pass

    def appraise_house(self):
        # the agent would appraise the value of house based on damage factor * damage level
        # or could this be at the HM level?
        # look at district discount
        if self.model.RUN_CONFIG == "A":
            location = self.model.h_sellers.loc[self.choice, 'district']
            if location in self.model.d_discounts.index:
                self.choice_discounted = True
                self.discount = self.model.d_discounts.discount[location]

        elif self.model.RUN_CONFIG == "B":
            # get agent object
            house: agent_household.HouseholdAgent = self.model.h_sellers.loc[self.choice, 'h_obj']

            if house.fld_dmg_current > 0: # house is damaged, apply discount
                self.choice_discounted = True
                self.discount = max([-1., house.fld_dmg_current * self.model.DAMAGE_DISCOUNT_RATE])
                # fld_dmg_current is 0 <= x <= 1, discount rate is negative float.

        elif self.model.RUN_CONFIG == "C":
            house: agent_household.HouseholdAgent = self.model.h_sellers.loc[self.choice,'h_obj']
            if house.fld_discount < 0.:
                self.choice_discounted = True
                self.discount = max([-1, house.fld_discount])


    def set_bid(self):
        # todo introduce states for damaged/modified state
        bids_list = self.housing_market.bids[self.choice]

        if len(bids_list) > 0:
            bids, _ = zip(*bids_list)
            bid = max(bids) + self.housing_market.BID_INCREMENT / 100
        else:  # if no one has bidded yet
            if self.choice_discounted:
                bid = (1 + self.discount) * self.housing_market.BID_MIN/100

            else:
                bid = self.housing_market.BID_MIN / 100
        # else:  # house IS flooded
        #     bid = 1 + self.discount
            # essentially flat rate, Housing Market should select the first one

        # since prices are externally governed now, bid as usual

        self.model.housing_market.bids[self.choice].append((bid, self))

    def derive_SE_metrics(self):
        # if won the bid, reverse-engineer the SE level

        pass

