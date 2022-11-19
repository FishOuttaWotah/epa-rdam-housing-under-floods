from __future__ import annotations
from typing import TYPE_CHECKING, Union, Sequence

import mesa

if TYPE_CHECKING:
    from model_init import RHuCC_Model
from agent_base import CustomAgent
from enum import Enum


# ------------------------------------------------------------------
## House (inanimate) object, as an item of
# -------------------------------------------------------------------
# class House:
#     def __init__(self,
#                  location: str,
#                  house_value: float,
#                  household: HouseholdAgent = None):
#         self.LOCATION: str = location  # ID linking to ledger of wijken or ENUM
#         self.current_value = house_value
#         self.value_history = [house_value]
#         self.flood_exp = ()
#         self.flood_dmg = ()
#         self.current_occupant = household  # occupant household, check if necessary
#

# ---------------------------------------------------------------------
# Mesa Agent, but not encapsulated within the Agent class. Would be standalone for transparency (ie. not hiding attributes in a superclass.
# ----------------------------------------------------------------------
class HouseholdAgent(CustomAgent):
    # households need a unique id. Could that id be shared with the big array? would be useful to have a common id for mesa, np-array (if necessary) and dict (if necessary)

    # class SearchMode(Enum):
    #     DORMANT = 1
    #     BUYING = 2
    #     SELLING = 3

    def __init__(self,
                 unique_id: str,
                 model: RHuCC_Model,
                 SE_percentile: float):
        super().__init__(unique_id=unique_id, model=model)

        ## environment attributes
        # location is logged under the owned_house variable!
        # need to think about syncing between the Ledger and here?
        self.SE_percentile = SE_percentile
        self.location = None

        # personal attributes
        self.income_disposable = None  # current iteration: no income growth, but kept as variable
        self.income_gross = None
        self.income_d_ratio = None
        self.income_g_ratio = None
        # self.mortgage_debt = None  # dummy, not included
        # self.current_damages: Union[int, None] = None  # current amount of home damages
        # self.house = None  # owned House object
        self.value = None
        self.buy_price = None  # buy price at the time
        self.flood_exp = (-1, -1)  # Nb: assumption
        self.flood_dmg = None
        self.fld_dmg_current: float = 0
        self.fld_discount = 0
        self.fld_discount_rate = 0
        self.fld_discount_elapsed = 0

        # items to record over history (even though datacollector should collect these...?)
        # self.value_history = []
        # self.wage_history = []
        # self.SE_history = []

        # self.flood_awareness: bool = False  # someone might be interested in setting this as a continuum...

        # Dummy attributes not used in current model iteration
        # self.savings = None
        # self.SAVINGS_RATE = None
        # self.current_sentiment = None  # modified by external code??

        ## Mortgage (expansion point)
        # self.mortgage_period = None
        # self.mortgage_timestep = None  # current progress on mortgage repayments
        # self.mortgage_lti = None
        # self.mortgage_credit_range = None

        # migration
        self.wants_to_move = False  # expansion point


