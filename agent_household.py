from typing import TYPE_CHECKING, Union
import mesa

# ------------------------------------------------------------------
## House (inanimate) object, as an item of
# -------------------------------------------------------------------
class House:
    def __init__(self, location: str, house_value:int):
        self.LOCATION: str = location #  ID linking to ledger of wijken
        self.HOUSE_VALUE_BASE: int = house_value
        self.house_flood_hazard: list[float] = []
        self.house_flood_history: list[float] = [] # contains flood depth/flood damage proportion?
        self.house_flood_damages: list[float] = []
        self.current_occupant: str = None # occupant household, check if necessary



# ---------------------------------------------------------------------
# Mesa Agent, but not encapsulated within the Agent class. Would be standalone for transparency (ie. not hiding attributes in a superclass.
# ----------------------------------------------------------------------
class HouseholdAgent:
    # households need a unique id. Could that id be shared with the big array? would be useful to have a common id for mesa, np-array (if necessary) and dict (if necessary)
    def __init__(self, household_ID, model):

        ## standard mesa.Agent attributes
        self.unique_id = household_ID
        self.model = model

        ## environment attributes
        # location is logged under the owned_house variable!
        self.AGENT_LEDGER = None
        # need to think about syncing between the Ledger and here?

       # personal attributes
        self.wage = None  #  current iteration: no income growth, but kept as variable
        self.mortgage_debt = None # would this include the interest rates??
        self.current_damages: Union[int, None] = None # current amount of home damages
        self.owned_house = None # owned House object
        self.OWNED_HOUSE_BUY_PRICE = None # buy price at the time
        self.flood_awareness: bool = False  # someone might be interested in setting this as a continuum...

        # Dummy attributes not used in current model iteration
        self.wage_history = None
        self.savings = None
        self.SAVINGS_RATE = None
        self.mortgage_period = None
        self.mortgage_timestep = None  # current progress on mortgage repayments
        # self.mortgage_lti = None
        # self.mortgage_credit_range = None

    # ---------------------------------------------------------------------
    ## Step functions for the agent, if activated to be buyer/seller
    # ---------------------------------------------------------------------
    def step_buyer(self):
        # some form of check to ensure that the household doesn't look to buy if already owns a house, only retrieve from non-occupants
        # looks at housing market (set up a market)
        # filters based on the prices (or already pre-filtered)
        # shortlist filtered stuff
        # decide bid based on makelaars' (set up makelaars' analysis) and flood_awareness

        #
        pass

    def step_seller(self):
        # check needed to ensure that household has a house to sell
        pass

    # ---------------------------------------------------------------------
    ## Smaller sub-functions
    # ---------------------------------------------------------------------
    def assign_house(self):
        # assigns house id and accompanying attributes. Used as
        pass


    def update_ledger(self):
        # update the ledger on wage and other accounts?
        return

    def search_houses(self):
        # make a skim of the housing market
        # if found a good selection, make a bid
        return

    def bid_house(self):
        # bid on chosen house
        return

    def offer_house(self):
        # puts house on sale
        # probably with a surcharge
        # some conditionals if a short sale could be done
        return

def generate_socioeconomic_level():
    return

