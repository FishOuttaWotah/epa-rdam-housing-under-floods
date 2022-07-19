
import mesa

class House():
    def __init__(self):
        return



class HouseholdAgent():
    def __init__(self, household_ID):
        self.id = household_ID
        self.value = None
        self.wage = None
        self.wage_history = None
        self.area = None
        self.savings = None
        self.savings_rate = None
        self.mortgage_debt = None
        self.mortgage_lti = None
        self.mortgage_period = None
        self.mortgage_credit_range = None
        self.damages = None
        self.flood_risk = None


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

def regex_cubic_2018_besteedbaar(decile):
    """
    Excel-derived cubic regression for household disposable income, given an input representing citizen decile.
    :param decile:
    :return: household disposable income
    """
    return 0.1807 * (decile * decile * decile) - 18.6247 * (decile * decile) + 1074.5260 * decile + 7334.1725