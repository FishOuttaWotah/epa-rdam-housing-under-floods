# -*- coding: utf-8 -*-

"""

@author: TabernaA

This script is part of the Climate-economy Regional Agent-Based (CRAB) model
and contains functions for the dynamics of the goods market.

"""

# --------
# COMMENT: Check necessity of these functions,
#          most of them can maybe be removed
# --------


# --------
# COMMENT: not used
# --------
def calc_prod_cost(wage, prod):
    """Calculate unit cost of production.

    Args:
        wage    : Firm wage
        prod    : Firm labor productivity
    """
    return wage/prod


# --------
# COMMENT: not used
# --------
def calc_price(cost, markup=0.05):
    """Calculate unit price.

    Args:
        cost        : Unit cost of production
        markup      : Difference between cost and selling price
    """
    return round((1 + markup) * cost, 6)


def calc_competitiveness(price, r, trade_cost,
                         trade_cost_exp, unfilled_demand):
    """Calculate firm competitiveness in both regions (Coastal and Inland).

    Args:
        price               : Firm unit price
        r                   : Firm region
        trade_cost          : Regional transport cost
        trade_cost_exp      : Export transport cost
        unfilled_demand     :

    """
    if r == 0:
        return [round(-price - unfilled_demand, 8),
                round(-price * (1 + trade_cost) - unfilled_demand, 8),
                round(-price * (1 + trade_cost_exp) - unfilled_demand, 8)]
    elif r == 1:
        return [round(-price * (1 + trade_cost) - unfilled_demand, 8),
                round(-price - unfilled_demand, 8),
                round(-price * (1 + trade_cost_exp + trade_cost) -
                      unfilled_demand, 8)]


def compete_and_sell(productivity, wage, market_share_history,
                     price, markup, v=0.05):
    """Calculates firm cost, markup and price.

    Args:
        productivity            : Firm productivity
        wage                    : Firm wage
        market_share_history    : History of firm market share
        price                   : Firm price
        markup                  : Firm markup
        v                       :

    Returns:
        cost                    : New firm cost
        markup                  : New firm markup
        price                   : New firm price

    COMMENT: Also here: change arguments to firm object, adjust from here.
             So do not return, but firm.cost = ... etc.
    """

    # Cost calculation; avoid division by zero
    if productivity[1] > 0:
        cost = wage / productivity[1]
    else:
        cost = wage

    # Markup calculation
    if len(market_share_history) < 10:
        # Keep markup fixed for the first 10 timesteps (for smooth start)
        markup = 0.125
    else:
        # Calculate markup from market share history,
        # bounded between 0.05 and 0.4
        markup = round(markup * (1 + v * ((market_share_history[-1] -
                                           market_share_history[-2]) /
                                          market_share_history[-2])), 5)
        markup = max(0.01, min(0.4, markup))

    # Adjust price based on new cost and markup, bounded
    # between 0.7 and 1.3 times the old price to avoid large oscillations
    price = max(0.7 * price, min(1.3 * price, round((1 + markup) * cost, 8)))

    return cost, markup, price


def calc_market_share_cons(model, lifecycle, MS_prev, comp,
                           comp_avg, K, K_total, chi=0.5):
    """Calculates market share of consumption good firms.

    Args:
        model           : CRAB_model object
        lifecycle       : Firm lifetime
        MS_prev         : Firm market share from the previous time step
        comp            : Firm competitiveness
        comp_avg        : Average sector competitiveness
        K               : Capital stock
        K_total         : Total capital amount in economy
        chi             : Scaling factor for level of competitiveness
    """

    # Initial market shares
    if (lifecycle == 0):
        market_share = [max(K/K_total[0], 1e-4),
                        max(K/K_total[0], 1e-4),
                        max(K/K_total[0], 1e-4)]
    else:
        # TODO: comment; rewrite to vector operations
        ms0 = MS_prev[0] * (1 + chi*(comp[0] - comp_avg[0]) / comp_avg[0])
        ms1 = MS_prev[1] * (1 + chi*(comp[1] - comp_avg[1]) / comp_avg[1])
        ms2 = MS_prev[2] * (1 + chi*(comp[2] - comp_avg[2]) / comp_avg[2])
        market_share = [round(ms0, 8), round(ms1, 8), round(ms2, 8)]

    return market_share


# --------
# COMMENT: not used
# --------
def calc_global_market_share(MS):
    """Calculate firm global market share.

    Args:
        MS      : Market share
    """
    return (MS[0] + MS[1]) / 2


# --------
# COMMENT: not used
# --------
def remove_myself(self):
    """Remove firm from market. """
    self.model.kill_agents.append(self)

    # Fire employees
    households = self.model.schedule.agents_by_type["Household"]
    for employee_id in self.employees_IDs:
        employee = households[employee_id]
        employee.employer_ID = None
    self.employees_IDs = []

    # Remove offers
    for offer in self.offers:
        supplier_agent = self.model.schedule.agents_by_type["Cap"][offer[2]]
        supplier_agent.client_IDs.remove(self.unique_id)
    self.offers = []
