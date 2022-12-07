# -*- coding: utf-8 -*-

"""

@author: TabernaA

This script is part of the Climate-economy Regional Agent-Based (CRAB) model
and contains accounting functions for all Firm agents.

"""

from scipy.stats import bernoulli


def individual_demands(size, lifecycle, past_sales, market_shares,
                       total_demand, price, productivity, min_lifecycle=10):
    """TODO: write description.

    Args:
        size                :
        lifecycle           : Firm lifetime
        past_sales          :
        market_shares       :
        total_demand        :
        price               : Firm price
        productivity        :
        min_lifecycle       : COMMENT: not used

    Returns:
        monetary_demand     :
        regional_demand     :
        real_demand         :
        production_made     :
        past_sales          :

    COMMENT: also here, work from firm object where possible
             adjust directly there, or not at all.
             TODO: check function, more calculations than
                   description mentioned.
                   Past_sales --> reassigned in function, confusing?

    """
    # Get demand for this firm in both regions
    regional_demand = [round(total_demand[0] * market_shares[0], 3),
                       round(total_demand[1] * market_shares[1], 3),
                       round(total_demand[3] * market_shares[2], 3)]
    # Calculate monetary and real demand from this regional demand
    monetary_demand = round(sum(regional_demand), 3)
    real_demand = round(monetary_demand / price, 3)

    # Actual production made is constrained by productivity
    production_made = size * productivity

    return (monetary_demand, regional_demand, real_demand,
            production_made, past_sales)


def production_filled_unfilled(production_made, inventories,
                               real_demand, lifecycle):
    """Calculate part of demand that is filled and part that is unfilled.

    Args:
        production_made     :
        inventories         :
        real_demand         :
        lifecycle           : Firm lifetime

    Returns:
        demand_filled       : Part of demand that can be filled
        unfilled_demand     : Part of demand that can not be filled
        inventories         :
    """
    stock_available = production_made + inventories
    demand_filled = min(stock_available, real_demand)

    if lifecycle > 3:
        unfilled_demand = max(0, real_demand - stock_available)
        inventories = max(0, stock_available - real_demand)
    else:
        unfilled_demand = 0
        inventories = 0

    return demand_filled, unfilled_demand, inventories


# --------
# COMMENT: remove? redundant function? Or adjust to something useful
#          TODO: check what this does,
#                it seems like the subsidiary attribute later is not used
# --------
def new_entry(profits, size, subsidiary, threshold):
    """TODO: write description

    Args:
        profits
        size
        subsidiary
        threshold
    """
    if profits > threshold * 2:
        # if bernoulli.rvs(ratio) == 1:
        subsidiary += 1
    else:
        subsidiary = 0
    return subsidiary


def remove_employees(employees, schedule):
    """Remove employees from firm.

    Args:
        employees       : List of employees to remove
        schedule        : Model scheduler (StagedActivationByType object)
        --> change to model

    COMMENT: again work from firm object rather than returning empty list
    COMMENT: use this function everywhere, now also typed out at many places
    """
    for employee_id in employees:
        employee = schedule._agents[employee_id]
        employee.employer_ID = None
    return []


def remove_offers(supplier_ids, schedule, unique_id):
    """Remove firm offers.

    Args:
        supplier_ids    : IDs of firm suppliers
        schedule        : Model scheduler (StagedActivationByType object)
        --> change to model
        unique_id       : Firm ID
    """
    for supplier_id in supplier_ids:
        supplier = schedule._agents[supplier_id]
        supplier.client_IDs.remove(unique_id)
