# -*- coding: utf-8 -*-

"""

@author: TabernaA

Class for Vintage objects. This non-agent class represents the machines
produces by CapitalGoodFirms.

"""
import time

import random
import numpy as np

from scipy.stats import bernoulli, beta
from random import choice, seed

seed_value = 12345678
seed(seed_value)
np.random.seed(seed=seed_value)


class Vintage():
    """Class representing a Vintage object. """

    def __init__(self, prod, amount=1):
        """Initialization of a Vintage object.

        Args:
            prod    : Productivity of the machine
            amount  : Number of machines (a Vintage object can represent
                      more than one machine).
        """
        self.productivity = prod
        self.amount = amount
        self.age = 0
        self.lifetime = 15 + random.randint(1, 10)


# --------
# COMMENT: 1. Change "self" to "firm" to avoid confusion??
#          2. Check way these functions are used, not part of class?
#              --> why not put in Firm files?
#          3. Check if variables like quantity_ordered and scrapping_machines
#             have to be stored in firm object.
# --------
def update_capital(self):
    """Update capital by consumption and service firms. """
    if self.damage_coeff > 0:
        for vintage in self.capital_vintage:
            # If Bernoulli is successful: vintage is destroyed
            if bernoulli.rvs(self.damage_coeff) == 1:
                self.capital_vintage.remove(vintage)
                vintage.amount == 0
                # # COMMENT:
                # del vintage

    # Handle orders if they are placed
    if self.supplier_id is not None and self.quantity_ordered > 0:
        # Add new vintage with amount ordered and supplier productivity
        supplier = self.model.schedule.agents_by_type["Cap"][self.supplier_id]
        # COMMENT: remove this check?
        if not supplier.productivity[0] == round(supplier.productivity[0], 3):
            print("FALSE")
        new_machine = Vintage(prod=round(supplier.productivity[0], 3),
                              amount=round(self.quantity_ordered))
        self.capital_vintage.append(new_machine)

        # Replace according to the replacement investment
        while self.scrapping_machines > 0:
            vintage = self.capital_vintage[0]
            if self.scrapping_machines < vintage.amount:
                vintage.amount -= round(self.scrapping_machines)
                self.scrapping_machines = 0
            else:
                self.scrapping_machines -= vintage.amount
                self.capital_vintage.remove(vintage)
                # # COMMENT:
                # del vintage

    # Remove the machines that are too old
    for vintage in self.capital_vintage:
        # COMMENT: again; why increase here, not at end of timestep?
        vintage.age += 1
        if vintage.age > vintage.lifetime:
            self.capital_vintage.remove(vintage)
            # # COMMENT:
            # del vintage
    self.investment_cost = 0


def capital_investments(time, capital_stock, capital_output_ratio,
                        past_demands, real_demand, inventories,
                        unfilled_demand):
    """Capital investment, choose supplier, make order
       COMMENT: TODO: check if this function can be made more efficient.

    Args:
        time                    :
        capital_stock           :
        capital_output_ratio    :
        past_demands            :
        real_demand             :
        inventories             :
        unfilled_demand         :

    Returns:
        past_demands            :
        inventories             :
        unfilled_demand         :
        desired_production      :
        feasible_production     :
        expansion               :
    """
    # Take mean of last three demands to form adaptive demand expectations
    past_demands.append(real_demand)
    if len(past_demands) > 2:
        past_demands = past_demands[-3:]
    expected_production = np.ceil(np.mean(past_demands))

    # # Handle flood
    # # COMMENT: Remove?, destruction already in machines (function above)?
    # if self.flooded == True:
    #     # # Draw individual shock from beta distribution
    #     shock = beta.rvs(self.model.beta_a,self.model.beta_b)
    #     # # (Optional): direct destruction of production
    #     # self.production_made = self.production_made * (1 - shock)
    #     # # Destruction of inventories
    #     self.inventories = (1 - shock) * self.inventories

    # Set desired level of inventories (10 percent of production)
    fraction_inventories = 0.1
    desired_level_inventories = fraction_inventories * expected_production

    # Used to let the model start smoothly
    # to be removed with model calibration
    if time < 10:
        inventories = desired_level_inventories
        # COMMENT: take this out of function? Variable not used here
        unfilled_demand = 0

    # Compute how many units firm can produce vs how many it wants to produce
    desired_production = max(0, expected_production +
                             desired_level_inventories - inventories)
    feasible_production = np.ceil(min(desired_production,
                                      (capital_stock / capital_output_ratio)))

    # If capital stock is too low to produce desired amount:
    # Expand firm by buying more capital
    if (feasible_production < desired_production):
        expansion = (np.ceil(desired_production - feasible_production) *
                     capital_output_ratio)
    else:
        expansion = 0  # constrain productivity with the capital output ratio
    return (past_demands, inventories, unfilled_demand,
            desired_production, feasible_production, expansion)


def calc_replacement_investment(self):
    """TODO: write description.

    Returns:
        replacement_investment  :
    """
    if self.offers:
        ratios = [prod/price for prod, price, u_id in self.offers]
        # Pick machine with the best product/price ratio from offers
        best_i = ratios.index(max(ratios))
        new_machine = self.offers[ratios.index(max(ratios))]
    else:
        # If there are no offers: pick random capital good firms as supplier
        supplier_id = random.choice(self.model.ids_firms1)
        supplier = self.model.schedule._agents[supplier_id]
        supplier.client_IDs.append(self.unique_id)
        if supplier.region == self.region:
            new_machine = supplier.brochure_regional
        else:
            new_machine = supplier.brochure_export

    replacement_investment = 0
    for vintage in self.capital_vintage:
        # Unit cost advantage of new machines (UCA)

        # COMMENT: remove use of self.wage here --> does not matter for ratio
        UCA = self.wage / vintage.productivity - self.wage / new_machine[0]
        # Payback rule
        # Don't consider if productivity is equal, prevent division by zero
        if ((UCA > 0 and (new_machine[1]/UCA <= 3)) or
            (vintage.age == vintage.lifetime - 1 and
             vintage.amount > (self.desired_production -
                               self.feasible_production) *
                              self.capital_output_ratio)):
            replacement_investment += vintage.amount
        # # Consider if they also replace capital because it is old
        # # and gonna be destroyed soon
        # if ((UCA > 0 and (new_machine[1]/UCA <= 3)) or
        #     (vintage.age == vintage.lifetime - 1 and
        #      vintage.amount > (self.desired_production -
        #                        self.feasible_production) *
        #                       self.capital_output_ratio))
        #     or vintage.age >= vintage.lifetime - 1:
        #     replacement_investment += vintage.amount
    return replacement_investment


def choose_supplier_and_place_order(self):
    """Choose supplier and place the order of machines. """
    self.investment_cost = 0
    self.quantity_ordered = 0

    # Choose based on highest productivity / price ratio
    ratios = [prod/price for prod, price, u_id in self.offers]
    # --------
    # COMMENT: partially does the same as function above?
    # --------
    if ratios:
        # Get supplier ID and price from brochure with best prod/price ratio
        self.supplier_id = self.offers[ratios.index(max(ratios))][2]
        supplier_price = self.offers[ratios.index(max(ratios))][1]
        supplier = self.model.schedule._agents[self.supplier_id]
    else:
        # If there are no offers: pick random capital good firms as supplier
        self.supplier_id = random.choice(self.model.ids_firms1)
        supplier = self.model.schedule._agents[self.supplier_id]
        supplier_price = supplier.price
        supplier.client_IDs.append(self.unique_id)
        if supplier.region == self.region:
            self.offers.append(supplier.brochure_regional)
        else:
            self.offers.append(supplier.brochure_export)

    # Calculate how many machines can be bought based on desired amount
    # and affordable amount.
    total_number_machines_wanted = self.expansion + self.replacements
    total_quantity_affordable_own = max(0, self.net_worth // supplier_price)
    quantity_bought = min(total_number_machines_wanted,
                          total_quantity_affordable_own)

    # If more machines desired than can be bought, make debt to invest
    if quantity_bought < total_number_machines_wanted and self.net_worth > 0:
        # Compute affordable debt and adjust number of machines bought
        debt_affordable = self.sales * self.model.debt_sales_ratio
        maximum_debt_quantity = debt_affordable // supplier_price
        quantity_bought = min(total_number_machines_wanted,
                              total_quantity_affordable_own +
                              maximum_debt_quantity)

        # Set debt based on bought machines
        self.debt = (quantity_bought -
                     total_quantity_affordable_own) * supplier_price
        if self.debt >= debt_affordable:
            self.credit_rationed = True

    self.quantity_ordered = np.ceil(quantity_bought)
    # Machines that will be replaced (expansion investments have priority)
    self.scrapping_machines = max(0, quantity_bought - self.expansion)

    # # Remove offers (next period will arrive updated ones)
    # COMMENT: remove?
    # self.offers = []

    # Add order to suppliers list
    if self.quantity_ordered > 0:
        # Convert quantity into cost
        self.investment_cost = self.quantity_ordered * supplier_price
        if supplier.region == self.region:
            supplier.regional_orders.append([self.quantity_ordered,
                                             self.unique_id])
        else:
            supplier.export_orders.append([self.quantity_ordered,
                                           self.unique_id])
    elif self.quantity_ordered == 0:
        self.supplier_id = None
    else:
        print("ERROR: Quantity ordered is negative", self.quantity_ordered)
