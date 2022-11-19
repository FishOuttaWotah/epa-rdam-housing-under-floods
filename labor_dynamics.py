# -*- coding: utf-8 -*-

"""

@author: TabernaA

This script is part of the Climate-economy Regional Agent-Based (CRAB) model
and contains functions for labor dynamics of Household and Firm agents.

"""

import numpy as np

from random import seed, sample
from math import ceil

seed_value = 12345678
seed(seed_value)


def labor_search(unique_id, suitable_employers):
    """ Labor search performed by Household agents.
        Every unemployed household searches through available suitable
        employers in all sectors and picks the firm
        offering the highest wage.

    Args:
        unique_id               : Household ID
        suitable_employers      : List of firms with open vacancies
    Returns:
        employer_ID             : ID of new employer
    """
    # Check if there are firms with open vacancies
    if suitable_employers:
        # Choose from subset of firm (bounded rationality)
        potential_employers = sample(suitable_employers,
                                     ceil(len(suitable_employers)/3))
        # Choose firm with highest wage
        wages = [firm.wage for firm in potential_employers]
        employer = potential_employers[np.argmax(wages)]
        employer.employees_IDs.append(unique_id)
        # print(employer.employees_IDs)

        # Check if the employer has enough workers, if so:
        # remove it from the list of firms with vacancies
        if employer.desired_employees == len(employer.employees_IDs):
            employer.open_vacancies = False
            suitable_employers.remove(employer)

        return employer.unique_id

    else:
        return None


def labor_demand(capital_vintage, feasible_production):
    """Labor demand determined by Firm agents.

    Args:
        capital_vintage         : Machine stock
        feasible_production     : Affordable production given capital stock

    Returns:
        labor_demand            : Labor needed to satisfy production
        average_productivity    : Average productivity of firm machines
    """

    # Find the most productive machines for the quantity I want to produce
    # Fewest machines needed to satisfy feasible_production
    Q = 0
    machines_used = []
    # Loop through machine stock backwards, starting with most productive one
    for vintage in capital_vintage[::-1]:
        # Stop when desired amount is reached
        if Q < feasible_production:
            machines_used.append(vintage)
            Q += vintage.amount
            vintage.amount = int(vintage.amount)

    # Weighted average productivity of chosen machines
    average_productivity = round(sum([v.amount * v.productivity for v in
                                      machines_used]) /
                                 sum([a.amount for a in machines_used]), 3)

    # Compute labor needed to satisfy feasible production
    labor_demand = max(0, ceil(feasible_production / average_productivity))
    return labor_demand, average_productivity


def hire_and_fire(labor_demand, employees_IDs, open_vacancies, model,
                  profits, wage):
    """Hire and fire employees.

    Args:
        labor_demand        : Number of desired employees
        employees_IDs       : List of IDs of my employees
        open_vacancies      : Boolean; True if looking for workers,
                              False if not looking for workers
        model               :
        profits             :
        wage                :

    Returns:
        labor_demand        : Number of desired employees
        employees_IDs       : List of IDs of my employees
        open_vacancies      : Boolean; True if looking for workers,
                              False if not looking for workers
    """
    # Number of desired employees
    desired_employees = round(labor_demand)

    # Open vacancies or fire employees, depending on demand and
    # current number of employees
    if desired_employees == len(employees_IDs):
        open_vacancies = False
    elif desired_employees > len(employees_IDs):
        open_vacancies = True
    elif desired_employees < len(employees_IDs):
        open_vacancies = False
        # Fire employees if profits are too low
        if profits < wage:
            firing_employees = abs(desired_employees - len(employees_IDs))
            for i in range(firing_employees):
                j = employees_IDs[0]
                employee = model.schedule.agents_by_type["Household"][j]
                employee.employer_ID = None
                del employees_IDs[0]

    return desired_employees, employees_IDs, open_vacancies


def wage_determination(r, gov, productivity, regional_av_prod,
                       wage, price, lifecycle):
    """Determine firm wage.

    Args:
        r                   : Firm region
        gov                 : Government agent
        productivity        : Firm productivity
        regional_av_prod    : Average production within region
        wage                : Current firm wage
        price               :
        lifecycle           :

    Returns:
        wage                : New firm wage

    COMMENT: TODO: remove unused arguments
    """
    # Get all necessary information from government
    minimum_wage = gov.min_wage[r]

    # # Delta price not used for now (not influential in wage determination)
    # con_price_avg = data["Consumption_price_average"]
    # current_average_price = con_price_avg[int(self.model.schedule.time)][r]
    # prev_average_price = con_price_avg[int(self.model.schedule.time)-1][r]
    # delta_price_average = ((current_average_price - prev_average_price) /
    #                        previous_average_price)

    # Keep change in productivity
    previous_productivity = productivity[0]
    current_productivity = productivity[1]
    # max_wage = price * current_productivity

    delta_my_productivity = max(-0.25, min(0.25, (current_productivity -
                                                  previous_productivity) /
                                                 previous_productivity))

    # # Delta unemployment, not used for now
    # if (previous_unemployment_rate_my_region or
    #     current_unemployment_rate_my_region) < 0.01:
    #     delta_unemployment = 0
    # else:
    # delta_unemployment = max(-0.025,
    #                          min(0.025,
    #                              (current_unemployment_rate_my_region -
    #                               previous_unemployment_rate_my_region) /
    #                              max(previous_unemployment_rate_my_region,
    #                                  current_unemployment_rate_my_region)))

    delta_unemployment = 0
    # unemploy_vars = data["Unemployment_Regional"]
    # delta_unemployment = unemploy_vars[int(self.model.schedule.time)][r+2]

    # Regional productivity change, calculated by government
    delta_productivity_average = regional_av_prod[r+2]
    b = 0.1

    # max_wage = previous_productivity * self.price
    # -- wage is  upper bounded by the governmental minimum wage -- #
    # TODO: check correctness of function, 1 + b not between brackets?
    #       and why (-0.0)???
    wage = max(minimum_wage,
               round(wage * (1 + b * delta_my_productivity +
                     (1 - b) * delta_productivity_average +
                     (-0.0) * delta_unemployment + 0.0), 3))

    # wage = min(wage, max_wage)
    return wage


def update_employees_wage(employees_list, schedule, wage):
    for i in range(len(employees_list)):
        j = employees_list[i]
        employee = schedule[j]
        employee.wage = wage
