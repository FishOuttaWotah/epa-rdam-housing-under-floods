# -*- coding: utf-8 -*-

"""

@author: TabernaA

This script is part of the Climate-economy Regional Agent-Based (CRAB) model
and contains functions for the research and development (R&D) process perfomed
by the Firm agents in the model.

"""

import random
import math
import bisect
import numpy as np

from scipy.stats import bernoulli
from scipy.stats import beta

seed_value = 12345678
random.seed(seed_value)
np.random.seed(seed=seed_value)


def calculateRDBudget(sales, net_worth, rd_fraction=0.04, in_fraction=0.5):
    """Calculate total research and development budget (RD),
       innovation budget (IN) and imitation budget (IM).

    Args:
        sales           : Sales in previous period
        net_worth       : Available firm money (net worth)
        rd_fraction     : Fraction of previous sales used for RD
        in_fraction     : Fraction of budget used for innovation (IN),
                          rest is used for imitation (IM)

    Returns:
        rd_budget   : Total research and development (RD) budget
        in_budget   : Budget used for innovation
        im_budget   : Budget used for imitation

    COMMENT: remove returning of rd_budget (is total of other two)
    """
    if sales > 0:
        rd_budget = rd_fraction * sales
    elif net_worth < 0:
        rd_budget = 0
    else:
        rd_budget = rd_fraction * net_worth

    in_budget = in_fraction * rd_budget
    im_budget = (1 - in_fraction) * rd_budget

    return rd_budget, in_budget, im_budget


# --------
# COMMENT: also implemented in firm itself?? Also see comments there
# --------
def innovate(IN, prod, Z=0.3, a=3, b=3, x_low=-0.15, x_up=0.15):
    """ Innovation process perfomed by Firm agents.

    Args:
        in_budget       : Firms innovation budget
        prod            : Firms productivity
    Params:
        Z               : Budget scaling factor for Bernoulli draw
        a               : Alpha parameter for Beta distribution
        b               : Beta parameter for Beta distribution
        x_low           : Lower bound of support vector
        x_up            : Upper bound of support vector
    """
    in_productivity = [0, 0]

    # Bernoulli draw to determine success (1) or failure (0)
    p = 1 - math.exp(-Z * IN)
    # p1 = 1-math.exp(-Z*IN/2)
    if bernoulli.rvs(p) == 1:
        # new machine productivity (A) from innovation
        a_0 = (1 + x_low + beta.rvs(a, b) * (x_up-x_low))
        in_productivity[0] = prod[0] * a_0

    # # new production productivity (B) from innovation
    # if bernoulli.rvs(p1) == 1:
    #     a1 = (1 + x_low + beta.rvs(a, b) * (x_up-x_low))
    #     in_productivity[1] = prod[1] * a1

    return in_productivity


# --------
# COMMENT: not used, written in firm classes as well; remove one of those
# --------
def imitate(im_budget, firm_ids, agents, prod, region, Z=0.3, e=2):
    """ Imitation process performed by Firm agents.

    Args:
        im_budget       : Firm imitation budget
        firm_ids        : Firm IDs in firms sector
        agents          : Dict of all agents
        prod            : Firms productivity ([A, B] pair)
        region          : Firm region

    Parameters:
        Z               : Budget scaling factor for Bernoulli draw
        e               : Distance scaling factor for firms in other region
    """
    im_productivity = [0, 0]

    # Bernoulli draw to determine success (1) or failure (0)
    p = 1 - math.exp(-Z * im_budget)
    if bernoulli.rvs(p) == 1:
        # store imitation probabilities and the corresponding firms
        imiProb = []
        imiProbID = []

        # Compute inverse Euclidean distance
        for id in firm_ids:
            firm = agents[id]
            distance = (math.sqrt(pow(prod[0] - firm.productivity[0], 2) +
                        pow(prod[0] - firm.productivity[0], 2)))
            if distance == 0:
                imiProb.append(0)
            else:
                # increase distance if the firm is in another region
                if firm.region != region:
                    imiProb.append(1/e*distance)
                else:
                    imiProb.append(1/distance)
            imiProbID.append(firm.unique_id)

        # cumulative probability
        _sum = sum(imiProb)

        if (_sum > 0):
            acc = 0
            for i in range(len(imiProb)):
                acc += imiProb[i] / _sum
                imiProb[i] = acc

            # randomly pick a firm to imitate (index j)
            rnd = random.uniform(0, 1)
            j = bisect.bisect_right(imiProb, rnd)

            # copy that firm's technology
            if j < len(imiProb):
                firm = agents[imiProbID[j]]
                im_productivity[0] = firm.productivity[0]
                im_productivity[1] = firm.productivity[1]

    return im_productivity


# --------
# COMMENT: is same as calculateRDBudget function?
# --------
def calculateRDBudgetCCA(sales, net_worth, v=0.005, e=0.5):
    if sales > 0:
        rd_budget = v * sales
    elif net_worth < 0:
        rd_budget = 0
    else:
        rd_budget = v * net_worth

    in_budget = e * rd_budget
    im_budget = (1 - e) * rd_budget

    return rd_budget, in_budget, im_budget


def innovate_CCA(IN, R, Z=0.3, a=3, b=3, x_low=-0.10, x_up=0.10):
    """
    RD : CCA resilience coefficient
    COMMENT: check how this differs from normal innovate function
    """
    in_R = [0, 0]

    # Bernoulli draw to determine success (1) or failure (0)
    p = 1 - math.exp(-Z * IN / 2)
    # Labor productivity resilience
    if bernoulli.rvs(p) == 1:
        # New resilience coefficient from innovation
        in_R[0] = R[0] * (1 + x_low + beta.rvs(a, b) * (x_up - x_low))

    # Capital stock resilience
    if bernoulli.rvs(p) == 1:
        in_R[1] = R[1] * (1 + x_low + beta.rvs(a, b) * (x_up - x_low))

    return in_R


def imitate_CCA(IM, firm_ids, agents, R, reg, Z=0.3, e=1.5):
    """CCA RD: imitation
    COMMENT: check how this differs from normal imitate function
    """
    im_R = [0, 0]

    # Bernoulli draw to determine success (1) or failure (0)
    p = 1 - math.exp(-Z * IM)
    if bernoulli.rvs(p) == 1:
        # Compute inverse Euclidean distances
        imiProb = []
        imiProbID = []
        for id in firm_ids:
            firm = agents[id]
            distance = (math.sqrt(pow(R[0] - firm.CCA_resilience[0], 2) +
                        pow(R[0] - firm.CCA_resilience[0], 2)))
            if distance == 0:
                imiProb.append(0)
            else:
                # Increase distance if the firm is in another region
                if firm.region != reg:
                    imiProb.append(1/e*distance)
                else:
                    imiProb.append(1/distance)
            imiProbID.append(firm.unique_id)

        # Cumulative probability
        if (sum(imiProb) > 0):
            acc = 0
            for i in range(len(imiProb)):
                acc += imiProb[i] / sum(imiProb)
                imiProb[i] = acc

            # Randomly pick a firm to imitate (index j)
            rnd = random.uniform(0, 1)
            j = bisect.bisect_right(imiProb, rnd)
            # Copy that firm's technology
            if j < len(imiProb):
                firm = agents[imiProbID[j]]
                im_R[0] = firm.CCA_resilience[0]
                im_R[1] = firm.CCA_resilience[1]

    return im_R
