import numpy as np
import numpy.random
def cubic_disposable_wage_func(x):
    """
    3rd-order polynomial curve fitted with CBS disposable income data. Curve fitted in MS Excel for transparency
    :param x: input x value for fitting, 1<= x <= 100
    :return: associated disposable wage
    """
    return 0.1807*x*x*x - 18.6247*x*x + 1074.5260*x + 7334.1725

def generate_agent_wages(n_agents, wage_dist_func):
    """
    Generate agents with distribution of wages based off the cubic disposable wage function. Agents are first generated via uniform distribution and then applied to the wage function.
    :param n_agents: number of agents to generate
    :param wage_dist_func: function used to generate wages from uniform distribution
    :return:
    """
    # set up random generator
    gen = np.random.default_rng()

    # create uniform distribution
    xs = gen.random(size=n_agents) * 100

    # vectorize and return disposable wage
    return np.vectorize(wage_dist_func)(xs)