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

def generate_flood_distribution_in_bins(input_arr: np.ndarray,
                                        bins,
                                        outlier_ceil: float = None,
                                        exclude_outliers: bool=True):
    """
    Fits the flood distribution array into pre-defined bins. Bins must be monotonically increasing. Two classes of outliers exist: acceptable outliers and unaccepted outliers.

     Accepted outliers can be included/excluded in the binning process, but will be passed on to the user (debugging, corrections). Unaccepted outliers are defined by the outlier_ceil parameter, and will be eliminated from consideration
    :param input_arr: input array of flood distributions
    :param bins: bins to be sorted into, must be monotonic
    :param outlier_ceil: Ceiling where outliers beyond will be eliminated
    :param exclude_outliers: if acceptable outliers should be excluded from binning
    :return: 3-tuple for binned counts, accepted outliers and unaccepted outliers
    """
    # truncate means to eliminate the overflow points
    # overflow means flood points that are higher than required (could be something weird though)
    eliminated = [] # generate empty array
    if outlier_ceil is not None:
        # modifies in place
        mask = input_arr < outlier_ceil
        eliminated = input_arr[~mask] # tilde = inverse of mask
        input_arr = input_arr[mask]

    # Warning to catch if manually defined ceiling is lower than bin maximum
    print(f'Warning: defined ceil ({outlier_ceil}) higher than bin maximum({bins.max()})') if outlier_ceil < bins.max() else None

    # get population within bins
    in_bins = input_arr[input_arr<bins.max()]
    overflow = list(set(in_bins) ^ set(input_arr)) # outliers preserved for debug or correction
    if exclude_outliers:
        input_stuff = in_bins # not include overflow stuff
    else: # will preserve, and the outliers are captured in the max bin size (document!)
        input_stuff = input_arr

    return np.histogram(input_stuff, bins)[0], overflow, eliminated

