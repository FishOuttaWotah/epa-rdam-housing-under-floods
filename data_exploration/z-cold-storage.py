# place to drop things that are not necessary anymore
import numpy as np

# simple outliers rejection
def reject_outliers(data, m=2.):
    # credit to shaneb here https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

# depreciated due to alternative linkage found
def regex_cubic_2018_besteedbaar(decile):
    """
    Excel-derived cubic regression for household disposable income, given an input representing citizen decile.
    :param decile:
    :return: household disposable income
    """
    return 0.1807 * (decile * decile * decile) - 18.6247 * (decile * decile) + 1074.5260 * decile + 7334.1725
