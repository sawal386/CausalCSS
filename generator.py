## This file contains functions for generating data via SCMs
import numpy as np
import pandas as pd

def generate_data_stress_sleep(N, t_name, y_name, m_name, conf_names, coll_names):
    """
    generates data for the query: "Does high stress affect sleep?"

    Args:
        N: (int) total number of data points
        t_name: (str) the name of the treatment variable
        y_name: (str) the name of the outcome variable
        m_name: (str) the name of the mediator variable
        conf_names: (List[str]) the list of confounders
        coll_names: (List[str]) the list of colliders

    Returns:
        (df) a pandas DataFrame whose columns are the model variables
    """
    np.random.seed(111)
    dict_conf = {}
    dict_coll = {}

    T = np.zeros(N)
    Y = np.zeros(N)
    for var in conf_names:
        dict_conf[var] = np.random.normal(0, 2 , N)
        T = T + dict_conf[var]
        Y = Y + dict_conf[var]
    T = T + np.random.normal(0, 2, N)
    T = np.where(T > 1, 1, 0)
    M = 2 * T + np.random.normal(0, 1, N)
    Y = 2 * M + np.random.normal(0, 1, N)

    for var in coll_names:
        dict_coll[var] = Y + 5 * T + np.random.normal(N)

    final_dict = {t_name: T, y_name: Y, m_name: M}
    for name in conf_names:
        final_dict[name] = dict_conf[name]

    for name in coll_names:
        final_dict[name] = dict_coll[name]

    return pd.DataFrame(final_dict)








