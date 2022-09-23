import numpy as np
import pandas as pd
from scipy.optimize import nnls

class Base():
    @staticmethod
    def nn_opt_func(a, b):
        if isinstance(a, pd.DataFrame):
            a = a.values
        if isinstance(b, pd.DataFrame):
            b = b.values
        results = np.zeros_like(b)
        for idx, protein in enumerate(b.T):
            results[:, idx] = nnls(a, protein)[0]
        return results
