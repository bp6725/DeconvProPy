from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.stats import entropy
from functools import reduce
import pandas as pd



class PpSvmSignature(TransformerMixin,BaseEstimator) :
    def __init__(self,n_features = 20,with_norm=True):
        self.n_features=n_features
        self.with_norm = with_norm

    def transform(self, data, *_):
        if data[0].empty:
            return None, None

        _A_from_pipe, _B= data[0], data[1]
        _A = _A_from_pipe.deconv.original_data[0]
        _A = _A.loc[_A_from_pipe.index]

        # filter by intra var
        relvent_cells = None
        if _A.deconv.intra_variance is not None:
            method, intra_var = _A.deconv.intra_variance.popitem()
            trh = _A.deconv.intra_variance_trh[method]

            relvent_cells = {}
            for protein, cell in _A.idxmax(axis=1).to_dict().items():
                _val = intra_var.loc[protein, cell]
                if _val < trh:
                    if cell in relvent_cells.keys():
                        relvent_cells[cell].append(protein)
                    else:
                        relvent_cells[cell] = [protein]

        clf = LinearSVC(max_iter = 100000)
        list_of_genes_list = []

        _samples = _A.copy(deep=True).T
        for cell in set(_A.copy().T.index.map(lambda x: x.split("_0")[0])):
            if relvent_cells is not None :
                if cell not in relvent_cells.keys() :
                    continue
                _samples = _samples[relvent_cells[cell]]

            y = pd.Series(index=_samples.index, data=0)
            y.loc[y.index.map(lambda x: cell in x)] = 1

            model_res = clf.fit(_samples, y)
            list_of_genes_list.append(
                pd.Series(abs(model_res.coef_[0]), index=_samples.columns).nlargest(self.n_features).index)

        genes_list_idx = reduce(lambda x, y: x.union(y), list_of_genes_list)

        filt_A = _A.copy(deep=True).loc[genes_list_idx]
        filt_B = _B.copy(deep=True).loc[genes_list_idx]

        if not self.with_norm:
            return [filt_A, filt_B]
        else:
            norm_filt_A = filt_A.div(filt_A.max(axis=1), axis=0)
            norm_filt_B = filt_B.div(filt_A.max(axis=1), axis=0)
        return [norm_filt_A, norm_filt_B]

    def fit(self, *_):
        return self