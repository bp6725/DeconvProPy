from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin
import pandas as pd
import scipy
from deconv_py.infras.cellMix.cellMix_coordinator import CellMixCoordinator

class BasicDeconv(BaseEstimator, ClassifierMixin):
    def __init__(self, normalize = True,cellMix = False):
        self.normalize = normalize
        self.cellMix =cellMix
        self.cmc = CellMixCoordinator()

        self.result = None

    def fit(self, X=None, y=None):
        """
        cell proportions using scipy.optimize.nnls
        :param cell_specific: [number_of_proteins,number_of_cells]
        :param mass_spec: [number_of_proteins,number_of_samples]
        :return: cell_abundance_over_samples:[number_of_proteins,number_of_samples]
        """

        cell_specific, mass_spec = X[0],X[1]

        if (cell_specific is None ) or (mass_spec is None) :
            return None

        cell_abundance_over_samples = []

        if not self.cellMix :
            for sample in mass_spec:
                cell_abundance = scipy.optimize.nnls(cell_specific, mass_spec[sample])[0]
                cell_abundance_df = pd.DataFrame(data=cell_abundance, index=cell_specific.columns, columns=[sample])
                cell_abundance_over_samples.append(cell_abundance_df)

            cell_abundance_over_samples_df = pd.concat(cell_abundance_over_samples, axis=1)
        else :
            try :
                cell_abundance_over_samples_df = self.cmc.cell_prop_with_bash(mass_spec, cell_specific).rename({"Unnamed: 0": "cells"},
                                                                                             axis=1).set_index("cells")
            except  Exception as e:
                print("cant use cellmix")
                print(e)
                return None

        if not self.normalize :
            return  cell_abundance_over_samples_df

        cell_abundance_over_samples_df = cell_abundance_over_samples_df / cell_abundance_over_samples_df.sum(
            axis=0)
        cell_abundance_over_samples_df = cell_abundance_over_samples_df.round(2)
        return cell_abundance_over_samples_df

    def predict(self, X):
        return self.fit(X)

    def fit_predict(self,X=None,y=None):
        return self.predict("None")

    def transform(self,data):
        return data