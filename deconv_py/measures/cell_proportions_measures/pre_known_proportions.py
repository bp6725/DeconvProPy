import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class PreKnownProportions(TransformerMixin,BaseEstimator) :

    def __init__(self,known_prop = None):
        self.known_prop = known_prop
        pass

    def transform(self, data, *_):
        return self.known_prop

    def fit(self, *_):
        return self
