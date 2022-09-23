import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals.six import StringIO
from IPython.display import Image,display

from sklearn.tree import export_graphviz
import pydotplus
from graphviz import Source
from sklearn.tree import DecisionTreeClassifier


class HyperParameterMeasures():
    def __init__(self):
        pass

    @staticmethod
    def plot_hyperparameter_tree(results_df,measure_columns = "corrMean",measure_trh = 0.6,feature_columns = "all"):
        print(f"there are {sum(results_df[measure_columns].isna())} None {measure_columns}")

        results_df = results_df.dropna(subset=[measure_columns])
        results_df = results_df.fillna("NONE")

        if feature_columns == "all" :
            X_set = results_df[results_df.columns.difference(pd.Index([measure_columns]))]
        else :
            X_set = results_df[feature_columns]

        Y_set = (results_df[measure_columns] > measure_trh).astype(int)
        dummi_result_df = pd.get_dummies(X_set)

        dtree = DecisionTreeClassifier(class_weight="balanced")
        dtree.fit(dummi_result_df, Y_set)

        dot_data = StringIO()
        export_graphviz(dtree, out_file=dot_data,
                        filled=True, rounded=True, feature_names=dummi_result_df.columns,
                        special_characters=True)
        # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # Image(graph.create_png())
        s = Source(dot_data.getvalue(), format="png")
        s.view()

        plt.show()






