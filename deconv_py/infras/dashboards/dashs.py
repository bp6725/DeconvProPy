import sys
import os
sys.path.append('../infras/dashboards/')
# import exploration_plots


def cells_profile_dash(A,B,X):
    import exploration_plots
    exploration_plots.cells_profiles(A,X,B)

    exploration_plots.values_histogram_per_cell_profile(A)

def mixture_based_dash(A,B,X):
    import exploration_plots

    expcted_b = A.dot(X)
    exploration_plots.corr_expcted_vs_mixture(expcted_b,B)
    exploration_plots.corr_expcted_vs_mixture(expcted_b, B, method="spearman")
    exploration_plots.unexpcted_zeros_per_mixture(expcted_b,B)
    exploration_plots.scatter_mixture_vs_expcted_mixture(expcted_b,B)
    exploration_plots.expected_values_of_unexpected_zeros(expcted_b,B)




