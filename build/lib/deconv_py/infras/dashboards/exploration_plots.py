import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def cells_profiles(A,cell_proportions_df = None,B = None):
    n_var = len([1 for i in locals().values() if i is not None])

    res= (A.T/A.mean(axis=1)).T
    res[A==0]=0

    d = sch.distance.pdist(res)
    L = sch.linkage(res, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    indexes = [res.index.tolist()[i] for i in list((np.argsort(ind)))]
    df = res.reindex_axis(indexes, axis=0)

    args = plt.subplots(1, n_var)
    args[0].set_figheight(10)
    args[0].set_figwidth(10)

    if n_var >1 :
        sns.heatmap(df,ax=args[1][0])
    else :
        sns.heatmap(df,ax=args[1])
    if cell_proportions_df is not None :
        sns.heatmap(cell_proportions_df,ax=args[1][1])

    if B is not None:
        B_ordered = B.copy(deep=True).T/B.mean(axis=1)
        B_ordered[B.T==0] = 0
        B_ordered = B_ordered.T.reindex_axis(indexes)
        sns.heatmap(B_ordered,ax=args[1][2])
    plt.subplots_adjust(wspace = 2)

def values_histogram_per_cell_profile(A):
    fig = plt.figure(figsize = (15,20))
    ax = fig.gca()
    np.log2(A+1).hist(bins=100,ax=ax)

def corr_expcted_vs_mixture(expcted_b,B,method  = "pearson"):
    corr_per_mixture_df = pd.DataFrame(index= B.columns)

    for i in np.linspace(0.05,0.95,19) :
        trh = round(i,2)

        for mixture_idx in B:
            mixture =np.log2(B[mixture_idx])
            mixture = B[mixture_idx]
            mixture[B[mixture_idx] == 0] = 0
            mixture_trh = mixture.quantile(trh)
            relevant_proteins = mixture[mixture>mixture_trh].index

            _expcted_b = np.log2(expcted_b[mixture_idx])
            _expcted_b = expcted_b[mixture_idx]
            _expcted_b[expcted_b[mixture_idx] ==0] = 0
            corr = _expcted_b.loc[relevant_proteins].corr(mixture.loc[relevant_proteins],method = method)
            corr_per_mixture_df.ix[mixture_idx,trh] = corr
    corr_per_mixture_df.T.plot(title=f"correlation mixture vs expcted_mixture - {method}")
    plt.show()

def unexpcted_zeros_per_mixture(expcted_b,B) :
    number_of_unexpcted_zeros = ((expcted_b != 0)&(B==0)).sum(axis=1)
    display((number_of_unexpcted_zeros >0).sum())
    display(number_of_unexpcted_zeros.hist())
    plt.title("number of mixtures where the gene is zero (but not in the profile)")
    plt.show()

def scatter_mixture_vs_expcted_mixture(expcted_b,B):
    for i,mixture_idx in enumerate(B):
        mixture =np.log2(1+B[mixture_idx])
        expcted_mixture = np.log2(expcted_b[mixture_idx])

        sns.regplot(mixture,expcted_mixture)
        plt.title(mixture_idx)
        plt.xlabel( "mixture log2")
        plt.ylabel ("expected mixture log2")
        plt.show()

def expected_values_of_unexpected_zeros(expcted_b,B):
    for mixture_idx in B:
        zeros = np.log2(1+expcted_b[mixture_idx][((expcted_b[mixture_idx] != 0)&(B[mixture_idx]==0))])
        all_data = np.log2(1+B[mixture_idx][B[mixture_idx] != 0])

        plt.hist(zeros, 100, alpha=0.5, label="expected values of unexpected zeros")
        plt.hist(all_data, 100, alpha=0.5, label="expected values")
        plt.title(mixture_idx)
        plt.xlabel("log2 values")
        plt.legend(loc = "upper right")
        plt.show()