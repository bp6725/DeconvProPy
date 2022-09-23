from functools import reduce


class CellSpecificMetricsPlot():
    def __init__(self):
        pass

    @staticmethod
    def std_over_permuts(cell_specific_per_permutation_dfs,cell_name):
        eva_cell_specific = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True),
                                   map(lambda x: x.T, cell_specific_per_permutation_dfs))

        return eva_cell_specific[[col for col in eva_cell_specific.columns if cell_name in col]].std(axis=1) / \
                              eva_cell_specific[[col for col in eva_cell_specific.columns if cell_name in col]].\
                                  mean(axis=1)

    @staticmethod
    def mixture_correlation_to_results(cell_specific_per_permutation_dfs,permuts,X,real_A):
        real_results = []
        eva_results = []

        index = 0
        for cell_spec, permut in zip(cell_specific_per_permutation_dfs, permuts):
            relvent_proportions = X[X.columns[list(permuts[1])]]
            real_results.append(real_A.dot(relvent_proportions).rename(columns={0: index}))
            eva_results.append(cell_spec.T.dot(relvent_proportions).rename(columns={0: index}))
            index += 1

        _real = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True), real_results)
        _eva = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True), eva_results)

        return _real.corrwith(_eva,axis=1).hist(bins=30)

