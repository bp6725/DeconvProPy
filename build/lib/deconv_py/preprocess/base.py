from abc import abstractmethod

class BasePreprocess() :
    def __init__(self):
        pass

    @abstractmethod
    def return_mutual_proteins(A,B):
        corresponds_proteins = B.index[~B.index.isna()].intersection(A.index[~A.index.isna()])

        _B = B.loc[corresponds_proteins]
        _A = A.loc[corresponds_proteins]

        return _A,_B

    @abstractmethod
    def return_mutual_proteins_by_index(A,B,index = 'Majority protein IDs',index_func = lambda x:x):

        A['major_protein_id'] = A[index].apply(index_func)
        A_with_index = A.set_index('major_protein_id', drop=True, append=False)

        B['major_protein_id'] = B[index].apply(index_func)
        B_with_index = B.set_index('major_protein_id', drop=True, append=False)

        return BasePreprocess.return_mutual_proteins(A_with_index,B_with_index)

    @abstractmethod
    def get_major_protein_to_gene_name_dict(mixture_data):
        return mixture_data[['major_protein_id', 'Gene names']].set_index('major_protein_id').to_dict()['Gene names']

if __name__ == '__main__':
    pass