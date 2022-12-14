import pandas as pd

@pd.api.extensions.register_dataframe_accessor("deconv")
class DecovAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

        self.original_data = []
        self.intra_variance = None
        self.intra_variance_trh = None
        self.is_agg_cells_profile = False
        self.must_contain_genes = None

        self.signature_of_pipeline = None

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        return True

    def sign_transformer_to_df(self,new_sign:str):
        self.signature_of_pipeline = f"{self.signature_of_pipeline}_{new_sign}"

    def keep_original_data(self,original_data):
        self.original_data = original_data

    def set_must_contain_gene(self,must_contain_genes):
        self.must_contain_genes = must_contain_genes

    def transfer_all_relevant_properties(self,source_deconv_df):
        self.update_intra_variance_dict(source_deconv_df.deconv.intra_variance)
        self.set_intra_variance_trh(source_deconv_df.deconv.intra_variance_trh)
        self.is_agg_cells_profile = source_deconv_df.deconv.is_agg_cells_profile
        self.original_data = source_deconv_df.deconv.original_data
        self.must_contain_genes = source_deconv_df.deconv.must_contain_genes

        if self.signature_of_pipeline is None :
            self.signature_of_pipeline = source_deconv_df.deconv.signature_of_pipeline

    def set_agg_cells(self,is_agg = None):
       self.is_agg_cells_profile = is_agg

    @property
    def calc_and_set_intra_var(self,method = "std"):
        full_profile_data = self._obj.copy(deep=True).T
        full_profile_data["cell_for_gb"] = full_profile_data .index.map(lambda x: x.split('_0')[0])

        if method == "std" :
            intra_var = (full_profile_data.groupby("cell_for_gb").std() / (
                    full_profile_data.groupby("cell_for_gb").mean() + 0.001)).T

        if method == "range" :
            intra_var = ((full_profile_data.groupby("cell_for_gb").max() - full_profile_data("cell_for_gb").min()) / (
                    full_profile_data.groupby("cell_for_gb").mean() + 0.001)).T

        if self.how == "number_of_zeros" :
            intra_var = full_profile_data.groupby("cell_for_gb").agg(lambda x: x.eq(0).sum())

        self.update_intra_variance_dict({method : intra_var})

    def update_intra_variance_dict(self,new_intra_var_item):
        if self.intra_variance is None:
            self.intra_variance = new_intra_var_item
        if new_intra_var_item is None :
            return
        else :
            self.intra_variance = {**new_intra_var_item, **self.intra_variance}

    def set_intra_variance_trh(self,params):
        if (params is None) and (self.intra_variance_trh is not None) : return

        self.intra_variance_trh = params
