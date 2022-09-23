import sys
import os
import pickle as pkl

def tree_cache_deconv_pipe(func):
    def _decorator(self, *args, **kwargs):
        _A,_B = args[0]
        pipeline_signature = _A.deconv.signature_of_pipeline

        if pipeline_signature is None :
            return func(self, *args, **kwargs)

        cache_path = os.path.join(r"C:\Repos\deconv_py\deconv_py\experiments\pipe_tree_cache" , pipeline_signature + ".pkl")

        if os.path.exists(cache_path) :
            with open(cache_path,"rb") as f :
                res = pkl.load(f)
        else :
            res = func(self, *args, **kwargs)
            with open(cache_path, "wb") as f:
                pkl.dump(res,f)

        res[0].deconv.transfer_all_relevant_properties(_A)
        res[0].deconv.sign_transformer_to_df(str("_".join([f"{k}_{v}" for k,v in self.__dict__.items()] + [str(self.__class__)]).__hash__()))
        return res
    return _decorator