from lm_tournament_eval.api.model import TournamentLM
from hflm import HFLM

from typing import List, Tuple, Union, Optional

from lm_tournament_eval.api.registry import register_model

@register_model("hf-auto", "hf", "huggingface")
class TournamentHFLM(TournamentLM, HFLM):
    def __init__(self,
        model,
        *args,
        **kwargs,                                  
    ):
        TournamentLM.__init__(self)
        HFLM.__init__(self, model, *args, **kwargs)

    def loglikelihood(self, requests: List[Tuple[str,str]], disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
         return HFLM.loglikelihood(self, [req.args for req in requests], disable_tqdm=disable_tqdm)
    
    def loglikelihood_rolling(self, requests : List[str], disable_tqdm : bool = False) -> List[Tuple[float]]:
        return HFLM.loglikelihood_rolling(self, [req.args for req in requests], disable_tqdm=disable_tqdm)

    def generate_until(self, requests : List[Tuple[str, dict]], disable_tqdm : bool = False) -> List[str]:
        return HFLM.generate_until(self, [req.args for req in requests], disable_tqdm=disable_tqdm)
    
    def get_model_info(self) -> dict:
        """
        Method to get Hugging Face model information for experiment reproducibility.
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
        }
        return model_info        