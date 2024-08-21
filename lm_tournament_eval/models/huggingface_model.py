from hflm import HFLM

from typing import List, Tuple

from lm_tournament_eval.api.registry import register_model

@register_model("hf-auto", "hf", "huggingface")
class TournamentHFLM(HFLM):
    def __init__(self,
        model,
        *args,
        **kwargs,                                  
    ):
        super().__init__(model, *args, **kwargs)

    def loglikelihood(self, requests: List[Tuple[str,str]], disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        return super().loglikelihood([req.args for req in requests], disable_tqdm=disable_tqdm)
    
    def loglikelihood_rolling(self, requests : List[str], disable_tqdm : bool = False) -> List[Tuple[float]]:
        return super().loglikelihood_rolling([req.args for req in requests], disable_tqdm=disable_tqdm)

    def generate_until(self, requests : List[Tuple[str, dict]], disable_tqdm : bool = False) -> List[str]:
        return super().generate_until([req.args for req in requests], disable_tqdm=disable_tqdm)