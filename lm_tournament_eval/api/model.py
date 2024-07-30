import abc

from typing import Dict, List, Optional, Tuple, Type, TypeVar

from lm_tournament_eval import utils

T = TypeVar("T", bound="LM")

import logging

class LM(abc.ABC):

    def __init__(self) -> None:
        self._rank = 0 
        self._world_size = 1

    @abc.abstractmethod
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        pass

    @abc.abstractmethod
    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        pass

    @abc.abstractmethod
    def generate_until(self, requests) -> List[str]:
        pass

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k : v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    @classmethod
    def create_from_arg_obj(
        cls: Type[T], arg_dict: dict, additional_config: Optional[dict] = None
    ) -> T:
        additional_config = {} if additional_config is None else additional_config
        additional_config = {
            k: v for k, v in additional_config.items() if v is not None
        }
        return cls(**arg_dict, **additional_config)