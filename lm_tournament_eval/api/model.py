from lm_tournament_eval import utils
from typing import Optional, Type, TypeVar

T = TypeVar("T", bound="TournamentLM")

class TournamentLM:
    def __init__(self) -> None:
        self._rank = 0
        self._world_size = 1

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
