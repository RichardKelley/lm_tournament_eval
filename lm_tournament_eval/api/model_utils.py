import logging
import lm_tournament_eval.models
from lm_tournament_eval.api.registry import get_model
from lm_tournament_eval.utils import simple_parse_args_string

def load_model(model_type, model, model_args, batch_size: int = 1,
               max_batch_size: int = 1, device: str = "cuda:0"):

    # load model0 to device
    if isinstance(model, str):
        if model_args is None:
            logging.warning("model0_args not specified. Using defaults")
            model_args = ""

        if isinstance(model_args, dict):
            model_args.update({"model": model})
            logging.info(
                f"Initializing {model} model, with arguments: {model_args}."
            )

            lm = get_model(model_type).create_from_arg_obj(
                model_args,
                {
                    "batch_size" : batch_size,
                    "max_batch_size" : max_batch_size,
                    "device":  device,
                },
            )
        else:
            if model_args == "":
                model_args += f"model={model}"
            else:
                model_args += f",model={model}"
            logging.info(
                f"Initializing {model} model, with arguments: {simple_parse_args_string(model_args)}"
            )
            lm = get_model(model_type).create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                }
            )
    else:
        if not isinstance(model, lm_tournament_eval.api.model.LM):
            raise TypeError
        logging.info("Using pre-initialized model")
        lm = model

    return lm