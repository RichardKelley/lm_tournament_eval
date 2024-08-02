import logging
import lm_tournament_eval.models
from lm_tournament_eval.api.registry import get_model

def load_models(model_type, model0, model0_args, model1, model1_args, batch_size: int = 1,
                max_batch_size: int = 1, device: str = "cuda:0"):

    # load model0 to device
    if isinstance(model0, str):
        if model0_args is None:
            logging.warning("model0_args not specified. Using defaults")
            model0_args = ""

        if isinstance(model0_args, dict):
            model0_args.update({"model": model0})
            logging.info(
                f"Initializing {model0} model, with arguments: {model0_args}."
            )

            lm0 = get_model(model_type).create_from_arg_obj(
                model0_args,
                {
                    "batch_size" : batch_size,
                    "max_batch_size" : max_batch_size,
                    "device":  device,
                },
            )
            
        else:
            model0_args += f"model={model0}"
            logging.info(
                f"Initializing {model0} model, with arguments: {simple_parse_args_string(model0_args)}"
            )
            lm0 = get_model(model_type).create_from_arg_string(
                model0_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                }
            )
    else:
        if not isinstance(model0, lm_tournament_eval.api.model.LM):
            raise TypeError
        logging.info("Using pre-initialized model")
        lm0 = model0

    # now do model1
    if isinstance(model1, str):
        if model1_args is None:
            logging.warning("model1_args not specified. Using defaults")
            model1_args = ""

        if isinstance(model1_args, dict):
            model1_args.update({"model": model1})
            logging.info(
                f"Initializing {model1} model, with arguments: {model1_args}."
            )

            lm1 = get_model(model_type).create_from_arg_obj(
                model1_args,
                {
                    "batch_size" : batch_size,
                    "max_batch_size" : max_batch_size,
                    "device":  device,
                },
            )
            
        else:
            model1_args += f"model={model1}"
            logging.info(
                f"Initializing {model1} model, with arguments: {simple_parse_args_string(model1_args)}"
            )
            lm1 = get_model(model_type).create_from_arg_string(
                model1_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                }
            )
    else:
        if not isinstance(model1, lm_tournament_eval.api.model.LM):
            raise TypeError
        logging.info(f"Using pre-initialized model for model 1.")
        lm1 = model1

    return lm0, lm1    