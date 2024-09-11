import logging
import lm_tournament_eval.models
from lm_tournament_eval.api.registry import get_model
from lm_tournament_eval.utils import simple_parse_args_string

from typing import Tuple

# as of 22 August 2024
_ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20240620",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-5-sonnet@20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-opus@20240229",
    "claude-3-sonnet@20240229",
    "claude-3-haiku@20240307",

    # legacy models
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2",
]

_OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-instruct"
]

def parse_model_name(model_name) -> Tuple[str, str]:
    """
    Given a model name that may be either a Hugging Face model path or a specification of the
    form <api-name>:<model-name>, return a pair consisting of the 'model_type' and the 'model'
    that are expected by the load_model function.
    """
    if ':' in model_name:
        idx = model_name.find(":")
        model_type = model_name[:idx]
        model = model_name[(idx+1):]

        model_type = model_type.lower()
        assert(model_type in ['anthropic', 
                              'anthropic-chat', 
                              'anthropic-chat-completions',
                              'openai-completions',
                              'local-completions',
                              'openai-chat-completions',
                              'local-chat-completions'])

        match model_type:
            case "anthropic" | "anthropic-chat" | "anthropic-chat-completions":
                assert(model in _ANTHROPIC_MODELS)
                return (model_type, model)
            case "openai-completions" | "local-completions" | "openai-chat-completions" | "local-chat-completions":
                assert(model in _OPENAI_MODELS)
                return (model_type, model)

    else:
        # assume we're in the hf case.
        return ("hf", model_name)

def load_model(model_type, model, model_args, batch_size: int = 1, device: str = "cuda:0"):

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

            match model_type:
                case "hf":
                    lm = get_model(model_type).create_from_arg_obj(
                        model_args,
                        {
                            "batch_size" : batch_size,
                            "device":  device,
                        },
                    )
                case "anthropic" | "anthropic-chat" | "anthropic-chat-completions":
                    lm = get_model(model_type).create_from_arg_obj(
                        model_args,
                        { }
                    )
                case "openai-completions" | "local-completions" | "openai-chat-completions" | "local-chat-completions":
                    lm = get_model(model_type).create_from_arg_obj(
                        model_args,
                        { }
                    )

            
        else:
            if model_args == "":
                model_args += f"model={model}"
            else:
                model_args += f",model={model}"
            logging.info(
                f"Initializing {model} model, with arguments: {simple_parse_args_string(model_args)}"
            )

            match model_type:
                case "hf":
                    lm = get_model(model_type).create_from_arg_string(
                        model_args,
                        {
                            "batch_size": batch_size,
                            "device": device,
                        }
                    )
                case "anthropic" | "anthropic-chat" | "anthropic-chat-completions":
                    lm = get_model(model_type).create_from_arg_string(
                        model_args,
                        { }
                    )
                case "openai-completions" | "local-completions" | "openai-chat-completions" | "local-chat-completions":
                    lm = get_model(model_type).create_from_arg_string(
                        model_args,
                        { }
                    )

    else:
        if not isinstance(model, lm_tournament_eval.api.model.LM):
            raise TypeError
        logging.info("Using pre-initialized model")
        lm = model

    return lm