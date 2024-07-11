from .model import LM

import torch
import torch.nn.functional as F

import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer

import copy

from .lm_utils import (
    get_rolling_token_windows,
    make_disjoint_window,
    Collator,
    pad_and_concat,
    stop_sequences_criteria
)

from typing import List, Tuple, Union, Optional

# from lm_eval/models/utils.py
def get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, str) and dtype != "auto":
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype

class HFLM(LM):

    AUTO_MODEL_CLASS = None # this is set in _get_backend
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(self, 
        model: Union[str, PreTrainedModel],
        tokenizer: Optional[
            Union[
                str, PreTrainedTokenizer
            ]
        ] = None,
        truncation : Optional[bool] = False,
        device: Optional[str] = "cuda",
        device_map_option : Optional[str] = "auto",
        dtype: Optional[Union[str, torch.dtype]] = "auto", # NB: MLX incompatibility
        add_bos_token: Optional[bool] = False, # Need for Gemma-2
        max_length: Optional[int] = None,
        prefix_token_id: Optional[int] = None,
        batch_size: Optional[Union[int,str]] = 1,
        max_batch_size: Optional[int] = 64,
        **kwargs,                 
    ) -> None:
        super().__init__()

        if not isinstance(model, str):
            self._model = model
            self._device = model.device

            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                model_name = model.name_or_path
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name
                )
        else:
            assert isinstance(device, str)
            assert isinstance(model, str)

            # most of the rest of this branch in lm-evaluation-harness is for setting 
            # up accelerate. TODO later.

        self._device = device
        self._max_length = max_length
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = prefix_token_id
        self.batch_size_per_gpu = batch_size
        self.truncation = truncation

        # get backend
        self._get_backend()

        # create tokenizer
        self._create_tokenizer(
            model,
            tokenizer
        )

        # create model
        if isinstance(model, str):
            self._create_model(model, dtype=dtype, device_map_option=device_map_option, **kwargs)

    def _get_backend(
        self,
        backend = "default",
        trust_remote_code = False
    ) -> None:
        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            if backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
        else:
            self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM # TEMPORARY

        assert self.AUTO_MODEL_CLASS in [
            transformers.AutoModelForCausalLM,
            transformers.AutoModelForSeq2SeqLM,
        ]
        return None

    def _create_tokenizer(
            self,
            model : Union[str, PreTrainedModel],
            tokenizer: Optional[
                Union[
                    str,
                    PreTrainedTokenizer
                ]
            ]
    ) -> None:
        if tokenizer is not None:
            if isinstance(tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer
                )
            else:
                assert isinstance(tokenizer, PreTrainedTokenizer)
                self.tokenizer = tokenizer
        else:
            if isinstance(model, str):
                model_name = model
            else:
                model_name = model.name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return None

    def _create_model(
        self,
        model: str,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        device_map_option: str = "auto",
        **kwargs
    ) -> None:
        model_kwargs = kwargs if kwargs else {}

        if "device_map" not in kwargs:
            model_kwargs.update({"device_map" : device_map_option})

        self._model = self.AUTO_MODEL_CLASS.from_pretrained(
            model,
            torch_dtype=get_dtype(dtype),
            **model_kwargs
        )

        return None

    def to(self, new_device) -> None:
        assert new_device in ["cpu", "cuda"] # How will this work with multiple GPUs?
        if not hasattr(self._model.config, "quantization_config"): # has to be a no-op for quantized models - to(...) fails.
            match new_device:
                case "cpu":
                        self._model = self._model.to(new_device)
                        torch.cuda.empty_cache()
                case "cuda":
                    self._model = self._model.to(new_device)

    @property
    def model(self):
        return self._model
    
    @property
    def max_length(self):
        if self._max_length is not None:
            return self._max_length
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH
    
    @property
    def prefix_token_id(self):
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id
    
    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        special_tokens_kwargs = {}

        if add_special_tokens is None:
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                add_special_tokens = {
                    "add_special_tokens": False or self.add_bos_token
                }
        else:
            special_tokens_kwargs = {"add_special_tokens" : add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        if left_truncate_len is not None:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())

        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        model_class = getattr(self, "AUTO_MODEL_CLASS", None)

        if model_class == transformers.AutoModelForSeq2SeqLM:
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
        else:
            whole_enc = self.tok_encode(context + continuation)
            context_enc = self.tok_encode(context)

            context_enc_len = len(context_enc)
            continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Tuple[str,str]]) -> List[Tuple[float, bool]]:
        '''
        requests is a list of (context, continuation) pairs
        '''
        new_reqs = []
        for context, continuation in requests:
            if context == '':
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation)
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _model_call(self, inps, attn_mask=None, labels=None):
        with torch.no_grad():
            assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
            return self.model(inps).logits
        
    def _select_cont_toks(self, logits: torch.Tensor, contlen:int = None, inplen: int = None):
        assert (contlen and inplen)
        logits = logits[inplen - contlen : inplen]

        return logits

    def _loglikelihood_tokens(self, requests) -> List[float]:
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts",
            group_fn=_lookup_one_token_cont,
        )

        batch_size = self.batch_size
        chunks = re_ord.get_batched(n=batch_size, batch_fn=None)

        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None

            for _, context_enc, continuation_enc in chunk:
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # we assume we're in the causal case
                inp = torch.tensor(
                    (context_enc+continuation_enc)[-(self.max_length + 1):][:-1],
                    dtype=torch.long,
                    device=self.device
                )
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )
                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

                call_kwargs = {}

                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )

                multi_logits = F.log_softmax(
                    self._model_call(batched_inps, **call_kwargs), dim=-1
                )

                for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                    chunk, multi_logits, inplens, cont_toks_list
                ):
                    contlen = len(cont_toks)
                    ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                    logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                    logits = logits.unsqueeze(0) 

                    greedy_tokens = logits.argmax(dim=-1)

                    for request_str, cont_toks, logits in re_ord.get_cache(
                        req_str=request_str,
                        cxt_toks=ctx_tokens,
                        cont_toks=cont_toks,
                        logits=logits,
                    ):
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long, device=self.device).unsqueeze(0)
                        max_equal = (greedy_tokens == cont_toks).all()
                        
                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)

                        answer = (float(logits.sum()), bool(max_equal))
                        res.append(answer)

        return re_ord.get_original(res)
            
    def loglikelihood_rolling(self, requests : List[str]) -> List[Tuple[float]]:
        '''
        We will assume that `requests` has type List[str] for this implementation
        '''
        loglikelihoods = []

        for req in requests:
            string = req
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows
            )

            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods
        
    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def max_gen_toks(self) -> int:
        return 256
    
    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len is not None:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )

        return self.model.generate(
            input_ids = context,
            max_length = max_length,
            stopping_criteria = stopping_criteria,
            pad_token_id = self.tokenizer.pad_token_id,
            use_cache = True,
            **generation_kwargs
        )

    def generate_until(self, requests) -> List[str]:
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        batch_size = self.batch_size

        re_ords = Collator(
            requests,
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )

        chunks = re_ords.get_batched(n=batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            until = None

            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)

            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                max_ctx_len = self.max_length - max_gen_toks
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                max_ctx_len = self.max_length

            context_enc, attn_masks = self.tok_batch_encode(
                contexts, left_truncate_len=max_ctx_len, truncation=self.truncation
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    cont_toks = cont_toks[context_enc.shape[1] : ]

                s = self.tok_decode(cont_toks)

                for term in until:
                    if len(term) > 0:
                        s = s.split(term)[0]

                res.append(s)
        res = re_ords.get_original(res)
        return res