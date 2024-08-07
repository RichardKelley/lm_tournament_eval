# Tournament Evaluation of Language Models


## Installation

For a local installation, clone the repository and run

```
pip install -e .
```

## Running the evaluator

You can run the following script:

```
lm-tournament-eval <options>
```

You can use the `limit` command line option to test if your setup works:

```
lm_tournament_eval --tasks hellaswag --include_path /home/richard/tmp/lm-evaluation-harness/lm_eval/tasks/ --model0 microsoft/Phi-3-mini-4k-instruct --limit 100 --model1 microsoft/phi-2
```

You can remove the `limit` option to run matches on the entire data set. 