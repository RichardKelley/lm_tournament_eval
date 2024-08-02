# Accuracy Vs Elo score 

Comparing quantized models with their non-quantized versions on hellaswag shows a minimal drop in accuracy between 4bit and full precision models. (at least with phi-3 as a start)

| Model      | Accuracy |
| --------   | -------  |
| phi-3      | 0.6056562437761402 |
| phi-3-4bit | 0.5966938856801434 |

This difference becomes more apparent in the ELO score. 

After running 20 matches of 100 samples per round, we see:

| Model      | ELO      |
| --------   | -------  |
| phi-3      | 1242.05  |
| phi-3-4bit | 1157.95  |


## Other results

Below are some other models' hellaswag results (in no particular order), using `lm_eval`:

| Model      | Accuracy |
| --------   | -------  |
| phi-3-mini      | 0.6056562437761402 |
| phi-3-mini-4bit | 0.5966938856801434 |
| mistral-7b-instruct | 0.6479784903405696|
| mistral-7b-instruct-4bit | 0.6391157140011949 |
| phi-3-medium    | 0.6495717984465246 |
| llama3-8b       | 0.6013742282413862 |
| llama3-70b-4bit | 0.6618203545110536 | 





