# Exercise 1: concept-mapping table (T0)

Copy the table in the README §3 into your notes and **extend it to function level**: for each row, name the function(s) in nano-vllm, #0 v1, vLLM v1 and SGLang that do the work, with a one-line description in your own words.

Must-have rows (at least):

- admit a waiting request · schedule a decode · choose a preemption victim · free a finished request
- compute a full block's hash · look up a cached prefix · copy-on-write
- build the slot mapping for a prefill · build block tables for a decode · capture a CUDA graph
- apply temperature/top-p · verify draft tokens

For each vLLM/SGLang path, write the commit you read it at. If a path in README §3 has moved at the pin, fix it in a PR to this repo (and mark it verified).

**Check:** the [quiz](../quiz.md) questions 1–4 should be answerable straight from your table.
