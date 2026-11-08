# Offline app checks

```sh
python -m pytest src/llm/tests src/app/tests
node src/app/tests/test_example_switching.cjs
```

Pytest blocks real model transports: every model response must come from an
injected fake client. Local application endpoints, the session and the inference
engine (all on the clique-tree sampler, src/inference/cliques.py) can still be
exercised. The sampler imports rsr.mcmc, so put the rsr working tree on
PYTHONPATH (see .env). Do not use `evaluate --live` for these checks.
