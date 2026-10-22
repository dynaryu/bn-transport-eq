# Offline Ask checks

```sh
python -m pytest src/llm/tests src/app/tests
node --test src/app/tests/test_ask.cjs
```

Pytest blocks real model transports: every model response must come from an
injected fake client. Local application endpoints and the inference engine can
still be exercised. Do not use `evaluate --live` for these checks.

For a browser check, prepare a local fixture with `src.app.server.example("toy-ca/demo")`
and `src.inference.engine.answer` for a small toy query, saved as JSON with keys
`example` and `result`. Neither requires a model. Then run:

```sh
node src/app/tests/ask_browser_smoke.cjs "<Chrome or Edge executable>" .cache/ask-gui-fixture.json
```

The browser script serves the page locally and intercepts every fetch, including
Ask, with deterministic fixtures. Page requests outside localhost are blocked.
It checks query tables, review decisions, clarification follow-ups, analyses,
proposals, activation without losing conversation history, interrupted streams,
exports and mobile layout. Screenshots are saved under `.cache/ask-gui-*.png`.
