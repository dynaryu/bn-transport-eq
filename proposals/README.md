# Reviewed additions

Drafts here are excluded from the active instance loader. Ask can propose a
parameter-set file for an existing function, or a saved scenario query. It must
identify its source, assumptions and purpose. Source values are supplied or
transcribed for review; the app does not download or invent model coefficients.

The app shows the JSON and validation errors before activation. A named reviewer
activates the exact displayed version; the server revalidates it. Parameter sets
are added under `instances/<role>/`; named scenario variants under
`examples/<network>/*.scenario.json`. Activation never overwrites an existing file.
New mathematical forms still require a kernel, schema declaration and tests.

Example dialogues:

- “Evaluate a 1,000-year hazard.” → Ask for the hazard study or ground-motion
  product; never infer a magnitude from return period alone. Draft after its
  inputs, units and applicability have been supplied.
- “Save a policy that closes moderate damage too.” → Propose a functionality
  instance, with the explicit usable states and the user's stated policy as source.
- “Save access to N1 only, allowing 30 extra minutes.” → Ask for the origin;
  propose a scenario with destination flags and the time allowance. Validate
  against the existing singleton SystemEvent; do not append a second row.
- “Which bridge should we inspect?” → Ask for actions, loss table, loss units,
  inspection accuracy and costs. Use VOI once supplied; this is a decision query,
  not a new fragility instance.
