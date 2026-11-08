# Five short quakenet walkthroughs — recording draft

Aim: show one useful action per clip. Target **1 minute 45 seconds total**, with different lengths for each subject. Timings are editing targets, to be adjusted after a recording rehearsal.

Use **EMA (`ema-highway/gmm`)** throughout: the Eastern Massachusetts highway example, with 74 nodes, 129 roads/bridges and two airport destinations, n22 and n66. Keep one continuous story: an earthquake scenario, an inspection decision, a station observation, and updated priorities. Frame selected branches of the larger diagrams so the labels remain readable.

Format: screen recording with concise captions; no narration for the first cut. Start directly in the app, without an opening title card. **Fast-forward computation waits**, keeping control selections and result-reading holds at normal speed. Keep actual computation times visible in Roadmap. Export five separate MP4 files, with a still thumbnail for each.

### Fast-forward treatment

- Record each actual computation from submission through completion. Preserve its genuine inputs, outputs and timestamps.
- Compress each waiting interval to roughly 1–2 seconds. Overlay **“Inference wait fast-forwarded”** for that interval; if displaying a speed factor, calculate it from the recorded interval rather than guessing it.
- Show the click that starts the work and the completed state at normal speed. The result must remain still long enough to read.
- Apply this to **Run**, task calculation, evidence updates and the custom-network rerun. Actual recording may take minutes; the edited clips keep their short target lengths.
- If a long wait is visually static, an explicitly labelled time cut is an acceptable fallback. Do not speed up dropdown selection, numerical results or caption reading.

## 1. Problem and model — 15 seconds

**Viewer takeaway:** the physical network and its probabilistic model describe the same system.

| Time | Screen/action | Caption |
|---|---|---|
| 0–5 s | EMA map, fitted to the full network. Point to airport destinations n22 and n66. | “After an earthquake, which bridge inspections would help restore access to airports?” |
| 5–8 s | Briefly indicate the roads and bridges. | “Eastern Massachusetts: 129 roads, each represented by a bridge.” |
| 8–15 s | Click **BN**. Frame bridge-e0073’s shaking → damage → road state chain and its connection to access. | “The Bayesian network links shaking, damage, road availability and access.” |

Small persistent label: **“Illustrative earthquake scenario”**. EMA includes population weights: use **people**, rather than a count of towns, when explaining decision scores. The supplied network includes assumed bridge locations and road speeds. Access means reaching a destination within the allowed additional travel time (30 minutes in the default scenario), rather than merely having a connected path.

## 2. Clique tree and inference — 20 seconds

**Viewer takeaway:** the clique tree organises inference, and message details expose the methods used.

| Time | Screen/action | Caption |
|---|---|---|
| 0–5 s | Click **Clique tree**; keep **Inference** checked. Select bridge-e0073’s observation/bridge cliques using the available selectors; frame that branch and the root. | “The clique tree organises the inference computation.” |
| 5–10 s | Click an upward message from the observation branch. Hold its details. | “An observation contributes a likelihood message over the shared variable.” |
| 10–15 s | Close the popup, then click the root’s heading. | “System evidence: certificate MCMC.” |
| 15–20 s | Close the popup, then open the shaking update’s downward-message details. | “Shaking updates: elliptical slice sampling. Click messages to inspect their methods.” |

Recording notes: the diagram explains the computation; its arrows are not a live execution trace. Certificate MCMC concerns conditioned sampling with system evidence. Without evidence the sampler can draw directly; ordinary session updates can reweight existing samples. Do not describe every update as MCMC, or the whole inference as exact. Use the implemented method names rather than “latest techniques”.

## 3. Decision task and evidence — 25 seconds

**Viewer takeaway:** select a decision task, then add an observation that updates its priorities.

| Time | Screen/action | Caption |
|---|---|---|
| 0–5 s | Map tab, EMA selected, fresh session. Click **Run**, fast-forward its wait, then show the ready state. | “Start an EMA analysis.” |
| 5–10 s | Click **Prioritise bridge inspection** in the left panel. Hold the active task. | “Choose the task: inspect bridges and repair them if damaged.” |
| 10–18 s | In **Dropdowns**, choose **Ground-motion reading**, asset **bridge-e0073**, reading **0.05**, source **demo station**. | “Example observation: a station at bridge-e0073 reads 0.05 g.” |
| 18–25 s | Click **Add observation**; fast-forward the wait, then hold the completed map and updated result list. | “The evidence base and inspection priorities update together.” |

Use the dropdown workflow for a predictable, short first cut. The **Agent chat (beta)** button remains visible as an alternative. A chat demonstration can be a separate later clip once the configured model has been rehearsed.

Fast-forward the task calculation at 5–10 seconds as well. The station reading is an illustrative input for this walkthrough, not a report of an actual measurement. Rehearse it from a fresh EMA session before recording; do not use the working app's existing evidence history. Do not promise a particular winning bridge or score: record the numbers actually computed. The task values the expected number of people whose access is restored under the inspection-and-repair assumption; it does not optimise crew routes, inspection costs or repair budgets.

## 4. Read the results — 20 seconds

**Viewer takeaway:** Roadmap shows the decision’s history; BN shows how an individual variable changed.

| Time | Screen/action | Caption |
|---|---|---|
| 0–7 s | Click **Roadmap**. Frame the newest evidence entry and its top inspection candidates. | “Newest analysis first. Grey: previous round. Blue: current round.” |
| 7–10 s | Briefly point to the milestone’s computation time. | “Each milestone records its computation time.” |
| 10–20 s | Click **BN**, locate **SA(1.0)::bridge-e0073**, then click it. Hold the overlapping distributions and legend. | “Compare the shaking distribution before and after the station observation.” |

Continue from clip 3’s completed station update. That supplies a genuine previous round for the BN overlay. Click **SA(1.0)::bridge-e0073**, the latent shaking variable, so the viewer sees how the measurement updates uncertainty; the observation variable itself is fixed to the entered value. Frame the top inspection candidates rather than scrolling through all 129 options. The recorded computation time remains the actual duration despite the video’s fast-forwarded waits.

## 5. Customise the system — 25 seconds

**Viewer takeaway:** network rows are editable inputs; validate and start a fresh analysis after changing them.

| Time | Screen/action | Caption |
|---|---|---|
| 0–6 s | Editor: open `examples/ema-highway/network.json`. Briefly show **Node**, **Edge** and **Bridge** table names. | “Describe your system in network.json: nodes, roads and bridges.” |
| 6–12 s | Frame edge **e0073**. Change only its `avg_speed_kph` from **60** to **40**, then save. | “For example, change a road’s assumed travel speed.” |
| 12–17 s | Terminal in the activated project environment: `python -m src check`. Hold the successful output. | “Validate the inputs.” |
| 17–25 s | Cut to the restarted app; reload EMA and click **New session**. Fast-forward its wait, then hold the completed analysis. | “Restart the server and begin a new session to analyse the changed network.” |

Use an isolated recording copy for this edit. Update its `assumptions.avg_speed_kph` description to identify the e0073 override before validation; the visible shot can stay focused on the numeric edit. The running server caches compiled views, so a browser refresh alone is insufficient. Never present an old saved session as the new network’s results. Hold the real rerun output without promising that this single speed change will alter the leading inspection candidate.

Small end caption: **“network.json = system data · assembly.json = model choices”**. Reuse the existing assembly for this simple input change. A different model family or a completely new example needs matching assembly configuration as well.

## Recording setup and continuity

- Use a separate local recording copy and its own session storage, preserving the working app’s saved sessions. Choose **ema-highway** explicitly; example selection restores its saved memory.
- Start with 100,000 samples, the existing EMA model settings and a fresh session. Rehearse runtime and readability before capture; retain actual output rather than recreating scores in an overlay. Keep the full sample count and fast-forward the waits instead of lowering the sample count just for speed.
- Record clips 1 and 2 with the diagrams already fitted. Record clip 3 from a fresh session, then clip 4 from its completed observation. Record clip 5 separately in the editable recording copy.
- Capture at 1920 × 1080 if the display allows. Frame the useful area tightly; enlarge text before recording rather than showing a whole desktop. Keep the cursor still during reading holds.
- Use a single caption at a time, away from the controls and numerical results. Keep the app’s tabs visible during transitions. No music or decorative transitions in the first cut.
- Before exporting, check that the cursor, selected values, message details and previous/current legends can be read at the blog player’s size. Keep screenshots and captions free of unrelated desktop content.

## Planned GitHub Pages presentation

| Button | Video file | Thumbnail |
|---|---|---|
| **1 · Problem & model** | `01-problem-model.mp4` | EMA map |
| **2 · Clique tree** | `02-clique-tree.mp4` | Tree with inference messages |
| **3 · Task & evidence** | `03-task-evidence.mp4` | Decision and observation panels |
| **4 · Read results** | `04-read-results.mp4` | Previous/current distribution overlay |
| **5 · Customise a system** | `05-custom-system.mp4` | The relevant network.json row |

Each button opens the matching video in a popup with playback controls, a close button, Escape-to-close and a short text description. Load a video when opened, and pause it when closed. Final durations appear beside buttons after editing. Hosting and page implementation follow after the clips are reviewed.

## Source checks for this draft

- App labels, selectable task, observation options, newest-first Roadmap and BN comparisons: `src/app/quakenet.html`.
- Clique structure and implemented message/sampling methods: `src/inference/cliques.py`.
- Task calculation and sequential posterior updates: `src/inference/session.py`.
- EMA inventory (74 nodes, 129 edges, 129 bridges), destinations, population fields, bridge-e0073 and edge e0073 at 60 km/h: `examples/ema-highway/network.json`.
- EMA model bindings: `examples/ema-highway/gmm.assembly.json`.
- Validation command: `src/__main__.py`; server view caching: `src/app/server.py`.

## First recording trial

An **18-second EMA introduction** has been recorded and exported locally:

- Video: `dist/videos/01-ema-introduction-draft.mp4`.
- Local preview player: `dist/videos/index.html`.
- Caption sources: `dist/videos/01-ema-introduction.ass` and `.vtt`.
- Format: H.264 MP4, 1920 × 1080, 25 fps, burned-in captions, no audio.
- Content: existing EMA map → BN layers, captured from the running app. No inference was started and no saved evidence was changed. This introduction has no inference waiting interval to accelerate.
- Validation: the finished MP4 decodes fully; map and BN frames were checked for image quality and caption placement.

The Snipping Tool's selection overlay was not accessible through automation. The usable take was captured with a portable FFmpeg encoder placed under `.cache/video-tools`, without changing the app's Python environment. An initial black window-capture attempt was replaced by capture of the verified app display area. Raw takes and capture checks remain under `dist/videos/raw`; only the edited draft above is intended for review.

Status: **first introduction draft available; four other clips pending**. The app code, model inputs and blog have not been changed. Videos are local drafts and have not been published.

### Requested introduction revision

The revised introduction will start with **no results**, then compute a baseline to inspect example distributions. Narration will read the captions. Each of the five BN layers will receive a matching animated highlight while it is explained, with real node clicks and distribution popups captured from the app. Keep the opening physically empty of prior results; computing a baseline later in the clip is needed for the distribution popups.

Prepared locally:

- Isolated recording server on port **8767**, with session storage in `.cache/recording-sessions/ema-intro-v2`; the working server on 8766 and its evidence are untouched.
- A capture of the fresh EMA map with **“Not run yet: grey”**, stored as `dist/videos/raw/v2-fresh.jpg`.
- Narration with Microsoft Hazel, generated locally and edited to **44.2 seconds**: `dist/videos/narration-preview.mp3` (also `.wav`).
- Eight caption/narration cues covering the empty start, the decision purpose, the baseline, shaking, station readings, damage, road state and access. Sources: `dist/videos/narrated-script.json`, `narrated-timing.json` and `01-ema-narrated.vtt`.

**Capture currently blocked:** Windows now denies desktop capture (`gdigrab` error 5; Desktop Duplication access denied) and browser mouse input (`GetCursorPos failed: Access is denied`, 0x80070005). Window screenshots remain readable, but the baseline Run and node clicks have not been captured. One window reconnection/retry also failed. User was asked to unlock the computer if locked and confirm the desktop is available. Do not present the old silent draft as this completed revision, or recreate unperformed node clicks in the video.

### Current revision: guided callouts beside the BN layers

The user replaced the live node-click approach with nearby explanatory captions and boxes. A new **34.96-second narrated guide** is ready:

- `dist/videos/01-ema-guided-bn.mp4`: fresh, unrun opening followed by all five BN layer explanations. Each layer has a matching coloured box, a nearby caption card connected to it, and synchronized local narration. Inactive layers are dimmed.
- `dist/videos/guided-bn.html` (also `index.html`): the local review player, with buttons to jump to the fresh start, the question, shaking, station readings, bridge damage, road availability and community access.
- `dist/videos/callout-script.json` and `callout-timing.json`: wording, annotation positions and exact timing.
- `dist/videos/01-ema-guided-bn.vtt`: matching accessible captions.

This guide uses captured app views and editorial video annotations. The opening is the actual fresh EMA view; the BN background is the captured structural diagram. It does not show a baseline calculation or unperformed node clicks, and the narration no longer refers to those actions. The app code, calculations and saved evidence are unchanged.

Validation: all seven scenes were extracted for visual review; connectors and caption placement were refined; colour range was normalized across scenes. The complete video/audio decodes, narration peaks below clipping, and the review player's chapter selection, initial loading and end-of-video state were checked.

### Narration refinement and silent option

The user found the desktop narration unnatural. A new version uses **en-GB-SoniaNeural**, at its normal speaking pace, with the same wording and callouts. The seven passages were generated through Microsoft's online speech service using a portable `edge-tts` installation in `.cache/speech-tools`; only the tutorial script was sent. The app environment and model inputs were not changed.

- `dist/videos/01-ema-guided-bn-natural.mp4`: updated neural narration, **35.76 seconds**.
- `dist/videos/01-ema-guided-bn-silent.mp4`: identical visuals and timing, **no audio stream**.
- `dist/videos/01-ema-guided-bn.mp4`: updated to the new narrated version so the existing review link remains useful.
- Both review pages include a **Narration: on/off** button, downloads for both versions, and chapter jumps aligned to the new voice.

Both MP4 files decode fully. The narration toggle and all seven retimed chapter jumps were checked. The earlier desktop-voice export is retained under `callout-parts/previous-desktop-voice.mp4` for reference.

## Clip 2 completed: clique tree and message passing

A **34.32-second** second clip is ready, using the same en-GB-SoniaNeural voice, matching highlight boxes and nearby caption cards:

- `dist/videos/02-ema-clique-tree.mp4`: narrated clip.
- `dist/videos/02-ema-clique-tree-silent.mp4`: matching silent version.
- `dist/videos/guided-cliques.html`: interactive chapter player, narration toggle and a link back to clip 1.
- `dist/videos/clique-script.json`, `clique-timing.json` and `02-ema-clique-tree.vtt`: script, annotation geometry, timing and captions.

Six steps: the inference circuit; a station reading's likelihood on shared shaking; collection of local messages; access constraints on road-state combinations; the certificate-MCMC root update when access evidence is present; and the elliptical-slice update of shaking with samples passed back to the local cliques.

The background was captured from the fresh EMA recording instance's **Clique tree** tab with **Inference** enabled. Moving dots illustrate the directions of the existing message arrows. They are labelled **“Illustrated message flow”**. No inference run, evidence update, app-code change or blog publication was performed. The recording browser was returned to the Map tab.

Technical wording was checked against the message definitions and sampling sequence in `src/inference/cliques.py`. All six scenes were visually reviewed, the full narrated export decodes, and the chapter jumps, narration toggle and final playback state were checked. Three clips remain: decision/evidence, results and custom system.

### Clip 2 revision: every message direction

The user requested coverage of arrows skipped by the short draft. The expanded clip is **55.60 seconds** and covers all four connection types in both directions, with a distinct numbered callout and animation for each of the eight directed messages, plus an overview and root-update explanation.

| Direction | Implementation |
|---|---|
| Measurement → bridge | `obs_up` |
| Bridge → hub | `edge_up` |
| Hub → root | `sa_e_up` |
| Access → root | `system_up` |
| Root → hub | `sa_e_down` |
| Hub → bridge | `edge_down` |
| Bridge → measurement | `obs_down` |
| Root → access | `system_down` |

Coverage was checked against every `@message` decorator in `src/inference/cliques.py`: each implemented direction is explained exactly once. The repeated stack members share these message types; the video explains the complete representative circuit. The root explanation also distinguishes certificate MCMC with access evidence from direct drawing otherwise.

Current outputs:

- `dist/videos/02-ema-clique-tree-all-edges.mp4` and `02-ema-clique-tree-all-edges-silent.mp4`.
- `dist/videos/guided-cliques.html`: updated with all ten chapters and narration control.
- Existing `02-ema-clique-tree.mp4` and its silent counterpart now point to the expanded content. The original shorter export is retained as `02-ema-clique-tree-brief.mp4`.
- Script and timing: `clique-all-edges-script.json`, `clique-all-edges-timing.json`; scene previews and annotation sources: `clique-all-edges-parts/`.

All eight arrow scenes were visually reviewed. Caption widths and animation endpoints were checked, the full video decodes, and all ten chapter jumps and the narration toggle passed their checks. No application or inference code changed.

## Clip 3 completed: decision task and evidence

The third clip is **37.16 seconds**, with the same en-GB-SoniaNeural narration, nearby caption cards and matching highlight boxes.

- `dist/videos/03-ema-decision-evidence.mp4`: narrated video.
- `dist/videos/03-ema-decision-evidence-silent.mp4`: identical visuals without audio.
- `dist/videos/guided-decision.html`: eight chapter buttons, narration toggle and downloads.
- `dist/videos/decision-script.json`, `decision-timing.json` and `03-ema-decision-evidence.vtt`: wording, annotations, timings and captions.

The walkthrough starts fresh, runs a baseline, selects the bridge-inspection task, enters an example ground-motion reading of **0.05 g at bridge-e0073** with source **demo station**, adds it, then shows the actual recorded evidence. All actions were performed in the isolated recording instance on **8767**. The production session on 8766 was untouched. The completed recording session is retained for the results clip.

Actual calculation times were **2.81 s** for the baseline, **18.50 s** for task selection and **275.47 s** for incorporating the observation and updating priorities. Waiting states are shown for about one second each and explicitly labelled as shortened. These are captured app states with editorial video callouts; no model results or unperformed interactions were fabricated.

Validation: all eight scenes were visually reviewed; caption widths were checked; both MP4 exports decode completely; chapter jumps, initial loading, narration toggle and final playback state were checked. The completed session was verified to contain exactly one observation at the intended value and one inspection query. No application or calculation code changed. The next clip covers Roadmap and BN results.

## Clip 4 completed: reading Roadmap and BN results

The fourth clip is **34.28 seconds**, continuing clip 3's actual EMA recording session and using the same neural narration, highlight boxes and nearby caption cards.

- `dist/videos/04-ema-read-results.mp4`: narrated clip.
- `dist/videos/04-ema-read-results-silent.mp4`: identical visuals without audio.
- `dist/videos/guided-results.html`: six chapter buttons, narration toggle and downloads.
- `dist/videos/results-script.json`, `results-timing.json` and `04-ema-read-results.vtt`: script, geometry, timing and captions.

Six explanations cover newest-first history; previous/current inspection scores; bridge-e0125 remaining the leading candidate with expected benefit changing from **51.13 to 50.67 people**; the update's actual **275.47 s** computation time; clicking **SA(1.0)::bridge-e0073** in BN; and its previous/current shaking distributions. The median changes from **0.121 g to 0.051 g**, with a visibly narrower distribution after the example station reading.

The BN popup was opened through the actual interface and captured after browser control recovered. Captures were cropped to remove browser chrome and annotated for the tutorial. The comparisons and displayed numbers were verified against the recording server's session and node-distribution replies. No new inference or evidence update was needed, and no application or calculation code changed.

All six scenes were visually reviewed. Caption widths, full decoding of both MP4 versions, all chapter jumps, narration toggle, initial metadata loading and end state were checked. The remaining clip is customising the system through network.json.

## Clip 5 completed: customise a system

The final clip is **36.60 seconds**, with the same neural narration, coloured highlight boxes and nearby explanatory cards.

- `dist/videos/05-ema-custom-system.mp4`: narrated clip.
- `dist/videos/05-ema-custom-system-silent.mp4`: identical visuals without audio.
- `dist/videos/guided-custom.html`: seven chapter buttons, narration toggle and downloads.
- `dist/videos/custom-script.json`, `custom-timing.json` and `05-ema-custom-system.vtt`: script, annotations, timing and captions.

The actual source file was opened in VS Code from an isolated project copy under `.cache/custom-system-recording`. Road **e0073**, connecting **n71 and n36**, was edited from **60 to 40 km/h** and saved. Its speed-assumption description was also updated. The actual `python -m src check` output confirms that the changed EMA assembly validates with **588 BN nodes**. The validation log is displayed as captured output, rather than presented as a fabricated terminal session.

The isolated server on **8768** was restarted after the edit, verified to expose the 40 km/h value, and **New session** was clicked in the app. The fresh **100,000-sample** baseline completed in **161.40 s**, with no evidence or decision queries carried over. The real waiting view is compressed to about one second and labelled as shortened. The closing scene distinguishes system data in `network.json` from model choices in `*.assembly.json`.

Validation: all seven scenes were visually reviewed; captions fit their cards; both MP4 versions decode completely; chapter jumps, narration toggle, initial metadata loading and final state passed checks. The recording copy's only network-data differences are the intended speed and its assumption. Its Python sources and EMA assembly match the working project, whose network hash remains unchanged. The working servers and earlier recording sessions were preserved.

**All five tutorial clips are now complete as local review drafts.** They have not been published to the blog or GitHub Pages.

### Revised example title and clip 1 wording

The guided page and document title now use **Earthquake inspection prioritisation for the Eastern Massachusetts highway benchmark network**. This title stays visible when selecting any of the five clips in the single player.

Clip 1's opening narration and captions introduce the full network name, replacing “EMA analysis.” Its next passage asks which bridge inspections would help restore airport access after an earthquake. The same approved en-GB-SoniaNeural voice was used. The rebuilt clip is **40.96 seconds**; the narrated alias, silent version, VTT captions, previews, player chapter timings and displayed duration were updated together.

The opening two scenes were visually reviewed, both exports decode fully, and the title and all chapter jumps were checked across the five clip selections. The recorded app views and inference calculations are unchanged.

Clip 3's opening was subsequently changed to **“Start with a fresh session, then run the baseline.”** Its opening caption reads **“Start a fresh session.”** The narration, narrated and silent exports, VTT captions, previews and player timings were regenerated together; the clip is now **36.60 seconds**. The clip's badge and standalone player also use “Decision task and evidence” without the unexplained EMA abbreviation. The opening frame was visually checked and both exports decode fully.
