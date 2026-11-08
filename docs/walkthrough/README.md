# Bridge network walkthrough

The public tutorial belongs at
https://jieunbyun.github.io/research/bridge-networks/.
Its title in the Research menu is **Risk modelling of bridge networks**.

The blog repository hosts the finished player and media under
`assets/research/bridge-networks/`. This repository retains the editing sources;
MP4s and original screen captures stay out of this repository's Git history.
The five buttons switch a single inline video, with chapter buttons for each clip.

## Editing sources

| Clip | Script/timing prefix | Video filename stem |
|---|---|---|
| Problem and Bayesian network | `callout` | `01-ema-guided-bn` |
| Clique tree and inference | `clique-all-edges` | `02-ema-clique-tree-all-edges` |
| Decision task and evidence | `decision` | `03-ema-decision-evidence` |
| Read the results | `results` | `04-ema-read-results` |
| Customise your system | `custom` | `05-ema-custom-system` |

Each `*-script.json` contains the narration (`text`, or `speech` when overridden),
caption lines, screenshot path, highlight box and caption-card geometry at
1920 × 1080. The corresponding `*-timing.json` and VTT files preserve the approved
timings and subtitles. `walkthrough-clips.json` records the player metadata.

The captures show real application states and results. Calculation waits are
shortened and labelled. Editing these assets does not change inference code.

## Rebuild a clip

Use Python, FFmpeg with libass/libx264 support, and `pip install edge-tts`.
Voice generation needs internet access; the approved voice is
`en-GB-SoniaNeural` at its normal rate.

Run from the project root. Restore the original captured backgrounds to
`dist/videos/`, preserving the relative `background` paths in each script.
For older steps without that field, the builder uses `raw/v2-fresh.jpg` for the
map and `bn-guide-background.png` for the BN. These local source images are
required to regenerate footage; the published package alone cannot rebuild it.
Copy the script JSON files from this directory into `dist/videos/` before editing.

For example, rebuild the decision clip:

```sh
python tools/walkthrough/generate_neural_narration.py --script decision-script.json --output-dir decision-narration
python tools/walkthrough/build_callout_video.py --script decision-script.json --narration-dir decision-narration --parts decision-parts --timing decision-timing.json --audio-prefix decision-narration --name 03-ema-decision-evidence --badge "Decision task and evidence"
ffmpeg -y -i dist/videos/03-ema-decision-evidence.mp4 -an -c:v copy -movflags +faststart dist/videos/03-ema-decision-evidence-silent.mp4
```

Use the same pattern for another row in the table. Rebuild all affected parts
after changing wording or geometry; `--reuse-existing` deliberately skips
existing part videos and is only appropriate when their visuals and duration
remain valid. Retain or regenerate each poster at the path in
`walkthrough-clips.json` (a JPG frame exported with FFmpeg is sufficient).

Once all five final videos, silent versions, poster images and timing JSONs are
available in `dist/videos/`, rebuild the shared player:

```sh
python tools/walkthrough/build_video_library.py
```

Review the narration, captions, all clip switches and chapter jumps. Copy the
updated scripts, timings, VTTs and metadata back here. Copy the final
`guided-bn.html` and only the video/silent/poster files named in its metadata
into the blog's `assets/research/bridge-networks/`, retaining relative paths.
The player includes automatic height reporting when embedded in the research
page. Do not copy raw captures, speech intermediates or Python caches to the blog.

The blog uses its existing Jekyll GitHub Pages workflow. Publishing changes to
its `master` branch rebuilds the site; editing local files alone does not publish.
