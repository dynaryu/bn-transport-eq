import html
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dist/videos'
entries=[
    ('Problem and Bayesian network','The physical system and the model that describes it.','01-ema-guided-bn','callout-timing.json','callout-parts/preview-fresh.jpg'),
    ('Clique tree and inference','Follow every message direction through the inference circuit.','02-ema-clique-tree-all-edges','clique-all-edges-timing.json','clique-all-edges-parts/preview-overview.jpg'),
    ('Decision task and evidence','Select an inspection task and add a station observation.','03-ema-decision-evidence','decision-timing.json','decision-parts/preview-observation.jpg'),
    ('Read the results','Compare inspection priorities and BN distributions.','04-ema-read-results','results-timing.json','results-parts/preview-distribution.jpg'),
    ('Customise your system','Edit network.json, validate it and run a fresh analysis.','05-ema-custom-system','custom-timing.json','custom-parts/preview-edit.jpg'),
]
clips=[]
for title,summary,name,timing,poster in entries:
    steps=json.loads((OUT/timing).read_text(encoding='utf-8'))
    if not (OUT/poster).exists():
        choices=sorted((OUT/Path(poster).parent).glob('preview-*.jpg'))
        poster=choices[0].relative_to(OUT).as_posix()
    clip={'title':title,'summary':summary,'video':name+'.mp4','silent':name+'-silent.mp4','poster':poster,
          'duration':math.ceil(steps[-1]['end']),
          'steps':[{k:s[k] for k in ('id','title','text','start','end','colour')} for s in steps]}
    for key in ('video','silent','poster'):
        assert (OUT/clip[key]).is_file(),clip[key]
    clips.append(clip)
labels=['Problem & model','Clique tree','Task & evidence','Read results','Custom system']
buttons='\n'.join(
    f'  <button type="button" data-clip="{i}" aria-controls="guide" aria-pressed="{str(i==0).lower()}">'
    f'{i+1}. {html.escape(labels[i])}<small>{c["duration"]} sec</small></button>' for i,c in enumerate(clips))
page=(Path(__file__).parent/'guided-player-template.html').read_text(encoding='utf-8')
controller=(Path(__file__).parent/'inline-clip-player.js').read_text(encoding='utf-8')
controller=controller.replace('__CLIPS__',json.dumps(clips,ensure_ascii=False).replace('</','<\\/'))
page=re.sub(r'<script>.*?</script>',lambda _: '<script>\n'+controller+'</script>',page,count=1,flags=re.S)
extra_styles='''
  .clip-nav { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 24px; }
  .clip-nav button { --step-colour: #8fbcff; }
  .clip-nav small { color: #a7b8d0; margin-left: 7px; font-variant-numeric: tabular-nums; }
'''
nav='<nav class="clip-nav" aria-label="Five walkthrough clips">\n'+buttons+'\n</nav>'
page=page.replace('</style>',extra_styles+'\n</style>',1)
page=page.replace('<main>','<main>\n'+nav,1)
page=page.replace('<a href="01-ema-guided-bn.mp4">','<a id="video-fallback" href="01-ema-guided-bn.mp4">')
page=page.replace('<a href="01-ema-guided-bn.mp4" download>','<a id="download-narrated" href="01-ema-guided-bn.mp4" download>')
page=page.replace('<a href="01-ema-guided-bn-silent.mp4" download>','<a id="download-silent" href="01-ema-guided-bn-silent.mp4" download>')
page=re.sub(r'<footer>.*?</footer>','<footer>Guided preview &middot; neural narration &middot; illustrative earthquake scenario.<br>Calculation waits are shortened where shown.</footer>',page,flags=re.S)
for filename in ('guided-bn.html','index.html'):
    (OUT/filename).write_text(page,encoding='utf-8')
(OUT/'walkthrough-clips.json').write_text(json.dumps(clips,indent=2,ensure_ascii=False),encoding='utf-8')
print('Built five clip buttons that switch videos in the original single player; no popups.')
