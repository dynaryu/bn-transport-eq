"""Generate narration for the non-sensitive tutorial script; app state is not accessed."""
import asyncio
import json
import shutil
import subprocess
import argparse
import wave
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'dist/videos'
parser = argparse.ArgumentParser()
parser.add_argument('--script', default='callout-script.json')
parser.add_argument('--output-dir', default='neural-narration')
args = parser.parse_args()
DEST = OUT/args.output_dir
DEST.mkdir(exist_ok=True)
ENCODER = shutil.which('ffmpeg') or next((ROOT/'.cache/video-tools/imageio_ffmpeg/binaries').glob('ffmpeg*.exe'), None)
if not ENCODER:
    raise SystemExit('Install FFmpeg and add it to PATH before building tutorial media.')
STEPS = json.loads((OUT/args.script).read_text(encoding='utf-8'))


async def main():
    for step in STEPS:
        if step.get('silent'):
            with wave.open(str(DEST/f"{step['id']}-timed.wav"),'wb') as wav:
                wav.setparams((1,2,22050,0,'NONE','not compressed'))
                wav.writeframes(bytes(round(step.get('duration',.8)*22050)*2))
            continue
        mp3 = DEST/f"{step['id']}.mp3"
        await asyncio.wait_for(edge_tts.Communicate(
            step.get('speech',step['text']),voice='en-GB-SoniaNeural',rate='+0%'
        ).save(str(mp3)),timeout=45)
        subprocess.run([str(ENCODER),'-y','-hide_banner','-loglevel','error','-i',str(mp3),
                        '-ar','22050','-ac','1','-c:a','pcm_s16le',str(DEST/f"{step['id']}-timed.wav")],check=True)
        print(f"Generated {step['id']}",flush=True)


asyncio.run(main())
