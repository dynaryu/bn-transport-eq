"""Build review media from captured app views; no application or inference changes."""
import json
import shutil
import math
import subprocess
import wave
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'dist/videos'
FFMPEG = shutil.which('ffmpeg') or next((ROOT/'.cache/video-tools/imageio_ffmpeg/binaries').glob('ffmpeg*.exe'), None)
if not FFMPEG:
    raise SystemExit('Install FFmpeg and add it to PATH before building tutorial media.')
parser = argparse.ArgumentParser()
parser.add_argument('--narration-dir', default='callout-narration')
parser.add_argument('--script', default='callout-script.json')
parser.add_argument('--name', default='01-ema-guided-bn')
parser.add_argument('--parts', default='callout-parts')
parser.add_argument('--timing', default='callout-timing.json')
parser.add_argument('--audio-prefix', default='guided-narration')
parser.add_argument('--badge', default='Illustrative earthquake scenario')
parser.add_argument('--reuse-existing', action='store_true')
args = parser.parse_args()
STEPS = json.loads((OUT/args.script).read_text(encoding='utf-8'))
NARRATION = OUT/args.narration_dir
PARTS = OUT/args.parts
PARTS.mkdir(exist_ok=True)
RATE, FPS = 22050, 25


def ffmpeg(*args):
    subprocess.run([str(FFMPEG), '-y', '-hide_banner', '-loglevel', 'error', *map(str, args)],
                   cwd=ROOT, check=True)


def ass_colour(hex_colour):
    r, g, b = [hex_colour[i:i+2] for i in (1, 3, 5)]
    return f'&H00{b}{g}{r}'


HEADER = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Body,Arial,30,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''


def at(seconds):
    cs = round(seconds*100)
    return f'{cs//360000}:{cs//6000%60:02}:{cs//100%60:02}.{cs%100:02}'


def rect_path(x, y, w, h):
    return f'm {x} {y} l {x+w} {y} {x+w} {y+h} {x} {y+h} {x} {y}'


def connector(step):
    x, y, w, h = step['box']
    cx, cy, cw, ch = step['card']
    if cx+cw <= x:
        yy = min(max(y+h//2,cy+15),cy+ch-15)
        return [(cx+cw,yy),(x,yy)]
    if cx >= x+w:
        yy = min(max(y+h//2,cy+15),cy+ch-15)
        return [(x+w,yy),(cx,yy)]
    xx = min(max(x+w//2,cx+20),cx+cw-20)
    if cy < y:
        return [(xx,cy+ch),(xx,y)]
    return [(xx,y+h),(xx,cy)]


def annotation(step, duration):
    x, y, w, h = step['box']
    cx, cy, cw, ch = step['card']
    c = ass_colour(step['colour'])
    events = []

    def event(layer, tags, text):
        events.append(f'Dialogue: {layer},0:00:00.00,{at(duration)},Body,,0,0,0,,{{{tags}}}{text}')

    event(1, rf'\an7\pos(0,0)\p1\bord4\1a&HFF&\3c{c}\3a&H00&\fad(130,100)',
          rect_path(x,y,w,h))
    points = connector(step)
    event(2, rf'\an7\pos(0,0)\p1\bord2\1a&HFF&\3c{c}\fad(180,100)',
          'm '+' l '.join(f'{px} {py}' for px,py in points))
    event(3, rf'\an7\pos({cx},{cy})\p1\bord1\3c{c}\1c&H00221C17&\1a&H08&\fad(180,100)',
          rect_path(0,0,cw,ch))
    event(4, rf'\an7\pos({cx},{cy})\p1\bord0\1c{c}\fad(180,100)',
          rect_path(0,0,5,ch))
    event(5, rf'\an7\pos({cx+22},{cy+15})\fs30\b1\1c{c}\fad(180,100)',step['title'])
    event(5, rf'\an7\pos({cx+22},{cy+59})\fs29\1c&H00FFFFFF&\fad(180,100)',r'\N'.join(step['lines']))
    event(6, r'\an7\pos(0,0)\p1\bord0\1c&H00222222&\1a&H20&',rect_path(790,4,340,36))
    event(7, r'\an8\pos(960,10)\fs20\1c&H00DDDDDD&\bord0',
          step.get('badge',args.badge))
    # Editorial dots follow the app's message arrows; they illustrate direction, not live progress.
    for flow in step.get('flows', []):
        (sx,sy),(ex,ey) = flow
        t = .35
        while t+1.0 < duration-.1:
            circle = 'm 0 -5 b 3 -5 5 -3 5 0 5 3 3 5 0 5 -3 5 -5 3 -5 0 -5 -3 -3 -5 0 -5'
            tags = rf'\an7\move({sx},{sy},{ex},{ey},0,1000)\p1\bord2\3c&H00FFFFFF&\1c{c}\fad(80,80)'
            events.append(f'Dialogue: 4,{at(t)},{at(t+1)},Body,,0,0,0,,{{{tags}}}{circle}')
            t += 1.45
    return HEADER+'\n'.join(events)+'\n'


timeline, pcm, total_frames = [], [], 0
for step in STEPS:
    with wave.open(str(NARRATION/f"{step['id']}-timed.wav"),'rb') as wav:
        assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth()) == (RATE,1,2)
        speech_frames = wav.getnframes()
        data = wav.readframes(speech_frames)
    frames = math.ceil((speech_frames/RATE+.42)*FPS)
    duration = frames/FPS
    start = total_frames/FPS
    timeline.append({**step,'start':start,'end':start+duration,
                     'speech_start':start+.12,'speech_end':start+.12+speech_frames/RATE,
                     'connector':connector(step)})
    lead = round(.12*RATE)
    pad = frames*(RATE//FPS)-lead-speech_frames
    assert pad >= 0
    pcm += [bytes(lead*2),data,bytes(pad*2)]
    total_frames += frames

    ass_file = PARTS/f"{step['id']}.ass"
    if args.reuse_existing and (PARTS/f"{step['id']}.mp4").exists():
        continue
    ass_file.write_text(annotation(step,duration),encoding='utf-8')
    background = OUT/step['background'] if 'background' in step else OUT/'raw/v2-fresh.jpg' if step['scene']=='map' else OUT/'bn-guide-background.png'
    filters = ['scale=1920:1080:in_range=auto:out_range=tv','setsar=1','format=yuv420p']
    if step['scene'] in ('bn','clique','decision'):
        x,y,w,h=step['box']
        for dx,dy,dw,dh in [(0,90,1920,y-90),(0,y,x,h),(x+w,y,1920-x-w,h),(0,y+h,1920,1080-y-h)]:
            filters.append(f'drawbox=x={dx}:y={dy}:w={dw}:h={dh}:color=black@0.48:t=fill')
    filters.append('ass='+ass_file.relative_to(ROOT).as_posix())
    ffmpeg('-loop','1','-framerate',FPS,'-i',background,'-vf',','.join(filters),
           '-frames:v',frames,'-an','-c:v','libx264','-preset','medium','-crf','20',
           '-pix_fmt','yuv420p','-color_range','tv',PARTS/f"{step['id']}.mp4")

with wave.open(str(OUT/f'{args.audio_prefix}.wav'),'wb') as wav:
    wav.setparams((1,2,RATE,0,'NONE','not compressed'))
    wav.writeframes(b''.join(pcm))
ffmpeg('-i',OUT/f'{args.audio_prefix}.wav','-c:a','libmp3lame','-b:a','128k',OUT/f'{args.audio_prefix}.mp3')
(PARTS/'concat.txt').write_text('\n'.join(f"file '{s['id']}.mp4'" for s in STEPS)+'\n')
ffmpeg('-f','concat','-safe','0','-i',PARTS/'concat.txt','-i',OUT/f'{args.audio_prefix}.wav',
       '-c:v','copy','-c:a','aac','-b:a','128k','-shortest','-movflags','+faststart',
       OUT/f'{args.name}.mp4')
(OUT/args.timing).write_text(json.dumps(timeline,indent=2),encoding='utf-8')

def vtt_time(seconds):
    ms=round(seconds*1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}'

(OUT/f'{args.name}.vtt').write_text('WEBVTT\n\n'+'\n\n'.join(
    f"{vtt_time(s['speech_start'])} --> {vtt_time(s['speech_end'])}\n{s['text']}" for s in timeline)+'\n')
print(f'Exported narrated callout video: {total_frames/FPS:.2f} seconds, {len(STEPS)} guided steps.')
