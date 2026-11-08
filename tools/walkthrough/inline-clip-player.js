const clips = __CLIPS__;
const video = document.querySelector('#guide'), description = document.querySelector('#description');
const voice = document.querySelector('#voice');
const clipButtons = [...document.querySelectorAll('[data-clip]')];
const firstLabels = {fresh:'Fresh start',purpose:'The question',shaking:'Shaking',reading:'Station readings',damage:'Bridge damage',road:'Road availability',access:'Community access'};
let activeClip = 0, steps = clips[0].steps, buttons = [], pendingStep = null, selectionEpoch = 0;

function updateVoice() {
  const audible = !video.muted && video.volume > 0;
  voice.textContent = audible ? 'Narration: on' : 'Narration: off';
  voice.setAttribute('aria-pressed',String(audible));
}
voice.addEventListener('click', () => {
  video.muted = !video.muted;
  if (!video.muted && video.volume === 0) video.volume = 1;
  updateVoice();
});
video.addEventListener('volumechange',updateVoice);
function play() {
  const epoch = selectionEpoch;
  video.play().catch(() => {
    if (epoch === selectionEpoch) description.textContent = 'Press Play to start the video.';
  });
}
function update() {
  let active = steps.findIndex(step => video.currentTime >= step.start && video.currentTime < step.end);
  if (active < 0) active = video.currentTime >= steps.at(-1).end ? steps.length-1 : 0;
  buttons.forEach((button,index) => button.setAttribute('aria-pressed',String(index===active)));
  description.textContent = steps[active].text;
}
function playStep(step) {
  if (!video.readyState) { pendingStep = step; return; }
  video.currentTime = step.start + .2;
  update();
  play();
}
video.addEventListener('loadedmetadata', () => {
  if (pendingStep) { const step = pendingStep; pendingStep = null; playStep(step); }
  else update();
});
function renderSteps() {
  document.querySelector('#steps').replaceChildren();
  buttons = steps.map((step,index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.style.setProperty('--step-colour',step.colour);
    button.setAttribute('aria-pressed',String(index===0));
    const dot = document.createElement('span');
    dot.className = 'dot'; dot.setAttribute('aria-hidden','true');
    button.append(dot,document.createTextNode(activeClip===0 ? firstLabels[step.id] : step.title));
    button.addEventListener('click',() => playStep(step));
    document.querySelector('#steps').append(button);
    return button;
  });
}
function selectClip(index) {
  const clip = clips[index];
  video.pause(); selectionEpoch++;
  activeClip = index; steps = clip.steps; pendingStep = null;
  clipButtons.forEach((button,i) => button.setAttribute('aria-pressed',String(i===index)));
  video.setAttribute('aria-label',clip.title);
  video.poster = clip.poster;
  video.src = clip.video;
  document.querySelector('#download-narrated').href = clip.video;
  document.querySelector('#download-silent').href = clip.silent;
  document.querySelector('#video-fallback').href = clip.video;
  renderSteps();
  description.textContent = steps[0].text;
  video.load();
  play();
}
clipButtons.forEach((button,index) => button.addEventListener('click',() => selectClip(index)));
video.addEventListener('timeupdate',update);
video.addEventListener('seeked',update);
video.addEventListener('error',() => {
  description.textContent = 'The video could not load. Keep the HTML and its video files together.';
});
renderSteps();
updateVoice();
update();
