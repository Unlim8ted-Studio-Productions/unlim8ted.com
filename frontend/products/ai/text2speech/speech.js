const text = document.getElementById('speechText');
const voices = document.getElementById('speechVoice');
const status = document.getElementById('speechStatus');
const play = document.getElementById('speechPlay');
const stop = document.getElementById('speechStop');
const synth = window.speechSynthesis;
function loadVoices() {
  const selected = voices.value;
  voices.replaceChildren(new Option('Default system voice', ''));
  for (const voice of synth.getVoices()) voices.add(new Option(`${voice.name} (${voice.lang})`, voice.voiceURI));
  voices.value = selected;
}
if (!synth || !window.SpeechSynthesisUtterance) {
  play.disabled = stop.disabled = true;
  status.textContent = 'Your browser does not support speech playback.';
} else {
  loadVoices(); synth.addEventListener('voiceschanged', loadVoices);
  play.addEventListener('click', () => {
    if (!text.value.trim()) { text.setAttribute('aria-invalid', 'true'); status.textContent = 'Enter some text first.'; text.focus(); return; }
    text.removeAttribute('aria-invalid'); synth.cancel();
    const utterance = new SpeechSynthesisUtterance(text.value);
    utterance.voice = synth.getVoices().find(v => v.voiceURI === voices.value) || null;
    utterance.onstart = () => { status.textContent = 'Reading aloud...'; };
    utterance.onend = () => { status.textContent = 'Finished.'; };
    utterance.onerror = () => { status.textContent = 'Playback stopped or the voice is unavailable. Try another voice.'; };
    synth.speak(utterance);
  });
  stop.addEventListener('click', () => { synth.cancel(); status.textContent = 'Stopped.'; });
  window.addEventListener('pagehide', () => synth.cancel());
}
text.setAttribute('aria-describedby', 'speechStatus');
