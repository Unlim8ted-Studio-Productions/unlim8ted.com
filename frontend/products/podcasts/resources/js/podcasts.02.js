const audio = new Audio();
audio.preload = 'none';
const play = document.getElementById('playBtn');
const name = document.getElementById('npName');
const seek = document.getElementById('podcastSeek');
document.querySelectorAll('.episode-play').forEach(button => {
  button.addEventListener('click', async () => {
    audio.src = `https://assets.unlim8ted.com/podcasts/${encodeURIComponent(button.dataset.file)}`;
    name.textContent = button.textContent;
    try { await audio.play(); } catch { name.textContent = 'Playback could not start. Please try again.'; }
  });
});
play.addEventListener('click', async () => {
  if (!audio.getAttribute('src')) { name.textContent = 'Choose an available episode first.'; return; }
  if (!audio.paused) { audio.pause(); return; }
  try { await audio.play(); } catch { name.textContent = 'Playback could not start. Please try again.'; }
});
audio.addEventListener('play', () => { play.textContent = 'Pause'; });
audio.addEventListener('pause', () => { play.textContent = 'Play'; });
audio.addEventListener('ended', () => { play.textContent = 'Play'; });
audio.addEventListener('error', () => { play.textContent = 'Play'; name.textContent = 'This episode is unavailable right now.'; });
audio.addEventListener('timeupdate', () => {
  if (seek && Number.isFinite(audio.duration) && audio.duration > 0) seek.value = audio.currentTime / audio.duration * 100;
});
seek?.addEventListener('input', () => { if (Number.isFinite(audio.duration)) audio.currentTime = Number(seek.value) / 100 * audio.duration; });
// Retained for the existing volume control; uses only a bounded numeric value.
window.setVolume = value => { audio.volume = Math.max(0, Math.min(1, Number(value) / 100)); };
const footer = document.getElementById('footer-text');
if (footer) footer.textContent = `© 2019–${new Date().getFullYear()} Unlim8ted Studio Productions.`;
window.addEventListener('pagehide', () => audio.pause());
