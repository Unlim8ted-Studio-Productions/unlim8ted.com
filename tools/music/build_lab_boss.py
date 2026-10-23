"""Build continuous Lab Boss audio and MIDI. Requires mido and ffmpeg on PATH."""
from pathlib import Path
import subprocess

import mido


ROOT = Path(__file__).resolve().parents[2]
MUSIC = ROOT / "assets" / "music"
PARTS = [mido.MidiFile(MUSIC / f"LabBoss{part}.mid") for part in ("Intro", "Loop", "Outro")]
# MIDI exports have no rendered reverb padding. Join at the score boundaries,
# removing the outro's initial pickup silence at its internal join.
outro_start = next_time = 0
for message in PARTS[2]:
    next_time += message.time
    if message.type == "note_on" and message.velocity:
        outro_start = next_time
        break

intro_end, loop_end = PARTS[0].length, PARTS[1].length
subprocess.run([
    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
    "-i", str(MUSIC / "LabBossIntro.wav"),
    "-i", str(MUSIC / "LabBossLoop.wav"),
    "-i", str(MUSIC / "LabBossOutro.wav"),
    "-filter_complex",
    f"[0:a]atrim=end={intro_end},asetpts=PTS-STARTPTS[a];"
    f"[1:a]atrim=end={loop_end},asetpts=PTS-STARTPTS[b];"
    f"[2:a]atrim=start={outro_start},asetpts=PTS-STARTPTS[c];"
    "[a][b][c]concat=n=3:v=0:a=1[out]",
    "-map", "[out]", "-c:a", "pcm_f32le", str(MUSIC / "LabBoss.wav"),
], check=True)

# Use one fixed tempo and convert all source events through real time, keeping
# the three slightly different source tempos and all note/controller events.
result = mido.MidiFile(type=1, ticks_per_beat=9600)
conductor = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000)])
result.tracks.append(conductor)
offset = 0
for index, source in enumerate(PARTS):
    events = mido.MidiTrack()
    result.tracks.append(events)
    elapsed = 0
    previous_tick = 0
    trim = outro_start if index == 2 else 0
    for message in source:
        elapsed += message.time
        if message.type in ("set_tempo", "end_of_track"):
            continue
        tick = round((offset + max(0, elapsed - trim)) * 19200)
        events.append(message.copy(time=tick - previous_tick))
        previous_tick = tick
    offset += source.length - trim

# Keep the final reverb tail in both timelines. ffprobe supports float WAVs.
duration = float(subprocess.check_output([
    "ffprobe", "-v", "error", "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1", str(MUSIC / "LabBoss.wav"),
], text=True))
conductor.append(mido.MetaMessage("end_of_track", time=round(duration * 19200)))
result.save(MUSIC / "LabBoss.mid")
print(f"Built Lab Boss: {duration:.3f}s; joins at {intro_end:.3f}s and {intro_end + loop_end:.3f}s")
