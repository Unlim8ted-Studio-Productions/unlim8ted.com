"""GUI/offline exporter for the music page's falling MIDI notes.

Setup: python -m pip install -r tools/midi_video_requirements.txt
Run:   python tools/midi_video_recorder.py
Requires ffmpeg and ffprobe on PATH. See docs/midi_video_recorder.md.
"""
import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#ff00c1", "#9600ff", "#4900ff", "#00b8ff", "#00fff9",
          "#ff7400", "#ffcc00", "#9cff00", "#00ff6b", "#ff003c",
          "#00ffcc", "#3c00ff", "#eaff00", "#ff0074", "#00ff96"]
FORMATS = {"Landscape": (1920, 1080), "Vertical": (1080, 1920)}
HIDDEN = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def dependencies():
    try:
        import mido
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Install dependencies: python -m pip install -r tools/midi_video_requirements.txt") from exc
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise RuntimeError(f"{name} was not found. Install FFmpeg and add its bin folder to PATH.")


def local_asset(url):
    parsed = urlparse(url)
    if parsed.netloc == "assets.unlim8ted.com":
        path = (ROOT / "assets" / unquote(parsed.path).lstrip("/")).resolve()
        if ROOT / "assets" not in path.parents:
            raise ValueError("Asset path is outside the assets folder.")
        return path
    if parsed.scheme:
        raise ValueError(f"Select a local file for {url}")
    path = Path(url)
    return path if path.is_absolute() else ROOT / path


def read_notes(path):
    """Convert tick times using the global tempo map, retaining track colors."""
    import mido
    midi = mido.MidiFile(path)
    if midi.type == 2:
        raise ValueError("Asynchronous type-2 MIDI files are not supported.")
    events = []
    # Tone.js exposes tracks containing notes; skip conductor/meta-only tracks.
    color_index = 0
    for track in midi.tracks:
        tick = 0
        has_notes = any(m.type == "note_on" and m.velocity for m in track)
        for order, msg in enumerate(track):
            tick += msg.time
            events.append((tick, color_index, order, msg))
        if has_notes:
            color_index += 1
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    tempo, previous, seconds = 500000, 0, 0.0
    active = defaultdict(deque)
    notes = []
    for tick, track, _, msg in events:
        seconds += mido.tick2second(tick - previous, midi.ticks_per_beat, tempo)
        previous = tick
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type in ("note_on", "note_off"):
            key = (track, msg.channel, msg.note)
            if msg.type == "note_on" and msg.velocity:
                active[key].append(seconds)
            elif active[key]:
                start = active[key].popleft()
                notes.append((start, seconds, msg.note, track))
    for (track, _, pitch), starts in active.items():
        notes.extend((start, seconds, pitch, track) for start in starts)
    if not notes:
        raise ValueError("The MIDI file contains no notes.")
    return sorted(notes)


def audio_duration(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "json", str(path)], capture_output=True, text=True,
                            check=True, **HIDDEN)
    duration = float(json.loads(result.stdout)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio has no valid duration.")
    return duration


class Cancelled(Exception):
    pass


def render_frame(notes, size, time, glow=False):
    """Rasterize moving edges with pixel coverage, without integer snapping."""
    from PIL import Image, ImageDraw
    width, height = size
    speed = 100 * height / 1080
    image = Image.new("RGB", size, (10, 8, 20))
    rectangles = []
    for start, end, pitch, track in notes:
        x = pitch * width / 128
        top = height - (end - time) * speed
        bottom = max(top + 1, height - (start - time) * speed)
        top, bottom = max(0.0, top), min(float(height), bottom)
        if bottom <= top:
            continue
        left = int(x)
        right = int(x + max(1, width / 128 - 2)) - 1
        color = COLORS[track % len(COLORS)]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        rectangles.append((left, right, top, bottom, rgb))

    if glow and rectangles:
        from PIL import ImageChops, ImageFilter
        # A broad pool of colored light sits BEHIND the crisp foreground.
        # Widen the light source so narrow MIDI columns still cast a visible halo.
        radius = max(12, min(size) / 18)
        light = Image.new("RGB", size)
        sources = ImageDraw.Draw(light)
        for left, right, top, bottom, rgb in rectangles:
            sources.rounded_rectangle(
                (left - radius / 2, top - radius / 3,
                 right + radius / 2, bottom + radius / 3),
                radius=radius / 3, fill=rgb)
        light = light.filter(ImageFilter.GaussianBlur(radius))
        image = ImageChops.add(image, light)

    # Composite sharp, original-color notes last: glow never washes out their fill.
    draw = ImageDraw.Draw(image, "RGBA")
    for left, right, top, bottom, rgb in rectangles:
        first, last = math.floor(top), math.ceil(bottom) - 1
        if first == last:
            draw.line((left, first, right, first), fill=rgb + (round(255 * (bottom - top)),))
        else:
            draw.line((left, first, right, first), fill=rgb + (round(255 * (first + 1 - top)),))
            if last > first + 1:
                draw.rectangle((left, first + 1, right, last - 1), fill=rgb + (255,))
            draw.line((left, last, right, last), fill=rgb + (round(255 * (bottom - last)),))
    return image


def export_video(notes, audio, output, size, fps, duration, cancel, progress, glow=False):
    width, height = size
    frames = math.ceil(duration * fps)
    # Match drawMidiNotes(): 128 pitch columns, 100 pixels/second,
    # scaled from a 1080px canvas for consistent motion at every resolution.
    speed = 100 * height / 1080
    lookahead = height / speed
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="midi-render-", dir=output.parent) as tmp:
        partial = Path(tmp) / "render.mp4"
        with (Path(tmp) / "ffmpeg.log").open("w+b") as log:
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                       "-r", str(fps), "-i", "pipe:0", "-i", str(audio),
                       "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                       "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                       "-threads", "2",
                       "-c:a", "aac", "-b:a", "320k", "-t", str(duration),
                       "-movflags", "+faststart", str(partial)]
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=log, **HIDDEN)
            # Interrupt even when the worker is blocked writing a frame to FFmpeg.
            stopped = threading.Event()

            def watch_cancel():
                while not stopped.wait(0.1):
                    if cancel.is_set():
                        if proc.poll() is None:
                            proc.terminate()
                        return

            watcher = threading.Thread(target=watch_cancel, daemon=True)
            watcher.start()
            try:
                visible, cursor = [], 0
                for frame in range(frames):
                    if cancel.is_set():
                        raise Cancelled()
                    time = frame / fps
                    while cursor < len(notes) and notes[cursor][0] <= time + lookahead:
                        visible.append(notes[cursor])
                        cursor += 1
                    visible = [n for n in visible if n[1] >= time]
                    image = render_frame(visible, size, time, glow=glow)
                    proc.stdin.write(image.tobytes())
                    if frame % max(1, fps // 4) == 0:
                        progress((frame + 1) / frames)
                proc.stdin.close()
                code = proc.wait()
                if cancel.is_set():
                    raise Cancelled()
                if code:
                    log.seek(0)
                    raise RuntimeError(log.read().decode(errors="replace")[-4000:])
            except BrokenPipeError as exc:
                if cancel.is_set():
                    raise Cancelled() from exc
                proc.wait()
                log.seek(0)
                raise RuntimeError(log.read().decode(errors="replace")[-4000:]) from exc
            finally:
                stopped.set()
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                if not proc.stdin.closed:
                    try:
                        proc.stdin.close()
                    except BrokenPipeError:
                        pass
                watcher.join()
        # Never replace an existing export, including one created during rendering.
        with output.open("xb") as destination, partial.open("rb") as source:
            shutil.copyfileobj(source, destination)
    progress(1.0)


@dataclass
class ExportJob:
    number: int
    midi: Path
    audio: Path
    output: Path
    format: str
    size: tuple
    fps: int
    glow: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    state: str = "Queued"
    progress: float = 0.0
    error: str = ""
    thread: object = None


class Recorder(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Unlim8ted MIDI Video Recorder")
        self.geometry("1000x820")
        self.minsize(880, 720)
        self.events = queue.Queue()
        self.jobs = {}
        self.next_number = 1
        self.closing = False
        self.midi = tk.StringVar()
        self.audio = tk.StringVar()
        self.output = tk.StringVar(value=str(ROOT / "tools/generated/midi-videos"))
        self.fps = tk.StringVar(value="60")
        self.glow = tk.BooleanVar(value=False)
        self.resolution = tk.StringVar(value="1080p")
        self.status = tk.StringVar(value="Select songs (Ctrl/Shift for multiple), then add them to the export queue.")
        self.formats = {name: tk.BooleanVar(value=True) for name in FORMATS}
        body = ttk.Frame(self, padding=20)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(9, weight=1)
        ttk.Label(body, text="MIDI Video Recorder", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        self.products = []
        try:
            self.products = [p for p in json.loads((ROOT / "assets/data/products.json").read_text(encoding="utf-8"))
                             if p.get("product-type") == "music" and p.get("midi") and p.get("file")]
        except (OSError, ValueError) as exc:
            self.status.set(f"Catalog unavailable; select local files. {exc}")
        ttk.Label(body, text="Songs\nCtrl/Shift to select").grid(row=1, column=0, sticky="nw", padx=(0, 12))
        song_frame = ttk.Frame(body)
        song_frame.grid(row=1, column=1, columnspan=2, sticky="ew", pady=5)
        self.songs = tk.Listbox(song_frame, selectmode="extended", exportselection=False, height=5)
        self.songs.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(song_frame, orient="vertical", command=self.songs.yview)
        scrollbar.pack(side="right", fill="y")
        self.songs.configure(yscrollcommand=scrollbar.set)
        for product in self.products:
            self.songs.insert("end", product["name"])
        self.songs.bind("<<ListboxSelect>>", self.select_song)
        self.file_entries = []
        for row, label, variable, kind in [(2, "MIDI file", self.midi, "midi"), (3, "Audio file", self.audio, "audio"), (4, "Save to", self.output, "folder")]:
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12))
            entry = ttk.Entry(body, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", pady=5)
            if kind != "folder":
                self.file_entries.append(entry)
            ttk.Button(body, text="Browse...", command=lambda v=variable, k=kind: self.browse(v, k)).grid(row=row, column=2, padx=(8, 0))
        options = ttk.Frame(body)
        options.grid(row=5, column=0, columnspan=3, sticky="w", pady=10)
        for name, value in self.formats.items():
            ttk.Checkbutton(options, text=name + (" (16:9)" if name == "Landscape" else " (9:16)"), variable=value).pack(side="left", padx=(0, 12))
        ttk.Combobox(options, textvariable=self.resolution, values=["1080p", "720p"], width=8, state="readonly").pack(side="left", padx=8)
        ttk.Combobox(options, textvariable=self.fps, values=["24", "30", "60"], width=4, state="readonly").pack(side="left")
        ttk.Label(options, text=" fps").pack(side="left")
        ttk.Checkbutton(options, text="Glow", variable=self.glow).pack(side="left", padx=12)
        self.bar = ttk.Progressbar(body, maximum=100)
        self.bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(body, textvariable=self.status, wraplength=900).grid(row=7, column=0, columnspan=3, sticky="w", pady=5)
        actions = ttk.Frame(body)
        actions.grid(row=8, column=0, columnspan=3, sticky="w", pady=8)
        self.export_button = ttk.Button(actions, text="Export / add selected songs", command=self.start)
        self.export_button.pack(side="left")
        ttk.Button(actions, text="Select all songs", command=lambda: (self.songs.select_set(0, "end"), self.select_song())).pack(side="left", padx=8)
        ttk.Button(actions, text="Cancel selected", command=self.cancel_selected).pack(side="left", padx=8)
        ttk.Button(actions, text="Cancel all", command=self.cancel_all).pack(side="left", padx=8)
        ttk.Button(actions, text="Clear finished", command=self.clear_finished).pack(side="left", padx=8)
        table_frame = ttk.Frame(body)
        table_frame.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=5)
        self.table = ttk.Treeview(table_frame, columns=("song", "format", "settings", "state", "progress"), show="headings", selectmode="extended", height=7)
        for column, title, width in [("song", "Song", 230), ("format", "Format", 90), ("settings", "Video", 210), ("state", "Status", 110), ("progress", "Progress", 80)]:
            self.table.heading(column, text=title)
            self.table.column(column, width=width, stretch=column == "song")
        self.table.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scroll.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=scroll.set)
        self.table.bind("<<TreeviewSelect>>", self.show_job)
        self.detail = tk.StringVar(value="Select an export to see its file path or error.")
        ttk.Label(body, textvariable=self.detail, wraplength=900).grid(row=10, column=0, columnspan=3, sticky="w", pady=5)
        if self.products:
            self.songs.selection_set(0)
            self.select_song()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_timer = self.after(100, self.poll)

    def select_song(self, event=None):
        selection = self.songs.curselection()
        if not selection:
            return
        multiple = len(selection) > 1
        for entry in self.file_entries:
            entry.configure(state="disabled" if multiple else "normal")
        if multiple:
            self.midi.set(f"{len(selection)} catalog songs selected")
            self.audio.set("Each song uses its matching catalog audio")
            return
        try:
            product = self.products[selection[0]]
            self.midi.set(str(local_asset(product["midi"])))
            self.audio.set(str(local_asset(product["file"])))
        except ValueError as exc:
            messagebox.showerror("Song unavailable", str(exc))

    def browse(self, variable, kind):
        value = filedialog.askdirectory() if kind == "folder" else filedialog.askopenfilename(
            filetypes=[("MIDI", "*.mid *.midi"), ("All files", "*.*")] if kind == "midi" else [("Audio", "*.mp3 *.wav *.flac *.m4a *.ogg"), ("All files", "*.*")])
        if value:
            if kind != "folder":
                if len(self.songs.curselection()) > 1:
                    self.midi.set("")
                    self.audio.set("")
                self.songs.selection_clear(0, "end")
                for entry in self.file_entries:
                    entry.configure(state="normal")
            variable.set(value)

    def start(self):
        if self.closing:
            return
        try:
            dependencies()
            selection = self.songs.curselection()
            if len(selection) > 1:
                sources = [(local_asset(self.products[i]["midi"]), local_asset(self.products[i]["file"])) for i in selection]
            else:
                sources = [(Path(self.midi.get()), Path(self.audio.get()))]
            formats = [name for name, enabled in self.formats.items() if enabled.get()]
            if not formats:
                raise ValueError("Select at least one video format.")
            if not self.output.get().strip():
                raise ValueError("Choose an output folder.")
            folder = Path(self.output.get()).resolve()
            fps = int(self.fps.get())
            scale = 2 / 3 if self.resolution.get() == "720p" else 1
            reserved = {job.output for job in self.jobs.values() if job.state in ("Queued", "Rendering", "Cancelling")}
            planned = []
            for midi, audio in sources:
                if not midi.is_file() or not audio.is_file():
                    raise ValueError(f"Missing MIDI or audio file for {midi.stem}. Select existing local files.")
                name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", midi.stem).strip(" .") or "song"
                for fmt in formats:
                    output = folder / f"{name}_{fmt.lower()}.mp4"
                    # Re-exporting after changing quality gets a fresh filename.
                    suffix = 2
                    while output.exists() or output in reserved:
                        output = folder / f"{name}_{fmt.lower()}_{suffix}.mp4"
                        suffix += 1
                    size = tuple(round(v * scale) for v in FORMATS[fmt])
                    planned.append((midi.resolve(), audio.resolve(), output, fmt, size, fps, self.glow.get()))
                    reserved.add(output)
        except Exception as exc:
            messagebox.showerror("Cannot export", str(exc))
            return
        for settings in planned:
            job = ExportJob(self.next_number, *settings)
            self.next_number += 1
            self.jobs[job.number] = job
            self.table.insert("", "end", iid=str(job.number), values=self.job_values(job))
        self.dispatch()
        self.update_summary()

    @staticmethod
    def job_values(job):
        return (job.midi.stem, job.format, f"{job.size[0]}x{job.size[1]} / {job.fps} fps" + (" / Glow" if job.glow else ""), job.state, f"{job.progress:.0%}")

    def run_job(self, job):
        try:
            if job.cancel.is_set():
                raise Cancelled()
            notes = read_notes(job.midi)
            duration = audio_duration(job.audio)
            export_video(notes, job.audio, job.output, job.size, job.fps, duration, job.cancel,
                         lambda fraction: self.events.put((job.number, "progress", fraction)), glow=job.glow)
            self.events.put((job.number, "Completed", ""))
        except Cancelled:
            self.events.put((job.number, "Cancelled", ""))
        except Exception as exc:
            self.events.put((job.number, "Failed", str(exc)))

    def dispatch(self):
        if self.closing:
            return
        for job in self.jobs.values():
            if job.state == "Queued":
                job.state = "Rendering"
                self.table.item(str(job.number), values=self.job_values(job))
                job.thread = threading.Thread(target=self.run_job, args=(job,), name=f"render-{job.number}", daemon=True)
                job.thread.start()

    def cancel_jobs(self, numbers):
        for number in numbers:
            job = self.jobs[number]
            if job.state in ("Queued", "Rendering"):
                job.cancel.set()
                job.state = "Cancelled" if job.state == "Queued" else "Cancelling"
                self.table.item(str(number), values=self.job_values(job))
        self.update_summary()

    def cancel_selected(self):
        self.cancel_jobs([int(number) for number in self.table.selection()])

    def cancel_all(self):
        self.cancel_jobs(list(self.jobs))

    def clear_finished(self):
        for number, job in list(self.jobs.items()):
            if job.state in ("Completed", "Cancelled", "Failed") and not (job.thread and job.thread.is_alive()):
                self.table.delete(str(number))
                del self.jobs[number]
        self.update_summary()

    def show_job(self, event=None):
        selection = self.table.selection()
        if selection:
            job = self.jobs[int(selection[0])]
            self.detail.set(f"{job.state}: {job.output}" + (f"\n{job.error}" if job.error else ""))

    def update_summary(self):
        if not self.jobs:
            self.bar["value"] = 0
            self.status.set("Select songs, then add them to the export queue.")
            return
        counts = {state: sum(j.state == state for j in self.jobs.values()) for state in ("Queued", "Rendering", "Cancelling", "Completed", "Cancelled", "Failed")}
        self.bar["value"] = 100 * sum(j.progress if j.state in ("Queued", "Rendering", "Cancelling") else 1 for j in self.jobs.values()) / len(self.jobs)
        self.status.set(" | ".join(f"{state}: {count}" for state, count in counts.items() if count))

    def poll(self):
        try:
            while True:
                number, kind, value = self.events.get_nowait()
                job = self.jobs[number]
                if kind == "progress":
                    job.progress = value
                else:
                    job.state, job.error = kind, value
                    if kind == "Completed":
                        job.progress = 1
                self.table.item(str(number), values=self.job_values(job))
                self.show_job()
        except queue.Empty:
            pass
        self.dispatch()
        if self.jobs:
            self.update_summary()
        if self.closing and not any(j.thread and j.thread.is_alive() for j in self.jobs.values()):
            self.destroy()
            return
        self.poll_timer = self.after(100, self.poll)

    def destroy(self):
        self.after_cancel(self.poll_timer)
        super().destroy()

    def close(self):
        self.closing = True
        self.export_button.configure(state="disabled")
        self.cancel_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", type=Path, metavar="OUTPUT_DIR", help="Export a two-second sample in both orientations without opening the GUI")
    args = parser.parse_args()
    if args.smoke_test:
        dependencies()
        products = json.loads((ROOT / "assets/data/products.json").read_text(encoding="utf-8"))
        song = next(p for p in products if p.get("id") == "unlim8ted-stinger")
        notes = read_notes(local_asset(song["midi"]))
        def sample(name, size):
            path = args.smoke_test / (name + ".mp4")
            export_video(notes, local_asset(song["file"]), path, size, 60, 2, threading.Event(), lambda _: None, glow=True)
            print(path)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(sample, name, size) for name, size in {"landscape": (640, 360), "vertical": (360, 640)}.items()]
            for future in futures:
                future.result()
    else:
        Recorder().mainloop()


if __name__ == "__main__":
    main()
