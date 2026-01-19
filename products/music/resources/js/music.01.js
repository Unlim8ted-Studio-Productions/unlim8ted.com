import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";
    import {
      collection,
      query,
      orderBy,
      onSnapshot,
      doc,
      setDoc,
      getDoc,
      deleteDoc,
      Timestamp,
    } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";
    import { getFirebase } from "/components/firebase-init.js";

    // ============================
    // Basic page helpers
    // ============================
    const $ = (id) => document.getElementById(id);

    document.addEventListener("DOMContentLoaded", function () {
      const currentYear = new Date().getFullYear();
      const footerText = `&copy; 2019-${currentYear} Unlim8ted Studio Productions. All rights reserved.`;
      const el = $("footer-text");
      if (el) el.innerHTML = footerText;

      const updateNavSpace = () => {
        const spacer = document.querySelector(".site-navbar-spacer");
        const h = spacer ? spacer.getBoundingClientRect().height : 56;
        document.documentElement.style.setProperty("--navSpace", `${Math.round(h)}px`);
      };
      updateNavSpace();
      window.addEventListener("resize", updateNavSpace, { passive: true });
      setTimeout(updateNavSpace, 250);
    });

    // ============================
    // Epilepsy warning (dismiss)
    // ============================
    const warnEl = $("epWarn");
    const dismissWarnBtn = $("dismissWarn");
    const WARN_KEY = "vizWarnDismissed_v1";

    function showWarnIfNeeded() {
      try {
        const dismissed = localStorage.getItem(WARN_KEY) === "1";
        warnEl.style.display = dismissed ? "none" : "flex";
      } catch {
        warnEl.style.display = "flex";
      }
    }
    dismissWarnBtn.addEventListener("click", () => {
      try { localStorage.setItem(WARN_KEY, "1"); } catch { }
      warnEl.style.display = "none";
    });
    showWarnIfNeeded();

    // ============================
    // Visualizer + playback state
    // ============================
    const canvas = $("midiCanvas");
    const ctx2d = canvas.getContext("2d");

    let notes = [];
    let startTime = null;
    let audio = null;
    let isPlaying = false;
    let durationMs = 0;

    let vizMode = "midi"; // "midi" | "audio"
    let rafId = null;

    // WebAudio analyser for audio mode (also OK for normal tracks)
    let audioCtx = null;
    let analyser = null;
    let srcNode = null;
    let freqData = null;
    let timeData = null;

    const nowPlayingEl = $("nowPlaying");
    const statusChip = $("statusChip");
    const timeChip = $("timeChip");
    const timeline = $("timeline");
    const vizSubtitle = $("vizSubtitle");
    const autoplayNextEl = $("autoplayNext");

    const pauseBtn = $("pauseBtn");
    const playBtn = $("playBtn");
    const restartBtn = $("restartBtn");
    const volumeEl = $("volume");

    const songListEl = $("songList");

    const colorPalette = [
      "#ff00c1", "#9600ff", "#4900ff", "#00b8ff", "#00fff9",
      "#ff7400", "#ffcc00", "#9cff00", "#00ff6b", "#ff003c",
      "#00ffcc", "#3c00ff", "#eaff00", "#ff0074", "#00ff96",
    ];

    function fmtTime(sec) {
      sec = Math.max(0, sec || 0);
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return `${m}:${String(s).padStart(2, "0")}`;
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener("resize", resizeCanvas, { passive: true });
    setTimeout(resizeCanvas, 50);

    function stopRaf() {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
    }

    function ensureAnalyser() {
      if (!audio) return;

      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === "suspended") audioCtx.resume().catch(() => { });

      if (srcNode) { try { srcNode.disconnect(); } catch { } srcNode = null; }
      if (!analyser) {
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.85;
      }

      // IMPORTANT: MediaElementSource is tied to the specific <audio> element instance.
      srcNode = audioCtx.createMediaElementSource(audio);
      srcNode.connect(analyser);
      analyser.connect(audioCtx.destination);

      freqData = new Uint8Array(analyser.frequencyBinCount);
      timeData = new Uint8Array(analyser.fftSize);
    }

    function clearCanvas() {
      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;
      ctx2d.clearRect(0, 0, w, h);
    }

    // ===== MIDI "falling notes" =====
    function drawMidiNotes() {
      if (!isPlaying || vizMode !== "midi") return;

      const currentTime = performance.now() - startTime;
      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;

      ctx2d.clearRect(0, 0, w, h);

      const noteWidth = w / 128;

      for (const note of notes) {
        const timeOffset = note.startTime - currentTime;
        const y = h - (timeOffset / 10) - (note.duration / 10);
        const noteHeight = note.duration / 10;

        if (y > h || y + noteHeight < 0) continue;

        const instrumentColor = colorPalette[note.instrument % colorPalette.length];
        const x = note.midi * noteWidth;

        ctx2d.fillStyle = instrumentColor;
        ctx2d.fillRect(x, y, Math.max(1, noteWidth - 2), Math.max(1, noteHeight));
      }

      rafId = requestAnimationFrame(drawMidiNotes);
    }

    // ===== Audio-reactive "cool" visualizer =====
    function drawAudioViz() {
      if (!isPlaying || vizMode !== "audio" || !analyser || !freqData || !timeData) return;

      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;

      analyser.getByteFrequencyData(freqData);
      analyser.getByteTimeDomainData(timeData);

      // Background fade for trails
      ctx2d.fillStyle = "rgba(0,0,0,0.18)";
      ctx2d.fillRect(0, 0, w, h);

      // Energy (0..1-ish)
      let sum = 0;
      for (let i = 0; i < freqData.length; i++) sum += freqData[i];
      const avg = sum / (freqData.length * 255);
      const energy = Math.min(1, Math.max(0, avg * 1.8));

      // Center glow rings
      const cx = w * 0.5;
      const cy = h * 0.5;
      const baseR = Math.min(w, h) * (0.10 + energy * 0.14);
      const rings = 5;

      for (let r = 0; r < rings; r++) {
        const t = (performance.now() * 0.001) + r * 0.35;
        const wobble = (Math.sin(t * 1.7) + Math.cos(t * 1.1)) * 0.5;
        const radius = baseR + r * (18 + energy * 22) + wobble * (8 + energy * 10);

        ctx2d.beginPath();
        ctx2d.arc(cx, cy, Math.max(6, radius), 0, Math.PI * 2);
        const c1 = colorPalette[(r * 3) % colorPalette.length];

        ctx2d.strokeStyle = `rgba(255,255,255,${0.05 + energy * 0.10})`;
        ctx2d.lineWidth = 1 + energy * 2;
        ctx2d.stroke();

        ctx2d.strokeStyle = `${c1}`;
        ctx2d.globalAlpha = 0.05 + energy * 0.18;
        ctx2d.lineWidth = 1 + energy * 2;
        ctx2d.stroke();
        ctx2d.globalAlpha = 1;
      }

      // Spectrum bars
      const bars = Math.min(96, freqData.length);
      const barW = w / bars;
      for (let i = 0; i < bars; i++) {
        const v = freqData[i] / 255;
        const bh = v * (h * 0.45);
        const x = i * barW;
        const y = h - bh;

        const col = colorPalette[i % colorPalette.length];
        ctx2d.globalAlpha = 0.25 + v * 0.65;
        ctx2d.fillStyle = col;
        ctx2d.fillRect(x + 1, y, Math.max(1, barW - 3), bh);
      }
      ctx2d.globalAlpha = 1;

      // Waveform
      ctx2d.beginPath();
      for (let i = 0; i < timeData.length; i++) {
        const v = (timeData[i] - 128) / 128;
        const x = (i / (timeData.length - 1)) * w;
        const y = cy + v * (h * 0.16 + energy * h * 0.08);
        if (i === 0) ctx2d.moveTo(x, y);
        else ctx2d.lineTo(x, y);
      }
      ctx2d.lineWidth = 2;
      ctx2d.strokeStyle = `rgba(233,231,255,${0.35 + energy * 0.45})`;
      ctx2d.stroke();

      rafId = requestAnimationFrame(drawAudioViz);
    }

    function setMode(mode) {
      vizMode = mode;
      stopRaf();
      clearCanvas();
      if (mode === "midi") {
        vizSubtitle.textContent = "MIDI notes falling in real-time.";
        if (isPlaying) drawMidiNotes();
      } else {
        vizSubtitle.textContent = "Audio-reactive mode (spectrum + waveform).";
        if (isPlaying) drawAudioViz();
      }
    }

    function teardownAudio() {
      stopRaf();

      if (audio) {
        try { audio.pause(); } catch { }
        audio.src = "";
        audio.load();
      }
      audio = null;

      if (srcNode) { try { srcNode.disconnect(); } catch { } srcNode = null; }
      // keep analyser/audioCtx for speed
    }

    function setupAudioElement(audioUrl) {
      teardownAudio();

      audio = new Audio(audioUrl);
      audio.crossOrigin = "anonymous";
      audio.volume = parseFloat(volumeEl.value || "1");

      audio.onplay = () => { ensureAnalyser(); };

      audio.ontimeupdate = () => {
        timeline.value = audio.currentTime * 1000;
        timeChip.textContent = fmtTime(audio.currentTime);
      };

      audio.onended = () => {
        isPlaying = false;
        statusChip.textContent = "Ended";
        stopRaf();
        // Auto-play next if enabled
        if (autoplayNextEl.checked) playNext();
      };
    }

    pauseBtn.addEventListener("click", pauseTrack);
    playBtn.addEventListener("click", resumeTrack);
    restartBtn.addEventListener("click", restartTrack);
    volumeEl.addEventListener("input", (e) => setVolume(e));
    timeline.addEventListener("change", (e) => seekTrack(e));

    // ============================
    // Products.json -> build list
    // ============================
    const PRODUCTS_URL = "/tools/data/products.json";

    /** full array from products.json (only "music" items) */
    let allMusic = [];

    /** index currently selected/playing inside allMusic */
    let currentIndex = -1;

    function isMidiTrack(item) {
      return !!(item && item.midi && String(item.midi).trim());
    }

    function displayTitle(item) {
      return (item?.name || "").trim() || "Untitled";
    }

    function safeUrl(u) {
      return String(u || "").trim();
    }

    function clearPlayingHighlight() {
      const prev = songListEl.querySelector(".song-item.playing");
      if (prev) prev.classList.remove("playing");
    }

    function setPlayingHighlight(index) {
      clearPlayingHighlight();
      const el = songListEl.querySelector(`.song-item[data-index="${index}"]`);
      if (el) {
        el.classList.add("playing");
        // keep visible (especially when autoplaying)
        el.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }

    function buildSongItem(item, index) {
      const wrap = document.createElement("div");
      wrap.className = "song-item";
      wrap.dataset.index = String(index);

      const name = document.createElement("div");
      name.className = "song-name";

      // If it's TimeCat-style, keep the name as-is (you can encode TRACK numbers in name if you want)
      name.textContent = displayTitle(item);

      const actions = document.createElement("div");
      actions.className = "song-actions";

      // PDF button (only if present)
      if (item.pdf) {
        const a = document.createElement("a");
        a.className = "pill";
        a.href = safeUrl(item.pdf);
        a.target = "_blank";
        a.rel = "noopener";
        a.textContent = "View PDF";
        actions.appendChild(a);
      }

      // Play button
      const play = document.createElement("span");
      play.className = "pill pill-btn";
      play.textContent = "Play";
      play.addEventListener("click", (e) => {
        e.stopPropagation();
        playByIndex(index);
      });
      actions.appendChild(play);

      wrap.addEventListener("click", () => playByIndex(index));

      wrap.appendChild(name);
      wrap.appendChild(actions);
      return wrap;
    }

    function buildDivider(kicker, title, sub) {
      const d = document.createElement("div");
      d.className = "section-divider";

      const k = document.createElement("div");
      k.className = "section-kicker";
      k.textContent = kicker;

      const t = document.createElement("div");
      t.className = "section-title";
      t.textContent = title;

      const s = document.createElement("div");
      s.className = "section-sub";
      s.textContent = sub;

      d.appendChild(k);
      d.appendChild(t);
      d.appendChild(s);
      return d;
    }

    function inferIsTimecat(item) {
      // No special flag in JSON, so infer by path or lack of midi (you can change this any time)
      const file = safeUrl(item.file);
      return file.includes("/products/games/TimeCat/") || file.includes("/TimeCat/");
    }

    function renderSongList() {
      songListEl.innerHTML = "";

      const normal = allMusic.filter(i => isMidiTrack(i));
      const timecat = allMusic.filter(i => !isMidiTrack(i) && inferIsTimecat(i));
      const otherAudio = allMusic.filter(i => !isMidiTrack(i) && !inferIsTimecat(i));

      if (!normal.length && !timecat.length && !otherAudio.length) {
        songListEl.innerHTML = `<div class="empty">No tracks found in products.json.</div>`;
        return;
      }

      if (normal.length) {
        const d = buildDivider("Library", "Scores + MIDI", "MIDI + MP3 with falling-note visuals and optional PDFs.");
        songListEl.appendChild(d);
        normal.forEach(item => {
          const idx = allMusic.indexOf(item);
          songListEl.appendChild(buildSongItem(item, idx));
        });
      }

      if (timecat.length) {
        const d = buildDivider("Soundtrack", "TimeCat", "The soundracks to Unlim8ted's newest game in development, Time Cat.");
        songListEl.appendChild(d);
        timecat.forEach(item => {
          const idx = allMusic.indexOf(item);
          songListEl.appendChild(buildSongItem(item, idx));
        });
      }

      if (otherAudio.length) {
        const d = buildDivider("Audio", "Other Tracks", "Audio-only tracks with reactive visuals.");
        songListEl.appendChild(d);
        otherAudio.forEach(item => {
          const idx = allMusic.indexOf(item);
          songListEl.appendChild(buildSongItem(item, idx));
        });
      }

      // restore highlight if something is selected
      if (currentIndex >= 0) setPlayingHighlight(currentIndex);
    }

    async function loadProducts() {
      try {
        const res = await fetch(PRODUCTS_URL, { cache: "no-store" });
        if (!res.ok) throw new Error(`Failed to load products.json (${res.status})`);
        const json = await res.json();

        // Keep only music
        allMusic = Array.isArray(json) ? json.filter(p => p && p["product-type"] === "music") : [];
        renderSongList();

        // If you want: auto-select first track without playing
        if (allMusic.length && currentIndex < 0) {
          setActiveProduct(0, { autoplay: false, keepStatus: true });
        }
      } catch (e) {
        console.error("products load error:", e);
        songListEl.innerHTML = `<div class="empty">Could not load <b>/tools/data/products.json</b>.</div>`;
      }
    }

    // ============================
    // Play by item (from JSON)
    // ============================
    async function playByIndex(index) {
      index = Number(index);
      if (!Number.isFinite(index) || index < 0 || index >= allMusic.length) return;

      setActiveProduct(index, { autoplay: true, keepStatus: false });
    }

    function playNext() {
      if (!allMusic.length) return;
      const next = (currentIndex >= 0) ? (currentIndex + 1) % allMusic.length : 0;
      // autoplay next
      setActiveProduct(next, { autoplay: true, keepStatus: false });
    }

    async function setActiveProduct(index, opts = { autoplay: true, keepStatus: false }) {
      const item = allMusic[index];
      if (!item) return;

      currentIndex = index;
      setPlayingHighlight(index);

      // Update "now playing" line immediately
      const title = displayTitle(item);
      nowPlayingEl.textContent = opts.autoplay ? `Now playing: ${title}` : `Selected: ${title}`;

      // Switch reviews/rating context to this product id
      activateReviewsFor(item);

      // Start playback (or just prepare)
      if (isMidiTrack(item)) {
        await playMidiAndAudio(item, title, opts.autoplay);
      } else {
        await playAudioOnly(item, title, opts.autoplay);
      }
    }

    // Normal track (MIDI + MP3)
    async function playMidiAndAudio(item, title, autoplay = true) {
      const midiFileUrl = safeUrl(item.midi);
      const audioFileUrl = safeUrl(item.file);

      statusChip.textContent = autoplay ? "Loading…" : "Ready";
      setMode("midi");

      // Reset timeline to avoid weird jump
      timeline.value = 0;
      timeline.max = 100;

      try {
        const r = await fetch(midiFileUrl);
        const data = await r.arrayBuffer();
        const midi = new Midi(data);

        notes = midi.tracks.flatMap((track, index) =>
          track.notes.map(n => ({
            midi: n.midi,
            startTime: n.time * 1000,
            duration: n.duration * 1000,
            instrument: index
          }))
        );

        durationMs = midi.duration * 1000;
        timeline.max = durationMs;
        timeline.value = 0;

        resizeCanvas();
        setupAudioElement(audioFileUrl);

        audio.onloadedmetadata = () => {
          // If the MP3 duration differs, timeline still works (we keep durationMs from MIDI for visuals)
          if (!durationMs && audio.duration) timeline.max = audio.duration * 1000;
        };

        startTime = performance.now();

        if (!autoplay) {
          isPlaying = false;
          statusChip.textContent = "Ready";
          clearCanvas();
          return;
        }

        await audio.play();
        isPlaying = true;
        statusChip.textContent = "Playing";
        drawMidiNotes();
      } catch (err) {
        console.error("Error loading/playing MIDI track:", err);
        isPlaying = false;
        statusChip.textContent = "Error";
        nowPlayingEl.textContent = "Could not load MIDI/audio";
      }
    }

    // TimeCat / audio-only track
    async function playAudioOnly(item, title, autoplay = true) {
      const audioFileUrl = safeUrl(item.file);

      statusChip.textContent = autoplay ? "Loading…" : "Ready";
      setMode("audio");
      resizeCanvas();

      // duration unknown until metadata loads
      durationMs = 0;
      timeline.value = 0;
      timeline.max = 100;

      setupAudioElement(audioFileUrl);

      audio.onloadedmetadata = () => {
        durationMs = (audio.duration || 0) * 1000;
        if (durationMs > 0) timeline.max = durationMs;
      };

      startTime = performance.now();

      if (!autoplay) {
        isPlaying = false;
        statusChip.textContent = "Ready";
        clearCanvas();
        return;
      }

      try {
        await audio.play();
        isPlaying = true;
        statusChip.textContent = "Playing";
        drawAudioViz();
      } catch (err) {
        console.error("Error playing audio-only track:", err);
        isPlaying = false;
        statusChip.textContent = "Error";
        nowPlayingEl.textContent = "Playback error";
      }
    }

    // ============================
    // Controls
    // ============================
    function pauseTrack() {
      if (audio && isPlaying) {
        audio.pause();
        isPlaying = false;
        statusChip.textContent = "Paused";
        stopRaf();
      }
    }

    function resumeTrack() {
      if (audio && !isPlaying) {
        audio.play().then(() => {
          isPlaying = true;
          statusChip.textContent = "Playing";
          startTime = performance.now() - (audio.currentTime * 1000);
          if (vizMode === "midi") drawMidiNotes();
          else drawAudioViz();
        }).catch(() => { });
      }
    }

    function restartTrack() {
      if (!audio) return;
      audio.currentTime = 0;
      timeline.value = 0;
      startTime = performance.now();

      if (!isPlaying) {
        audio.play().then(() => {
          isPlaying = true;
          statusChip.textContent = "Playing";
          if (vizMode === "midi") drawMidiNotes();
          else drawAudioViz();
        }).catch(() => { });
      } else {
        statusChip.textContent = "Playing";
        stopRaf();
        if (vizMode === "midi") drawMidiNotes();
        else drawAudioViz();
      }
    }

    function seekTrack(event) {
      const seekTime = Number(event.target.value || 0);
      if (!audio) return;

      audio.currentTime = seekTime / 1000;
      startTime = performance.now() - seekTime;

      if (isPlaying) {
        stopRaf();
        if (vizMode === "midi") drawMidiNotes();
        else drawAudioViz();
      }
    }

    function setVolume(event) {
      if (audio) audio.volume = parseFloat(event.target.value);
    }

    // ============================
    // Reviews / ratings (Firestore)
    // products/{productId}/comments
    // ============================
    const { auth, db } = getFirebase();

    // --- DOM (comments/ratings)
    const avgNum = $("avgNum");
    const avgLabel = $("avgLabel");
    const countLabel = $("countLabel");
    const avgStars = $("avgStars");
    const avgStarsBig = $("avgStarsBig");
    const barsEl = $("bars");

    const commentList = $("commentList");
    const commentForm = $("commentForm");
    const mustSignIn = $("mustSignIn");
    const ratingSel = $("ratingSel");
    const displayNameInput = $("displayName");
    const commentText = $("commentText");
    const formMsg = $("formMsg");
    const deleteMyReviewBtn = $("deleteMyReviewBtn");
    const postBtn = $("postBtn");

    function starsText(value) {
      const v = Math.max(0, Math.min(5, value));
      let out = "";
      for (let i = 1; i <= 5; i++) out += (i <= Math.round(v)) ? "★" : "☆";
      return out;
    }

    function renderBars(counts, total) {
      barsEl.innerHTML = "";
      for (let star = 5; star >= 1; star--) {
        const n = counts[star] || 0;
        const pct = total > 0 ? (n / total) * 100 : 0;

        const row = document.createElement("div");
        row.className = "barRow";

        const left = document.createElement("div");
        left.textContent = `${star} star`;

        const track = document.createElement("div");
        track.className = "barTrack";

        const fill = document.createElement("div");
        fill.className = "barFill";
        fill.style.width = `${pct.toFixed(1)}%`;

        const right = document.createElement("div");
        right.style.textAlign = "right";
        right.textContent = String(n);

        track.appendChild(fill);
        row.appendChild(left);
        row.appendChild(track);
        row.appendChild(right);
        barsEl.appendChild(row);
      }
    }

    function renderComments(docs, uid) {
      commentList.innerHTML = "";

      if (!docs.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No reviews yet. Be the first.";
        commentList.appendChild(empty);
        return;
      }

      docs.forEach(d => {
        const data = d.data();
        const rating = Number(data.rating || 0);
        const text = (data.text || "").trim();
        const who = (data.displayName || "").trim() || "Anonymous";

        let when = "";
        const ts = data.createdAt;
        if (ts && typeof ts.toDate === "function") when = ts.toDate().toLocaleString();

        const el = document.createElement("div");
        el.className = "comment";

        const top = document.createElement("div");
        top.className = "commentTop";

        const whoEl = document.createElement("div");
        whoEl.className = "who";
        whoEl.style.color = "#ffcc66";
        whoEl.textContent = `${who} • ${starsText(rating)}`;

        if (uid && d.id === uid) {
          const badge = document.createElement("span");
          badge.className = "mineBadge";
          badge.textContent = "Yours";
          whoEl.appendChild(badge);
        }

        const whenEl = document.createElement("div");
        whenEl.className = "when";
        whenEl.textContent = when;

        const p = document.createElement("p");
        p.className = "commentText";
        p.textContent = text;

        top.appendChild(whoEl);
        top.appendChild(whenEl);
        el.appendChild(top);
        el.appendChild(p);
        commentList.appendChild(el);
      });
    }

    function setFormBusy(busy) {
      postBtn.disabled = busy;
      deleteMyReviewBtn.disabled = busy;
      ratingSel.disabled = busy;
      displayNameInput.disabled = busy;
      commentText.disabled = busy;
    }

    function showFormMessage(msg, isError = false) {
      formMsg.textContent = msg || "";
      formMsg.style.color = isError ? "rgba(255,120,120,.95)" : "rgba(255,255,255,.65)";
    }

    // State for reviews
    let currentUid = null;
    let myReviewCache = null;
    let unsubComments = null;

    // Track which product's comments are active
    let activeProductId = null;

    // One auth listener for the whole page
    onAuthStateChanged(auth, async (user) => {
      currentUid = user ? user.uid : null;

      if (user) {
        mustSignIn.style.display = "none";
        commentForm.style.display = "grid";
        displayNameInput.value = user.displayName || "";
      } else {
        myReviewCache = null;
        commentForm.style.display = "none";
        mustSignIn.style.display = "block";
        deleteMyReviewBtn.style.display = "none";
        showFormMessage("");
      }

      // Reload "my review" when auth state changes for current product
      if (activeProductId) {
        await loadMyReviewFor(activeProductId);
      }
    });

    async function loadMyReviewFor(productId) {
      const user = auth.currentUser;
      if (!user) {
        myReviewCache = null;
        deleteMyReviewBtn.style.display = "none";
        showFormMessage("");
        return;
      }

      try {
        const myRef = doc(db, "products", String(productId), "comments", user.uid);
        const mySnap = await getDoc(myRef);

        if (mySnap.exists()) {
          myReviewCache = mySnap.data();
          ratingSel.value = String(myReviewCache.rating || "5");
          displayNameInput.value = (myReviewCache.displayName || "").trim() || (user.displayName || "");
          commentText.value = (myReviewCache.text || "").trim();
          deleteMyReviewBtn.style.display = "inline-block";
          showFormMessage("Editing your review.");
        } else {
          myReviewCache = null;
          ratingSel.value = "5";
          displayNameInput.value = user.displayName || "";
          commentText.value = "";
          deleteMyReviewBtn.style.display = "none";
          showFormMessage("");
        }
      } catch (e) {
        console.error("load my review error:", e?.code, e?.message, e);
        myReviewCache = null;
        deleteMyReviewBtn.style.display = "none";
        showFormMessage("");
      }
    }

    function activateReviewsFor(item) {
      const productId = String(item?.id || "");
      if (!productId) return;

      activeProductId = productId;

      // Reset UI immediately
      avgNum.textContent = "—";
      avgLabel.textContent = "Loading ratings…";
      countLabel.textContent = "";
      avgStars.textContent = "☆☆☆☆☆";
      avgStarsBig.textContent = "☆☆☆☆☆";
      barsEl.innerHTML = "";
      commentList.innerHTML = `<div class="empty">Loading reviews…</div>`;
      showFormMessage("");
      deleteMyReviewBtn.style.display = "none";

      // Kill old listener
      if (unsubComments) { try { unsubComments(); } catch { } unsubComments = null; }

      // Start new listener
      const commentsRef = collection(db, "products", productId, "comments");
      const commentsQ = query(commentsRef, orderBy("createdAt", "desc"));

      unsubComments = onSnapshot(
        commentsQ,
        (snap) => {
          const docs = snap.docs;
          const total = docs.length;
          let sum = 0;
          const counts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };

          for (const d of docs) {
            const r = Number(d.data().rating || 0);
            if (r >= 1 && r <= 5) {
              sum += r;
              counts[r] = (counts[r] || 0) + 1;
            }
          }

          const avg = total ? sum / total : 0;
          avgNum.textContent = total ? avg.toFixed(1) : "—";
          avgLabel.textContent = total ? "Average rating" : "No ratings yet";
          countLabel.textContent = `${total} review${total === 1 ? "" : "s"}`;
          avgStars.textContent = starsText(avg);
          avgStarsBig.textContent = starsText(avg);

          renderBars(counts, total);
          renderComments(docs, currentUid);
        },
        (err) => {
          console.error("Comments listener error:", err?.code, err?.message, err);
          commentList.innerHTML = `<div class="empty">Reviews unavailable.</div>`;
          avgLabel.textContent = "Reviews unavailable";
        }
      );

      // Load my review state (if signed in)
      loadMyReviewFor(productId);
    }

    // Post/edit review
    commentForm.onsubmit = async (e) => {
      e.preventDefault();
      const user = auth.currentUser;
      if (!user || !activeProductId) return;

      const rating = Math.trunc(parseInt(ratingSel.value, 10) || 0);
      const text = commentText.value.trim();
      const rawName = displayNameInput.value.trim();
      const finalName = rawName ? rawName : "Anonymous";

      if (!text) { showFormMessage("Please write a comment.", true); return; }
      if (text.length > 2000) { showFormMessage("Comment is too long (max 2000 chars).", true); return; }
      if (rating < 1 || rating > 5) { showFormMessage("Please select a rating 1–5.", true); return; }
      if (finalName.length > 80) { showFormMessage("Name too long (max 80 chars).", true); return; }

      showFormMessage("Saving…");
      setFormBusy(true);

      try {
        const myRef = doc(db, "products", String(activeProductId), "comments", user.uid);
        const existing = await getDoc(myRef);

        if (existing.exists()) {
          await setDoc(myRef, { rating, text, displayName: finalName }, { merge: true });
        } else {
          await setDoc(myRef, {
            userId: user.uid,
            rating,
            text,
            displayName: finalName,
            createdAt: Timestamp.now(),
          });
        }

        const snap = await getDoc(myRef);
        myReviewCache = snap.exists() ? snap.data() : null;

        deleteMyReviewBtn.style.display = myReviewCache ? "inline-block" : "none";
        showFormMessage("Saved");
        setTimeout(() => showFormMessage(myReviewCache ? "Editing your review." : ""), 900);
      } catch (err) {
        console.error("save review error:", err?.code, err?.message, err);
        showFormMessage("Failed to save review.", true);
      } finally {
        setFormBusy(false);
      }
    };

    // Delete my review
    deleteMyReviewBtn.onclick = async () => {
      const user = auth.currentUser;
      if (!user || !activeProductId) return;
      if (!confirm("Delete your review?")) return;

      showFormMessage("Deleting…");
      setFormBusy(true);

      try {
        const myRef = doc(db, "products", String(activeProductId), "comments", user.uid);
        await deleteDoc(myRef);

        myReviewCache = null;
        ratingSel.value = "5";
        displayNameInput.value = user.displayName || "";
        commentText.value = "";
        deleteMyReviewBtn.style.display = "none";
        showFormMessage("Deleted");
        setTimeout(() => showFormMessage(""), 900);
      } catch (err) {
        console.error("delete review error:", err?.code, err?.message, err);
        showFormMessage("Failed to delete.", true);
      } finally {
        setFormBusy(false);
      }
    };

    // ============================
    // Boot
    // ============================
    await loadProducts();

    // Expose a tiny API (optional; list uses internal click listeners anyway)
    window.playByIndex = playByIndex;
