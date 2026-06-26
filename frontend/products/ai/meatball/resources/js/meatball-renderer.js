(function () {
  const DEFAULT_LINES = {
    neutral: "<strong>Neutral.</strong> The sauce is awake and ready to answer.",
    excited: "<strong>Excited.</strong> Yes. Absolutely. The meatball has located several dramatic facts.",
    confused: "<strong>Confused.</strong> Something about that question made the sauce blink twice.",
    suspicious: "<strong>Suspicious.</strong> The answer is probably correct, but the sauce does not fully trust that signal.",
    angry: "<strong>Angry.</strong> The Glitch matters. The sauce refuses to pretend otherwise.",
    sad: "<strong>Sad.</strong> The sauce is still awake, but it is lower to the floor now.",
    overwhelmed: "<strong>Overwhelmed.</strong> Too many signals entered the tiny meatball brain at once."
  };

  const VALID_EMOTIONS = new Set(Object.keys(DEFAULT_LINES));

  function createMarkup() {
    return `
      <div class="meatballEmotionStage" data-emotion="neutral" data-talking="false">
        <div class="cutsceneLayer">
          <div class="blackout"></div>
          <div class="eyeReveal left"></div>
          <div class="eyeReveal right"></div>
          <div class="letterbox top"></div>
          <div class="letterbox bottom"></div>
          <div class="moon"></div>
          <div class="aura"></div>
          <div class="kanji">SAUCE</div>
          <div class="speedline s1"></div>
          <div class="speedline s2"></div>
          <div class="speedline s3"></div>
          <div class="speedline s4"></div>
          <div class="slash one"></div>
          <div class="slash two"></div>
          <div class="slash three"></div>
          <div class="impact"></div>
          <div class="shock"></div>
          <div class="subtitle">forbidden pasta technique: sauce rupture</div>
        </div>

        <div class="world">
          <div class="orbital"></div>
          <div class="orbital two"></div>

          <div class="fx">
            <div class="ring"></div>
            <div class="glitch"></div>
            <div class="flash"></div>
          </div>

          <div class="symbol a">?</div>
          <div class="symbol b">!</div>
          <div class="symbol c">&infin;</div>

          <div class="wrap">
            <div class="shadow"></div>
            <div class="arm left"></div>
            <div class="arm right"></div>

            <div class="meatball">
              <div class="sauce"></div>
              <div class="brow left"></div>
              <div class="brow right"></div>
              <div class="eye left"><div class="pupil"></div></div>
              <div class="eye right"><div class="pupil"></div></div>
              <div class="mouth"><div class="tongue"></div></div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function resolveRoot(target) {
    if (!target) return null;
    if (target.classList?.contains("meatballEmotionStage")) return target;
    return target.querySelector?.(".meatballEmotionStage") || null;
  }

  function getState(root) {
    if (!root) return null;
    if (!root.__meatballState) {
      root.__meatballState = {
        emotion: "neutral",
        talking: false,
        cutscene: false,
        glitching: false,
        bubbleHtml: DEFAULT_LINES.neutral,
        statusHtml: "emotion: neutral<br>talking: false",
        cutsceneTimer: null,
        glitchTimer: null,
        talkTimer: null
      };
    }
    return root.__meatballState;
  }

  function render(root) {
    const state = getState(root);
    if (!root || !state) return;

    root.dataset.emotion = state.emotion;
    root.dataset.talking = String(state.talking && !state.cutscene && !state.glitching);
    root.classList.toggle("is-cutscene", state.cutscene);
    root.classList.toggle("is-glitching", state.glitching);

  }

  function setEmotion(target, emotion) {
    const root = resolveRoot(target);
    const state = getState(root);
    if (!root || !state) return;
    state.emotion = VALID_EMOTIONS.has(emotion) ? emotion : "neutral";
    render(root);
  }

  function setTalking(target, talking) {
    const root = resolveRoot(target);
    const state = getState(root);
    if (!root || !state) return;
    state.talking = Boolean(talking);
    render(root);
  }

  function setBubble(target, html) {
    const root = resolveRoot(target);
    const state = getState(root);
    if (!root || !state) return;
    state.bubbleHtml = html || DEFAULT_LINES[state.emotion] || DEFAULT_LINES.neutral;
    render(root);
  }

  function setStatus(target, html) {
    const root = resolveRoot(target);
    const state = getState(root);
    if (!root || !state) return;
    state.statusHtml = html || `emotion: ${state.emotion}<br>talking: ${state.talking}`;
    render(root);
  }

  function inferEmotionFromText(text, options) {
    if (options?.isError) return "angry";
    if (options?.isLoading) return "suspicious";

    const input = String(text || "").toLowerCase();
    if (!input) return "neutral";
    if (input.includes("?")) return "suspicious";
    if (/(error|failed|jammed|refuse|wrong|cannot|can't)/.test(input)) return "angry";
    if (/(too many|overwhelmed|flood|everything|all at once)/.test(input)) return "overwhelmed";
    if (/(sorry|sad|lower|quiet|miss|lost)/.test(input)) return "sad";
    if (/(confused|blink|unclear|maybe|not sure|unknown)/.test(input)) return "confused";
    if (/(yes|absolutely|dramatic|great|strong|ready|awake|here)/.test(input)) return "excited";
    return "neutral";
  }

  function speakFor(target, text, maxMs, options) {
    const root = resolveRoot(target);
    const state = getState(root);
    if (!root || !state) return 0;

    window.clearTimeout(state.talkTimer);

    state.emotion = VALID_EMOTIONS.has(options?.emotion)
      ? options.emotion
      : inferEmotionFromText(text, options);
    state.talking = true;
    state.bubbleHtml = text ? `<strong>${state.emotion}.</strong> ${String(text)}` : DEFAULT_LINES[state.emotion];
    state.statusHtml = `emotion: ${state.emotion}<br>talking: true`;
    render(root);

    const duration = Math.min(
      typeof maxMs === "number" ? maxMs : 4200,
      Math.max(900, String(text || "").length * 34)
    );

    state.talkTimer = window.setTimeout(() => {
      state.talking = false;
      state.statusHtml = `emotion: ${state.emotion}<br>talking: false`;
      render(root);
    }, duration);

    return duration;
  }

  function playCutscene(target) {
    const root = resolveRoot(target);
    const state = getState(root);
    if (!root || !state) return;

    window.clearTimeout(state.cutsceneTimer);
    state.cutscene = false;
    render(root);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        state.cutscene = true;
        state.talking = false;
        render(root);
      });
    });

    state.cutsceneTimer = window.setTimeout(() => {
      state.cutscene = false;
      state.emotion = "angry";
      state.statusHtml = "emotion: angry<br>talking: false";
      state.bubbleHtml = DEFAULT_LINES.angry;
      render(root);
    }, 5000);
  }

  function playGlitch(target, options) {
    const root = resolveRoot(target);
    const state = getState(root);
    if (!root || !state) return 0;

    const duration = Math.max(1600, Number(options?.duration) || 2600);
    const settleEmotion = VALID_EMOTIONS.has(options?.settleEmotion) ? options.settleEmotion : "angry";
    const bubbleHtml = options?.bubbleHtml || "<strong>Glitch.</strong> The signal split into sauce static.";

    window.clearTimeout(state.cutsceneTimer);
    window.clearTimeout(state.glitchTimer);
    window.clearTimeout(state.talkTimer);

    state.cutscene = false;
    state.glitching = true;
    state.talking = false;
    state.emotion = "overwhelmed";
    state.bubbleHtml = bubbleHtml;
    state.statusHtml = "emotion: overwhelmed<br>talking: false<br>state: glitch";
    render(root);

    state.glitchTimer = window.setTimeout(() => {
      state.glitching = false;
      state.emotion = settleEmotion;
      state.statusHtml = `emotion: ${settleEmotion}<br>talking: false`;
      state.bubbleHtml = DEFAULT_LINES[settleEmotion] || DEFAULT_LINES.neutral;
      render(root);
    }, duration);

    return duration;
  }

  function mount(target) {
    const container = typeof target === "string" ? document.querySelector(target) : target;
    if (!container) return null;
    container.innerHTML = createMarkup();
    const root = resolveRoot(container);
    render(root);
    return root;
  }

  window.MeatballEmotionRenderer = {
    DEFAULT_LINES,
    inferEmotionFromText,
    mount,
    playCutscene,
    playGlitch,
    render,
    resolveRoot,
    setBubble,
    setEmotion,
    setStatus,
    setTalking,
    speakFor
  };
})();
