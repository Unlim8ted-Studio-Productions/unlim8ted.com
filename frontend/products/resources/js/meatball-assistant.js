(function () {
  const d = document;
  if (!d.body) return;
  const desktopOnly = window.matchMedia("(min-width: 821px) and (pointer: fine)").matches;
  if (!desktopOnly) {
    const existing = d.getElementById("meatballAssistant");
    if (existing) existing.remove();
    return;
  }

  const assistant = d.getElementById("meatballAssistant") || createAssistant();
  const bubble = assistant.querySelector(".meatballAssistant-bubble");
  const textEl = assistant.querySelector(".meatballAssistant-text");
  const statusEl = assistant.querySelector(".meatballAssistant-status");
  const avatar = assistant.querySelector(".meatballAvatar");
  const storage = window.sessionStorage;
  const STORAGE_KEY = "meatball-assistant-state-v2";

  if (!bubble || !textEl || !avatar) return;

  const state = {
    x: 0,
    y: 0,
    tx: 0,
    ty: 0,
    side: "left",
    speakingUntil: 0,
    lastSpokeAt: 0,
    pointerX: window.innerWidth * 0.5,
    pointerY: window.innerHeight * 0.5,
    hoverKey: "",
    pendingHoverKey: "",
    pendingHoverTimer: 0,
    recentLines: [],
    recentConcepts: [],
    bubblePinned: false,
    sweatUntil: 0,
    hypeUntil: 0,
    bounceUntil: 0,
    jamUntil: 0,
    jamVisibleUntil: 0,
    currentJamTitle: "",
    anchorLockedUntil: 0,
    anchorZone: "",
    raf: 0
  };

  const actionableSelectors = [
    "#addBtn",
    "#buyBtn",
    "#variantWrap",
    ".spinnerCard",
    ".btn",
    "button",
    "a[href*='/products']",
    ".item-box",
    ".product-box",
    ".card"
  ];

  const namedReactions = [
    {
      match: /life of a meatball|meatball puzzle/i,
      lines: {
        card: "that is the meatball. show some respect.",
        cart: "yes YES. add the meatball one.",
        buy: "straight to the meatball. correct."
      }
    },
    {
      match: /timecat/i,
      lines: {
        card: "timecat is weird in the right way.",
        cart: "timecat in the cart. dangerous levels of wisdom."
      }
    },
    {
      match: /square pixels/i,
      lines: {
        card: "square pixels looks like a proper rabbit hole.",
        buy: "buying square pixels immediately. bold."
      }
    },
    {
      match: /text generator/i,
      lines: {
        card: "text generator is the loud one here.",
        variant: "good. text generator choices actually matter."
      }
    },
    {
      match: /text summarizer/i,
      lines: {
        card: "text summarizer. less mess. strong."
      }
    },
    {
      match: /text classifier/i,
      lines: {
        card: "text classifier sounds like it came prepared."
      }
    },
    {
      match: /the glitch/i,
      lines: {
        card: "the glitch still has a mean little aura.",
        cart: "the glitch. yes. take the suspicious one."
      }
    },
    {
      match: /alien/i,
      lines: {
        card: "alien. solid pick if you want things to get worse."
      }
    },
    {
      match: /unicornia/i,
      lines: {
        card: "unicornia has confidence. i respect it."
      }
    },
    {
      match: /chess/i,
      lines: {
        card: "chess. no notes. just stress.",
        buy: "buying chess without blinking. terrifying."
      }
    },
    {
      match: /air hockey/i,
      lines: {
        card: "air hockey online could get petty fast.",
        buy: "buying air hockey energy. reckless."
      }
    },
    {
      match: /puzzle square/i,
      lines: {
        card: "puzzle square looks innocent. i do not trust it."
      }
    },
    {
      match: /film script generator/i,
      lines: {
        card: "film script generator is trying to start something."
      }
    },
    {
      match: /download any youtube video/i,
      lines: {
        card: "extremely direct name. points for honesty."
      }
    },
    {
      match: /face stuff/i,
      lines: {
        card: "face stuff. alarming title. memorable though."
      }
    },
    {
      match: /chatapp/i,
      lines: {
        card: "chatapp is giving homemade troublemaker energy."
      }
    },
    {
      match: /music ai gen/i,
      lines: {
        card: "music ai gen sounds like it already has opinions."
      }
    },
    {
      match: /confuzzled/i,
      lines: {
        card: "confuzzled. fair warning right in the name."
      }
    },
    {
      match: /the crystal'?s conflict/i,
      lines: {
        card: "the crystal's conflict sounds appropriately dramatic."
      }
    },
    {
      match: /hidden magic school/i,
      lines: {
        card: "hidden magic school. solid suspicious-school setup."
      }
    },
    {
      match: /holes? in space and time/i,
      lines: {
        card: "holes in space and time. finally, a calm title."
      }
    },
    {
      match: /stranded/i,
      lines: {
        card: "stranded feels like a commit."
      }
    },
    {
      match: /ancient/i,
      lines: {
        card: "ancient. simple. ominous. efficient."
      }
    },
    {
      match: /the pool world war/i,
      lines: {
        card: "the pool world war is a wildly confident title."
      }
    },
    {
      match: /multiplayer physics simulator/i,
      lines: {
        card: "multiplayer physics simulator feels like planned chaos."
      }
    },
    {
      match: /easy pygame ui creator/i,
      lines: {
        card: "easy pygame ui creator might actually spare somebody pain."
      }
    },
    {
      match: /copy keyframes to selected blender addon/i,
      lines: {
        card: "that one sounds painfully specific. good sign."
      }
    },
    {
      match: /organisms sim/i,
      lines: {
        card: "organisms sim has experiment-gone-strange energy."
      }
    },
    {
      match: /ftl node based modding/i,
      lines: {
        card: "ftl node based modding sounds like a proper rabbit hole."
      }
    },
    {
      match: /ftl chooseyourside/i,
      lines: {
        card: "choose your side. already confrontational. nice."
      }
    },
    {
      match: /wrighting/i,
      lines: {
        card: "wrighting. bold spelling. stronger commitment."
      }
    },
    {
      match: /chessvr/i,
      lines: {
        card: "chessvr sounds like stress with depth perception."
      }
    },
    {
      match: /kindel e-ink games/i,
      lines: {
        card: "e-ink games is a very specific kind of ambition."
      }
    },
    {
      match: /turbo-octo-funicular/i,
      lines: {
        card: "turbo-octo-funicular. impossible to ignore that name."
      }
    },
    {
      match: /unlim8ted phone/i,
      lines: {
        card: "the phone is trying very hard to look expensive.",
        buy: "buying the phone. serious move."
      }
    },
    {
      match: /what is unlim8ted/i,
      lines: {
        card: "that title is asking for trouble in a good way."
      }
    },
    {
      match: /guide to the zipper economy/i,
      lines: {
        card: "zipper economy. elite niche material."
      }
    },
    {
      match: /jokes/i,
      lines: {
        card: "jokes. dangerous confidence."
      }
    },
    {
      match: /the gusian language/i,
      lines: {
        card: "the gusian language sounds like a commitment."
      }
    },
    {
      match: /computers/i,
      lines: {
        card: "computers. finally, a modest topic."
      }
    },
    {
      match: /the meatball adventures/i,
      lines: {
        card: "meatball adventures. that is my guy."
      }
    },
    {
      match: /ai and you/i,
      lines: {
        card: "ai and you. ominous. relevant."
      }
    }
  ];

  const hashReactions = {
    "the-life-of-a-meatball": {
      card: "the meatball film still clears half this site.",
      cart: "yes YES. the meatball one. obviously.",
      buy: "straight to the meatball. no delay."
    },
    "the-life-of-a-meatball-jigsaw-puzzle": {
      card: "meatball puzzle. absurdly solid.",
      cart: "put the meatball puzzle in the cart.",
      buy: "buying the meatball puzzle instantly. strong."
    },
    "theglitch": {
      card: "the glitch still feels slightly untrustworthy.",
      cart: "yes. take the glitch one."
    },
    "theglitchbook": {
      card: "the glitch book looks like it could spiral properly."
    },
    "theglitchscreenplay": {
      card: "screenplay version. sharper choice than most people make."
    },
    "download-any-youtube-video": {
      card: "that title gets right to the point."
    },
    "music-ai-gen": {
      card: "music ai gen looks ready to start an argument."
    },
    "chatapp": {
      card: "chatapp feels like a little chaos machine."
    }
  };

  const pageReactions = {
    "": {
      card: "",
      wheel: ""
    },
    "index": {
      card: "that category might have something decent.",
      wheel: "the wheel is looking for trouble."
    },
    "ai": {
      card: "that one is trying hard to be useful.",
      variant: "ai options can go bad fast. pick carefully."
    },
    "books": {
      card: "that book at least looks like it knows what it is doing."
    },
    "films": {
      card: "that one has enough pull to click."
    },
    "games": {
      card: "that one looks like a time sink.",
      buy: "buying a game without hesitation. risky confidence."
    },
    "software": {
      card: "that might actually earn its space on a machine."
    },
    "music": {
      card: "that track might be worth the click.",
      buy: "committing to a track immediately. dramatic."
    },
    "podcasts": {
      card: "that episode title knows what it is doing."
    },
    "images": {
      card: "that image has enough pull for a closer look."
    },
    "hardware": {
      card: "that hardware pick has some weight to it."
    },
    "physical-items": {
      card: "that one feels like a real-object gamble."
    },
    "hardware/unlim8ted-phone": {
      card: "the phone page is doing a lot to impress me."
    },
    "games/timecat": {
      card: "timecat has protagonist energy and i am not arguing."
    },
    "games/square-pixels": {
      card: "square pixels looks like sleep is about to lose."
    },
    "product": {
      cart: "yes. this is where you stop browsing and add it.",
      buy: "right to checkout. no sightseeing."
    }
  };

  function createAssistant() {
    const wrap = d.createElement("aside");
    wrap.className = "meatballAssistant";
    wrap.id = "meatballAssistant";
    wrap.setAttribute("aria-live", "polite");
    wrap.innerHTML = `
      <div class="meatballAssistant-bubble">
        <div class="meatballAssistant-label">
          <span>Meatball</span>
        </div>
        <p class="meatballAssistant-text">I will keep quiet unless this gets interesting.</p>
        <div class="meatballAssistant-actions">
          <a class="meatballAssistant-link" href="/products/ai/meatball">Talk to Meatball</a>
        </div>
      </div>
      <button class="meatballAvatar" type="button" aria-label="Talk to Meatball">
        <span class="meatballAvatar-shadow"></span>
        <span class="meatballAvatar-body">
          <span class="meatballAvatar-eye left"><span class="pupil"></span></span>
          <span class="meatballAvatar-eye right"><span class="pupil"></span></span>
          <span class="meatballAvatar-mouth"></span>
        </span>
      </button>
    `;
    d.body.appendChild(wrap);
    return wrap;
  }

  ensureAssistantControls();

  function clamp(n, min, max) {
    return Math.max(min, Math.min(max, n));
  }

  function ensureAssistantControls() {
    if (!bubble.querySelector(".meatballAssistant-actions")) {
      const actions = d.createElement("div");
      actions.className = "meatballAssistant-actions";
      const link = d.createElement("a");
      link.className = "meatballAssistant-link";
      link.href = "/products/ai/meatball";
      link.textContent = "Talk to Meatball";
      actions.appendChild(link);
      bubble.appendChild(actions);
    }
  }

  function rememberLine(line, concept) {
    const normalized = String(line || "").trim().toLowerCase();
    if (normalized) {
      state.recentLines.unshift(normalized);
      state.recentLines = state.recentLines.slice(0, 8);
    }

    const conceptKey = String(concept || "").trim().toLowerCase();
    if (conceptKey) {
      state.recentConcepts.unshift(conceptKey);
      state.recentConcepts = state.recentConcepts.slice(0, 8);
    }
  }

  function recentlySaid(line) {
    return state.recentLines.includes(String(line || "").trim().toLowerCase());
  }

  function recentlyCovered(concept) {
    return state.recentConcepts.includes(String(concept || "").trim().toLowerCase());
  }

  function setMood(mood) {
    if (statusEl) statusEl.textContent = mood || "";
  }

  function persistState() {
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify({
        x: state.x,
        y: state.y,
        tx: state.tx,
        ty: state.ty,
        side: state.side,
        anchorZone: state.anchorZone,
        bubblePinned: state.bubblePinned,
        text: textEl.textContent,
        status: statusEl.textContent,
        speaking: assistant.classList.contains("is-speaking"),
        speakingUntil: state.speakingUntil,
        savedAt: Date.now()
      }));
    } catch (_err) {
      // Ignore storage failures.
    }
  }

  function restoreState() {
    try {
      const raw = storage.getItem(STORAGE_KEY);
      if (!raw) return false;
      const saved = JSON.parse(raw);
      if (!saved || typeof saved !== "object") return false;

      const bounds = viewportBounds();
      state.x = clamp(Number(saved.x) || bounds.maxX, bounds.minX, bounds.maxX);
      state.y = clamp(Number(saved.y) || bounds.minY, bounds.minY, bounds.maxY);
      state.tx = clamp(Number(saved.tx) || state.x, bounds.minX, bounds.maxX);
      state.ty = clamp(Number(saved.ty) || state.y, bounds.minY, bounds.maxY);
      state.side = saved.side === "left" ? "left" : "right";
      state.anchorZone = typeof saved.anchorZone === "string" ? saved.anchorZone : `${state.side}-top`;
      state.bubblePinned = Boolean(saved.bubblePinned);

      if (typeof saved.text === "string" && saved.text.trim()) {
        textEl.textContent = saved.text;
      }
      if (typeof saved.status === "string" && saved.status.trim()) {
        setMood(saved.status);
      }

      const age = Date.now() - (Number(saved.savedAt) || 0);
      if (saved.speaking && age < 1800) {
        state.speakingUntil = Date.now() + Math.max(700, 1800 - age);
        assistant.classList.add("is-speaking");
      }

      return true;
    } catch (_err) {
      return false;
    }
  }

  function say(line, mood, concept) {
    textEl.textContent = line;
    setMood(mood);
    state.lastSpokeAt = Date.now();
    state.speakingUntil = state.lastSpokeAt + 2200;
    if (mood === "Locked in") {
      state.hypeUntil = state.lastSpokeAt + 1500;
    }
    if (mood === "Sweating") {
      state.bounceUntil = state.lastSpokeAt + 1200;
    }
    assistant.classList.add("is-speaking");
    rememberLine(line, concept);
    persistState();
  }

  function calmIfNeeded() {
    const now = Date.now();
    assistant.classList.toggle("is-sweating", now < state.sweatUntil);
    assistant.classList.toggle("is-hyped", now < state.hypeUntil);
    assistant.classList.toggle("is-bouncing", now < state.bounceUntil);
    assistant.classList.toggle("is-jamming", now < state.jamUntil);
    assistant.classList.toggle("is-jam-visible", now < state.jamVisibleUntil);
    if (now < state.jamVisibleUntil) {
      assistant.classList.add("is-speaking");
    }
    if (!state.bubblePinned && now > state.speakingUntil) {
      if (now >= state.jamVisibleUntil) {
        assistant.classList.remove("is-speaking");
        setMood(now < state.sweatUntil ? "Sweating" : "Watching");
        persistState();
      }
    }
  }

  function viewportBounds() {
    const bubbleWidth = Math.min(260, Math.max(180, bubble.offsetWidth || 220));
    const avatarWidth = avatar.offsetWidth || 84;
    const avatarHeight = avatar.offsetHeight || 84;
    const gap = 12;
    const totalWidth = bubbleWidth + avatarWidth + gap;
    const minX = 12;
    const maxX = Math.max(minX, window.innerWidth - totalWidth - 12);
    const minY = 84;
    const bottomSafePad = 34;
    const maxY = Math.max(minY, window.innerHeight - avatarHeight - bottomSafePad);
    return { minX, maxX, minY, maxY, totalWidth, avatarHeight };
  }

  function cornerTargets() {
    const bounds = viewportBounds();
    const top = bounds.minY;
    const bottom = bounds.maxY;
    return [
      { x: bounds.minX, y: top, side: "left" },
      { x: bounds.maxX, y: top, side: "right" },
      { x: bounds.minX, y: bottom, side: "left" },
      { x: bounds.maxX, y: bottom, side: "right" }
    ];
  }

  function distanceToRect(px, py, rect) {
    if (!rect) return 999999;
    const dx = px < rect.left ? rect.left - px : px > rect.right ? px - rect.right : 0;
    const dy = py < rect.top ? rect.top - py : py > rect.bottom ? py - rect.bottom : 0;
    return Math.hypot(dx, dy);
  }

  function currentAssistantRect(candidate) {
    const bubbleWidth = Math.min(260, Math.max(180, bubble.offsetWidth || 220));
    const avatarWidth = avatar.offsetWidth || 84;
    const avatarHeight = avatar.offsetHeight || 84;
    const gap = 12;
    return {
      left: candidate.x,
      top: candidate.y,
      right: candidate.x + bubbleWidth + avatarWidth + gap,
      bottom: candidate.y + avatarHeight
    };
  }

  function setTargetPosition(x, y, side, maxStep = 110) {
    const dx = x - state.tx;
    const dy = y - state.ty;
    const distance = Math.hypot(dx, dy);
    const ratio = distance > maxStep ? maxStep / distance : 1;

    state.tx += dx * ratio;
    state.ty += dy * ratio;
    if (side) state.side = side;
  }

  function chooseAnchor(targetEl, force = false) {
    if (Date.now() < state.anchorLockedUntil) return;
    if (!targetEl && !force) return;

    const targets = cornerTargets();
    const pointerRect = targetEl?.getBoundingClientRect?.() || null;
    const verticalMid = window.innerHeight * 0.5;
    let verticalZone = state.pointerY < verticalMid ? "top" : "bottom";

    if (state.anchorZone) {
      const [, anchoredVertical] = state.anchorZone.split("-");
      if (Math.abs(state.pointerY - verticalMid) < window.innerHeight * 0.3) verticalZone = anchoredVertical;
    }

    const anchoredSide = "right";
    const preferred = `${anchoredSide}-${verticalZone}`;
    const fallback = `${anchoredSide}-${verticalZone === "top" ? "bottom" : "top"}`;
    const order = [preferred, fallback];
    const map = {
      "left-top": targets[0],
      "right-top": targets[1],
      "left-bottom": targets[2],
      "right-bottom": targets[3]
    };

    let best = map[order[0]];
    for (const zone of order) {
      const candidate = map[zone];
      if (!candidate) continue;
      if (!pointerRect) {
        best = candidate;
        break;
      }

      const rect = currentAssistantRect(candidate);
      const overlaps =
        rect.left < pointerRect.right + 18 &&
        rect.right > pointerRect.left - 18 &&
        rect.top < pointerRect.bottom + 18 &&
        rect.bottom > pointerRect.top - 18;

      if (!overlaps) {
        best = candidate;
        break;
      }
    }

    setTargetPosition(best.x, best.y, "right", force ? 72 : 44);
    state.anchorZone = `${best.side}-${best.y <= viewportBounds().minY + 8 ? "top" : "bottom"}`;
    state.anchorLockedUntil = Date.now() + 2600;
    persistState();
  }

  function applyPosition() {
    const bounds = viewportBounds();
    state.x = clamp(state.x, bounds.minX, bounds.maxX);
    state.y = clamp(state.y, bounds.minY, bounds.maxY);
    assistant.style.transform = `translate3d(${state.x}px, ${state.y}px, 0)`;
    assistant.classList.toggle("is-right", state.side === "right");
    keepAssistantOnscreen();
  }

  function keepAssistantOnscreen() {
    const rect = assistant.getBoundingClientRect();
    const pad = 12;
    let nextX = state.x;
    let nextY = state.y;

    if (rect.right > window.innerWidth - pad) {
      nextX -= rect.right - (window.innerWidth - pad);
    }
    if (rect.left < pad) {
      nextX += pad - rect.left;
    }
    if (rect.bottom > window.innerHeight - pad) {
      nextY -= rect.bottom - (window.innerHeight - pad);
    }
    if (rect.top < pad) {
      nextY += pad - rect.top;
    }

    const bounds = viewportBounds();
    nextX = clamp(nextX, bounds.minX, bounds.maxX);
    nextY = clamp(nextY, bounds.minY, bounds.maxY);

    if (nextX !== state.x || nextY !== state.y) {
      state.x = nextX;
      state.y = nextY;
      state.tx = clamp(state.tx, bounds.minX, bounds.maxX);
      state.ty = clamp(state.ty, bounds.minY, bounds.maxY);
      assistant.style.transform = `translate3d(${state.x}px, ${state.y}px, 0)`;
    }
  }

  function updateEyes() {
    const rect = avatar.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = clamp((state.pointerX - cx) / Math.max(22, rect.width / 2), -1, 1);
    const dy = clamp((state.pointerY - cy) / Math.max(22, rect.height / 2), -1, 1);
    avatar.style.setProperty("--pupil-x", `${(dx * 3.1).toFixed(2)}px`);
    avatar.style.setProperty("--pupil-y", `${(dy * 3.1).toFixed(2)}px`);
  }

  function cleanLabel(text, max) {
    const value = String(text || "").replace(/\s+/g, " ").trim();
    if (!value) return "";
    if (value.length <= (max || 44)) return value;
    return `${value.slice(0, (max || 44) - 1).trim()}...`;
  }

  function pageKey() {
    const parts = window.location.pathname.toLowerCase().split("/").filter(Boolean);
    const productsIndex = parts.indexOf("products");
    if (productsIndex < 0) return "";
    return parts.slice(productsIndex + 1).join("/");
  }

  function currentHashKey() {
    return String(window.location.hash || "").replace(/^#/, "").toLowerCase();
  }

  function currentProductName() {
    return cleanLabel(
      d.getElementById("pName")?.textContent ||
      d.querySelector("h1")?.textContent ||
      d.title,
      48
    );
  }

  function labelFromElement(el) {
    if (!el) return "";

    const variantLabel = el.closest("label, button, a, option");
    if (variantLabel) {
      const named = cleanLabel(variantLabel.getAttribute("aria-label") || variantLabel.textContent, 42);
      if (named) return named;
    }

    const card = el.closest(".card, .item-box, .product-box");
    if (card) {
      return cleanLabel(
        card.querySelector("h1, h2, h3, .title, [data-title]")?.textContent ||
        card.getAttribute("aria-label") ||
        card.textContent,
        42
      );
    }

    return cleanLabel(el.getAttribute?.("aria-label") || el.textContent, 42);
  }

  function topicFromElement(el) {
    if (!el) return "";
    if (el.closest("#addBtn")) return "cart";
    if (el.closest("#buyBtn")) return "buy";
    if (el.closest("#variantWrap")) return "variant";
    if (el.closest(".spinnerCard")) return "wheel";
    if (el.closest(".item-box, .product-box, .card")) return "card";
    return "";
  }

  function reactionFor(label, topic) {
    const text = String(label || currentProductName() || "").trim();
    if (!text) return "";

    const hashKey = currentHashKey();
    if (hashReactions[hashKey]?.[topic]) {
      return hashReactions[hashKey][topic];
    }

    for (const reaction of namedReactions) {
      if (reaction.match.test(text) && reaction.lines[topic]) {
        return reaction.lines[topic];
      }
    }

    const page = pageKey();
    const topPage = page.split("/")[0] || "index";
    if (pageReactions[page]?.[topic]) return pageReactions[page][topic];
    if (pageReactions[topPage]?.[topic]) return pageReactions[topPage][topic];

    if (topic === "card") {
      if (page.startsWith("games")) return `${cleanLabel(text, 32)} feels playable.`;
      if (page.startsWith("books")) return `${cleanLabel(text, 32)} looks like a real pick.`;
      if (page.startsWith("films")) return `${cleanLabel(text, 32)} has some pull.`;
      if (page.startsWith("software")) return `${cleanLabel(text, 32)} might actually be useful.`;
      if (page.startsWith("ai")) return `${cleanLabel(text, 32)} is doing a lot right now.`;
      if (page === "product") return `${cleanLabel(text, 32)} is getting a serious look.`;
    }

    return "";
  }

  function shouldSpeakAboutCard(label) {
    const text = String(label || "").trim();
    if (!text) return false;
    if (reactionFor(text, "card")) return true;
    return namedReactions.some((reaction) => reaction.match.test(text));
  }

  function pickLine(candidates, fallback) {
    const list = (candidates || []).map((line) => String(line || "").trim()).filter(Boolean);
    const fresh = list.filter((line) => !recentlySaid(line));
    return fresh[0] || list[0] || fallback;
  }

  function buildThought(topic, label) {
    const short = cleanLabel(label, 36);
    const concept = `${topic}:${short}`;
    const specific = reactionFor(short, topic);

    if (specific) {
      return {
        concept,
        mood: topic === "cart" ? "Sweating" : topic === "buy" ? "Locked in" : "Watching",
        line: specific
      };
    }

    if (topic === "cart") {
      return {
        concept,
        mood: "Sweating",
        line: pickLine([
          short ? `yes YES. add ${short}.` : "",
          "yes YES. put it in the cart.",
          "yes YES. add it. i am getting a little worked up."
        ], "yes YES. put it in the cart.")
      };
    }

    if (topic === "buy") {
      return {
        concept,
        mood: "Locked in",
        line: pickLine([
          short ? `${short}. straight to checkout. wild behavior.` : "",
          "straight to checkout. dangerous.",
          "bold. i respect that.",
          "no hesitation. good."
        ], "bold. i respect that.")
      };
    }

    if (topic === "variant") {
      return {
        concept,
        mood: "Watching",
        line: pickLine([
          short ? `${short}. keep that one in play.` : "",
          short ? `${short}. that is at least not the weak one.` : "",
          "careful. one of these is clearly better.",
          "good. now pick the strong version."
        ], "good. now pick the strong version.")
      };
    }

    return {
      concept,
      mood: "Watching",
      line: pickLine([
        "the wheel is reckless. continue.",
        "pure chaos. i approve.",
        "spin it. obviously.",
        "this wheel has no business being this tempting."
      ], "pure chaos. i approve.")
    };
  }

  function canSpeak(topic) {
    if (!topic) return false;
    return Date.now() - state.lastSpokeAt > (topic === "card" ? 6200 : 4400);
  }

  function maybeSweat(topic) {
    if (topic === "cart") {
      state.sweatUntil = Date.now() + 1800;
      state.bounceUntil = Date.now() + 1200;
      if (!assistant.classList.contains("is-speaking")) {
        setMood("Sweating");
      }
    }
  }

  function handleMusicState(detail) {
    const page = pageKey();
    if (!page.startsWith("music")) return;

    if (detail?.selecting) {
      state.jamVisibleUntil = Date.now() + 2600;
      assistant.classList.add("is-speaking");
      const title = cleanLabel(detail.selectedTitle || detail.title || "", 40);
      if (title) {
        textEl.textContent = `${title}. loading the groove.`;
      }
      persistState();
    }

    if (detail?.isPlaying) {
      state.jamUntil = Date.now() + 3600000;
      state.jamVisibleUntil = Date.now() + 3600000;
      state.hypeUntil = Date.now() + 1400;
      state.bounceUntil = Date.now() + 1800;
      const title = cleanLabel(detail.selectedTitle || detail.title || "", 40);
      if (title && title !== state.currentJamTitle && Date.now() - state.lastSpokeAt > 5200) {
        state.currentJamTitle = title;
        say(pickLine([
          `${title}. all right. this one jams.`,
          `${title}. yes. i can work with this.`,
          `${title}. acceptable groove.`
        ], `${title}. all right. this one jams.`), "Watching", `music:${title}`);
      } else if (title) {
        state.currentJamTitle = title;
      }
      return;
    }

    if (detail?.paused || detail?.ended) {
      state.jamUntil = 0;
      state.jamVisibleUntil = Date.now() + 1200;
      state.currentJamTitle = "";
      textEl.textContent = detail.paused ? "paused. the groove is on hold." : "all right. track over.";
      persistState();
    }
  }

  function scheduleSpeech(topic, label, el) {
    const key = `${topic}:${label}`;
    if (key === state.hoverKey || key === state.pendingHoverKey) return;

    clearTimeout(state.pendingHoverTimer);
    state.pendingHoverKey = key;
    state.pendingHoverTimer = window.setTimeout(() => {
      state.pendingHoverKey = "";
      state.pendingHoverTimer = 0;
      if (!canSpeak(topic)) return;

      const thought = buildThought(topic, label);
      if (!thought.line) return;
      if (topic === "card" && !shouldSpeakAboutCard(label)) return;
      if (recentlyCovered(thought.concept)) return;

      state.hoverKey = key;
      chooseAnchor(el);
      say(thought.line, thought.mood, thought.concept);
    }, topic === "cart" || topic === "buy" ? 220 : topic === "card" ? 520 : 320);
  }

  function nudgeAwayFromPointer() {
    const rect = assistant.getBoundingClientRect();
    const distance = distanceToRect(state.pointerX, state.pointerY, rect);
    if (distance < 34) {
      state.anchorLockedUntil = 0;
      chooseAnchor(null, true);
    }
  }

  function animate() {
    const bounds = viewportBounds();
    state.tx = clamp(state.tx, bounds.minX, bounds.maxX);
    state.ty = clamp(state.ty, bounds.minY, bounds.maxY);
    state.x += (state.tx - state.x) * 0.028;
    state.y += (state.ty - state.y) * 0.028;
    applyPosition();
    updateEyes();
    calmIfNeeded();
    nudgeAwayFromPointer();
    state.raf = window.requestAnimationFrame(animate);
  }

  d.addEventListener("mousemove", (event) => {
    state.pointerX = event.clientX;
    state.pointerY = event.clientY;
    const topic = topicFromElement(event.target);
    if (topic) {
      const label = labelFromElement(event.target);
      maybeSweat(topic);
      chooseAnchor(event.target);
      scheduleSpeech(topic, label, event.target);
      return;
    }

    clearTimeout(state.pendingHoverTimer);
    state.pendingHoverTimer = 0;
    state.pendingHoverKey = "";
  }, { passive: true });

  d.addEventListener("click", (event) => {
    const topic = topicFromElement(event.target);
    if (!topic) return;

    clearTimeout(state.pendingHoverTimer);
    state.pendingHoverTimer = 0;
    state.pendingHoverKey = "";

    const label = labelFromElement(event.target);
    const thought = buildThought(topic, label);
    maybeSweat(topic);
    chooseAnchor(event.target);

    if (!recentlyCovered(`${thought.concept}:click`)) {
      say(thought.line, thought.mood, `${thought.concept}:click`);
    }
  }, true);

  avatar.addEventListener("click", () => {
    state.bubblePinned = !state.bubblePinned;
    if (state.bubblePinned) {
      say("i only speak when the moment is good.", "On", "manual:on");
    } else {
      assistant.classList.remove("is-speaking");
      setMood(Date.now() < state.sweatUntil ? "Sweating" : "Watching");
      persistState();
    }
    state.anchorLockedUntil = 0;
    chooseAnchor(null, true);
  });

  avatar.addEventListener("dblclick", () => {
    window.location.href = "/products/ai/meatball";
  });

  window.addEventListener("resize", () => {
    state.anchorLockedUntil = 0;
    chooseAnchor(null, true);
  }, { passive: true });

  window.addEventListener("meatball:music-state", (event) => {
    handleMusicState(event.detail || {});
  });

  window.addEventListener("beforeunload", persistState);

  const start = cornerTargets()[1] || { x: 18, y: 92, side: "right" };
  if (!restoreState()) {
    state.x = start.x;
    state.y = start.y;
    state.tx = start.x;
    state.ty = start.y;
    state.side = start.side;
    state.anchorZone = `${start.side}-${start.y <= viewportBounds().minY + 8 ? "top" : "bottom"}`;
  }
  applyPosition();
  animate();
})();
