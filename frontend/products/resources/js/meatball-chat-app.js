const SPECIALIZED_TOPICS = [
  "ai", "animals", "art", "biology", "birds", "books", "chemistry", "countries",
  "earth_science", "food", "games", "general", "health_basics", "history",
  "insects", "math", "movies", "music", "objects", "physics", "plants",
  "space", "sports", "technology", "unlim8ted", "vehicles"
];

const MODEL_PATHS = {
  selectorModel: "/assets/models/specialized_meatball_chunks/selector.onnx",
  selectorConfig: "/assets/models/specialized_meatball_chunks/selector_config.json",
  selectorLabels: "/assets/models/specialized_meatball_chunks/selector_labels.json",
  selectorVocab: "/assets/models/specialized_meatball_chunks/selector_vocab.json",
  subjectInserterModel: "/assets/models/subject_inserter/subject_inserter.onnx",
  subjectInserterConfig: "/assets/models/subject_inserter/config.json",
  subjectInserterLabels: "/assets/models/subject_inserter/labels.json",
  subjectInserterVocab: "/assets/models/subject_inserter/vocab.json",
  subjectFinderConfig: "/assets/models/subject_finder/config.json",
  subjectFinderVocab: "/assets/models/subject_finder/vocab.json",
  specializedDataBase: "/assets/data/specialized_QA"
};

const DEBUG_ENABLED = new URLSearchParams(location.search).has("debug");

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendButton = document.getElementById("sendButton");
const debugPanel = document.getElementById("debugPanel");
const bigMeatballAvatar = document.getElementById("bigMeatballAvatar");
const modelMeta = document.getElementById("modelMeta");

let selectorModel = null;
let subjectRewriterModel = null;
let subjectFinderConfig = null;
let subjectFinderVocab = null;
let modelsLoaded = false;
let modelsLoadingPromise = null;

let answerBankCache = new Map();
let historyTurns = [];
let lastSubject = "";
let lastTopic = "general";

function setModelMeta(text) {
  if (modelMeta) modelMeta.textContent = text;
}

function setDebug(data) {
  if (!DEBUG_ENABLED || !debugPanel) return;
  debugPanel.classList.add("show");
  debugPanel.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

function addMessage(role, text) {
  if (!chatLog) return;
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  msg.textContent = text;
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
  if (role === "bot") animateMeatballTalk(text);
}

function setMeatballTalking(isTalking) {
  if (window.MeatballAvatarEmbed?.setTalking) {
    window.MeatballAvatarEmbed.setTalking(bigMeatballAvatar, isTalking);
    return;
  }
  if (bigMeatballAvatar) bigMeatballAvatar.classList.toggle("talking", Boolean(isTalking));
}

function animateMeatballTalk(text) {
  const talkDuration = Math.min(4200, Math.max(900, String(text || "").length * 42));
  if (window.MeatballAvatarEmbed?.speakFor) {
    window.MeatballAvatarEmbed.speakFor(bigMeatballAvatar, text, talkDuration);
    return;
  }
  setMeatballTalking(true);
  window.setTimeout(() => setMeatballTalking(false), talkDuration);
}

function normalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^\w\s?!.,:'"/&()-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeFeatureText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9_'?!.:,/ -]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenizeFeatureText(text) {
  return normalizeFeatureText(text).match(/[a-z0-9_']+|[?!.:,/]/g) || [];
}

function tokenizeSelectorText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/\n/g, " ")
    .replace(/[^a-z0-9_!?.,' -]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

function fnv1aHash(str) {
  let hash = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(value => Math.exp(value - max));
  const sum = exp.reduce((a, b) => a + b, 0) || 1;
  return exp.map(value => value / sum);
}

function makeTokenNgrams(tokens, ngrams) {
  const output = [];
  for (const n of ngrams) {
    if (!Number.isFinite(n) || n < 1 || tokens.length < n) continue;
    for (let i = 0; i <= tokens.length - n; i += 1) {
      output.push(tokens.slice(i, i + n).join("_"));
    }
  }
  return output;
}

function vectorFromFeatureVocab(vocab, features) {
  const size = Object.keys(vocab || {}).length;
  const vec = new Float32Array(size);
  for (const feature of features) {
    const index = vocab[feature];
    if (typeof index === "number" && index >= 0 && index < size) vec[index] += 1;
  }
  let norm = 0;
  for (let i = 0; i < vec.length; i += 1) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < vec.length; i += 1) vec[i] /= norm;
  return vec;
}

function labelsFromIndexObject(raw) {
  return Object.entries(raw || {})
    .map(([key, value]) => [Number(key), String(value)])
    .filter(([key]) => Number.isFinite(key))
    .sort((a, b) => a[0] - b[0])
    .map(([, value]) => value);
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to fetch ${url}`);
  return response.json();
}

async function fetchText(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to fetch ${url}`);
  return response.text();
}

async function createSession(url) {
  return ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
}

async function runSessionWithFirstInput(session, tensor) {
  const inputName = session.inputNames?.[0] || "input";
  return session.run({ [inputName]: tensor });
}

function parseAnswerBank(rawText, url = "") {
  const trimmed = String(rawText || "").trim();
  if (!trimmed) return [];
  if (url.endsWith(".json")) {
    const parsed = JSON.parse(trimmed);
    return Array.isArray(parsed) ? parsed : (Array.isArray(parsed.rows) ? parsed.rows : []);
  }
  return trimmed
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

function normalizeRows(rows) {
  return rows.map((row, index) => ({
    id: row.id || `answer_${index}`,
    question: row.question || row.input || "",
    answer: row.answer || row.response || row.text || "",
    category: row.category || row.topic || "general",
    tags: Array.isArray(row.tags) ? row.tags : []
  }));
}

async function loadTopicRows(topic) {
  const safeTopic = SPECIALIZED_TOPICS.includes(topic) ? topic : "general";
  if (answerBankCache.has(safeTopic)) return answerBankCache.get(safeTopic);
  const url = `${MODEL_PATHS.specializedDataBase}/${safeTopic}.jsonl`;
  const rows = normalizeRows(parseAnswerBank(await fetchText(url), url));
  answerBankCache.set(safeTopic, rows);
  return rows;
}

function lexicalScore(question, row, subject) {
  const q = normalizeText(question);
  const rowText = normalizeText(`${row.question} ${row.answer} ${row.category} ${(row.tags || []).join(" ")}`);
  if (!q || !rowText) return -1;
  let score = 0;
  if (normalizeText(row.question) === q) score += 80;
  const words = q.split(/\s+/).filter(word => word.length > 2);
  for (const word of words) {
    if (rowText.includes(word)) score += 3;
  }
  if (subject && rowText.includes(subject)) score += 9;
  score += Math.min(row.answer.length / 220, 3);
  return score;
}

function pickBestRow(question, rows, subject) {
  let best = null;
  let bestScore = -Infinity;
  for (const row of rows) {
    const score = lexicalScore(question, row, subject);
    if (score > bestScore) {
      best = row;
      bestScore = score;
    }
  }
  return { row: best, score: bestScore };
}

function getFallbackAnswer(topic, subject) {
  if (topic === "unlim8ted") {
    return subject
      ? `The sauce is on: ${subject} is in the Unlim8ted lane, but I need a cleaner question to plate the right answer.`
      : "The sauce is on: ask a sharper Unlim8ted question and I will plate a cleaner answer.";
  }
  return subject
    ? `The sauce is on, but "${subject}" needs a more direct question before I plate the answer.`
    : "The sauce is on, but that question needs a clearer subject before I plate the answer.";
}

function getHistoryText() {
  return historyTurns
    .slice(-6)
    .map(turn => `${turn.role}: ${turn.text}`)
    .join(" ");
}

function extractCandidatePhrases(text) {
  const clean = normalizeText(text);
  if (!clean) return [];
  const patterns = [
    /\b(?:about|on|for|regarding|with|of)\s+([a-z0-9][a-z0-9\s'-]{1,48})/g,
    /\b(?:what is|who is|tell me about|explain|describe)\s+([a-z0-9][a-z0-9\s'-]{1,48})/g,
    /\b([a-z0-9][a-z0-9\s'-]{1,48})\s+(?:sell|make|do|work|mean|is|are|was|were|has|have)\b/g
  ];
  const found = [];
  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(clean))) {
      found.push(match[1].trim());
    }
  }
  return found;
}

function sanitizeSubject(candidate) {
  const stop = new Set([
    "it", "this", "that", "they", "them", "he", "she", "him", "her", "these", "those",
    "something", "anything", "everything", "nothing", "question", "answer", "thing", "stuff"
  ]);
  const words = normalizeText(candidate)
    .split(/\s+/)
    .filter(Boolean)
    .filter(word => !stop.has(word));
  if (!words.length) return "";
  return words.slice(0, 4).join(" ");
}

function findSubject(historyText, inputText) {
  const combined = `${historyText} ${inputText}`.trim();
  const candidates = [
    ...extractCandidatePhrases(inputText),
    ...extractCandidatePhrases(historyText),
    lastSubject
  ].map(sanitizeSubject).filter(Boolean);
  if (candidates.length) return candidates[0];

  const vocabWords = new Set(Object.keys(subjectFinderVocab || {}));
  const tokens = normalizeText(combined).split(/\s+/).filter(Boolean);
  const filtered = tokens.filter(token => token.length > 2 && vocabWords.has(token));
  if (filtered.length) return filtered.slice(-3).join(" ");
  return "";
}

async function predictRewriteAction(inputText, subject) {
  const inputTextNormalized = normalizeFeatureText(inputText);
  const subjectNormalized = normalizeFeatureText(subject);
  const maxNgram = Number(subjectRewriterModel.config?.max_ngram || 3);
  const inputForModel = `message: ${inputTextNormalized} subject: ${subjectNormalized}`;
  const tokens = tokenizeFeatureText(inputForModel);
  const vec = vectorFromFeatureVocab(
    subjectRewriterModel.vocab,
    makeTokenNgrams(tokens, Array.from({ length: maxNgram }, (_, index) => index + 1))
  );
  const tensor = new ort.Tensor("float32", vec, [1, vec.length]);
  const outputs = await runSessionWithFirstInput(subjectRewriterModel.session, tensor);
  const firstKey = Object.keys(outputs)[0];
  const logits = Array.from(outputs[firstKey].data);
  const probs = softmax(logits);
  const top = probs
    .map((score, index) => ({
      action: subjectRewriterModel.labels[index] || String(index),
      score
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
  return {
    action: top[0]?.action || "no_rewrite",
    confidence: top[0]?.score || 0,
    top
  };
}

function rewriteStandalone(inputText, subject, action) {
  const clean = String(inputText || "").trim();
  if (!clean || !subject) return clean;
  const lower = clean.toLowerCase();
  switch (action) {
    case "replace_it_with_subject":
      return clean.replace(/\bit\b/gi, subject);
    case "replace_this_with_subject":
      return clean.replace(/\bthis\b/gi, subject);
    case "replace_that_with_subject":
      return clean.replace(/\bthat\b/gi, subject);
    case "replace_they_with_subject":
      return clean.replace(/\bthey\b/gi, subject);
    case "replace_he_she_with_subject":
      return clean.replace(/\bhe\b/gi, subject).replace(/\bshe\b/gi, subject);
    case "what_is_subject":
      return `what is ${subject}`;
    case "what_does_subject_do":
      return `what does ${subject} do`;
    case "where_is_subject":
      return `where is ${subject}`;
    case "when_was_subject":
      return `when was ${subject}`;
    case "who_made_subject":
      return `who made ${subject}`;
    case "is_subject":
      return lower.startsWith("is ") ? `is ${subject} ${clean.slice(3).trim()}` : `is ${subject} ${clean}`.trim();
    case "does_subject_have":
      return `does ${subject} have ${clean.replace(/^does\s+/i, "")}`.trim();
    case "can_subject":
      return `can ${subject} ${clean.replace(/^can\s+/i, "")}`.trim();
    case "append_about_subject":
      return `${clean} about ${subject}`.trim();
    case "append_for_subject":
      return `${clean} for ${subject}`.trim();
    case "already_standalone":
    case "no_rewrite":
    default:
      return clean;
  }
}

async function predictTopic(question) {
  const config = selectorModel.config || {};
  const tokens = tokenizeSelectorText(`question: ${question} history:`);
  const ngrams = Array.isArray(config.input_ngrams) && config.input_ngrams.length ? config.input_ngrams : [1, 2, 3];
  const vec = vectorFromFeatureVocab(selectorModel.vocab, makeTokenNgrams(tokens, ngrams));
  const tensor = new ort.Tensor("float32", vec, [1, vec.length]);
  const outputs = await runSessionWithFirstInput(selectorModel.session, tensor);
  const firstKey = Object.keys(outputs)[0];
  const logits = Array.from(outputs[firstKey].data);
  const probs = softmax(logits);
  const top = probs
    .map((score, index) => ({
      topic: selectorModel.labels[index] || "general",
      score
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);
  return {
    topic: top[0]?.topic || "general",
    confidence: top[0]?.score || 0,
    top
  };
}

async function answerQuestion(inputText) {
  const historyText = getHistoryText();
  const subject = findSubject(historyText, inputText);
  const rewritePrediction = await predictRewriteAction(inputText, subject);
  const standaloneInput = rewriteStandalone(inputText, subject, rewritePrediction.action);
  const topicPrediction = await predictTopic(standaloneInput);
  const rows = await loadTopicRows(topicPrediction.topic);
  const picked = pickBestRow(standaloneInput, rows, subject);

  lastSubject = subject || lastSubject;
  lastTopic = topicPrediction.topic || lastTopic;

  const answer = picked.row && picked.score >= 6
    ? picked.row.answer
    : getFallbackAnswer(topicPrediction.topic, subject);

  historyTurns.push({ role: "user", text: inputText });
  historyTurns.push({ role: "bot", text: answer });
  historyTurns = historyTurns.slice(-12);

  setModelMeta(`Finder: ${subject || "none"}. Rewrite: ${standaloneInput}. Topic: ${topicPrediction.topic}.`);

  return {
    text: answer,
    debug: {
      historyText,
      subject,
      rewritePrediction,
      standaloneInput,
      topicPrediction,
      pickedRow: picked.row ? {
        id: picked.row.id,
        question: picked.row.question,
        category: picked.row.category,
        score: picked.score
      } : null
    }
  };
}

async function loadModels() {
  if (modelsLoaded) return;
  if (modelsLoadingPromise) return modelsLoadingPromise;

  setModelMeta("Loading finder, rewriter, selector, and topic banks.");
  modelsLoadingPromise = (async () => {
    const [
      selectorConfig,
      selectorLabelsRaw,
      selectorVocab,
      subjectInserterConfig,
      subjectInserterLabels,
      subjectInserterVocab,
      finderConfig,
      finderVocab,
      selectorSession,
      subjectInserterSession
    ] = await Promise.all([
      fetchJson(MODEL_PATHS.selectorConfig),
      fetchJson(MODEL_PATHS.selectorLabels),
      fetchJson(MODEL_PATHS.selectorVocab),
      fetchJson(MODEL_PATHS.subjectInserterConfig),
      fetchJson(MODEL_PATHS.subjectInserterLabels),
      fetchJson(MODEL_PATHS.subjectInserterVocab),
      fetchJson(MODEL_PATHS.subjectFinderConfig),
      fetchJson(MODEL_PATHS.subjectFinderVocab),
      createSession(MODEL_PATHS.selectorModel),
      createSession(MODEL_PATHS.subjectInserterModel)
    ]);

    selectorModel = {
      session: selectorSession,
      config: selectorConfig,
      labels: labelsFromIndexObject(selectorLabelsRaw),
      vocab: selectorVocab
    };

    subjectRewriterModel = {
      session: subjectInserterSession,
      config: subjectInserterConfig,
      labels: Array.isArray(subjectInserterLabels) ? subjectInserterLabels : labelsFromIndexObject(subjectInserterLabels),
      vocab: subjectInserterVocab
    };

    subjectFinderConfig = finderConfig;
    subjectFinderVocab = finderVocab;
    modelsLoaded = true;
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.placeholder = "Ask Meatball...";
    setModelMeta("Finder, rewriter, selector, and topic banks are ready.");
  })();

  try {
    await modelsLoadingPromise;
  } finally {
    if (!modelsLoaded) modelsLoadingPromise = null;
  }
}

chatForm?.addEventListener("submit", async event => {
  event.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  chatInput.value = "";
  chatInput.disabled = true;
  sendButton.disabled = true;
  addMessage("user", text);

  try {
    await loadModels();
    const result = await answerQuestion(text);
    addMessage("bot", result.text);
    setDebug(result.debug);
  } catch (error) {
    addMessage("bot", "The sauce jammed. Ask again with a cleaner plate.");
    setDebug({ error: String(error?.message || error) });
  } finally {
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.focus();
  }
});

chatInput?.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

window.addEventListener("DOMContentLoaded", async () => {
  chatInput.disabled = false;
  sendButton.disabled = false;
  chatInput.placeholder = "Ask Meatball...";
  setModelMeta("Models stay asleep until you talk to Meatball.");
  if (chatLog) chatLog.innerHTML = "";
  addMessage("bot", "I am here. Ask me something and I will wake the sauce.");
});
