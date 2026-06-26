const ASSET_BASE = "https://assets.unlim8ted.com";

const MODEL_PATHS = {
  reactionModel: `${ASSET_BASE}/models/meatball_reaction_model/reaction_model.onnx`,
  reactionVocab: `${ASSET_BASE}/models/meatball_reaction_model/input_vocab.json`,
  reactionLabels: `${ASSET_BASE}/models/meatball_reaction_model/labels.json`,

  complexityModel: `${ASSET_BASE}/models/complexity_classifier/complexity_classifier.onnx`,
  complexityVocab: `${ASSET_BASE}/models/complexity_classifier/input_vocab.json`,
  complexityLabels: `${ASSET_BASE}/models/complexity_classifier/labels.json`,

  mathClassifierModel: `${ASSET_BASE}/models/math_classifier/math_classifier.onnx`,
  mathClassifierVocab: `${ASSET_BASE}/models/math_classifier/input_vocab.json`,
  mathClassifierLabels: `${ASSET_BASE}/models/math_classifier/labels.json`,

  subjectFinderModel: `${ASSET_BASE}/models/subject_finder/subject_finder.onnx`,
  subjectFinderConfig: `${ASSET_BASE}/models/subject_finder/config.json`,
  subjectFinderVocab: `${ASSET_BASE}/models/subject_finder/vocab.json`,

  subjectInserterModel: `${ASSET_BASE}/models/subject_inserter/subject_inserter.onnx`,
  subjectInserterConfig: `${ASSET_BASE}/models/subject_inserter/config.json`,
  subjectInserterVocab: `${ASSET_BASE}/models/subject_inserter/vocab.json`,
  subjectInserterLabels: `${ASSET_BASE}/models/subject_inserter/labels.json`,

  generatorModel: `${ASSET_BASE}/models/general_cover_chunks_noisy_continue/model.onnx`,
  generatorConfig: `${ASSET_BASE}/models/general_cover_chunks_noisy_continue/config.json`,
  generatorInputVocab: `${ASSET_BASE}/models/general_cover_chunks_noisy_continue/input_vocab.json`,
  generatorOutputChunks: `${ASSET_BASE}/models/general_cover_chunks_noisy_continue/output_chunks.json`,

  mathTranslatorModel: `${ASSET_BASE}/models/math_equation_translator/math_equation_translator_final.onnx`,
  mathTranslatorInputVocab: `${ASSET_BASE}/models/math_equation_translator/input_vocab.json`,
  mathTranslatorOutputVocab: `${ASSET_BASE}/models/math_equation_translator/output_vocab.json`
};

const DEBUG_ENABLED = new URLSearchParams(location.search).has("debug");
const PAD_ID = 0;
const BOS_ID = 1;
const EOS_ID = 2;
const UNK_ID = 3;
const GENERATOR_INPUT_NGRAMS = [1, 2, 3];
const SUBJECT_FINDER_THRESHOLD = 0.55;
const MATH_ROUTE_THRESHOLD = 0.72;
const REACTION_EMOTION_MAP = {
  neutral: "neutral",
  excited: "excited",
  confused: "confused",
  suspicious: "suspicious",
  angry: "angry",
  sad: "sad",
  overwhelmed: "overwhelmed"
};

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendButton = document.getElementById("sendButton");
const debugPanel = document.getElementById("debugPanel");
const meatballStage = document.getElementById("bigMeatballMount");
const modelMeta = document.getElementById("modelMeta");
const loadingPanel = document.getElementById("loadingPanel");
const loadingBar = document.getElementById("loadingBar");
const loadingBarFill = document.getElementById("loadingBarFill");

const runtimeMemory = {
  history: [],
  subjects: [],
  lastAnswer: "",
  previousReaction: "neutral",
  angryStreak: 0,
  sauceAttackCooldown: 0
};

let models = null;
let modelsLoadingPromise = null;
let loadingUiRemoved = false;

function setModelMeta(text) {
  if (modelMeta) modelMeta.textContent = text;
  if (window.MeatballEmotionRenderer?.setBubble && text) {
    window.MeatballEmotionRenderer.setBubble(meatballStage, `<strong>System.</strong> ${text}`);
  }
}

function setLoadingProgress(value, label, options = {}) {
  if (loadingUiRemoved) return;
  const indeterminate = options.indeterminate === true;
  if (label) setModelMeta(label);
  if (loadingPanel) loadingPanel.hidden = false;
  if (!loadingBar || !loadingBarFill) return;

  loadingBar.hidden = false;
  loadingBar.classList.toggle("is-indeterminate", indeterminate);

  if (indeterminate) {
    loadingBarFill.style.width = "42%";
    return;
  }

  const clamped = Math.max(0, Math.min(1, Number(value) || 0));
  loadingBarFill.style.width = `${Math.round(clamped * 100)}%`;
}

function clearLoadingProgress(label) {
  if (loadingUiRemoved) return;
  if (label) setModelMeta(label);
  if (!loadingBar || !loadingBarFill) return;
  loadingBar.classList.remove("is-indeterminate");
  loadingBarFill.style.width = "100%";
  window.setTimeout(() => {
    loadingBar.hidden = true;
    loadingBarFill.style.width = "0%";
    if (loadingPanel) loadingPanel.hidden = true;
  }, 220);
}

function removeLoadingUi() {
  if (loadingUiRemoved) return;
  loadingUiRemoved = true;
  loadingPanel?.remove();
}

function setDebug(data) {
  if (!DEBUG_ENABLED || !debugPanel) return;
  debugPanel.classList.add("show");
  debugPanel.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

function addMessage(role, text, options = {}) {
  if (!chatLog) return;
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  msg.textContent = text;
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
  if (role === "user") setMeatballEmotion("suspicious");
  if (role === "bot" && !options.skipAnimation) animateMeatballTalk(text, options.emotion);
}

function setMeatballTalking(isTalking) {
  if (window.MeatballEmotionRenderer?.setTalking) {
    window.MeatballEmotionRenderer.setTalking(meatballStage, isTalking);
  }
}

function setMeatballEmotion(emotion, statusHtml) {
  const mapped = REACTION_EMOTION_MAP[emotion] || "neutral";
  if (!window.MeatballEmotionRenderer) return;
  window.MeatballEmotionRenderer.setEmotion(meatballStage, mapped);
  window.MeatballEmotionRenderer.setStatus(
    meatballStage,
    statusHtml || `emotion: ${mapped}<br>talking: false`
  );
}

function animateMeatballTalk(text, emotion) {
  const talkDuration = Math.min(4200, Math.max(900, String(text || "").length * 40));
  const mapped = REACTION_EMOTION_MAP[emotion] || emotion;
  if (mapped) setMeatballEmotion(mapped, `emotion: ${mapped}<br>talking: true`);
  if (window.MeatballEmotionRenderer?.speakFor) {
    window.MeatballEmotionRenderer.speakFor(meatballStage, text, talkDuration, { emotion: mapped });
    return;
  }
  setMeatballTalking(true);
  window.setTimeout(() => setMeatballTalking(false), talkDuration);
}

function sleep(ms) {
  return new Promise(resolve => window.setTimeout(resolve, ms));
}

async function playAnimationPath(animationPath, finalReaction, answer) {
  if (animationPath === "error_glitch") {
    if (window.MeatballEmotionRenderer?.playGlitch) {
      const duration = window.MeatballEmotionRenderer.playGlitch(meatballStage, {
        duration: 2600,
        settleEmotion: "angry",
        bubbleHtml: "<strong>Glitch.</strong> The sauce signal just shredded itself."
      });
      await sleep(duration || 2600);
      return;
    }
    setMeatballEmotion("overwhelmed", "emotion: overwhelmed<br>talking: false");
    await sleep(1400);
    setMeatballEmotion("angry", "emotion: angry<br>talking: false");
    return;
  }

  if (animationPath !== "sad_to_sauce_attack_cutscene") {
    animateMeatballTalk(answer, finalReaction);
    return;
  }
  setMeatballEmotion("angry", "emotion: angry<br>talking: false");
  await sleep(180);
  if (window.MeatballEmotionRenderer?.playCutscene) {
    window.MeatballEmotionRenderer.playCutscene(meatballStage);
  }
  await sleep(5000);
  setMeatballEmotion("angry", "emotion: angry<br>talking: false");
}

function normalize(text) {
  return String(text || "").toLowerCase().replace(/\n/g, " ").replace(/\s+/g, " ").trim();
}

function normalizeNoPunc(text) {
  text = normalize(text);
  text = text.replace(/[!?.,:;"'`()\[\]{}]/g, " ");
  return text.replace(/\s+/g, " ").trim();
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

function classifierFeatures(text, charNgrams = [2, 3, 4, 5], wordNgrams = [1, 2, 3]) {
  const clean = normalize(text);
  const noPunc = normalizeNoPunc(text);
  const features = [];

  const wrapped = `<${clean}>`;
  for (const n of charNgrams) {
    for (let i = 0; i <= wrapped.length - n; i += 1) {
      features.push(`c:${wrapped.slice(i, i + n)}`);
    }
  }

  const words = clean.split(/\s+/).filter(Boolean);
  for (const n of wordNgrams) {
    for (let i = 0; i <= words.length - n; i += 1) {
      features.push(`w:${words.slice(i, i + n).join("_")}`);
    }
  }

  if (!noPunc) features.push("flag:empty");
  if (clean.includes("?")) features.push("flag:question");
  if (clean.includes("!")) features.push("flag:bang");
  if (/\d/.test(clean)) features.push("flag:number");
  if (/[+\-*/=]/.test(clean)) features.push("flag:operator");
  if (/(.)\1\1/.test(clean)) features.push("flag:repeated_chars");
  if (/\b(facts|list|examples|features|types|projects)\b/.test(noPunc)) features.push("flag:list_word");
  if (/\b(compare|contrast|vs|versus|difference|different|better)\b/.test(noPunc)) features.push("flag:compare_word");
  if (/\b(and|also|plus)\b/.test(noPunc)) features.push("flag:connector");
  if (/\b(it|that|this|more|they|them|their)\b/.test(noPunc)) features.push("flag:followup_pronoun");
  if (["hi", "hello", "hey", "yo", "sup", "thanks", "thank you"].includes(noPunc)) features.push("flag:smalltalk_exact");
  if (["what does that mean", "what do you mean", "explain that", "what was that", "tell me more", "more"].includes(noPunc)) {
    features.push("flag:followup_exact");
  }

  return features;
}

function countMap(items) {
  const map = new Map();
  for (const item of items) {
    map.set(item, (map.get(item) || 0) + 1);
  }
  return map;
}

function vectorizeClassifier(text, vocab, charNgrams, wordNgrams) {
  const vec = new Float32Array(Object.keys(vocab || {}).length);
  const counts = countMap(classifierFeatures(text, charNgrams, wordNgrams));
  for (const [feature, count] of counts.entries()) {
    const idx = typeof vocab[feature] === "number" ? vocab[feature] : 0;
    if (idx >= 0 && idx < vec.length) vec[idx] = Math.min(Number(count) || 0, 5);
  }
  return vec;
}

function makeTokenNgrams(tokens, ngrams) {
  const out = [];
  for (const n of ngrams) {
    if (!Number.isFinite(n) || n < 1 || tokens.length < n) continue;
    for (let i = 0; i <= tokens.length - n; i += 1) {
      out.push(tokens.slice(i, i + n).join("_"));
    }
  }
  return out;
}

function vectorFromFeatureVocab(vocab, features) {
  const size = Object.keys(vocab || {}).length;
  const vec = new Float32Array(size);
  const counts = countMap(features);
  for (const [feature, count] of counts.entries()) {
    const idx = vocab[feature];
    if (typeof idx === "number" && idx >= 0 && idx < size) {
      vec[idx] = Math.min(Number(count) || 0, 5);
    }
  }
  return vec;
}

function inputNormalize(text) {
  text = String(text || "").toLowerCase().replace(/\n/g, " ");
  text = text.replace(/[^a-z0-9_!?.,' -]+/g, " ");
  return text.replace(/\s+/g, " ").trim();
}

function inputTokenize(text) {
  return inputNormalize(text).split(/\s+/).filter(Boolean);
}

function vectorizeGeneratorQuestion(question, inputVocab) {
  const feats = makeTokenNgrams(inputTokenize(`question: ${question}`), GENERATOR_INPUT_NGRAMS);
  const vec = new Float32Array(Object.keys(inputVocab || {}).length);
  const counts = countMap(feats);
  const unk = inputVocab["<UNK>"] ?? 1;
  for (const [feat, count] of counts.entries()) {
    const idx = typeof inputVocab[feat] === "number" ? inputVocab[feat] : unk;
    if (idx >= 0 && idx < vec.length) vec[idx] = Math.min(Number(count) || 0, 5);
  }
  return vec;
}

function joinChunkTexts(chunks) {
  let out = "";
  for (const raw of chunks) {
    const chunk = String(raw || "").trim();
    if (!chunk) continue;
    if ([".", ",", "!", "?", ":", ";", "%", ")", "]", "}"].includes(chunk)) {
      out = out.replace(/\s+$/, "") + chunk;
      continue;
    }
    if (["(", "[", "{"].includes(chunk)) {
      if (out && !out.endsWith(" ")) out += " ";
      out += chunk;
      continue;
    }
    if (/^['’]/.test(chunk)) {
      out = out.replace(/\s+$/, "") + chunk;
      continue;
    }
    if (out && !/[ ([{'’]$/.test(out)) out += " ";
    out += chunk;
  }
  return out.replace(/\s+/g, " ").trim();
}

function decodeGeneratorIds(ids, outputChunks) {
  const texts = [];
  for (const rawId of ids) {
    const idx = Number(rawId);
    if (idx === EOS_ID) break;
    if ([PAD_ID, BOS_ID, UNK_ID].includes(idx)) continue;
    if (idx >= 0 && idx < outputChunks.length) texts.push(outputChunks[idx]?.text || "");
  }
  return joinChunkTexts(texts);
}

function softmax(values) {
  const max = Math.max(...values);
  const exp = values.map(value => Math.exp(value - max));
  const sum = exp.reduce((a, b) => a + b, 0) || 1;
  return exp.map(value => value / sum);
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function tokenizeWithSpans(text) {
  const source = String(text || "");
  const regex = /[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*|[^\w\s]/g;
  const tokens = [];
  let match;
  while ((match = regex.exec(source))) {
    tokens.push({
      text: match[0],
      norm: match[0].toLowerCase(),
      start: match.index,
      end: match.index + match[0].length
    });
  }
  return tokens;
}

function normalizeSubjectText(text) {
  return String(text || "")
    .trim()
    .replace(/\s+/g, " ")
    .replace(/^[\"']+|[\"'?.!,;:]+$/g, "")
    .trim();
}

function subjectFinderInputText(message) {
  return `message: ${String(message || "").trim()}`.trim();
}

function encodeSubjectFinderInput(text, vocab, maxLen) {
  const tokens = tokenizeWithSpans(text);
  const ids = new BigInt64Array(maxLen);
  const mask = new Float32Array(maxLen);
  for (let i = 0; i < Math.min(tokens.length, maxLen); i += 1) {
    const token = tokens[i];
    const id = typeof vocab[token.norm] === "number" ? vocab[token.norm] : UNK_ID;
    ids[i] = BigInt(id);
    mask[i] = 1;
  }
  return { tokens, ids, mask };
}

function copySpanText(inputText, tokens, startIdx, endIdx) {
  if (!tokens.length || startIdx < 0 || endIdx < startIdx || startIdx >= tokens.length || endIdx >= tokens.length) {
    return "";
  }
  return inputText.slice(tokens[startIdx].start, tokens[endIdx].end).trim();
}

function sanitizeExtractedSubject(subject) {
  const clean = normalizeSubjectText(subject);
  if (!clean) return "NONE";
  const lower = clean.toLowerCase();
  if (["message", "history", "question", "answer", "none", "it", "this", "that", "they"].includes(lower)) return "NONE";
  return clean;
}

function mapInserterLabelToAction(label, subject, previousSubject) {
  const pronounOps = new Set([
    "replace_it_with_subject",
    "replace_this_with_subject",
    "replace_that_with_subject",
    "replace_they_with_subject",
    "replace_he_she_with_subject",
    "is_subject",
    "does_subject_have",
    "can_subject"
  ]);
  const insertOps = new Set([
    "what_is_subject",
    "what_does_subject_do",
    "where_is_subject",
    "when_was_subject",
    "who_made_subject",
    "append_about_subject",
    "append_for_subject"
  ]);
  if (!label || label === "already_standalone" || label === "no_rewrite") return "keep";
  if (pronounOps.has(label)) return subject !== "NONE" ? "replace_pronoun" : (previousSubject !== "NONE" ? "use_previous_subject" : "keep");
  if (insertOps.has(label)) return subject !== "NONE" ? "insert" : (previousSubject !== "NONE" ? "use_previous_subject" : "keep");
  return previousSubject !== "NONE" && subject === "NONE" ? "use_previous_subject" : "keep";
}

function applyInserterRewrite(text, label, subject, previousSubject) {
  const chosenSubject = subject !== "NONE" ? subject : previousSubject;
  if (!chosenSubject || chosenSubject === "NONE") return text;
  const clean = String(text || "").trim();
  const lower = clean.toLowerCase();

  switch (label) {
    case "replace_it_with_subject":
      return clean.replace(/\bit\b/gi, chosenSubject);
    case "replace_this_with_subject":
      return clean.replace(/\bthis\b/gi, chosenSubject);
    case "replace_that_with_subject":
      return clean.replace(/\bthat\b/gi, chosenSubject);
    case "replace_they_with_subject":
      return clean.replace(/\bthey\b|\bthem\b/gi, chosenSubject);
    case "replace_he_she_with_subject":
      return clean.replace(/\bhe\b|\bshe\b|\bhim\b|\bher\b/gi, chosenSubject);
    case "what_is_subject":
      return `what is ${chosenSubject}`;
    case "what_does_subject_do":
      return `what does ${chosenSubject} do`;
    case "where_is_subject":
      return `where is ${chosenSubject}`;
    case "when_was_subject":
      return `when was ${chosenSubject}`;
    case "who_made_subject":
      return `who made ${chosenSubject}`;
    case "is_subject":
      return lower.startsWith("is ") ? `is ${chosenSubject} ${clean.slice(3).trim()}`.trim() : `is ${chosenSubject} ${clean}`.trim();
    case "does_subject_have":
      return clean.replace(/^does\s+it\b/i, `does ${chosenSubject}`).replace(/^does\s+/i, `does ${chosenSubject} have `).trim();
    case "can_subject":
      return clean.replace(/^can\s+it\b/i, `can ${chosenSubject}`).replace(/^can\s+/i, `can ${chosenSubject} `).trim();
    case "append_about_subject":
      return `${clean} about ${chosenSubject}`.trim();
    case "append_for_subject":
      return `${clean} for ${chosenSubject}`.trim();
    default: {
      const replaced = clean
        .replace(/\bit\b/gi, chosenSubject)
        .replace(/\bthis\b/gi, chosenSubject)
        .replace(/\bthat\b/gi, chosenSubject)
        .replace(/\bthey\b|\bthem\b/gi, chosenSubject);
      return replaced === clean ? `${clean} about ${chosenSubject}`.trim() : replaced;
    }
  }
}

function splitMulti(text) {
  const parts = String(text || "")
    .split(/\s+\band\s+|\s+\balso\s+|\s+\bplus\s+|;/i)
    .map(part => part.trim())
    .filter(Boolean);
  return parts.length > 1 ? parts : [String(text || "").trim()];
}

function formatList(answer) {
  const parts = String(answer || "")
    .split(/(?<=[.!?])\s+/)
    .map(part => part.trim())
    .filter(Boolean);
  if (parts.length <= 1) return parts[0] ? `- ${parts[0]}` : "";
  return parts.slice(0, 8).map(part => `- ${part}`).join("\n");
}

function stripRoutePrefix(text) {
  return String(text || "")
    .trim()
    .replace(/^(please\s+)?(list|facts|examples|features)\s+(out\s+)?/i, "")
    .replace(/^(give me|tell me)\s+(a\s+)?(list of\s+|facts about\s+|examples of\s+|features of\s+)/i, "")
    .trim();
}

function parseCompareSubjects(text) {
  const clean = String(text || "").trim().replace(/\?+$/g, "");
  if (!clean) return [];

  const patterns = [
    /\bcompare\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$/i,
    /\bcomparison\s+of\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$/i,
    /\bdifference\s+between\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$/i,
    /^(.+?)\s+(?:vs|versus)\s+(.+)$/i,
    /^(.+?)\s+and\s+(.+)$/i
  ];

  for (const pattern of patterns) {
    const match = clean.match(pattern);
    if (!match) continue;
    const subjects = [match[1], match[2]]
      .map(part => normalizeSubjectText(part.replace(/^(between|of)\s+/i, "")))
      .filter(Boolean);
    if (subjects.length === 2) return subjects;
  }

  return [];
}

function buildGeneratorQuestionForRoute(rawInput, staged) {
  const base = String(staged.rewrittenQuestion || rawInput || "").trim();
  const normalized = normalizeNoPunc(rawInput);
  const previousSubject = staged.inserter?.previousSubject || "NONE";

  if (staged.complexity.label === "list") {
    const stripped = stripRoutePrefix(base);
    if (/\babout\b/i.test(stripped) || /\bof\b/i.test(stripped)) return stripped;
    if (stripped) return `facts about ${stripped}`;
    return base;
  }

  if ((staged.complexity.label === "followup" || staged.complexity.label === "normal_qa") && staged.subject === "NONE" && previousSubject !== "NONE") {
    if (/\bit\b/i.test(rawInput)) return applyInserterRewrite(rawInput, "replace_it_with_subject", staged.subject, previousSubject);
    if (/\bthis\b/i.test(rawInput)) return applyInserterRewrite(rawInput, "replace_this_with_subject", staged.subject, previousSubject);
    if (/\bthat\b/i.test(rawInput)) return applyInserterRewrite(rawInput, "replace_that_with_subject", staged.subject, previousSubject);
    if (/\bthey\b|\bthem\b|\btheir\b/i.test(rawInput)) return applyInserterRewrite(rawInput, "replace_they_with_subject", staged.subject, previousSubject);
  }

  if (staged.complexity.label === "normal_qa" && previousSubject !== "NONE" && ["what is it", "who is it", "what is this", "what is that"].includes(normalized)) {
    if (normalized.includes("this")) return applyInserterRewrite(rawInput, "replace_this_with_subject", staged.subject, previousSubject);
    if (normalized.includes("that")) return applyInserterRewrite(rawInput, "replace_that_with_subject", staged.subject, previousSubject);
    return applyInserterRewrite(rawInput, "replace_it_with_subject", staged.subject, previousSubject);
  }

  return base;
}

function isLikelyMathQuestion(text) {
  const normalized = mathNormalizeQuestion(text);
  if (!normalized) return false;
  if (/\b(calculate|solve|evaluate|simplify|equation|math|plus|minus|times|divided by|square root|squared|cubed)\b/i.test(text)) return true;
  if (/\d/.test(normalized) && /[+\-*/^=]/.test(normalized)) return true;
  if (/^\s*what is\s+[-(]?\d/.test(normalized)) return true;
  return false;
}

function tryDirectMathAnswer(text) {
  const normalized = mathNormalizeQuestion(text);
  const directExpr = normalized
    .replace(/^(what is|whats|calculate|solve|evaluate|simplify)\s+/i, "")
    .replace(/\?+$/g, "")
    .trim();

  if (!directExpr || !/\d/.test(directExpr) || !/[+\-*/^()]/.test(directExpr)) return "";

  try {
    return String(safeEvalExpression(directExpr));
  } catch (error) {
    return "";
  }
}

function resolveComplexityLabel(rawInput, predictedLabel) {
  const normalized = normalizeNoPunc(rawInput);
  const previousSubject = runtimeMemory.subjects.length ? runtimeMemory.subjects[runtimeMemory.subjects.length - 1] : "NONE";

  if (/\b(compare|contrast|vs|versus|difference|different|better)\b/i.test(rawInput)) return "compare";
  if (/^(list|facts|examples|features)\b/i.test(normalized) || /\b(list|facts|examples|features)\s+(about|of)\b/i.test(normalized)) return "list";
  if (splitMulti(rawInput).length > 1 && /\b(and|also|plus)\b/i.test(rawInput)) return "multi_part";
  if (["what does that mean", "what do you mean", "explain that", "what was that", "tell me more", "more"].includes(normalized)) return "followup";
  if (previousSubject !== "NONE" && /\b(it|this|that|they|them|their)\b/i.test(rawInput)) return "followup";
  return predictedLabel || "unknown";
}

function postProcessAnswer(text) {
  let out = String(text || "");
  out = out.replace(/\s+/g, " ").trim();
  out = out.replace(/\s+([,.;:!?%])/g, "$1");
  out = out.replace(/([(\[{])\s+/g, "$1");
  out = out.replace(/\s+([)\]}])/g, "$1");
  out = out.replace(/([,.;:!?])([A-Za-z0-9])/g, "$1 $2");
  out = out.replace(/\s+'\s*/g, " '");
  out = out.replace(/\bask'([^']+)/gi, "Ask '$1");
  out = out.replace(/\bi\b/g, "I");
  const firstAlpha = out.search(/[A-Za-z]/);
  if (firstAlpha >= 0) out = out.slice(0, firstAlpha) + out[firstAlpha].toUpperCase() + out.slice(firstAlpha + 1);
  out = out.replace(/([.!?]\s+)([a-z])/g, (_, prefix, letter) => `${prefix}${letter.toUpperCase()}`);
  out = out.replace(/ i’m /gi, " I’m ").replace(/ i'm /gi, " I'm ");
  return out.trim();
}

function postProcessAnswerPreserveLines(text) {
  const lines = String(text || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map(line => postProcessAnswer(line))
    .filter(Boolean);
  return lines.join("\n").trim();
}

function applyAnswerOverrides(answer, reaction, animation) {
  let nextAnswer = String(answer || "").trim();
  let nextReaction = reaction;
  let nextAnimation = animation;

  if (nextAnswer === "The Meatball chooses to interpret that as") {
    nextAnswer = "The Meatball chooses to interpret that as completely true.";
  }

  if (nextAnswer === "I'm not" || nextAnswer === "I’m not") {
    nextAnswer = "I'm not quite sure what you mean. It might just be the sauce getting tangled up in itself, but idk...";
    nextReaction = "confused";
    nextAnimation = "confused";
  }

  if (nextAnswer === "Thank you. The sauce accepts the compliment.") {
    nextReaction = "excited";
    nextAnimation = "excited";
  }

  return {
    answer: nextAnswer,
    reaction: nextReaction,
    animation: nextAnimation
  };
}

function mathReplaceSymbols(text) {
  let out = String(text || "");
  for (const [from, to] of Object.entries({
    "×": "*",
    "÷": "/",
    "−": "-",
    "²": " ^ 2 ",
    "³": " ^ 3 ",
    "√": " sqrt "
  })) {
    out = out.split(from).join(to);
  }
  return out;
}

function mathNormalizeQuestion(text) {
  let out = mathReplaceSymbols(text).toLowerCase().replace(/\n/g, " ");
  for (const [pattern, replacement] of [
    [/\btimes\b/g, " * "],
    [/\bplus\b/g, " + "],
    [/\bminus\b/g, " - "],
    [/\bdivided\s+by\b/g, " / "],
    [/\bsquared\b/g, " ^ 2 "],
    [/\bcubed\b/g, " ^ 3 "]
  ]) {
    out = out.replace(pattern, replacement);
  }
  out = out.replace(/[^a-z0-9_+\-*/^().,?:;$%=\s']/g, " ");
  return out.replace(/\s+/g, " ").trim();
}

function mathTokenize(text) {
  return mathReplaceSymbols(String(text || "").toLowerCase()).match(/\d+\.\d+|\d+|\*\*|sqrt|pi|[a-z_]+|[+\-*/^=().,?:;$%]/g) || [];
}

function mathDetok(tokens) {
  let out = "";
  for (const token of tokens) {
    if ([".", ",", "?", "!", ":", ";", "%", ")"].includes(token)) {
      out = out.replace(/\s+$/, "") + token;
    } else if (token === "(") {
      if (out && !out.endsWith(" ")) out += " ";
      out += token;
    } else if (["+", "-", "*", "/", "**", "^", "="].includes(token)) {
      out += ` ${token} `;
    } else {
      if (out && !out.endsWith(" ") && !out.endsWith("(")) out += " ";
      out += token;
    }
  }
  return out.replace(/\s+/g, " ").trim();
}

function mathEncode(tokens, vocab, maxLen) {
  const ids = new BigInt64Array(maxLen);
  ids[0] = BigInt(BOS_ID);
  let cursor = 1;
  for (const token of tokens) {
    if (cursor >= maxLen - 1) break;
    const id = typeof vocab[token] === "number" ? vocab[token] : UNK_ID;
    ids[cursor] = BigInt(id);
    cursor += 1;
  }
  if (cursor < maxLen) ids[cursor] = BigInt(EOS_ID);
  return ids;
}

function mathDecode(ids, idToToken) {
  const tokens = [];
  for (const raw of ids) {
    const idx = Number(raw);
    if (idx === EOS_ID) break;
    if ([PAD_ID, BOS_ID, UNK_ID].includes(idx)) continue;
    tokens.push(idToToken[idx] || "");
  }
  return mathDetok(tokens);
}

function extractEquationAndAnswer(text) {
  const eqMatch = String(text || "").match(/equation\s*:\s*(.*?)(?:\s+answer\s*:|$)/i);
  const ansMatch = String(text || "").match(/answer\s*:\s*(.*)$/i);
  return {
    equation: eqMatch ? eqMatch[1].trim() : "",
    answer: ansMatch ? ansMatch[1].trim() : ""
  };
}

function safeEvalExpression(expr) {
  const source = String(expr || "").trim().replace(/\^/g, "**");
  if (!source || !/^[0-9+\-*/().\s*]+$/.test(source)) throw new Error("unsafe expression");
  let index = 0;

  function skip() {
    while (/\s/.test(source[index] || "")) index += 1;
  }

  function parsePrimary() {
    skip();
    if (source[index] === "(") {
      index += 1;
      const value = parseExpression();
      skip();
      if (source[index] !== ")") throw new Error("missing )");
      index += 1;
      return value;
    }
    const match = source.slice(index).match(/^\d+(\.\d+)?/);
    if (!match) throw new Error("bad number");
    index += match[0].length;
    return Number(match[0]);
  }

  function parseUnary() {
    skip();
    if (source.slice(index, index + 1) === "-") {
      index += 1;
      return -parseUnary();
    }
    return parsePrimary();
  }

  function parsePower() {
    let value = parseUnary();
    skip();
    while (source.slice(index, index + 2) === "**") {
      index += 2;
      value = value ** parseUnary();
      skip();
    }
    return value;
  }

  function parseTerm() {
    let value = parsePower();
    skip();
    while (["*", "/"].includes(source[index])) {
      const op = source[index];
      index += 1;
      const right = parsePower();
      value = op === "*" ? value * right : value / right;
      skip();
    }
    return value;
  }

  function parseExpression() {
    let value = parseTerm();
    skip();
    while (["+", "-"].includes(source[index])) {
      const op = source[index];
      index += 1;
      const right = parseTerm();
      value = op === "+" ? value + right : value - right;
      skip();
    }
    return value;
  }

  const result = parseExpression();
  skip();
  if (index !== source.length) throw new Error("trailing expression");
  return Math.abs(result - Math.round(result)) < 1e-9 ? Math.round(result) : result;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to fetch ${url}`);
  return response.json();
}

async function createSession(url) {
  return ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
}

async function loadGenericClassifier(modelUrl, vocabUrl, labelsUrl, configUrl) {
  const [session, vocab, labels, config] = await Promise.all([
    createSession(modelUrl),
    fetchJson(vocabUrl),
    fetchJson(labelsUrl),
    fetchJson(configUrl)
  ]);
  return { session, vocab, labels, config };
}

async function loadModels() {
  if (models) return models;
  if (modelsLoadingPromise) return modelsLoadingPromise;

  setLoadingProgress(0.04, "Loading the tiny meatball brain.");
  modelsLoadingPromise = (async () => {
    const reaction = await loadGenericClassifier(
      MODEL_PATHS.reactionModel,
      MODEL_PATHS.reactionVocab,
      MODEL_PATHS.reactionLabels,
      `${ASSET_BASE}/models/meatball_reaction_model/config.json`
    );
    setLoadingProgress(0.16, "Reaction model loaded.");

    const complexity = await loadGenericClassifier(
      MODEL_PATHS.complexityModel,
      MODEL_PATHS.complexityVocab,
      MODEL_PATHS.complexityLabels,
      `${ASSET_BASE}/models/complexity_classifier/config.json`
    );
    setLoadingProgress(0.28, "Complexity classifier loaded.");

    const mathClassifier = await loadGenericClassifier(
      MODEL_PATHS.mathClassifierModel,
      MODEL_PATHS.mathClassifierVocab,
      MODEL_PATHS.mathClassifierLabels,
      `${ASSET_BASE}/models/math_classifier/config.json`
    );
    setLoadingProgress(0.4, "Math classifier loaded.");

    const [subjectFinderConfig, subjectFinderVocab, subjectFinderSession] = await Promise.all([
      fetchJson(MODEL_PATHS.subjectFinderConfig),
      fetchJson(MODEL_PATHS.subjectFinderVocab),
      createSession(MODEL_PATHS.subjectFinderModel)
    ]);
    setLoadingProgress(0.55, "Subject finder loaded.");

    const [subjectInserterConfig, subjectInserterVocab, subjectInserterLabels, subjectInserterSession] = await Promise.all([
      fetchJson(MODEL_PATHS.subjectInserterConfig),
      fetchJson(MODEL_PATHS.subjectInserterVocab),
      fetchJson(MODEL_PATHS.subjectInserterLabels),
      createSession(MODEL_PATHS.subjectInserterModel)
    ]);
    setLoadingProgress(0.7, "Subject inserter loaded.");

    const [generatorConfig, generatorInputVocab, generatorOutputChunks, generatorSession] = await Promise.all([
      fetchJson(MODEL_PATHS.generatorConfig),
      fetchJson(MODEL_PATHS.generatorInputVocab),
      fetchJson(MODEL_PATHS.generatorOutputChunks),
      createSession(MODEL_PATHS.generatorModel)
    ]);
    setLoadingProgress(0.86, "Neutral generator loaded.");

    const [mathInputVocab, mathOutputVocab, mathSession] = await Promise.all([
      fetchJson(MODEL_PATHS.mathTranslatorInputVocab),
      fetchJson(MODEL_PATHS.mathTranslatorOutputVocab),
      createSession(MODEL_PATHS.mathTranslatorModel)
    ]);
    setLoadingProgress(0.97, "Math system loaded.");

    const mathIdToToken = {};
    for (const [token, id] of Object.entries(mathOutputVocab || {})) mathIdToToken[Number(id)] = token;

    models = {
      reaction,
      complexity,
      mathClassifier,
      subjectFinder: {
        session: subjectFinderSession,
        config: subjectFinderConfig,
        vocab: subjectFinderVocab
      },
      subjectInserter: {
        session: subjectInserterSession,
        config: subjectInserterConfig,
        vocab: subjectInserterVocab,
        labels: subjectInserterLabels
      },
      generator: {
        session: generatorSession,
        config: generatorConfig,
        inputVocab: generatorInputVocab,
        outputChunks: generatorOutputChunks
      },
      math: {
        session: mathSession,
        inputVocab: mathInputVocab,
        outputVocab: mathOutputVocab,
        idToToken: mathIdToToken,
        config: {
          max_input_len: 96,
          max_output_len: 64
        }
      }
    };
    clearLoadingProgress("Reaction, routing, subject, generator, and math systems are ready.");
    window.setTimeout(() => removeLoadingUi(), 260);
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.placeholder = "Ask Meatball...";
    return models;
  })();

  try {
    return await modelsLoadingPromise;
  } finally {
    if (!models) modelsLoadingPromise = null;
  }
}

async function runGenericClassifier(text, runtime) {
  const config = runtime.config || {};
  const charNgrams = Array.isArray(config.char_ngrams) ? config.char_ngrams : [2, 3, 4, 5];
  const wordNgrams = Array.isArray(config.word_ngrams) ? config.word_ngrams : [1, 2, 3];
  const vec = vectorizeClassifier(text, runtime.vocab, charNgrams, wordNgrams);
  const inputName = runtime.session.inputNames?.[0] || "input";
  const outputs = await runtime.session.run({ [inputName]: new ort.Tensor("float32", vec, [1, vec.length]) });
  const outputName = runtime.session.outputNames?.[0] || Object.keys(outputs)[0];
  const logits = Array.from(outputs[outputName].data);
  const probs = softmax(logits);
  const idx = probs.indexOf(Math.max(...probs));
  return {
    label: runtime.labels[idx],
    confidence: probs[idx] || 0,
    probs: Object.fromEntries(runtime.labels.map((label, i) => [label, probs[i] || 0]))
  };
}

async function runSubjectFinder(rawInput, runtime) {
  const inputText = subjectFinderInputText(rawInput);
  const maxLen = Number(runtime.config?.max_len || 96);
  const encoded = encodeSubjectFinderInput(inputText, runtime.vocab, maxLen);
  const inputNames = runtime.session.inputNames || ["input_ids", "attention_mask"];
  const feeds = {
    [inputNames[0]]: new ort.Tensor("int64", encoded.ids, [1, maxLen]),
    [inputNames[1] || "attention_mask"]: new ort.Tensor("float32", encoded.mask, [1, maxLen])
  };
  const outputs = await runtime.session.run(feeds);
  const outputNames = runtime.session.outputNames || ["has_subject_logits", "start_logits", "end_logits"];
  const hasSubjectLogit = outputs[outputNames[0]]?.data?.[0] ?? 0;
  const startLogits = Array.from(outputs[outputNames[1]]?.data || []);
  const endLogits = Array.from(outputs[outputNames[2]]?.data || []);
  const hasProb = sigmoid(hasSubjectLogit);
  if (hasProb < SUBJECT_FINDER_THRESHOLD || !startLogits.length || !endLogits.length) {
    return { subject: "NONE", confidence: hasProb, raw: { inputText, hasProb } };
  }
  let startIdx = 0;
  let endIdx = 0;
  let bestStart = -Infinity;
  let bestEnd = -Infinity;
  for (let i = 0; i < startLogits.length; i += 1) {
    if (startLogits[i] > bestStart) {
      bestStart = startLogits[i];
      startIdx = i;
    }
    if (endLogits[i] > bestEnd) {
      bestEnd = endLogits[i];
      endIdx = i;
    }
  }
  if (endIdx < startIdx) endIdx = startIdx;
  const extracted = sanitizeExtractedSubject(copySpanText(inputText, encoded.tokens, startIdx, endIdx));
  return {
    subject: extracted || "NONE",
    confidence: hasProb,
    raw: { inputText, hasProb, startIdx, endIdx, extracted }
  };
}

async function runSubjectInserter(rawInput, subject, complexity, runtime) {
  const previousSubject = runtimeMemory.subjects.length ? runtimeMemory.subjects[runtimeMemory.subjects.length - 1] : "NONE";
  const modelInput = [
    `question: ${rawInput}`,
    `subject: ${subject}`,
    `complexity: ${complexity}`,
    `previous_subject: ${previousSubject}`,
    `last_answer: ${runtimeMemory.lastAnswer || "NONE"}`,
    `message: ${rawInput}`,
    `subject_value: ${subject}`
  ].join(" ");
  const maxNgram = Number(runtime.config?.max_ngram || 3);
  const features = makeTokenNgrams(
    tokenizeFeatureText(modelInput),
    Array.from({ length: maxNgram }, (_, idx) => idx + 1)
  );
  const vec = vectorFromFeatureVocab(runtime.vocab, features);
  const inputName = runtime.session.inputNames?.[0] || "input";
  const outputs = await runtime.session.run({ [inputName]: new ort.Tensor("float32", vec, [1, vec.length]) });
  const outputName = runtime.session.outputNames?.[0] || Object.keys(outputs)[0];
  const logits = Array.from(outputs[outputName].data || []);
  const probs = softmax(logits);
  const idx = probs.indexOf(Math.max(...probs));
  const rawLabel = runtime.labels[idx] || "no_rewrite";
  const abstractAction = mapInserterLabelToAction(rawLabel, subject, previousSubject);
  const rewritten = applyInserterRewrite(rawInput, rawLabel, subject, previousSubject);
  return {
    action: abstractAction,
    rawLabel,
    confidence: probs[idx] || 0,
    previousSubject,
    rewritten,
    modelInput
  };
}

async function runGenerator(question, runtime) {
  const vec = vectorizeGeneratorQuestion(question, runtime.inputVocab);
  const inputName = runtime.session.inputNames?.[0] || "input";
  const outputs = await runtime.session.run({ [inputName]: new ort.Tensor("float32", vec, [1, vec.length]) });
  const outputName = runtime.session.outputNames?.[0] || Object.keys(outputs)[0];
  const data = outputs[outputName]?.data;
  const dims = outputs[outputName]?.dims || [];
  if (!data || dims.length < 2) return { text: "" };
  const stepCount = dims.length >= 3 ? dims[1] : dims[0];
  const vocabSize = dims.length >= 3 ? dims[2] : dims[1];
  const ids = [];
  for (let step = 0; step < Math.min(stepCount, Number(runtime.config?.max_output_chunks || 24) + 1); step += 1) {
    let bestId = 0;
    let bestLogit = -Infinity;
    for (let token = 0; token < vocabSize; token += 1) {
      const value = data[(step * vocabSize) + token];
      if (value > bestLogit) {
        bestLogit = value;
        bestId = token;
      }
    }
    if ([PAD_ID, BOS_ID, UNK_ID].includes(bestId) || bestId === EOS_ID) break;
    ids.push(bestId);
  }
  return { text: decodeGeneratorIds([...ids, EOS_ID], runtime.outputChunks), ids };
}

async function runMathSystem(question, runtime) {
  const normalized = mathNormalizeQuestion(question);
  const encoded = mathEncode(mathTokenize(normalized), runtime.inputVocab, runtime.config.max_input_len);
  const inputName = runtime.session.inputNames?.[0] || "input_ids";
  const outputs = await runtime.session.run({ [inputName]: new ort.Tensor("int64", encoded, [1, runtime.config.max_input_len]) });
  const outputName = runtime.session.outputNames?.[0] || Object.keys(outputs)[0];
  const data = outputs[outputName]?.data;
  const dims = outputs[outputName]?.dims || [];
  const ids = [];
  if (data && dims.length >= 2) {
    const stepCount = dims.length >= 3 ? dims[1] : dims[0];
    const vocabSize = dims.length >= 3 ? dims[2] : dims[1];
    for (let step = 0; step < Math.min(stepCount, runtime.config.max_output_len); step += 1) {
      let bestId = 0;
      let bestLogit = -Infinity;
      for (let token = 0; token < vocabSize; token += 1) {
        const value = data[(step * vocabSize) + token];
        if (value > bestLogit) {
          bestLogit = value;
          bestId = token;
        }
      }
      ids.push(bestId);
      if (bestId === EOS_ID) break;
    }
  }
  const decoded = mathDecode(ids, runtime.idToToken);
  const parsed = extractEquationAndAnswer(decoded);
  let computed = "";
  if (parsed.equation) {
    try {
      computed = String(safeEvalExpression(parsed.equation));
    } catch (error) {
      computed = "";
    }
  }
  const final = computed || parsed.answer || decoded || "The sauce blinked at the numbers and could not plate an answer.";
  return {
    normalized,
    decoded,
    equation: parsed.equation,
    predictedAnswer: parsed.answer,
    computedAnswer: computed,
    final
  };
}

function updateMemory(userText, answer, reaction, subject, options = {}) {
  runtimeMemory.history.push({ role: "user", text: userText });
  runtimeMemory.history.push({ role: "bot", text: answer });
  runtimeMemory.history = runtimeMemory.history.slice(-8);

  if (subject && subject !== "NONE") {
    runtimeMemory.subjects.push(subject);
    runtimeMemory.subjects = runtimeMemory.subjects.slice(-5);
  }

  runtimeMemory.lastAnswer = answer;
  runtimeMemory.previousReaction = reaction;

  if (options.preserveAngryState) return;

  if (reaction === "angry") runtimeMemory.angryStreak += 1;
  else runtimeMemory.angryStreak = 0;
}

function explainLastAnswer() {
  if (!runtimeMemory.lastAnswer) return "The sauce blinked twice. I need a clearer question.";
  return `That means: ${runtimeMemory.lastAnswer}`;
}

async function routeRequest(rawInput, staged) {
  const normalized = normalizeNoPunc(rawInput);
  const generatorQuestion = buildGeneratorQuestionForRoute(rawInput, staged);
  const route = {
    route: "normal_qa",
    answer: "",
    animation: staged.reaction.label,
    animationPath: "",
    rewrittenQuestion: generatorQuestion
  };

  if (normalized === "ok" || normalized === "okay") {
    route.route = "smalltalk";
    route.answer = "Yep.";
    route.animation = "neutral";
    return route;
  }

  if (normalized === "yep") {
    route.route = "smalltalk";
    route.answer = "Yes.";
    route.animation = "neutral";
    return route;
  }

  if (staged.reaction.label === "angry" && runtimeMemory.angryStreak >= 1 && runtimeMemory.sauceAttackCooldown <= 0) {
    route.route = "anger_escalation_attack";
    route.answer = "YOU DONT LIKE ME??? THEN FACE THE SAUCE.";
    route.animation = "angry";
    route.animationPath = "sad_to_sauce_attack_cutscene";
    runtimeMemory.sauceAttackCooldown = 15;
    runtimeMemory.angryStreak = 0;
    runtimeMemory.previousReaction = "angry";
    return route;
  }

  if ((staged.math.label === "math" && staged.math.confidence >= MATH_ROUTE_THRESHOLD) || isLikelyMathQuestion(rawInput)) {
    const directAnswer = tryDirectMathAnswer(staged.rewrittenQuestion);
    const math = directAnswer
      ? {
          normalized: mathNormalizeQuestion(staged.rewrittenQuestion),
          decoded: "",
          equation: "",
          predictedAnswer: "",
          computedAnswer: directAnswer,
          final: directAnswer
        }
      : await runMathSystem(staged.rewrittenQuestion, models.math);
    route.route = "math";
    route.answer = math.final;
    route.math = math;
    return route;
  }

  if (staged.complexity.label === "unknown") {
    route.route = "unknown";
    route.answer = "The sauce blinked twice. I need a clearer question.";
    return route;
  }

  if (staged.complexity.label === "compare") {
    route.route = "compare";
    const compareSubjects = parseCompareSubjects(rawInput);
    if (compareSubjects.length === 2) {
      const left = await runGenerator(`what is ${compareSubjects[0]}`, models.generator);
      const right = await runGenerator(`what is ${compareSubjects[1]}`, models.generator);
      const leftText = left.text || "";
      const rightText = right.text || "";
      route.answer = [leftText, rightText].filter(Boolean).join(" ");
      if (route.answer) return route;
    }
    route.answer = "Comparing two things at once might make this tiny meatball brain explode.";
    return route;
  }

  if (staged.complexity.label === "smalltalk") {
    route.route = "smalltalk";
    const generated = await runGenerator(generatorQuestion, models.generator);
    route.answer = generated.text || "The sauce is on, but I need a sharper question.";
    return route;
  }

  if (staged.complexity.label === "followup" && ["what does that mean", "what do you mean", "explain that", "what was that"].includes(normalized)) {
    route.route = "followup_explain_last_answer";
    route.answer = explainLastAnswer();
    return route;
  }

  if (staged.complexity.label === "multi_part") {
    route.route = "multi_part";
    const parts = splitMulti(generatorQuestion);
    const answers = [];
    for (const part of parts) {
      const generated = await runGenerator(part, models.generator);
      answers.push(generated.text || "The sauce stalled on one part of that question.");
    }
    route.answer = answers.join(" ");
    return route;
  }

  const generated = await runGenerator(generatorQuestion, models.generator);
  route.answer = generated.text || "The sauce is on, but I need a sharper question.";
  if (staged.complexity.label === "list") {
    route.route = "list";
    route.answer = formatList(route.answer);
    return route;
  }

  if (staged.complexity.label === "followup") {
    route.route = "followup";
    return route;
  }

  route.route = "normal_qa";
  return route;
}

async function processUserMessage(rawInput) {
  const loaded = await loadModels();
  setLoadingProgress(0.08, "Reading the plate.");
  if (runtimeMemory.sauceAttackCooldown > 0) runtimeMemory.sauceAttackCooldown -= 1;

  const reaction = await runGenericClassifier(rawInput, loaded.reaction);
  setLoadingProgress(0.2, "Reading reaction.");
  const complexityPrediction = await runGenericClassifier(rawInput, loaded.complexity);
  setLoadingProgress(0.32, "Classifying question shape.");
  const complexity = {
    ...complexityPrediction,
    modelLabel: complexityPrediction.label,
    label: resolveComplexityLabel(rawInput, complexityPrediction.label)
  };
  const mathPrediction = await runGenericClassifier(rawInput, loaded.mathClassifier);
  setLoadingProgress(0.44, "Checking for math routing.");
  const math = {
    ...mathPrediction,
    modelLabel: mathPrediction.label,
    heuristicMatch: isLikelyMathQuestion(rawInput)
  };
  const finder = await runSubjectFinder(rawInput, loaded.subjectFinder);
  setLoadingProgress(0.58, "Finding the subject.");
  const inserter = await runSubjectInserter(rawInput, finder.subject, complexity.label, loaded.subjectInserter);
  setLoadingProgress(0.72, "Rewriting follow-up context.");

  const rewrittenQuestion = inserter.action === "keep" ? rawInput : inserter.rewritten;

  const staged = {
    reaction,
    complexity,
    math,
    subject: finder.subject,
    finder,
    inserter,
    rewrittenQuestion
  };

  const routed = await routeRequest(rawInput, staged);
  setLoadingProgress(0.92, "Plating the answer.");
  const finalAnswer = postProcessAnswerPreserveLines(routed.answer);
  const overridden = applyAnswerOverrides(
    finalAnswer,
    reaction.label,
    routed.animation || reaction.label
  );
  setLoadingProgress(1, "Answer ready.");

  updateMemory(rawInput, overridden.answer, overridden.reaction, finder.subject, {
    preserveAngryState: routed.route === "anger_escalation_attack"
  });

  return {
    answer: overridden.answer,
    reaction: overridden.reaction,
    animation: overridden.animation,
    animationPath: routed.animationPath || "",
    route: routed.route,
    cooldown: runtimeMemory.sauceAttackCooldown,
    debug: {
      rawInput,
      reaction,
      complexity,
      math,
      subjectFinder: finder,
      subjectInserter: inserter,
      rewrittenQuestion,
      route: routed.route,
      memory: {
        history: runtimeMemory.history,
        subjects: runtimeMemory.subjects,
        lastAnswer: runtimeMemory.lastAnswer,
        previousReaction: runtimeMemory.previousReaction,
        angryStreak: runtimeMemory.angryStreak,
        sauceAttackCooldown: runtimeMemory.sauceAttackCooldown
      },
      mathRuntime: routed.math || null
    }
  };
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
    const result = await processUserMessage(text);
    addMessage("bot", result.answer, { skipAnimation: true, emotion: result.reaction });
    if (result.animationPath) await playAnimationPath(result.animationPath, result.reaction, result.answer);
    else animateMeatballTalk(result.answer, result.reaction);
    clearLoadingProgress(`Reaction: ${result.reaction}. Route: ${result.route}. Subjects: ${runtimeMemory.subjects.join(" | ") || "NONE"}. Cooldown: ${runtimeMemory.sauceAttackCooldown}.`);
    setDebug(result.debug);
  } catch (error) {
    addMessage("bot", "The sauce jammed. Ask again with a cleaner plate.", { skipAnimation: true, emotion: "angry" });
    await playAnimationPath("error_glitch", "angry", "The sauce jammed. Ask again with a cleaner plate.");
    clearLoadingProgress("The runtime glitched while plating that answer.");
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

window.addEventListener("DOMContentLoaded", () => {
  chatInput.disabled = false;
  sendButton.disabled = false;
  chatInput.placeholder = "Ask Meatball...";
  clearLoadingProgress("Models stay asleep until you talk to Meatball.");
  setMeatballEmotion("neutral");
  const initialBotMessage = chatLog?.querySelector(".msg.bot");
  if (initialBotMessage) initialBotMessage.remove();
  addMessage("bot", "I am here. Ask me something and I will wake the sauce.", { skipAnimation: true });

  requestAnimationFrame(() => {
    chatInput.blur();
    document.activeElement?.blur?.();
  });
});
