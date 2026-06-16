// browser_meatball_hybrid.js
// Requires:
// <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>

function mbNormalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[^a-z0-9?!.,'"\s:_/\-|=+#@&()%]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function mbBuildControllerInput(userText, memory = {}) {
  return (
    "last_intent=" + (memory.last_intent || "none") + " " +
    "last_project=" + (memory.last_project_key || "none") + " " +
    "last_category=" + (memory.last_category || "none") + " " +
    "user: " + mbNormalizeText(userText)
  );
}

function mbEncodeFixed(text, charToId, maxLen) {
  const unk = charToId["<UNK>"];
  const pad = charToId["<PAD>"];
  const ids = new BigInt64Array(maxLen);

  for (let i = 0; i < maxLen; i++) {
    if (i < text.length) {
      ids[i] = BigInt(charToId[text[i]] ?? unk);
    } else {
      ids[i] = BigInt(pad);
    }
  }

  return ids;
}

function mbEncodeVoiceContext(text, charToId, maxLen) {
  const unk = charToId["<UNK>"];
  const pad = charToId["<PAD>"];

  let chars = Array.from(String(text || ""));

  if (chars.length > maxLen) {
    chars = chars.slice(chars.length - maxLen);
  }

  const leftPad = maxLen - chars.length;
  const ids = new BigInt64Array(maxLen);

  for (let i = 0; i < maxLen; i++) {
    if (i < leftPad) {
      ids[i] = BigInt(pad);
    } else {
      ids[i] = BigInt(charToId[chars[i - leftPad]] ?? unk);
    }
  }

  return ids;
}

function mbSoftmax(logits, temperature = 1.0) {
  const arr = Array.from(logits).map(x => x / temperature);
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function mbTopLabels(logits, labels, temperature = 1.0, n = 5) {
  const probs = mbSoftmax(logits, temperature);

  return probs
    .map((score, index) => ({ label: labels[index], score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, n);
}

function mbPickTopK(logits, temperature = 0.72, topK = 24) {
  const arr = Array.from(logits);

  const indexed = arr.map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, Math.min(topK, arr.length));

  const scaled = indexed.map(x => x.v / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);

  let r = Math.random();
  let acc = 0;

  for (let i = 0; i < indexed.length; i++) {
    acc += exps[i] / sum;
    if (r <= acc) return indexed[i].i;
  }

  return indexed[0].i;
}

function mbBuildVoicePrompt(userText, facts, style, safety, answerKey) {
  return (
    "<STYLE=" + style + "> " +
    "<SAFETY=" + safety + "> " +
    "<KEY=" + answerKey + "> " +
    "<USER>" + String(userText || "") + "</USER> " +
    "<FACTS>" + String(facts || "") + "</FACTS> " +
    "<REPLY>"
  );
}

function mbBadGeneratedText(text) {
  const t = String(text || "").trim();

  if (t.length < 2) return true;
  if (t.length > 360) return true;
  if (/[<>]{2,}/.test(t)) return true;
  if ((t.match(/undefined|null|nan/gi) || []).length) return true;

  const words = t.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length >= 8) {
    const counts = {};
    for (const w of words) counts[w] = (counts[w] || 0) + 1;
    const maxRepeat = Math.max(...Object.values(counts));
    if (maxRepeat >= Math.max(5, words.length * 0.35)) return true;
  }

  return false;
}

async function loadMeatballHybridBrain({
  controllerUrl = "/dist/meatball-hybrid/controller.onnx",
  controllerMetadataUrl = "/dist/meatball-hybrid/controller_metadata.json",
  voiceUrl = "/dist/meatball-hybrid/voice_generator.onnx",
  voiceMetadataUrl = "/dist/meatball-hybrid/voice_metadata.json",
  answerBankUrl = "/dist/meatball-hybrid/answer_bank.json"
} = {}) {
  const controllerMetadata = await fetch(controllerMetadataUrl).then(r => {
    if (!r.ok) throw new Error("Failed controller metadata " + r.status);
    return r.json();
  });

  const voiceMetadata = await fetch(voiceMetadataUrl).then(r => {
    if (!r.ok) throw new Error("Failed voice metadata " + r.status);
    return r.json();
  });

  const answerBank = await fetch(answerBankUrl).then(r => {
    if (!r.ok) throw new Error("Failed answer bank " + r.status);
    return r.json();
  });

  const controller = await ort.InferenceSession.create(controllerUrl, {
    executionProviders: ["wasm"]
  });

  const voice = await ort.InferenceSession.create(voiceUrl, {
    executionProviders: ["wasm"]
  });

  return {
    controller,
    voice,
    controllerMetadata,
    voiceMetadata,
    answerBank
  };
}

async function predictMeatballController(brain, userText, memory = {}) {
  const inputText = mbBuildControllerInput(userText, memory);
  const ids = mbEncodeFixed(
    inputText,
    brain.controllerMetadata.char_to_id,
    brain.controllerMetadata.config.controller_max_len
  );

  const tensor = new ort.Tensor(
    "int64",
    ids,
    [1, brain.controllerMetadata.config.controller_max_len]
  );

  const outputs = await brain.controller.run({ input_ids: tensor });
  const labels = brain.controllerMetadata.labels;

  const intentTop = mbTopLabels(outputs.intent_logits.data, labels.intent);
  const topicTop = mbTopLabels(outputs.topic_logits.data, labels.topic);
  const safetyTop = mbTopLabels(outputs.safety_logits.data, labels.safety);
  const contextTop = mbTopLabels(outputs.context_action_logits.data, labels.context_action);
  const styleTop = mbTopLabels(outputs.style_logits.data, labels.style);
  const answerTop = mbTopLabels(outputs.answer_key_logits.data, labels.answer_key);

  return {
    input_text: inputText,

    intent: intentTop[0].label,
    intent_confidence: intentTop[0].score,
    intent_top: intentTop,

    topic: topicTop[0].label,
    topic_confidence: topicTop[0].score,
    topic_top: topicTop,

    safety: safetyTop[0].label,
    safety_confidence: safetyTop[0].score,
    safety_top: safetyTop,

    context_action: contextTop[0].label,
    context_action_confidence: contextTop[0].score,
    context_action_top: contextTop,

    style: styleTop[0].label,
    style_confidence: styleTop[0].score,
    style_top: styleTop,

    answer_key: answerTop[0].label,
    answer_key_confidence: answerTop[0].score,
    answer_key_top: answerTop
  };
}

async function generateMeatballVoice(brain, prompt, options = {}) {
  const maxNewChars = options.maxNewChars || brain.voiceMetadata.config.max_new_chars || 220;
  const temperature = options.temperature || brain.voiceMetadata.config.temperature || 0.72;
  const topK = options.topK || brain.voiceMetadata.config.top_k || 24;

  const charToId = brain.voiceMetadata.char_to_id;
  const idToChar = brain.voiceMetadata.id_to_char;
  const maxLen = brain.voiceMetadata.config.voice_max_len;

  let text = String(prompt || "");

  for (let step = 0; step < maxNewChars; step++) {
    const ids = mbEncodeVoiceContext(text, charToId, maxLen);

    const tensor = new ort.Tensor(
      "int64",
      ids,
      [1, maxLen]
    );

    const outputs = await brain.voice.run({ input_ids: tensor });
    const logits = outputs.logits.data;

    const vocabSize = brain.voiceMetadata.vocab_size;
    const offset = (maxLen - 1) * vocabSize;
    const lastLogits = logits.slice(offset, offset + vocabSize);

    const nextId = mbPickTopK(lastLogits, temperature, topK);
    const ch = idToChar[String(nextId)] || "";

    text += ch;

    if (text.includes("</REPLY>")) break;
  }

  let reply = text.split("<REPLY>").pop() || "";
  reply = reply.split("</REPLY>")[0] || reply;
  reply = reply.replace(/\s+/g, " ").trim();

  return reply;
}

function pickCannedAnswer(brain, prediction) {
  const entry =
    brain.answerBank[prediction.answer_key] ||
    brain.answerBank.generic_answer ||
    null;

  if (!entry) return "I’m not totally sure what you mean. Ask that a little more clearly.";

  if (Array.isArray(entry.answers) && entry.answers.length) {
    return entry.answers[Math.floor(Math.random() * entry.answers.length)];
  }

  return entry.answer || "I’m not totally sure what you mean. Ask that a little more clearly.";
}

async function answerWithMeatballHybrid(brain, userText, memory = {}) {
  const prediction = await predictMeatballController(brain, userText, memory);

  const canned = pickCannedAnswer(brain, prediction);

  if (
    prediction.safety !== "safe" ||
    prediction.answer_key_confidence < 0.20 ||
    prediction.safety_confidence < 0.40
  ) {
    updateMeatballHybridMemory(memory, prediction, canned);
    return { text: canned, prediction, used_generator: false, fallback_reason: "safety_or_low_confidence" };
  }

  const prompt = mbBuildVoicePrompt(
    userText,
    canned,
    prediction.style,
    prediction.safety,
    prediction.answer_key
  );

  let generated = "";

  try {
    generated = await generateMeatballVoice(brain, prompt);
  } catch (err) {
    generated = "";
  }

  if (mbBadGeneratedText(generated)) {
    updateMeatballHybridMemory(memory, prediction, canned);
    return { text: canned, prediction, used_generator: false, fallback_reason: "bad_generation" };
  }

  updateMeatballHybridMemory(memory, prediction, generated);
  return { text: generated, prediction, used_generator: true, fallback_reason: "" };
}

function updateMeatballHybridMemory(memory, prediction, answerText) {
  memory.last_intent = prediction.intent || "none";
  memory.last_project_key = prediction.topic || "general_knowledge";
  memory.last_category = prediction.style || "conversation";
  memory.last_answer_text = answerText || "";
  memory.message_count = (memory.message_count || 0) + 1;

  if (prediction.safety !== "safe" || prediction.context_action === "soft_refusal") {
    memory.confusion_count = (memory.confusion_count || 0) + 1;
  } else {
    memory.confusion_count = Math.max(0, (memory.confusion_count || 0) - 1);
  }

  return memory;
}
