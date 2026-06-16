const MEATBALL_CONTEXT_CONFIG = {
  "feature_size": 2048,
  "memory_feature_size": 256,
  "ngram_min": 2,
  "ngram_max": 5
};
const MEATBALL_CONTEXT_ACTIONS = [
  "clarify",
  "continue_previous",
  "correct_misunderstanding",
  "direct_answer",
  "expand_previous",
  "playful_bridge",
  "reset_needed",
  "same_project_followup",
  "soft_refusal",
  "topic_shift"
];

function mbNormalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[^a-z0-9?!.,'"\s:_/-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function mbHash(text) {
  let h = 2166136261;

  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }

  return h >>> 0;
}

function mbTextToVector(text) {
  const vec = new Float32Array(MEATBALL_CONTEXT_CONFIG.feature_size);
  const clean = ` ${mbNormalizeText(text)} `;

  for (let n = MEATBALL_CONTEXT_CONFIG.ngram_min; n <= MEATBALL_CONTEXT_CONFIG.ngram_max; n++) {
    for (let i = 0; i <= clean.length - n; i++) {
      const gram = clean.slice(i, i + n);
      const index = mbHash(gram) % MEATBALL_CONTEXT_CONFIG.feature_size;
      vec[index] += 1.0;
    }
  }

  const words = clean.trim().split(/\s+/).filter(Boolean);

  for (const word of words) {
    const index = mbHash(`word:${word}`) % MEATBALL_CONTEXT_CONFIG.feature_size;
    vec[index] += 1.5;
  }

  let sum = 0;

  for (let i = 0; i < vec.length; i++) {
    sum += vec[i] * vec[i];
  }

  const norm = Math.sqrt(sum) || 1;

  for (let i = 0; i < vec.length; i++) {
    vec[i] /= norm;
  }

  return vec;
}

function mbMemoryToVector(memory) {
  const vec = new Float32Array(MEATBALL_CONTEXT_CONFIG.memory_feature_size);

  const fields = [
    "last_intent",
    "last_project_key",
    "last_category",
    "current_intent",
    "current_project_key",
    "current_category"
  ];

  for (const field of fields) {
    const value = String(memory[field] || "none").toLowerCase();
    const index = mbHash(`${field}:${value}`) % MEATBALL_CONTEXT_CONFIG.memory_feature_size;
    vec[index] += 1.0;
  }

  vec[0] = Math.min((memory.message_count || 0) / 20.0, 1.0);
  vec[1] = Math.min((memory.confusion_count || 0) / 8.0, 1.0);
  vec[2] = Math.min((memory.topic_switch_count || 0) / 8.0, 1.0);

  return vec;
}

function mbContextInputVector(question, memory) {
  const textVec = mbTextToVector(question);
  const memoryVec = mbMemoryToVector(memory);

  const out = new Float32Array(
    MEATBALL_CONTEXT_CONFIG.feature_size +
    MEATBALL_CONTEXT_CONFIG.memory_feature_size
  );

  out.set(textVec, 0);
  out.set(memoryVec, MEATBALL_CONTEXT_CONFIG.feature_size);

  return out;
}

function mbSoftmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

async function loadMeatballContextModel(modelUrl = "/models/context-model/meatball_context.onnx") {
  return await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"]
  });
}

async function predictMeatballContextAction(session, question, memory) {
  const inputVector = mbContextInputVector(question, memory);

  const tensor = new ort.Tensor(
    "float32",
    inputVector,
    [1, inputVector.length]
  );

  const outputs = await session.run({
    input: tensor
  });

  const logits = Array.from(outputs.logits.data);
  const scores = mbSoftmax(logits);

  const ranked = scores
    .map((score, index) => ({
      action: MEATBALL_CONTEXT_ACTIONS[index],
      score
    }))
    .sort((a, b) => b.score - a.score);

  return {
    action: ranked[0].action,
    confidence: ranked[0].score,
    top: ranked.slice(0, 5)
  };
}

window.loadMeatballContextModel = loadMeatballContextModel;
window.predictMeatballContextAction = predictMeatballContextAction;