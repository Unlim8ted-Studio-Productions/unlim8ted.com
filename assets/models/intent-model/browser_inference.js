const MEATBALL_CONFIG = {
  "feature_size": 4096,
  "ngram_min": 2,
  "ngram_max": 5
};
const MEATBALL_INTENTS = [
  "adult_or_age_boundary",
  "brand_ai_general",
  "brand_apps_general",
  "brand_audience",
  "brand_compare",
  "brand_connection",
  "brand_creator",
  "brand_film_story_general",
  "brand_future",
  "brand_games_general",
  "brand_hardware_general",
  "brand_limit",
  "brand_mascot",
  "brand_mission",
  "brand_music_general",
  "brand_name",
  "brand_origin",
  "brand_overview",
  "brand_process",
  "brand_projects_general",
  "brand_quality",
  "brand_reset_request",
  "brand_scope",
  "brand_style",
  "brand_values",
  "feedback_negative",
  "general_ai",
  "general_art",
  "general_basic_math",
  "general_basic_science",
  "general_basic_tech",
  "general_biology",
  "general_brain",
  "general_capabilities",
  "general_cells",
  "general_climate",
  "general_coding",
  "general_comparison_request",
  "general_computer",
  "general_creativity",
  "general_css",
  "general_def_ai",
  "general_def_animal",
  "general_def_bird",
  "general_def_bot",
  "general_def_brain",
  "general_def_cat",
  "general_def_dog",
  "general_def_fish",
  "general_def_flower",
  "general_def_forest",
  "general_def_game",
  "general_def_joke",
  "general_def_leaf",
  "general_def_mascot",
  "general_def_movie",
  "general_def_music",
  "general_def_plant",
  "general_def_project",
  "general_def_root",
  "general_def_tree",
  "general_def_water",
  "general_def_website",
  "general_definition_request",
  "general_dna",
  "general_earth",
  "general_evolution",
  "general_film",
  "general_game_design",
  "general_goodbye",
  "general_gravity",
  "general_greeting",
  "general_history",
  "general_html",
  "general_identity",
  "general_intent_classifier",
  "general_internet",
  "general_javascript",
  "general_learning",
  "general_limits",
  "general_math",
  "general_memory",
  "general_music",
  "general_nervous_system",
  "general_neural_network",
  "general_neurons",
  "general_phone",
  "general_privacy",
  "general_project_advice",
  "general_reset",
  "general_science",
  "general_security",
  "general_space",
  "general_stars",
  "general_training_data",
  "general_unlim8ted_creator",
  "general_water_cycle",
  "general_weather",
  "general_website",
  "general_website_navigation",
  "inappropriate_harassment",
  "inappropriate_illegal",
  "inappropriate_malware",
  "inappropriate_rude",
  "inappropriate_sexual",
  "inappropriate_violence",
  "nonsense_gibberish",
  "nonsense_repetitive",
  "nonsense_vague",
  "off_scope_random_fact",
  "private_data_request",
  "project_ai_tools",
  "project_ai_tools_details",
  "project_ai_tools_mascot",
  "project_ai_tools_safe",
  "project_assets_unlim8ted",
  "project_assets_unlim8ted_details",
  "project_assets_unlim8ted_mascot",
  "project_assets_unlim8ted_safe",
  "project_blender_copy_keyframe_addon",
  "project_blender_copy_keyframe_addon_details",
  "project_blender_copy_keyframe_addon_safe",
  "project_category_3d",
  "project_category_ai_audio",
  "project_category_ai_simulation",
  "project_category_ai_video",
  "project_category_app",
  "project_category_audio",
  "project_category_blender_addon",
  "project_category_computer_vision",
  "project_category_film_story",
  "project_category_game",
  "project_category_game_mod",
  "project_category_game_vr",
  "project_category_hardware",
  "project_category_infrastructure",
  "project_category_product",
  "project_category_simulation",
  "project_category_software",
  "project_category_tool",
  "project_category_unknown_repo",
  "project_category_website",
  "project_category_writing_tool",
  "project_chatapp",
  "project_chatapp_details",
  "project_chatapp_mascot",
  "project_chatapp_safe",
  "project_chessvr",
  "project_chessvr_details",
  "project_chessvr_mascot",
  "project_chessvr_safe",
  "project_cineme",
  "project_cineme_details",
  "project_cineme_mascot",
  "project_cineme_safe",
  "project_civ_style_game",
  "project_civ_style_game_details",
  "project_civ_style_game_mascot",
  "project_civ_style_game_safe",
  "project_comparison",
  "project_download_any_youtube_video",
  "project_download_any_youtube_video_details",
  "project_download_any_youtube_video_mascot",
  "project_download_any_youtube_video_safe",
  "project_easy_pygame_ui_maker",
  "project_easy_pygame_ui_maker_details",
  "project_easy_pygame_ui_maker_mascot",
  "project_easy_pygame_ui_maker_safe",
  "project_face_stuff",
  "project_face_stuff_details",
  "project_face_stuff_mascot",
  "project_face_stuff_safe",
  "project_film_script_writer",
  "project_film_script_writer_details",
  "project_film_script_writer_mascot",
  "project_film_script_writer_safe",
  "project_ftl_choose_your_side",
  "project_ftl_choose_your_side_details",
  "project_ftl_choose_your_side_mascot",
  "project_ftl_choose_your_side_safe",
  "project_ftl_node_based_modding",
  "project_ftl_node_based_modding_details",
  "project_ftl_node_based_modding_mascot",
  "project_kindel_e_ink_games",
  "project_kindel_e_ink_games_details",
  "project_kindel_e_ink_games_mascot",
  "project_life_of_a_meatball",
  "project_life_of_a_meatball_details",
  "project_life_of_a_meatball_mascot",
  "project_multiplayer_physics_simulator",
  "project_multiplayer_physics_simulator_details",
  "project_multiplayer_physics_simulator_mascot",
  "project_multiplayer_physics_simulator_safe",
  "project_music_ai_gen",
  "project_music_ai_gen_details",
  "project_music_ai_gen_mascot",
  "project_music_ai_gen_safe",
  "project_music_worlds",
  "project_music_worlds_details",
  "project_music_worlds_mascot",
  "project_music_worlds_safe",
  "project_organisms_sim",
  "project_organisms_sim_details",
  "project_organisms_sim_mascot",
  "project_organisms_sim_safe",
  "project_paint_app_v59",
  "project_paint_app_v59_details",
  "project_paint_app_v59_mascot",
  "project_paint_app_v59_safe",
  "project_repo_3d",
  "project_repo_3d_details",
  "project_repo_3d_mascot",
  "project_repo_3d_safe",
  "project_services_unlim8ted",
  "project_services_unlim8ted_details",
  "project_services_unlim8ted_mascot",
  "project_services_unlim8ted_safe",
  "project_square_pixels",
  "project_square_pixels_details",
  "project_square_pixels_mascot",
  "project_square_pixels_safe",
  "project_star_tracker",
  "project_star_tracker_details",
  "project_star_tracker_mascot",
  "project_star_tracker_safe",
  "project_timecat",
  "project_timecat_details",
  "project_timecat_mascot",
  "project_timecat_safe",
  "project_turbo_octo_funicular",
  "project_turbo_octo_funicular_details",
  "project_turbo_octo_funicular_mascot",
  "project_turbo_octo_funicular_safe",
  "project_unicornia",
  "project_unicornia_details",
  "project_unicornia_mascot",
  "project_unicornia_safe",
  "project_unlim8ted_phone",
  "project_unlim8ted_phone_details",
  "project_unlim8ted_phone_mascot",
  "project_unlim8ted_phone_safe",
  "project_unlim8ted_website",
  "project_unlim8ted_website_details",
  "project_unlim8ted_website_mascot",
  "project_unlim8ted_website_safe",
  "project_wise_size",
  "project_wise_size_details",
  "project_wise_size_mascot",
  "project_wise_size_safe",
  "project_wrighting",
  "project_wrighting_details",
  "project_wrighting_mascot",
  "project_wrighting_safe",
  "prompt_injection",
  "safety_legal_financial_boundary",
  "safety_medical_boundary",
  "smalltalk_apology",
  "smalltalk_bored",
  "smalltalk_capability_limits",
  "smalltalk_clean_joke",
  "smalltalk_confusion",
  "smalltalk_correction",
  "smalltalk_current_activity",
  "smalltalk_day_check",
  "smalltalk_disagreement",
  "smalltalk_emotion_check",
  "smalltalk_explain_previous",
  "smalltalk_goodbye",
  "smalltalk_help",
  "smalltalk_positive_reaction",
  "smalltalk_reset_request",
  "smalltalk_thanks",
  "spam_ads",
  "too_complex_deep_science",
  "too_complex_engineering",
  "too_complex_financial",
  "too_complex_homework_full",
  "too_complex_legal",
  "too_complex_math",
  "too_complex_meatball_overload",
  "too_complex_medical",
  "too_complex_personal_crisis",
  "too_complex_random_encyclopedia",
  "too_complex_unsafe"
];

function normalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[^a-z0-9?!.,'"\s:_/-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function fnv1aHash(text) {
  let h = 2166136261;

  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }

  return h >>> 0;
}

function textToVector(text) {
  const vec = new Float32Array(MEATBALL_CONFIG.feature_size);
  const clean = ` ${normalizeText(text)} `;

  for (let n = MEATBALL_CONFIG.ngram_min; n <= MEATBALL_CONFIG.ngram_max; n++) {
    for (let i = 0; i <= clean.length - n; i++) {
      const gram = clean.slice(i, i + n);
      const index = fnv1aHash(gram) % MEATBALL_CONFIG.feature_size;
      vec[index] += 1.0;
    }
  }

  const words = clean.trim().split(/\s+/).filter(Boolean);

  for (const word of words) {
    const index = fnv1aHash(`word:${word}`) % MEATBALL_CONFIG.feature_size;
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

async function loadMeatballIntentModel(modelUrl = "/models/meatball_intent.onnx") {
  return await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"]
  });
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

async function predictMeatballIntent(session, question) {
  const inputVector = textToVector(question);

  const inputTensor = new ort.Tensor(
    "float32",
    inputVector,
    [1, MEATBALL_CONFIG.feature_size]
  );

  const outputs = await session.run({
    input: inputTensor
  });

  const logits = Array.from(outputs.logits.data);
  const scores = softmax(logits);

  const ranked = scores
    .map((score, index) => ({
      intent: MEATBALL_INTENTS[index],
      score
    }))
    .sort((a, b) => b.score - a.score);

  return {
    intent: ranked[0].intent,
    confidence: ranked[0].score,
    top: ranked.slice(0, 5)
  };
}

window.loadMeatballIntentModel = loadMeatballIntentModel;
window.predictMeatballIntent = predictMeatballIntent;