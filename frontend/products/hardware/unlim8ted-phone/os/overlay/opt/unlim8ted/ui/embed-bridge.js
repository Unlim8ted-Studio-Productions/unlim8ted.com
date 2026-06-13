(function () {
  const uiBase = new URL("./", window.location.href);
  const stateBase = new URL("../state/", uiBase);
  const originalFetch = window.fetch.bind(window);
  const clientState = {
    system: null,
    browser: null,
    appState: {}
  };

  const appConfig = {
    phone: { title: "Phone", template: "../apps/phone/index.html" },
    messages: { title: "Messages", template: "../apps/messages/index.html" },
    browser: { title: "Browser", template: "../apps/browser/index.html" },
    camera: { title: "Camera", template: "../apps/camera/index.html", view: "camera" },
    code: { title: "Code", template: "../apps/code/index.html", client: "../apps/code/client.js" },
    terminal: { title: "Terminal", template: "../apps/terminal/index.html", client: "../apps/terminal/client.js" },
    gallery: { title: "Gallery", template: "../apps/gallery/index.html", client: "../apps/gallery/client.js" },
    music: { title: "Music", template: "../apps/music/index.html" },
    maps: { title: "Maps", template: "../apps/maps/index.html" },
    mail: { title: "Mail", template: "../apps/mail/index.html" },
    notes: { title: "Notes", template: "../apps/notes/index.html" },
    clock: { title: "Clock", template: "../apps/clock/index.html" },
    files: { title: "Files", template: "../apps/files/index.html", client: "../apps/files/client.js" },
    store: { title: "Store", template: "../apps/store/index.html" },
    settings: { title: "Settings", template: "../apps/settings/index.html", client: "../apps/settings/client.js" }
  };

  const stateFiles = {
    system: "system.json",
    browser: "browser.json",
    messages: "messages_app.json",
    music: "music.json"
  };

  let stateReadyPromise;

  function jsonResponse(payload) {
    return Promise.resolve(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    );
  }

  function cloneValue(value) {
    return JSON.parse(JSON.stringify(value));
  }

  async function loadJsonFile(fileName, fallback) {
    try {
      const response = await originalFetch(new URL(fileName, stateBase));
      if (!response.ok) return cloneValue(fallback);
      return await response.json();
    } catch (_error) {
      return cloneValue(fallback);
    }
  }

  async function ensureState() {
    if (!stateReadyPromise) {
      stateReadyPromise = (async () => {
        clientState.system = await loadJsonFile(stateFiles.system, {
          brightness: 0.58,
          camera: {
            available: false,
            last_capture: "",
            last_error: "No backend connected",
            preview_active: false
          },
          display_awake: true,
          idle_timeout_sec: 15,
          last_interaction: Date.now() / 1000,
          last_sleep_reason: "",
          sleeping: false,
          toggles: {
            airplane: false,
            bluetooth: true,
            focus: false,
            wifi: true
          }
        });
        clientState.browser = await loadJsonFile(stateFiles.browser, {
          tabs: [{ id: "tab-1", title: "Example", url: "https://example.com" }],
          active_tab_id: "tab-1",
          history: ["https://example.com"],
          recent_searches: [],
          downloads: []
        });
        clientState.appState.messages = await loadJsonFile(stateFiles.messages, {});
        clientState.appState.music = await loadJsonFile(stateFiles.music, {});
      })();
    }

    return stateReadyPromise;
  }

  function systemPayload() {
    return { system: cloneValue(clientState.system) };
  }

  function getAppPayload(appId) {
    const config = appConfig[appId] || { title: appId || "App" };
    const payload = {
      title: config.title
    };

    if (config.template) payload.template_url = config.template;
    if (config.client) payload.client_script_url = config.client;
    if (config.view) payload.view = config.view;

    if (appId === "browser") {
      payload.browser = cloneValue(clientState.browser);
      payload.home_url = clientState.browser?.tabs?.[0]?.url || "https://example.com";
    }

    if (appId === "messages") {
      payload.html = `
        <div class="content-card">
          <div class="content-title">Messages</div>
          <div class="content-text">Thread source: ${escapeHtml(clientState.appState.messages?.thread_id || "default")}</div>
        </div>
      `;
    }

    if (appId === "music") {
      const queue = Array.isArray(clientState.appState.music?.queue)
        ? clientState.appState.music.queue.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
        : "";
      payload.html = `
        <div class="content-card">
          <div class="content-title">Music</div>
          <div class="content-text">Playback state from repo-backed JSON.</div>
          <ul style="margin-top:12px;padding-left:18px;color:rgba(223,232,255,.78);line-height:1.7;">${queue}</ul>
        </div>
      `;
    }

    if (!payload.html && !config.client && appId !== "camera") {
      payload.html = `
        <div class="content-card">
          <div class="content-title">${escapeHtml(config.title || "App")}</div>
          <div class="content-text">This panel is running through the real shell code with local repo-backed data and a browser bridge in place of the device backend.</div>
        </div>
      `;
    }

    return payload;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;"
    }[char]));
  }

  async function parseBody(init) {
    const body = init?.body;
    if (!body) return {};
    if (typeof body === "string") {
      try {
        return JSON.parse(body);
      } catch (_error) {
        return {};
      }
    }
    return {};
  }

  async function handleApiRequest(url, init) {
    await ensureState();

    if (url.pathname === "/api/state") {
      return jsonResponse(systemPayload());
    }

    if (url.pathname === "/api/system/activity") {
      clientState.system.last_interaction = Date.now() / 1000;
      return jsonResponse(systemPayload());
    }

    if (url.pathname === "/api/system/brightness") {
      const body = await parseBody(init);
      clientState.system.brightness = Number(body.brightness || clientState.system.brightness || 0.58);
      return jsonResponse(systemPayload());
    }

    if (url.pathname === "/api/system/sleep") {
      const body = await parseBody(init);
      clientState.system.sleeping = true;
      clientState.system.display_awake = false;
      clientState.system.last_sleep_reason = body.reason || "embed";
      return jsonResponse(systemPayload());
    }

    if (url.pathname === "/api/system/wake") {
      clientState.system.sleeping = false;
      clientState.system.display_awake = true;
      return jsonResponse(systemPayload());
    }

    if (
      url.pathname === "/api/system/reboot" ||
      url.pathname === "/api/system/shutdown" ||
      url.pathname === "/api/system/exit-kiosk"
    ) {
      return jsonResponse(systemPayload());
    }

    if (url.pathname === "/api/log/client") {
      return jsonResponse({ ok: true });
    }

    const appMatch = url.pathname.match(/^\/api\/apps\/([^/]+)$/);
    if (appMatch) {
      const appId = decodeURIComponent(appMatch[1]);
      return jsonResponse({
        app: getAppPayload(appId),
        ...systemPayload()
      });
    }

    const appActionMatch = url.pathname.match(/^\/api\/apps\/([^/]+)\/action$/);
    if (appActionMatch) {
      const appId = decodeURIComponent(appActionMatch[1]);
      const body = await parseBody(init);
      if (appId === "browser" && body.action === "save_browser_state" && body.payload) {
        clientState.browser = {
          ...clientState.browser,
          tabs: Array.isArray(body.payload.tabs) ? body.payload.tabs : clientState.browser.tabs,
          active_tab_id: body.payload.active_tab_id || clientState.browser.active_tab_id,
          history: Array.isArray(body.payload.history) ? body.payload.history : clientState.browser.history
        };
      }
      return jsonResponse({
        ok: true,
        app: getAppPayload(appId),
        ...systemPayload()
      });
    }

    if (url.pathname === "/cmd") {
      const body = await parseBody(init);
      const action = body.action;
      if (action && clientState.system.toggles && Object.prototype.hasOwnProperty.call(clientState.system.toggles, action)) {
        clientState.system.toggles[action] = !clientState.system.toggles[action];
      }
      return jsonResponse(systemPayload());
    }

    return null;
  }

  window.fetch = async function patchedFetch(input, init) {
    const url = new URL(typeof input === "string" ? input : input.url, window.location.href);
    if (url.origin === window.location.origin && (url.pathname.startsWith("/api/") || url.pathname === "/cmd")) {
      const response = await handleApiRequest(url, init);
      if (response) return response;
    }
    return originalFetch(input, init);
  };

  window.__unlim8tedEmbedBridge = {
    ensureState
  };

  window.addEventListener("load", () => {
    window.setTimeout(() => {
      const root = document.documentElement;
      root.style.setProperty("--safe-top", "0px");
      root.style.setProperty("--safe-bottom", "0px");

      if (typeof window.lockDevice === "function") {
        window.lockDevice();
      } else {
        document.getElementById("lockscreen")?.classList.remove("unlocked");
        document.getElementById("homeScreen")?.classList.remove("unlocked");
        document.getElementById("lockscreen")?.style?.setProperty("transform", "translateY(0)");
        document.getElementById("homeScreen")?.style?.setProperty("transform", "translateY(100%)");
      }
    }, 120);
  });
})();
