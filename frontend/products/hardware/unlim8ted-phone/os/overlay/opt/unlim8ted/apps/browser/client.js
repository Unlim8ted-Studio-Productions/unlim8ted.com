window.Unlim8tedAppClients = window.Unlim8tedAppClients || {};
window.Unlim8tedAppClients.browser = (() => {
    let currentCtx = null;
    let messageHookInstalled = false;

    function normalizeUrl(value) {
        const input = String(value || '').trim();
        if (!input) return 'https://example.com';
        if (/^https?:\/\//i.test(input)) return input;
        if (input.includes('.') && !input.includes(' ')) return `https://${input}`;
        return `https://duckduckgo.com/?q=${encodeURIComponent(input)}`;
    }

    function proxyUrl(url) {
        return `/api/apps/browser/render?url=${encodeURIComponent(normalizeUrl(url))}`;
    }

    function ensureState(payload = {}) {
        const ctx = currentCtx;
        const rawBrowser = payload.browser || {};
        const homeUrl = normalizeUrl(rawBrowser.home || payload.home_url || 'https://example.com');
        const tabs = Array.isArray(rawBrowser.tabs) && rawBrowser.tabs.length
            ? rawBrowser.tabs.slice(0, 12).map((tab, index) => ({
                id: String(tab?.id || `tab-${index + 1}`),
                title: String(tab?.title || 'Tab').slice(0, 80),
                url: normalizeUrl(tab?.url || homeUrl)
            }))
            : [{ id: 'tab-1', title: 'Home', url: homeUrl }];
        const activeTabId = tabs.some((tab) => tab.id === (rawBrowser.activeTabId || rawBrowser.active_tab_id))
            ? (rawBrowser.activeTabId || rawBrowser.active_tab_id)
            : tabs[0].id;
        const history = Array.isArray(rawBrowser.history) && rawBrowser.history.length
            ? rawBrowser.history.slice(0, 100).map((item) => normalizeUrl(item))
            : [tabs[0].url];
        ctx.state.browser = {
            tabs,
            activeTabId,
            history,
            index: Math.max(0, history.length - 1),
            home: homeUrl
        };
        return ctx.state.browser;
    }

    function browserState() {
        return currentCtx?.state?.browser || ensureState({});
    }

    function activeTab(browser = browserState()) {
        return browser.tabs.find((item) => item.id === browser.activeTabId) || browser.tabs[0];
    }

    function persist() {
        const browser = browserState();
        currentCtx?.requestJson('/api/apps/browser/action', {
            method: 'POST',
            body: JSON.stringify({
                action: 'save_browser_state',
                payload: {
                    tabs: browser.tabs,
                    active_tab_id: browser.activeTabId,
                    history: browser.history
                }
            })
        });
    }

    function loadFrame(url) {
        const frame = document.getElementById('browserFrame');
        if (!frame) return;
        frame.removeAttribute('srcdoc');
        frame.src = proxyUrl(url);
    }

    function syncFrame() {
        const browser = browserState();
        const tab = activeTab(browser);
        const currentUrl = tab?.url || browser.history[browser.index] || browser.home;
        const input = document.getElementById('browserUrl');
        if (input) input.value = currentUrl;
        if (currentUrl) loadFrame(currentUrl);
    }

    function closeTabSheet() {
        const sheet = document.getElementById('browserTabSheet');
        if (sheet) sheet.style.display = 'none';
    }

    function openTabSheet() {
        const sheet = document.getElementById('browserTabSheet');
        if (sheet) sheet.style.display = 'flex';
    }

    function rerender(keepMenuOpen = false) {
        const payload = {
            browser: browserState(),
            home_url: browserState().home
        };
        render(payload, currentCtx);
        if (keepMenuOpen) {
            setTimeout(openTabSheet, 0);
        }
    }

    function navigate(rawUrl) {
        const browser = browserState();
        const tab = activeTab(browser);
        const rawInput = String(rawUrl || '').trim();
        const url = normalizeUrl(rawInput);
        browser.history = browser.history.slice(0, browser.index + 1);
        browser.history.push(url);
        browser.history = browser.history.slice(-100);
        browser.index = browser.history.length - 1;
        if (tab) {
            tab.url = url;
            tab.title = rawInput && !rawInput.includes(' ') && rawInput.includes('.') ? (new URL(url)).hostname : 'Search';
        }
        currentCtx.state.browser = browser;
        persist();
        syncFrame();
    }

    function moveHistory(delta) {
        const browser = browserState();
        const nextIndex = browser.index + delta;
        if (nextIndex < 0 || nextIndex >= browser.history.length) return;
        browser.index = nextIndex;
        const tab = activeTab(browser);
        const url = browser.history[nextIndex];
        if (tab && url) tab.url = url;
        currentCtx.state.browser = browser;
        persist();
        syncFrame();
    }

    function reload() {
        const tab = activeTab();
        if (tab?.url) loadFrame(tab.url);
    }

    function newTab() {
        const browser = browserState();
        const id = `tab-${Date.now()}`;
        browser.tabs = [...browser.tabs, { id, title: 'New Tab', url: browser.home }].slice(-12);
        browser.activeTabId = id;
        browser.history = [...browser.history, browser.home].slice(-100);
        browser.index = browser.history.length - 1;
        currentCtx.state.browser = browser;
        persist();
        rerender(false);
    }

    function switchTab(tabId) {
        const browser = browserState();
        const tab = browser.tabs.find((item) => item.id === tabId);
        if (!tab) return;
        browser.activeTabId = tabId;
        browser.history = [...browser.history, tab.url].slice(-100);
        browser.index = browser.history.length - 1;
        currentCtx.state.browser = browser;
        persist();
        rerender(false);
    }

    function closeTab(tabId = null, keepMenuOpen = false) {
        const browser = browserState();
        if (browser.tabs.length <= 1) return;
        const targetTabId = tabId || browser.activeTabId;
        const index = browser.tabs.findIndex((item) => item.id === targetTabId);
        if (index < 0) return;
        browser.tabs.splice(index, 1);
        if (browser.activeTabId === targetTabId) {
            browser.activeTabId = browser.tabs[Math.max(0, index - 1)].id;
        }
        currentCtx.state.browser = browser;
        persist();
        rerender(keepMenuOpen);
    }

    function browserSvgIcon(name) {
        const icons = {
            plus: '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M12 5v14M5 12h14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
            close: '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M7 7l10 10M17 7L7 17" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
            back: '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M14.5 6l-6 6 6 6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
            forward: '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M9.5 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
            reload: '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M18 8V4m0 0h-4m4 0-4.5 4.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M19 12a7 7 0 1 1-2-4.9" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
            menu: '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="12" cy="5.5" r="1.6" fill="currentColor"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/><circle cx="12" cy="18.5" r="1.6" fill="currentColor"/></svg>',
            search: '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="11" cy="11" r="5.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M16 16l4 4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>'
        };
        return icons[name] || icons.menu;
    }

    function bindTabList(rootNode, keepMenuOpen) {
        rootNode.querySelectorAll('[data-browser-tab]').forEach((button) => button.addEventListener('click', () => {
            closeTabSheet();
            switchTab(button.dataset.browserTab || '');
        }));
        rootNode.querySelectorAll('[data-browser-close-tab]').forEach((button) => button.addEventListener('click', (event) => {
            event.stopPropagation();
            closeTab(button.dataset.browserCloseTab || '', keepMenuOpen);
        }));
    }

    function installMessageHook() {
        if (messageHookInstalled) return;
        messageHookInstalled = true;
        window.addEventListener('message', (event) => {
            if (!currentCtx || currentCtx.state.appId !== 'browser') return;
            if (!event?.data || event.data.type !== 'browser-location') return;
            const browser = browserState();
            const tab = activeTab(browser);
            const url = normalizeUrl(event.data.url || tab?.url || browser.home);
            const title = String(event.data.title || '').trim().slice(0, 80);
            if (!tab) return;
            tab.url = url;
            if (title) tab.title = title;
            if (browser.history[browser.index] !== url) {
                browser.history = [...browser.history.slice(0, browser.index + 1), url].slice(-100);
                browser.index = browser.history.length - 1;
            }
            currentCtx.state.browser = browser;
            const input = document.getElementById('browserUrl');
            if (input) input.value = url;
            const list = document.getElementById('browserTabList');
            if (list && currentCtx.state.appOpen) {
                list.innerHTML = browser.tabs.map((entry) => `
                    <div style="display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center;padding:10px 12px;border-radius:16px;border:1px solid rgba(140,186,255,.10);background:${entry.id === browser.activeTabId ? 'rgba(157,205,255,.18)' : 'rgba(255,255,255,.03)'};color:#eef3ff;">
                        <button type="button" data-browser-tab="${currentCtx.escapeHtml(entry.id)}" style="display:grid;gap:3px;min-width:0;text-align:left;border:0;background:transparent;color:inherit;padding:0;">
                            <span style="font-size:13px;font-weight:700;line-height:1.15;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${currentCtx.escapeHtml(entry.title || 'Tab')}</span>
                            <span style="font-size:11px;color:#93a8c4;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${currentCtx.escapeHtml(entry.url || '')}</span>
                        </button>
                        <button type="button" data-browser-close-tab="${currentCtx.escapeHtml(entry.id)}" aria-label="Close tab" title="Close tab" style="height:28px;width:28px;border-radius:14px;border:1px solid rgba(140,186,255,.12);background:rgba(255,255,255,.06);color:#eef3ff;display:grid;place-items:center;font-size:13px;font-weight:800;">x</button>
                    </div>
                `).join('');
                bindTabList(list, true);
            }
            persist();
        });
    }

    async function render(payload, ctx) {
        currentCtx = ctx;
        installMessageHook();
        const browser = ensureState(payload || {});
        const tab = activeTab(browser);
        const currentUrl = tab?.url || browser.home;

        ctx.appView.classList.remove('browser-chrome-only');
        if (ctx.appTop) ctx.appTop.style.display = '';
        ctx.appBody.style.padding = '0 0 calc(var(--safe-bottom) + 12px)';
        ctx.appBody.style.gap = '0';
        ctx.appBody.style.alignContent = 'stretch';

        ctx.appBody.querySelectorAll('[data-browser-icon]').forEach((node) => {
            node.innerHTML = browserSvgIcon(node.dataset.browserIcon || 'menu');
        });
        const urlInput = document.getElementById('browserUrl');
        if (urlInput) urlInput.value = currentUrl;
        const tabList = document.getElementById('browserTabList');
        if (tabList) {
            tabList.innerHTML = browser.tabs.map((entry) => `
                <div style="display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center;padding:10px 12px;border-radius:16px;border:1px solid rgba(140,186,255,.10);background:${entry.id === browser.activeTabId ? 'rgba(157,205,255,.18)' : 'rgba(255,255,255,.03)'};color:#eef3ff;">
                    <button type="button" data-browser-tab="${ctx.escapeHtml(entry.id)}" style="display:grid;gap:3px;min-width:0;text-align:left;border:0;background:transparent;color:inherit;padding:0;">
                        <span style="font-size:13px;font-weight:700;line-height:1.15;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${ctx.escapeHtml(entry.title || 'Tab')}</span>
                        <span style="font-size:11px;color:#93a8c4;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${ctx.escapeHtml(entry.url || '')}</span>
                    </button>
                    <button type="button" data-browser-close-tab="${ctx.escapeHtml(entry.id)}" aria-label="Close tab" title="Close tab" style="height:28px;width:28px;border-radius:14px;border:1px solid rgba(140,186,255,.12);background:rgba(255,255,255,.06);color:#eef3ff;display:grid;place-items:center;font-size:13px;font-weight:800;">x</button>
                </div>
            `).join('');
            bindTabList(tabList, true);
        }

        document.getElementById('browserForm')?.addEventListener('submit', (event) => {
            event.preventDefault();
            navigate(document.getElementById('browserUrl')?.value || '');
        });
        document.getElementById('browserMenu')?.addEventListener('click', () => openTabSheet());
        document.getElementById('browserCloseInline')?.addEventListener('click', () => ctx.closeApp());
        document.getElementById('browserSheetNew')?.addEventListener('click', () => {
            closeTabSheet();
            newTab();
        });
        document.getElementById('browserSheetBack')?.addEventListener('click', () => {
            closeTabSheet();
            moveHistory(-1);
        });
        document.getElementById('browserSheetForward')?.addEventListener('click', () => {
            closeTabSheet();
            moveHistory(1);
        });
        document.getElementById('browserSheetReload')?.addEventListener('click', () => {
            closeTabSheet();
            reload();
        });
        document.getElementById('browserSheetClose')?.addEventListener('click', () => closeTabSheet());
        document.getElementById('browserTabSheet')?.addEventListener('click', (event) => {
            if (event.target?.id === 'browserTabSheet') closeTabSheet();
        });
        syncFrame();
    }

    return { render };
})();
