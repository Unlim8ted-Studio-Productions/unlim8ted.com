const root = document.documentElement;
const os = document.getElementById('os');
const state = {
    gestureLocked: false,
    gestureAxis: '',
    unlocked: false,
    controlOpen: false,
    appOpen: false,
    appId: '',
    pageIndex: 0,
    startX: 0,
    startY: 0,
    currentY: 0,
    startTime: 0,
    draggingPanel: false,
    draggingUnlock: false,
    gestureMode: '',
    lastPanelTranslate: -1,
    sleeping: false,
    idleTimeoutMs: 45000,
    idleTimer: null,
    lastActivitySentAt: 0,
    cameraPoll: null,
    browser: null,
    recentApps: [],
    appSwitcherOpen: false,
    homeEditMode: false,
    homeEditTimer: null,
    selectedHomeApp: null,
    pagePointerId: null,
    pageDragStartX: 0,
    pageDragStartY: 0,
    pageDragOrigin: 0,
    pageDragMoved: false,
    suppressAppClick: false,
    keyboardVisible: false,
    debugVisible: false,
    viewportHeight: 0,
    hardwareKeyboardSeen: false,
    performanceMode: false
};

const lockscreen = document.getElementById('lockscreen');
const homeScreen = document.getElementById('homeScreen');
const homeShell = document.getElementById('homeShell');
const pages = document.getElementById('pages');
const dots = Array.from(document.querySelectorAll('.dot'));
const controlCenter = document.getElementById('controlCenter');
const ccSheet = document.getElementById('ccSheet');
const ccBackdrop = document.getElementById('ccBackdrop');
const topConfigHandle = document.getElementById('topConfigHandle');
const statusTimeButton = document.getElementById('statusTimeButton');
const appView = document.getElementById('appView');
const appBody = document.getElementById('appBody');
const appTitle = document.getElementById('appTitle');
const closeAppBtn = document.getElementById('closeAppBtn');
const switchAppsBtn = document.getElementById('switchAppsBtn');
const homeNavBtn = document.getElementById('homeNavBtn');
const quickSettingsBtn = document.getElementById('quickSettingsBtn');
const appSwitcherSheet = document.getElementById('appSwitcherSheet');
const keyboardLayer = document.getElementById('keyboardLayer');
const keyboardPredictions = document.getElementById('keyboardPredictions');
const keyboardClipboardBtn = document.getElementById('keyboardClipboardBtn');
const keyboardCopyBtn = document.getElementById('keyboardCopyBtn');
const keyboardHideBtn = document.getElementById('keyboardHideBtn');
const keyboardRowQ = document.getElementById('keyboardRowQ');
const keyboardRowA = document.getElementById('keyboardRowA');
const keyboardRowZ = document.getElementById('keyboardRowZ');
const keyboardRowBottom = document.getElementById('keyboardRowBottom');
const keyboardGlidePath = document.getElementById('keyboardGlidePath');
const debugToggleBtn = document.getElementById('debugToggleBtn');
const brightnessRange = document.getElementById('brightnessRange');
const brightnessValue = document.getElementById('brightnessValue');
const sleepButton = document.getElementById('sleepButton');
const restartButton = document.getElementById('restartButton');
const shutdownButton = document.getElementById('shutdownButton');
const sleepScreen = document.getElementById('sleepScreen');
const sleepTime = document.getElementById('sleepTime');
const networkStatusLabel = document.getElementById('networkStatusLabel');
const networkIcons = Array.from(document.querySelectorAll('#statusNetworkIcon, #ccNetworkIcon'));
const toggles = Array.from(document.querySelectorAll('.toggle'));

closeAppBtn.textContent = '×';

function shouldUsePerformanceMode() {
    const ua = navigator.userAgent || '';
    const platform = navigator.platform || '';
    return /Linux/i.test(ua) && /(arm|aarch64|armv7|armv8)/i.test(`${ua} ${platform}`);
}

function setPerformanceMode(enabled) {
    state.performanceMode = Boolean(enabled);
    os?.classList.toggle('cm4-performance', state.performanceMode);
}

setPerformanceMode(shouldUsePerformanceMode());

const appTitles = {
    phone: 'Phone',
    messages: 'Messages',
    browser: 'Browser',
    camera: 'Camera',
    code: 'Code',
    terminal: 'Terminal',
    gallery: 'Gallery',
    music: 'Music',
    maps: 'Maps',
    mail: 'Mail',
    notes: 'Notes',
    clock: 'Clock',
    files: 'Files',
    store: 'Store',
    settings: 'Settings'
};

const appTop = document.querySelector('.app-top');
const keyboardState = {
    target: null,
    suggestions: [],
    glideActive: false,
    glideKeys: [],
    glidePointerId: null,
    glideStartAt: 0,
    clipboardMemory: '',
    glidePoints: []
};
const predictionDictionary = [
    'the', 'and', 'you', 'your', 'hello', 'home', 'settings', 'browser', 'camera', 'messages', 'phone', 'clock',
    'notes', 'maps', 'files', 'mail', 'music', 'store', 'code', 'coding', 'project', 'editor', 'display', 'brightness', 'timeout', 'focus', 'wifi',
    'bluetooth', 'airplane', 'device', 'system', 'owner', 'search', 'route', 'save', 'create', 'delete', 'call',
    'send', 'draft', 'gallery', 'alarm', 'panel', 'open', 'close', 'sleep', 'wake', 'today', 'tomorrow',
    'unlim8ted', 'raspberry', 'keyboard', 'clipboard', 'typing', 'quick', 'toggle', 'connectivity', 'network'
];
const predictionWeights = {
    the: 1000, and: 980, you: 970, your: 950, hello: 920, home: 910, settings: 900, browser: 890,
    messages: 880, camera: 870, phone: 860, notes: 850, maps: 840, files: 830, mail: 820, music: 810,
    code: 808, coding: 806, project: 804, editor: 802, search: 800, open: 790, close: 780, sleep: 770, wake: 760, keyboard: 750, clipboard: 740,
    typing: 730, display: 720, brightness: 710, timeout: 700, wifi: 690, bluetooth: 680, network: 670,
    unlim8ted: 660, raspberry: 650, system: 640, device: 630, panel: 620, gallery: 610, alarm: 600
};
const keyboardLayout = {
    q: ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
    a: ['shift', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
    z: ['z', 'x', 'c', 'v', 'b', 'n', 'm', "'", 'backspace'],
    bottom: ['123', ',', 'space', '.', 'done']
};

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[char]));
}

function isTextEntryTarget(target) {
    if (!target) return false;
    const tagName = String(target.tagName || '').toLowerCase();
    if (tagName === 'textarea') return true;
    if (tagName === 'input') {
        const type = String(target.type || 'text').toLowerCase();
        return !['range', 'checkbox', 'radio', 'button', 'submit', 'color', 'file'].includes(type);
    }
    return !!target.isContentEditable;
}

function keyboardButtonMarkup(key) {
    const labels = {
        shift: 'Shift',
        backspace: 'Bksp',
        enter: 'Enter',
        space: 'Space',
        done: 'Hide'
    };
    const special = ['shift', 'backspace', 'enter', 'space', 'done', '123'].includes(key);
    const widthClass = ({
        shift: 'key-wide',
        backspace: 'key-wide',
        enter: 'key-wide',
        done: 'key-wide',
        '123': 'key-wide',
        space: 'key-wider'
    })[key];
    const className = ['keyboard-key', special ? 'special' : '', widthClass || ''].filter(Boolean).join(' ');
    return `<button type="button" class="${className}" data-keyboard-key="${escapeHtml(key)}">${escapeHtml(labels[key] || key)}</button>`;
}

function renderKeyboardLayout() {
    if (!keyboardRowQ) return;
    keyboardRowQ.innerHTML = keyboardLayout.q.map(keyboardButtonMarkup).join('');
    keyboardRowA.innerHTML = keyboardLayout.a.map(keyboardButtonMarkup).join('');
    keyboardRowZ.innerHTML = keyboardLayout.z.map(keyboardButtonMarkup).join('');
    keyboardRowBottom.innerHTML = keyboardLayout.bottom.map(keyboardButtonMarkup).join('');
}

function setKeyboardVisible(visible) {
    state.keyboardVisible = visible;
    keyboardLayer?.classList.toggle('visible', visible);
    keyboardLayer?.setAttribute('aria-hidden', String(!visible));
    stabilizeViewport();
    if (!visible) {
        keyboardState.glidePoints = [];
        if (keyboardGlidePath) keyboardGlidePath.innerHTML = '';
    }
}

function focusKeyboardTarget(target) {
    if (!isTextEntryTarget(target)) return;
    keyboardState.target = target;
    try {
        target.focus?.({ preventScroll: true });
    } catch (_error) {
        target.focus?.();
    }
    if (!state.hardwareKeyboardSeen || target.dataset.forceSoftKeyboard === 'true') {
        setKeyboardVisible(true);
    }
    window.setTimeout(stabilizeViewport, 0);
    updateKeyboardPredictions();
}

function blurKeyboardTarget() {
    keyboardState.target = null;
    setKeyboardVisible(false);
}

function baseViewportHeight() {
    const screenHeight = Number(window.screen?.height || 0);
    const innerHeight = Number(window.innerHeight || 0);
    return Math.max(screenHeight, innerHeight, 1);
}

function stabilizeViewport() {
    const current = Number(window.innerHeight || 0);
    if (!state.viewportHeight) {
        state.viewportHeight = baseViewportHeight();
    } else if (!state.keyboardVisible && current > state.viewportHeight) {
        state.viewportHeight = current;
    }
    root.style.setProperty('--os-viewport-height', `${state.viewportHeight}px`);
    if (window.scrollX || window.scrollY) window.scrollTo(0, 0);
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
}

function currentTextValue() {
    const target = keyboardState.target;
    if (!target) return '';
    return typeof target.value === 'string' ? target.value : (target.textContent || '');
}

function currentSelectionStart() {
    const target = keyboardState.target;
    if (!target) return 0;
    if (typeof target.selectionStart === 'number') return target.selectionStart;
    return currentTextValue().length;
}

function setTextValue(value, caret = value.length) {
    const target = keyboardState.target;
    if (!target) return;
    if (typeof target.value === 'string') {
        target.value = value;
        if (typeof target.setSelectionRange === 'function') target.setSelectionRange(caret, caret);
    } else {
        target.textContent = value;
    }
    target.dispatchEvent(new Event('input', { bubbles: true }));
}

function bindIframeKeyboardBridge(frame) {
    try {
        const doc = frame.contentDocument;
        if (!doc || doc.__unlim8tedKeyboardBound) return;
        doc.__unlim8tedKeyboardBound = true;
        doc.addEventListener('focusin', (event) => {
            if (isTextEntryTarget(event.target)) focusKeyboardTarget(event.target);
        });
        doc.addEventListener('pointerdown', (event) => {
            if (!state.keyboardVisible) return;
            if (!isTextEntryTarget(event.target)) blurKeyboardTarget();
        });
    } catch (_error) {
    }
}

function replaceSelection(insertText) {
    const target = keyboardState.target;
    if (!target) return;
    const value = currentTextValue();
    const start = typeof target.selectionStart === 'number' ? target.selectionStart : value.length;
    const end = typeof target.selectionEnd === 'number' ? target.selectionEnd : start;
    const next = value.slice(0, start) + insertText + value.slice(end);
    setTextValue(next, start + insertText.length);
    updateKeyboardPredictions();
}

function deleteBackward() {
    const target = keyboardState.target;
    if (!target) return;
    const value = currentTextValue();
    const start = typeof target.selectionStart === 'number' ? target.selectionStart : value.length;
    const end = typeof target.selectionEnd === 'number' ? target.selectionEnd : start;
    if (start !== end) {
        setTextValue(value.slice(0, start) + value.slice(end), start);
    } else if (start > 0) {
        setTextValue(value.slice(0, start - 1) + value.slice(end), start - 1);
    }
    updateKeyboardPredictions();
}

function currentWordInfo() {
    const value = currentTextValue();
    const caret = currentSelectionStart();
    const left = value.slice(0, caret);
    const match = left.match(/([A-Za-z']+)$/);
    if (!match) return { prefix: '', start: caret, end: caret };
    return { prefix: match[1].toLowerCase(), start: caret - match[1].length, end: caret };
}

function computePredictions() {
    const { prefix } = currentWordInfo();
    const valueWords = currentTextValue().toLowerCase().match(/[a-z']{3,}/g) || [];
    const learnedRecent = valueWords.slice(-24);
    const learnedWeights = learnedRecent.reduce((weights, word, index) => {
        weights[word] = (weights[word] || 0) + (learnedRecent.length - index) + 12;
        return weights;
    }, {});
    const seen = new Set();
    const source = [...learnedRecent.slice().reverse(), ...predictionDictionary].filter((word) => {
        if (seen.has(word)) return false;
        seen.add(word);
        return true;
    });
    const normalizedPrefix = prefix.toLowerCase();
    const ranked = source
        .map((word, index) => {
            const lower = word.toLowerCase();
            const startsWithPrefix = normalizedPrefix ? lower.startsWith(normalizedPrefix) : true;
            if (!startsWithPrefix && normalizedPrefix) return null;
            return {
                word,
                score: (predictionWeights[lower] || 0) + (learnedWeights[lower] || 0) - Math.max(0, lower.length - normalizedPrefix.length) - (index * 0.01)
            };
        })
        .filter(Boolean)
        .sort((left, right) => right.score - left.score);
    return ranked.slice(0, 3).map((entry) => entry.word);
}

function renderPredictions() {
    if (!keyboardPredictions) return;
    keyboardState.suggestions = computePredictions();
    keyboardPredictions.innerHTML = keyboardState.suggestions.map((word) =>
        `<button type="button" class="keyboard-suggestion" data-keyboard-suggestion="${escapeHtml(word)}">${escapeHtml(word)}</button>`
    ).join('') || [
        '<button type="button" class="keyboard-suggestion">the</button>',
        '<button type="button" class="keyboard-suggestion">and</button>',
        '<button type="button" class="keyboard-suggestion">you</button>'
    ].join('');
}

function updateKeyboardPredictions() {
    if (!state.keyboardVisible) return;
    renderPredictions();
}

function applySuggestion(word) {
    const info = currentWordInfo();
    const value = currentTextValue();
    const next = value.slice(0, info.start) + word + ' ' + value.slice(info.end);
    setTextValue(next, info.start + word.length + 1);
    updateKeyboardPredictions();
}

function decodeGlideWord(path) {
    const condensed = path.filter((key, index) => index === 0 || key !== path[index - 1]).join('');
    const words = Array.from(new Set([...predictionDictionary, ...(currentTextValue().toLowerCase().match(/[a-z']{3,}/g) || [])]));
    let best = '';
    let bestScore = -Infinity;
    words.forEach((word) => {
        if (word[0] !== condensed[0] || word[word.length - 1] !== condensed[condensed.length - 1]) return;
        let cursor = 0;
        for (const char of condensed) {
            cursor = word.indexOf(char, cursor);
            if (cursor === -1) return;
            cursor += 1;
        }
        const score = (condensed.length * 4) - Math.abs(word.length - condensed.length) + (predictionWeights[word] || 0);
        if (score > bestScore) {
            best = word;
            bestScore = score;
        }
    });
    return best || condensed;
}

function commitKeyboardKey(key) {
    if (!keyboardState.target) return;
    if (keyboardState.target.dataset.terminalInput === 'true') {
        const data = {
            backspace: '\x7f',
            space: ' ',
            enter: '\r'
        }[key] || (/^[a-z',.]$/.test(key) ? key : '');
        if (data) {
            keyboardState.target.dispatchEvent(new CustomEvent('terminal-input', {
                bubbles: true,
                detail: { data }
            }));
        }
        if (key === 'done') return blurKeyboardTarget();
        return;
    }
    if (key === 'backspace') return deleteBackward();
    if (key === 'space') return replaceSelection(' ');
    if (key === 'enter') return replaceSelection('\n');
    if (key === 'done') return blurKeyboardTarget();
    if (key === '123' || key === 'shift') return;
    replaceSelection(key);
}

function setDebugVisible(visible) {
    state.debugVisible = visible;
    document.getElementById('debugConsole')?.classList.toggle('visible', visible);
    debugToggleBtn?.setAttribute('aria-pressed', String(visible));
}

function appendGlidePoint(x, y) {
    if (!keyboardLayer || !keyboardGlidePath) return;
    const rect = keyboardLayer.getBoundingClientRect();
    const px = Math.max(0, Math.min(rect.width, x - rect.left));
    const py = Math.max(0, Math.min(rect.height, y - rect.top));
    keyboardState.glidePoints.push([px, py]);
    const points = keyboardState.glidePoints;
    if (!points.length) {
        keyboardGlidePath.innerHTML = '';
        return;
    }
    const d = points.map((point, index) => `${index ? 'L' : 'M'} ${point[0]} ${point[1]}`).join(' ');
    keyboardGlidePath.setAttribute('viewBox', `0 0 ${Math.max(1, rect.width)} ${Math.max(1, rect.height)}`);
    keyboardGlidePath.innerHTML = `<path d="${d}" fill="none" stroke="rgba(150,220,255,.96)" stroke-width="10" stroke-linecap="round" stroke-linejoin="round" opacity=".72"></path>`;
}

function saveHomeLayout() {
    const pagesData = Array.from(document.querySelectorAll('.page')).map((page) =>
        Array.from(page.querySelectorAll('.app')).map((app) => app.dataset.app)
    );
    try {
        localStorage.setItem('unlim8ted.home.layout', JSON.stringify(pagesData));
    } catch (_error) {
    }
}

function restoreHomeLayout() {
    try {
        const raw = localStorage.getItem('unlim8ted.home.layout');
        if (!raw) return;
        const layout = JSON.parse(raw);
        if (!Array.isArray(layout)) return;
        const pageNodes = Array.from(document.querySelectorAll('.page'));
        const appNodes = new Map(Array.from(document.querySelectorAll('.app')).map((node) => [node.dataset.app, node]));
        layout.forEach((pageLayout, index) => {
            const page = pageNodes[index];
            if (!page || !Array.isArray(pageLayout)) return;
            pageLayout.forEach((appId) => {
                const node = appNodes.get(appId);
                if (node) page.appendChild(node);
            });
        });
    } catch (_error) {
    }
}

function enterHomeEditMode(appNode = null) {
    state.homeEditMode = true;
    os.classList.add('home-edit');
    state.selectedHomeApp = appNode || null;
    document.querySelectorAll('.app').forEach((item) => item.classList.toggle('selected-edit', item === appNode));
}

function exitHomeEditMode() {
    state.homeEditMode = false;
    state.selectedHomeApp = null;
    os.classList.remove('home-edit');
    document.querySelectorAll('.app').forEach((item) => item.classList.remove('selected-edit'));
    if (state.homeEditTimer) {
        clearTimeout(state.homeEditTimer);
        state.homeEditTimer = null;
    }
}

function swapHomeApps(first, second) {
    if (!first || !second || first === second) return;
    const firstMarker = document.createElement('div');
    const secondMarker = document.createElement('div');
    first.parentNode.insertBefore(firstMarker, first);
    second.parentNode.insertBefore(secondMarker, second);
    firstMarker.parentNode.replaceChild(second, firstMarker);
    secondMarker.parentNode.replaceChild(first, secondMarker);
    saveHomeLayout();
}

function recentAppSubtitle(appId, payload = null) {
    if (payload?.subtitle) return payload.subtitle;
    if (payload?.path_label) return payload.path_label;
    if (payload?.gallery?.selected?.name) return payload.gallery.selected.name;
    if (payload?.camera?.available) return payload?.camera?.preview_active ? 'Live camera preview' : 'Camera ready';
    if (payload?.view === 'structured' && payload?.sections?.length) {
        return payload.sections[0]?.title || payload.sections[0]?.body || 'Ready';
    }
    if (payload?.view === 'camera') return 'Capture and preview';
    return appId === state.appId ? 'Currently open' : 'Ready';
}

function rememberRecentApp(appId, payload = null) {
    const title = payload?.title || appTitles[appId] || 'App';
    const subtitle = recentAppSubtitle(appId, payload);
    const preview = payload?.sections?.slice?.(0, 3)?.map((section) => section.title || section.body || '').filter(Boolean)
        || [payload?.notice, payload?.path_label, payload?.gallery?.selected?.created_label].filter(Boolean);
    state.recentApps = state.recentApps.filter((item) => item.id !== appId);
    state.recentApps.unshift({ id: appId, title, subtitle, preview });
    state.recentApps = state.recentApps.slice(0, 8);
}

function removeRecentApp(appId) {
    state.recentApps = state.recentApps.filter((item) => item.id !== appId);
}

function renderAppSwitcherPreview(item) {
    const previewRows = (item.preview?.length ? item.preview : [
        `${item.title} is ready to resume.`,
        state.appId === item.id ? 'This task is active right now.' : 'Tap to jump back in.',
        'Swipe up from the bottom anytime to return here.'
    ]).slice(0, 3).map((line) => `<div class="content-text" style="max-width:none;">${escapeHtml(line)}</div>`).join('');
    return `
        <div class="switcher-card ${item.id === state.appId ? 'active' : ''}" data-switch-app="${escapeHtml(item.id)}" tabindex="0" role="button">
            <div class="switcher-card-top">
                <div>
                    <div style="font-size:18px;font-weight:800;">${escapeHtml(item.title)}</div>
                    <div class="switcher-card-id">${escapeHtml(item.id)}</div>
                </div>
                <button type="button" class="switcher-btn" data-close-recent="${escapeHtml(item.id)}">Close</button>
            </div>
            <div class="switcher-preview">
                <div class="switcher-preview-bar"><span></span><span></span><span></span></div>
                <div class="content-title" style="margin:0;">${escapeHtml(item.subtitle || 'Task')}</div>
                ${previewRows}
            </div>
        </div>
    `;
}

function ensureAppSwitcher() {
    if (!appSwitcherSheet) return null;
    const items = state.recentApps.length ? state.recentApps : [];
    appSwitcherSheet.innerHTML = `
        <div class="switcher-shell">
            <div class="switcher-top">
                <div class="switcher-title">Recent Apps</div>
                <div class="switcher-actions">
                    <button type="button" class="switcher-btn" id="appSwitcherHome">Home</button>
                    <button type="button" class="switcher-btn" id="appSwitcherClear">Clear All</button>
                </div>
            </div>
            ${items.length ? `<div class="switcher-list">${items.map(renderAppSwitcherPreview).join('')}</div>` : `<div class="switcher-empty">No recent apps yet. Open an app, then swipe up from the bottom to see your recent apps.</div>`}
        </div>
    `;
    appSwitcherSheet.querySelectorAll('[data-switch-app]').forEach((button) => {
        button.addEventListener('click', async () => {
            closeAppSwitcher();
            await openApp(button.dataset.switchApp || '');
        });
        button.addEventListener('keydown', async (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            closeAppSwitcher();
            await openApp(button.dataset.switchApp || '');
        });
    });
    appSwitcherSheet.querySelectorAll('[data-close-recent]').forEach((button) => {
        button.addEventListener('click', (event) => {
            event.stopPropagation();
            const appId = button.dataset.closeRecent || '';
            removeRecentApp(appId);
            if (state.appId === appId) closeApp();
            ensureAppSwitcher();
            if (!state.recentApps.length) closeAppSwitcher();
        });
    });
    appSwitcherSheet.querySelector('#appSwitcherHome')?.addEventListener('click', () => {
        closeAppSwitcher();
        closeApp();
    });
    appSwitcherSheet.querySelector('#appSwitcherClear')?.addEventListener('click', () => {
        state.recentApps = [];
        closeApp();
        ensureAppSwitcher();
    });
    appSwitcherSheet.onclick = (event) => {
        if (event.target === appSwitcherSheet) closeAppSwitcher();
    };
    return appSwitcherSheet;
}

function openAppSwitcher() {
    if (state.sleeping || !state.unlocked) return;
    closeControlCenter();
    state.appSwitcherOpen = true;
    const node = ensureAppSwitcher();
    if (node) {
        node.classList.add('visible');
        node.setAttribute('aria-hidden', 'false');
    }
}

function closeAppSwitcher() {
    state.appSwitcherOpen = false;
    if (appSwitcherSheet) {
        appSwitcherSheet.classList.remove('visible');
        appSwitcherSheet.setAttribute('aria-hidden', 'true');
    }
}

async function requestJson(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
            ...options
        });
        const contentType = response.headers.get('content-type') || '';
        if (contentType.includes('application/json')) return await response.json();
    } catch (error) {
        return null;
    }
    return null;
}

function updateTime() {
    const now = new Date();
    const time = now.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    const date = now.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' });
    document.getElementById('sbTime').textContent = time;
    document.getElementById('lockClock').textContent = time;
    document.getElementById('lockDate').textContent = date;
    document.getElementById('ccTime').textContent = time;
    sleepTime.textContent = time;
}

function setPanelProgress(progress) {
    const clamped = Math.max(0, Math.min(1, progress));
    root.style.setProperty('--panel-progress', clamped.toFixed(4));
    if (state.performanceMode) {
        homeShell.style.transform = `translateY(${clamped * 10}px)`;
        homeShell.style.filter = '';
        return;
    }
    homeShell.style.transform = `translateY(${clamped * 18}px) scale(${1 - clamped * 0.035})`;
    homeShell.style.filter = `blur(${clamped * 1.4}px)`;
}

function setPanelTranslate(translatePercent) {
    const value = Math.max(-100, Math.min(0, translatePercent));
    state.lastPanelTranslate = value;
    ccSheet.style.transform = `translateY(${value}%)`;
    setPanelProgress((100 + value) / 100);
}

function unlock() {
    if (state.unlocked) return;
    state.unlocked = true;
    lockscreen.classList.add('unlocked');
    homeScreen.classList.add('unlocked');
    lockscreen.style.transform = '';
    homeScreen.style.transform = '';
}

function lockDevice() {
    if (state.appOpen) closeApp();
    closeControlCenter();
    blurKeyboardTarget();
    state.unlocked = false;
    lockscreen.classList.remove('unlocked');
    homeScreen.classList.remove('unlocked');
    lockscreen.style.transition = '';
    homeScreen.style.transition = '';
    lockscreen.style.transform = 'translateY(0)';
    homeScreen.style.transform = 'translateY(100%)';
}

function openControlCenter() {
    if (!state.unlocked || state.sleeping) return;
    closeAppSwitcher();
    state.controlOpen = true;
    controlCenter.classList.add('visible');
    controlCenter.setAttribute('aria-hidden', 'false');
    ccSheet.style.transition = 'transform 520ms cubic-bezier(.22,1,.36,1)';
    setPanelTranslate(0);
}

function closeControlCenter() {
    if (!state.controlOpen && !controlCenter.classList.contains('visible')) return;
    state.controlOpen = false;
    ccSheet.style.transition = 'transform 420ms cubic-bezier(.32,.72,0,1)';
    setPanelTranslate(-100);
    setTimeout(() => {
        if (!state.controlOpen) {
            controlCenter.classList.remove('visible');
            controlCenter.setAttribute('aria-hidden', 'true');
            setPanelProgress(0);
            homeShell.style.transform = '';
            homeShell.style.filter = '';
        }
    }, 220);
}

function setActivePage(index) {
    state.pageIndex = index;
    dots.forEach((dot, i) => dot.classList.toggle('active', i === index));
}

function applySystemState(system) {
    if (!system) return;
    state.sleeping = !!system.sleeping;
    state.idleTimeoutMs = Math.max(10000, (system.idle_timeout_sec || 45) * 1000);
    const percent = Math.round((system.brightness || 0.68) * 100);
    brightnessRange.value = percent;
    brightnessValue.textContent = `${percent}%`;
    toggles.forEach((toggle) => {
        const action = toggle.dataset.action;
        toggle.classList.toggle('active', !!system.toggles?.[action]);
    });
    applyNetworkState(system.network || {});
    document.querySelectorAll('.app').forEach((app) => {
        const appId = app.dataset.app;
        const count = Number(system.badges?.[appId] || 0);
        let badge = app.querySelector('.app-badge');
        if (count > 0) {
            if (!badge) {
                badge = document.createElement('div');
                badge.className = 'app-badge';
                badge.style.position = 'absolute';
                badge.style.top = '-4px';
                badge.style.right = '10px';
                badge.style.minWidth = '18px';
                badge.style.height = '18px';
                badge.style.padding = '0 6px';
                badge.style.borderRadius = '999px';
                badge.style.background = '#ff564f';
                badge.style.color = '#fff';
                badge.style.fontSize = '11px';
                badge.style.fontWeight = '800';
                badge.style.display = 'grid';
                badge.style.placeItems = 'center';
                app.appendChild(badge);
            }
            badge.textContent = String(count);
        } else if (badge) {
            badge.remove();
        }
    });
    os.classList.toggle('sleeping', state.sleeping);
    sleepScreen.classList.toggle('visible', state.sleeping);
    sleepScreen.setAttribute('aria-hidden', String(!state.sleeping));
}

function applyNetworkState(network) {
    const connected = Boolean(network.connected);
    const type = String(network.connection || '').toLowerCase();
    const signal = Number(network.signal || 0);
    const label = connected
        ? (network.ssid || network.interface || (type === 'ethernet' ? 'Ethernet' : 'Connected'))
        : (network.wifi_enabled === false ? 'Wi-Fi off' : 'Offline');
    if (networkStatusLabel) networkStatusLabel.textContent = label;
    networkIcons.forEach((icon) => {
        icon.classList.toggle('off', !connected);
        icon.classList.toggle('ethernet', connected && type === 'ethernet');
        icon.classList.toggle('weak', connected && type === 'wifi' && signal > 0 && signal < 45);
        icon.setAttribute('title', label);
        icon.setAttribute('aria-label', label);
    });
}

async function syncSystemState() {
    const payload = await requestJson('/api/state');
    if (payload?.system) applySystemState(payload.system);
}

function scheduleIdleSleep() {
    if (state.idleTimer) clearTimeout(state.idleTimer);
    if (state.sleeping) return;
    state.idleTimer = window.setTimeout(() => sleepSystem('idle'), state.idleTimeoutMs);
}

async function reportActivity(force = false) {
    const now = Date.now();
    if (!force && now - state.lastActivitySentAt < 1200) return;
    state.lastActivitySentAt = now;
    const payload = await requestJson('/api/system/activity', { method: 'POST', body: '{}' });
    if (payload?.system) applySystemState(payload.system);
}

function noteActivity(force = false) {
    if (state.sleeping && !force) return;
    scheduleIdleSleep();
    reportActivity(force);
}

async function sendSystemCommand(action, extra = {}) {
    const payload = await requestJson('/cmd', {
        method: 'POST',
        body: JSON.stringify({ action, ...extra })
    });
    if (payload?.system) applySystemState(payload.system);
    return payload;
}

async function setBrightness(percent) {
    brightnessValue.textContent = `${percent}%`;
    const payload = await requestJson('/api/system/brightness', {
        method: 'POST',
        body: JSON.stringify({ brightness: percent / 100 })
    });
    if (payload?.system) applySystemState(payload.system);
}

async function sleepSystem(reason = 'manual') {
    if (state.sleeping) return;
    stopCameraPreview(true);
    blurKeyboardTarget();
    lockDevice();
    closeControlCenter();
    const payload = await requestJson('/api/system/sleep', {
        method: 'POST',
        body: JSON.stringify({ reason })
    });
    if (payload?.system) applySystemState(payload.system);
}

async function wakeSystem(reason = 'tap') {
    const payload = await requestJson('/api/system/wake', {
        method: 'POST',
        body: JSON.stringify({ reason })
    });
    if (payload?.system) applySystemState(payload.system);
    lockDevice();
    noteActivity(true);
}

async function rebootSystem(reason = 'manual') {
    stopCameraPreview(true);
    blurKeyboardTarget();
    lockDevice();
    closeControlCenter();
    try {
        await requestJson('/api/system/reboot', {
            method: 'POST',
            body: JSON.stringify({ reason })
        });
    } catch (_error) {
        // The request may drop as the backend exits during reboot.
    }
}

async function shutdownSystem(reason = 'manual') {
    stopCameraPreview(true);
    blurKeyboardTarget();
    lockDevice();
    closeControlCenter();
    try {
        await requestJson('/api/system/shutdown', {
            method: 'POST',
            body: JSON.stringify({ reason })
        });
    } catch (_error) {
        // The request may drop as the backend powers off.
    }
}

async function exitKiosk(reason = 'keyboard') {
    stopCameraPreview(true);
    blurKeyboardTarget();
    lockDevice();
    closeControlCenter();
    try {
        await requestJson('/api/system/exit-kiosk', {
            method: 'POST',
            body: JSON.stringify({ reason })
        });
    } catch (_error) {
        // The request may drop as the backend stops the kiosk service.
    }
}

async function runAppAction(action, payload = {}) {
    if (!state.appId || state.appId === 'camera') return;
    noteActivity(true);
    const response = await requestJson(`/api/apps/${encodeURIComponent(state.appId)}/action`, {
        method: 'POST',
        body: JSON.stringify({ action, payload })
    });
    if (response?.system) applySystemState(response.system);
    if (response?.app) {
        renderAppPayload(response.app, state.appId);
    }
}

function renderHtmlApp(payload, appId) {
    const html = payload?.html || `
        <div class="content-card">
            <div class="content-title">${escapeHtml(appTitles[appId] || 'App')}</div>
            <div class="content-text">No app content is available for this module yet.</div>
        </div>
    `;
    appBody.innerHTML = html;
}

function renderStructuredSection(section) {
    const sectionTitle = escapeHtml(section.title || '');
    const sectionBody = escapeHtml(section.body || '');
    const renderRow = (item) => {
        const title = escapeHtml(item.title || '');
        const subtitle = escapeHtml(item.subtitle || '');
        const attrs = item.action
            ? `data-app-action="${escapeHtml(item.action)}" data-app-value="${escapeHtml(item.value || '')}"`
            : '';
        const tag = item.action ? 'button' : 'div';
        const buttonType = tag === 'button' ? ' type="button"' : '';
        return `
            <${tag}${buttonType} class="app-list-row" ${attrs}>
                <div class="app-list-row-main">
                    <div class="app-list-row-title">${title}</div>
                    ${subtitle ? `<div class="app-list-row-subtitle">${subtitle}</div>` : ''}
                </div>
                <div class="app-list-row-trailing">${item.action ? '&#8250;' : ''}</div>
            </${tag}>
        `;
    };
    const renderField = (field) => {
        const name = escapeHtml(field.name || '');
        const placeholder = escapeHtml(field.placeholder || '');
        const label = escapeHtml(field.label || field.name || 'Field');
        const lower = String(field.name || '').toLowerCase();
        const multiline = lower.includes('body') || lower.includes('message') || lower.includes('note');
        const input = multiline
            ? `<textarea class="app-textarea" name="${name}" placeholder="${placeholder}"></textarea>`
            : `<input class="app-input" name="${name}" placeholder="${placeholder}" />`;
        return `<label class="app-field"><span class="app-field-label">${label}</span>${input}</label>`;
    };

    if (section.type === 'hero') {
        const actions = (section.actions || []).map((item) =>
            `<button type="button" class="app-pill-btn" data-app-action="${escapeHtml(item.action || '')}" data-app-value="${escapeHtml(item.value || '')}">${escapeHtml(item.label || 'Action')}</button>`
        ).join('');
        return `
            <section class="app-section">
                <div class="app-hero-card">
                    <div class="content-title">Overview</div>
                    <div class="app-hero-title">${sectionTitle}</div>
                    ${sectionBody ? `<div class="app-hero-body">${sectionBody}</div>` : ''}
                    ${actions ? `<div class="app-inline-actions">${actions}</div>` : ''}
                </div>
            </section>
        `;
    }
    if (section.type === 'form') {
        const fields = (section.fields || []).map(renderField).join('');
        return `
            <section class="app-section">
                <div class="app-group">
                    <div class="app-group-header">
                        <div class="app-group-title">${sectionTitle}</div>
                        ${sectionBody ? `<div class="app-group-copy">${sectionBody}</div>` : ''}
                    </div>
                    <form class="app-form" data-app-form="${escapeHtml(section.action || '')}">
                        ${fields}
                        <button class="app-submit" type="submit">${escapeHtml(section.submit_label || 'Submit')}</button>
                    </form>
                </div>
            </section>
        `;
    }
    if (section.type === 'chips') {
        const items = (section.items || []).map((item) =>
            `<button type="button" class="app-pill-btn" data-app-action="${escapeHtml(item.action || '')}" data-app-value="${escapeHtml(item.value || '')}">${escapeHtml(item.label || '')}</button>`
        ).join('');
        return `
            <section class="app-section">
                <div class="app-group">
                    <div class="app-group-header">
                        <div class="app-group-title">${sectionTitle}</div>
                    </div>
                    <div class="app-form" style="padding-top:0;">
                        <div class="app-chip-row">${items}</div>
                    </div>
                </div>
            </section>
        `;
    }
    if (section.type === 'kv') {
        const rows = (section.rows || []).map((row) =>
            `<div class="app-list-row">
                <div class="app-list-row-main">
                    <div class="app-list-row-title">${escapeHtml(row.label || '')}</div>
                </div>
                <div class="app-list-row-subtitle">${escapeHtml(row.value || '')}</div>
            </div>`
        ).join('');
        return `
            <section class="app-section">
                <div class="app-group">
                    <div class="app-group-header">
                        <div class="app-group-title">${sectionTitle}</div>
                    </div>
                    <div class="app-list">${rows}</div>
                </div>
            </section>
        `;
    }
    if (section.type === 'grid') {
        const items = (section.items || []).map((item) => `
            <div class="app-grid-card">
                ${item.image_url ? `<img class="app-grid-image" src="${escapeHtml(item.image_url)}" alt="${escapeHtml(item.title || '')}" />` : ''}
                <div class="app-list-row-title">${escapeHtml(item.title || '')}</div>
                <div class="app-list-row-subtitle">${escapeHtml(item.subtitle || '')}</div>
            </div>
        `).join('');
        return `
            <section class="app-section">
                <div class="app-group-header" style="padding:0 4px;">
                    <div class="app-group-title">${sectionTitle}</div>
                </div>
                <div class="app-grid">${items}</div>
            </section>
        `;
    }
    if (section.type === 'text') {
        return `
            <section class="app-section">
                <div class="app-group">
                    <div class="app-group-header">
                        <div class="app-group-title">${sectionTitle}</div>
                        <div class="app-group-copy">${sectionBody}</div>
                    </div>
                </div>
            </section>
        `;
    }
    const items = (section.items || []).map(renderRow).join('');
    return `
        <section class="app-section">
            <div class="app-group">
                <div class="app-group-header">
                    <div class="app-group-title">${sectionTitle}</div>
                    ${sectionBody ? `<div class="app-group-copy">${sectionBody}</div>` : ''}
                </div>
                <div class="app-list">${items || '<div class="app-list-row"><div class="app-list-row-main"><div class="app-list-row-subtitle">No items</div></div></div>'}</div>
            </div>
        </section>
    `;
}

function renderStructuredApp(payload) {
    const sections = (payload?.sections || []).map(renderStructuredSection).join('');
    appBody.innerHTML = sections || `<div class="content-card"><div class="content-title">${escapeHtml(payload?.title || 'App')}</div><div class="content-text">No content available.</div></div>`;
}

const appTemplateCache = new Map();
const appClientLoaderCache = new Map();
window.__unlim8tedLogs = window.__unlim8tedLogs || [];
let clientLogSequence = 0;

async function sendClientLog(level, scope, message, extra = null) {
    clientLogSequence += 1;
    try {
        await fetch('/api/log/client', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                seq: clientLogSequence,
                level,
                scope,
                message,
                extra
            }),
            keepalive: true
        });
    } catch (_error) {
    }
}

function shellLog(level, scope, message, extra = null) {
    const entry = {
        at: new Date().toISOString(),
        level,
        scope,
        message,
        extra
    };
    window.__unlim8tedLogs.push(entry);
    if (window.__unlim8tedLogs.length > 200) window.__unlim8tedLogs.shift();
    const logger = typeof console[level] === 'function' ? console[level] : console.log;
    if (extra === null || typeof extra === 'undefined') {
        logger(`[UNLIM8TED/${scope}] ${message}`);
        const panel = document.getElementById('debugConsole');
        if (panel) {
            const row = document.createElement('div');
            row.textContent = `[${scope}] ${message}`;
            panel.prepend(row);
            while (panel.childElementCount > 24) panel.removeChild(panel.lastElementChild);
        }
        return;
    }
    logger(`[UNLIM8TED/${scope}] ${message}`, extra);
    const panel = document.getElementById('debugConsole');
    if (panel) {
        const row = document.createElement('div');
        const detail = typeof extra === 'string' ? extra : (extra?.message || JSON.stringify(extra));
        row.textContent = `[${scope}] ${message}${detail ? ` :: ${detail}` : ''}`;
        panel.prepend(row);
        while (panel.childElementCount > 24) panel.removeChild(panel.lastElementChild);
    }
    if (level === 'error') {
        sendClientLog(level, scope, message, extra);
    }
}

window.addEventListener('error', (event) => {
    shellLog('error', 'window', event?.message || 'Unhandled error', {
        source: event?.filename || '',
        line: event?.lineno || 0,
        column: event?.colno || 0,
        stack: event?.error?.stack || ''
    });
});

window.addEventListener('unhandledrejection', (event) => {
    const reason = event?.reason;
    shellLog('error', 'promise', 'Unhandled rejection', {
        message: reason?.message || String(reason || ''),
        stack: reason?.stack || '',
        detail: typeof reason === 'object' ? reason : String(reason || '')
    });
});

document.addEventListener('contextmenu', (event) => {
    event.preventDefault();
});

document.addEventListener('auxclick', (event) => {
    if (event.button === 2) event.preventDefault();
});

function renderAppFailure(appId, title, error) {
    const appName = escapeHtml(title || appTitles[appId] || appId || 'App');
    const detail = escapeHtml(error?.message || String(error || 'Unknown error'));
    shellLog('error', appId || 'app', 'App failure rendered in shell', error);
    appBody.innerHTML = `
        <div class="content-card">
            <div class="content-title">${appName}</div>
            <div class="content-text">This app failed to load. The shell is still running.</div>
            <div class="content-text" style="margin-top:12px;color:rgba(255,200,200,.82);">${detail}</div>
        </div>
    `;
}

async function ensureAppClient(appId, scriptUrl) {
    if (!appId || !scriptUrl) return null;
    if (window.Unlim8tedAppClients?.[appId]) return window.Unlim8tedAppClients[appId];
    if (!appClientLoaderCache.has(appId)) {
        appClientLoaderCache.set(appId, (async () => {
            shellLog('log', appId, 'Loading app client', { scriptUrl });
            const response = await fetch(scriptUrl, { cache: 'no-store' });
            if (!response.ok) throw new Error(`Failed to load app client: ${appId}`);
            const code = await response.text();
            const script = document.createElement('script');
            script.type = 'text/javascript';
            script.text = code + `\n//# sourceURL=${scriptUrl}`;
            document.head.appendChild(script);
            script.remove();
            if (!window.Unlim8tedAppClients?.[appId]) {
                throw new Error('App client did not register itself: ' + appId);
            }
            shellLog('log', appId, 'App client loaded');
            return window.Unlim8tedAppClients?.[appId] || null;
        })());
    }
    try {
        return await appClientLoaderCache.get(appId);
    } catch (error) {
        shellLog('error', appId, 'App client load failed', error);
        appClientLoaderCache.delete(appId);
        return null;
    }
}

function buildAppClientContext(appId, payload) {
    return {
        appId,
        payload,
        state,
        root,
        os,
        appView,
        appBody,
        appTop,
        appTitle,
        requestJson,
        noteActivity,
        closeApp,
        escapeHtml,
        syncSystemState,
        rememberRecentApp,
        fetchAppTemplate,
        renderStructuredSectionsMarkup,
        renderStructuredApp,
        renderHtmlApp
    };
}

async function renderAppClient(payload, appId) {
    const client = await ensureAppClient(appId, payload?.client_script_url || '');
    if (!client || typeof client.render !== 'function') return false;
    shellLog('log', appId, 'Rendering app client');
    await client.render(payload || {}, buildAppClientContext(appId, payload || {}));
    shellLog('log', appId, 'Rendered app client');
    return true;
}

async function fetchAppTemplate(url) {
    if (!url) return '';
    if (appTemplateCache.has(url)) return appTemplateCache.get(url);
    try {
        const response = await fetch(url, { cache: 'no-store' });
        const html = response.ok ? await response.text() : '';
        appTemplateCache.set(url, html);
        shellLog('log', 'template', 'Fetched app template', { url, ok: response.ok });
        return html;
    } catch (error) {
        shellLog('error', 'template', 'Failed to fetch app template', { url, error });
        return '';
    }
}

function renderStructuredSectionsMarkup(payload) {
    return (payload?.sections || []).map(renderStructuredSection).join('');
}

async function renderTemplateApp(payload, appId) {
    try {
        shellLog('log', appId, 'Rendering template app');
        const templateHtml = await fetchAppTemplate(payload?.template_url || '');
        if (!templateHtml) {
            if (await renderAppClient(payload || {}, appId)) {
                return;
            }
            if (payload?.view === 'structured') {
                renderStructuredApp(payload || {});
                return;
            }
            renderHtmlApp(payload, appId);
            return;
        }
        appBody.innerHTML = templateHtml;
        if (await renderAppClient(payload || {}, appId)) {
            return;
        }
        const slot = appBody.querySelector('[data-app-slot="content"]');
        if (!slot) return;
        if (payload?.view === 'structured') {
            slot.innerHTML = renderStructuredSectionsMarkup(payload || {});
            return;
        }
        slot.innerHTML = payload?.html || '';
    } catch (error) {
        renderAppFailure(appId, payload?.title, error);
    }
}

async function renderAppPayload(payload, appId) {
    try {
        appTitle.textContent = payload?.title || appTitles[appId] || 'App';
        appView.dataset.appId = appId || '';
        if (appTop) appTop.style.display = '';
        appView.classList.remove('browser-chrome-only');
        appBody.style.padding = '';
        appBody.style.gap = '';
        appBody.style.alignContent = '';
        appBody.scrollTop = 0;
        if (payload?.view === 'camera' || appId === 'camera') {
            renderCameraApp();
            startCameraPreview();
            return;
        }
        if (payload?.template_url || payload?.client_script_url) {
            await renderTemplateApp(payload || {}, appId);
            return;
        }
        if (payload?.view === 'structured') {
            renderStructuredApp(payload || {});
            return;
        }
        renderHtmlApp(payload, appId);
    } catch (error) {
        renderAppFailure(appId, payload?.title, error);
    }
}

async function openApp(appId) {
    if (state.sleeping) return;
    noteActivity(true);
    closeControlCenter();
    closeAppSwitcher();
    try {
        shellLog('log', appId, 'Opening app');
        const response = await requestJson(`/api/apps/${encodeURIComponent(appId)}`);
        const payload = response?.app || null;
        if (response?.system) applySystemState(response.system);
        rememberRecentApp(appId, payload || {});
        state.appOpen = true;
        state.appId = appId;
        await renderAppPayload(payload || {}, appId);
        appView.classList.add('open');
        shellLog('log', appId, 'App open complete');
    } catch (error) {
        state.appOpen = true;
        state.appId = appId;
        appTitle.textContent = appTitles[appId] || 'App';
        appView.classList.add('open');
        renderAppFailure(appId, appTitles[appId], error);
    }
}



function renderCameraApp() {
    appBody.innerHTML = `
        <div class="content-card">
            <div class="content-title">Camera</div>
            <div class="content-text">Camera preview is unavailable in this shell build.</div>
        </div>
    `;
}

function startCameraPreview() {
    if (state.cameraPoll) {
        clearInterval(state.cameraPoll);
        state.cameraPoll = null;
    }
}

function stopCameraPreview(clearOnly = false) {
    if (state.cameraPoll) {
        clearInterval(state.cameraPoll);
        state.cameraPoll = null;
    }
    return clearOnly;
}

async function closeApp() {
    stopCameraPreview(true);
    closeAppSwitcher();
    blurKeyboardTarget();
    state.appOpen = false;
    state.appId = '';
    appView.dataset.appId = '';
    appView.classList.remove('open');
    appBody.innerHTML = '';
    if (appTop) appTop.style.display = '';
}

function isInteractiveElement(target) {
    return !!target?.closest?.('button, input, textarea, select, a, [data-app-action], [data-app-form]');
}

function isTerminalKeyEvent(event) {
    return state.appId === 'terminal' && !!event.target?.closest?.('#appBody');
}

function snapToPage(index) {
    if (!pages) return;
    const maxIndex = Math.max(0, dots.length - 1);
    const nextIndex = Math.max(0, Math.min(maxIndex, index));
    const width = pages.clientWidth || 1;
    pages.scrollTo({ left: nextIndex * width, behavior: 'smooth' });
    setActivePage(nextIndex);
}

function attachSystemGestures() {
    document.addEventListener('pointerdown', (event) => {
        if (state.sleeping) return;
        state.startX = event.clientX;
        state.startY = event.clientY;
        state.currentY = event.clientY;
        state.startTime = Date.now();
        state.gestureMode = '';
        if (!state.unlocked && event.target.closest('#lockscreen')) {
            state.gestureMode = 'lock-unlock';
            return;
        }
        if (!state.unlocked || state.appOpen) return;
        if (state.controlOpen) {
            state.gestureMode = 'panel-close';
            return;
        }
        if (event.target.closest('#homeScreen')) {
            state.gestureMode = 'home-swipe';
        }
    }, { passive: true });

    document.addEventListener('pointermove', (event) => {
        if (!state.gestureMode) return;
        state.currentY = event.clientY;
    }, { passive: true });

    document.addEventListener('pointerup', (event) => {
        if (state.sleeping || !state.gestureMode) return;
        const deltaX = event.clientX - state.startX;
        const deltaY = event.clientY - state.startY;
        const elapsed = Date.now() - state.startTime;
        const mode = state.gestureMode;
        state.gestureMode = '';
        if (elapsed >= 1200) return;
        const horizontal = Math.abs(deltaX) > Math.abs(deltaY);
        if (mode === 'lock-unlock' && deltaY < -42) {
            unlock();
            return;
        }
        if (mode === 'panel-close' && deltaY < -42) {
            closeControlCenter();
            return;
        }
        if (mode === 'home-swipe' && horizontal && Math.abs(deltaX) > 28) {
            state.suppressAppClick = true;
            snapToPage(state.pageIndex + (deltaX < 0 ? 1 : -1));
            window.setTimeout(() => {
                state.suppressAppClick = false;
            }, 140);
            return;
        }
        if (mode === 'home-swipe' && deltaY > 42) {
            openControlCenter();
            return;
        }
        if (mode === 'home-swipe' && deltaY < -42) {
            openAppSwitcher();
        }
    }, { passive: true });
}

function bindShellUi() {
    stabilizeViewport();
    window.addEventListener('resize', stabilizeViewport, { passive: true });
    window.visualViewport?.addEventListener?.('resize', stabilizeViewport, { passive: true });
    window.visualViewport?.addEventListener?.('scroll', stabilizeViewport, { passive: true });
    renderKeyboardLayout();
    document.querySelectorAll('.app').forEach((button) => {
        button.addEventListener('click', async () => {
            if (state.suppressAppClick) {
                state.suppressAppClick = false;
                return;
            }
            if (state.sleeping) return;
            if (!state.unlocked) unlock();
            closeControlCenter();
            closeAppSwitcher();
            await openApp(button.dataset.app || '');
        });
    });

    toggles.forEach((toggle) => {
        toggle.addEventListener('click', async () => {
            const action = toggle.dataset.action || '';
            if (!action) return;
            const payload = await sendSystemCommand(action);
            if (payload?.system) applySystemState(payload.system);
        });
    });

    brightnessRange?.addEventListener('input', () => {
        brightnessValue.textContent = `${brightnessRange.value}%`;
    });
    brightnessRange?.addEventListener('change', () => {
        setBrightness(Number(brightnessRange.value || 68));
    });

    sleepButton?.addEventListener('click', () => sleepSystem('button'));
    restartButton?.addEventListener('click', () => rebootSystem('button'));
    shutdownButton?.addEventListener('click', () => shutdownSystem('button'));
    closeAppBtn?.addEventListener('click', () => closeApp());
    ccBackdrop?.addEventListener('click', () => closeControlCenter());
    topConfigHandle?.addEventListener('click', () => openControlCenter());
    statusTimeButton?.addEventListener('click', () => openControlCenter());
    switchAppsBtn?.addEventListener('click', () => openAppSwitcher());
    homeNavBtn?.addEventListener('click', () => closeApp());
    quickSettingsBtn?.addEventListener('click', () => openControlCenter());

    pages?.addEventListener('scroll', () => {
        const width = pages.clientWidth || 1;
        setActivePage(Math.round(pages.scrollLeft / width));
    }, { passive: true });

    document.addEventListener('pointerdown', (event) => {
        if (state.sleeping) {
            wakeSystem('tap');
            return;
        }
        noteActivity();
    }, { passive: true });

    document.addEventListener('focusin', (event) => {
        if (isTextEntryTarget(event.target)) {
            focusKeyboardTarget(event.target);
        }
    });

    document.addEventListener('load', (event) => {
        const frame = event.target;
        if (frame?.tagName === 'IFRAME') bindIframeKeyboardBridge(frame);
    }, true);
    document.querySelectorAll('iframe').forEach(bindIframeKeyboardBridge);

    document.addEventListener('pointerdown', (event) => {
        if (!state.keyboardVisible) return;
        if (keyboardLayer?.contains(event.target)) return;
        if (isTextEntryTarget(event.target)) return;
        blurKeyboardTarget();
    });

    keyboardPredictions?.addEventListener('click', (event) => {
        const suggestion = event.target?.closest?.('[data-keyboard-suggestion]')?.dataset?.keyboardSuggestion;
        if (suggestion) applySuggestion(suggestion);
    });

    keyboardHideBtn?.addEventListener('click', () => blurKeyboardTarget());
    debugToggleBtn?.addEventListener('click', () => setDebugVisible(!state.debugVisible));
    keyboardCopyBtn?.addEventListener('click', async () => {
        const target = keyboardState.target;
        if (!target) return;
        const value = currentTextValue();
        const start = typeof target.selectionStart === 'number' ? target.selectionStart : 0;
        const end = typeof target.selectionEnd === 'number' ? target.selectionEnd : value.length;
        keyboardState.clipboardMemory = value.slice(start, end) || value;
        if (navigator.clipboard?.writeText && keyboardState.clipboardMemory) {
            try { await navigator.clipboard.writeText(keyboardState.clipboardMemory); } catch (_error) {}
        }
    });
    keyboardClipboardBtn?.addEventListener('click', async () => {
        let pasteText = keyboardState.clipboardMemory;
        if (navigator.clipboard?.readText) {
            try {
                pasteText = await navigator.clipboard.readText() || pasteText;
            } catch (_error) {
            }
        }
        if (pasteText) replaceSelection(pasteText);
    });

    keyboardLayer?.addEventListener('pointerdown', (event) => {
        const keyButton = event.target?.closest?.('[data-keyboard-key]');
        if (!keyButton) return;
        noteActivity(true);
        keyboardState.glidePointerId = event.pointerId;
        keyboardState.glideStartAt = Date.now();
        keyboardState.glideActive = true;
        keyboardState.glideKeys = [];
        keyboardState.glidePoints = [];
        keyboardLayer.setPointerCapture?.(event.pointerId);
        const key = keyButton.dataset.keyboardKey || '';
        if (/^[a-z']$/.test(key)) {
            keyboardState.glideKeys.push(key);
            keyButton.classList.add('glide-hit');
            appendGlidePoint(event.clientX, event.clientY);
        } else {
            commitKeyboardKey(key);
            keyboardState.glideActive = false;
        }
        event.preventDefault();
    });

    keyboardLayer?.addEventListener('pointermove', (event) => {
        if (!keyboardState.glideActive || keyboardState.glidePointerId !== event.pointerId) return;
        const keyButton = document.elementFromPoint(event.clientX, event.clientY)?.closest?.('[data-keyboard-key]');
        const key = keyButton?.dataset?.keyboardKey || '';
        appendGlidePoint(event.clientX, event.clientY);
        if (!/^[a-z']$/.test(key)) return;
        if (keyboardState.glideKeys[keyboardState.glideKeys.length - 1] === key) return;
        keyboardState.glideKeys.push(key);
        keyButton.classList.add('glide-hit');
        event.preventDefault();
    });

    const endKeyboardPointer = (event) => {
        if (keyboardState.glidePointerId !== event.pointerId) return;
        keyboardLayer?.releasePointerCapture?.(event.pointerId);
        document.querySelectorAll('.keyboard-key.glide-hit').forEach((node) => node.classList.remove('glide-hit'));
        const elapsed = Date.now() - keyboardState.glideStartAt;
        const keys = keyboardState.glideKeys.slice();
        keyboardState.glideActive = false;
        keyboardState.glidePointerId = null;
        keyboardState.glideKeys = [];
        window.setTimeout(() => {
            keyboardState.glidePoints = [];
            if (keyboardGlidePath) keyboardGlidePath.innerHTML = '';
        }, 80);
        if (!keys.length) return;
        if (keys.length === 1 || elapsed < 120) {
            commitKeyboardKey(keys[0]);
            return;
        }
        applySuggestion(decodeGlideWord(keys));
    };
    keyboardLayer?.addEventListener('pointerup', endKeyboardPointer);
    keyboardLayer?.addEventListener('pointercancel', endKeyboardPointer);

    document.addEventListener('keydown', (event) => {
        if (!isTerminalKeyEvent(event)) return;
        if (event.key === 'Escape' || event.ctrlKey || event.metaKey || event.altKey) {
            event.preventDefault();
        }
    }, true);

    document.addEventListener('keydown', (event) => {
        if (!keyboardLayer?.contains(event.target) && event.isTrusted && event.key.length) {
            state.hardwareKeyboardSeen = true;
            if (state.keyboardVisible && isTextEntryTarget(event.target)) setKeyboardVisible(false);
        }
        if (state.sleeping) {
            wakeSystem('key');
            return;
        }
        noteActivity(false);
        if (isTerminalKeyEvent(event)) return;
        const typingTarget = isTextEntryTarget(event.target);
        if (event.key === 'Escape') {
            if (state.appSwitcherOpen) {
                closeAppSwitcher();
                return;
            }
            if (state.appOpen) {
                closeApp();
                return;
            }
            if (state.controlOpen) {
                closeControlCenter();
            }
        }
        if (typingTarget) return;
        if (event.key.toLowerCase() === 'q') {
            event.preventDefault();
            exitKiosk('keyboard-q');
            return;
        }
        if (event.key.toLowerCase() === 'p') {
            sleepSystem('power-key');
        }
    });

    attachSystemGestures();
}

restoreHomeLayout();
updateTime();
setInterval(updateTime, 1000);
syncSystemState();
scheduleIdleSleep();
bindShellUi();
shellLog('log', 'shell', 'Shell bootstrap restored');

