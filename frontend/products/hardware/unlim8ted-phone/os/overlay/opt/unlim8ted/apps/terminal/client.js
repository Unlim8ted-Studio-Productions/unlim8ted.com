window.Unlim8tedAppClients = window.Unlim8tedAppClients || {};
window.Unlim8tedAppClients.terminal = (() => {
    let ctx = null;
    let output = null;
    let input = null;
    let surface = null;
    let offset = 0;
    let pollTimer = null;
    let lastInputValue = '';
    let buffer = '';

    function normalizeTerminalText(text) {
        let value = String(text || '');
        value = value.replace(/\x1b\][^\x07]*(?:\x07|\x1b\\)/g, '');
        value = value.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');
        value = value.replace(/\x1b[()][A-Za-z0-9]/g, '');
        value = value.replace(/\x1bc/g, '');
        const lines = buffer.split('\n');
        let current = lines.pop() || '';
        for (let index = 0; index < value.length; index += 1) {
            const char = value[index];
            if (char === '\r') {
                current = '';
            } else if (char === '\n') {
                lines.push(current);
                current = '';
            } else if (char === '\b' || char === '\x7f') {
                current = current.slice(0, -1);
            } else if (char === '\t') {
                current += '    ';
            } else if (char >= ' ' || char === '\x1b') {
                current += char === '\x1b' ? '' : char;
            }
        }
        lines.push(current);
        buffer = lines.slice(-1200).join('\n');
    }

    function repaint() {
        if (!output) return;
        output.textContent = buffer;
        output.scrollTop = output.scrollHeight;
    }

    function append(text) {
        if (!text) return;
        normalizeTerminalText(text);
        repaint();
    }

    function terminalSize() {
        const style = output ? getComputedStyle(output) : null;
        const fontSize = parseFloat(style?.fontSize || '13') || 13;
        const lineHeight = parseFloat(style?.lineHeight || String(fontSize * 1.38)) || (fontSize * 1.38);
        const cols = Math.max(20, Math.floor((output?.clientWidth || 320) / (fontSize * 0.61)));
        const rows = Math.max(8, Math.floor((output?.clientHeight || 240) / lineHeight));
        return { rows, cols };
    }

    async function post(path, payload = {}) {
        const response = await ctx.requestJson(`/api/apps/terminal/${path}`, {
            method: 'POST',
            body: JSON.stringify(payload)
        });
        if (response?.output) append(response.output);
        if (response?.offset) offset = response.offset;
        const status = ctx.appBody.querySelector('#terminalStatus');
        if (status) status.textContent = response?.closed ? 'Session closed' : `${terminalSize().cols}x${terminalSize().rows} PTY`;
        return response;
    }

    async function poll() {
        if (ctx?.state?.appId !== 'terminal') {
            if (pollTimer) clearInterval(pollTimer);
            pollTimer = null;
            return;
        }
        try {
            await post('poll', { offset, ...terminalSize() });
        } catch (_error) {
        }
    }

    function focusInput() {
        try {
            input?.focus({ preventScroll: true });
        } catch (_error) {
            input?.focus();
        }
    }

    async function send(data) {
        if (!data) return;
        await post('input', { data, offset });
    }

    function controlKey(event) {
        const key = event.key.toLowerCase();
        if (key.length !== 1 || !event.ctrlKey || event.altKey || event.metaKey) return '';
        const code = key.charCodeAt(0);
        if (code >= 97 && code <= 122) return String.fromCharCode(code - 96);
        return '';
    }

    function specialKey(event) {
        const map = {
            ArrowUp: '\x1b[A',
            ArrowDown: '\x1b[B',
            ArrowRight: '\x1b[C',
            ArrowLeft: '\x1b[D',
            Home: '\x1b[H',
            End: '\x1b[F',
            PageUp: '\x1b[5~',
            PageDown: '\x1b[6~',
            Delete: '\x1b[3~',
            Escape: '\x1b'
        };
        return map[event.key] || '';
    }

    async function pasteFromClipboard() {
        let text = '';
        try {
            text = await navigator.clipboard?.readText?.() || '';
        } catch (_error) {
        }
        if (text) await send(text);
    }

    function softKeyValue(name) {
        return {
            paste: 'paste',
            tab: '\t',
            'ctrl-c': '\x03',
            esc: '\x1b',
            up: '\x1b[A',
            down: '\x1b[B',
            left: '\x1b[D',
            right: '\x1b[C',
            enter: '\r'
        }[name] || '';
    }

    function bindKeyboard() {
        surface?.addEventListener('pointerdown', () => window.setTimeout(focusInput, 0));
        surface?.addEventListener('keydown', (event) => {
            event.stopPropagation();
        });
        ctx.appBody.querySelectorAll('[data-terminal-send]').forEach((button) => {
            button.addEventListener('click', async () => {
                const value = softKeyValue(button.dataset.terminalSend || '');
                if (value === 'paste') await pasteFromClipboard();
                else await send(value);
                focusInput();
            });
        });
        ctx.appBody.querySelector('#terminalClear')?.addEventListener('click', async () => {
            buffer = '';
            repaint();
            offset = 0;
            await post('clear', { offset: 0 });
            focusInput();
        });
        ctx.appBody.querySelector('#terminalExitKiosk')?.addEventListener('click', async () => {
            const status = ctx.appBody.querySelector('#terminalStatus');
            if (status) status.textContent = 'Exiting kiosk...';
            await ctx.requestJson('/api/system/exit-kiosk', {
                method: 'POST',
                body: JSON.stringify({ reason: 'terminal-button' })
            });
        });
        input?.addEventListener('keydown', async (event) => {
            event.stopPropagation();
            const control = controlKey(event);
            if (control) {
                event.preventDefault();
                await send(control);
                input.value = '';
                lastInputValue = '';
                return;
            }
            const special = specialKey(event);
            if (special) {
                event.preventDefault();
                await send(special);
                return;
            }
            if (event.key === 'Enter') {
                event.preventDefault();
                await send('\r');
                input.value = '';
                lastInputValue = '';
                return;
            }
            if (event.key === 'Backspace') {
                event.preventDefault();
                await send('\x7f');
                input.value = '';
                lastInputValue = '';
                return;
            }
            if (event.key === 'Tab') {
                event.preventDefault();
                await send('\t');
                return;
            }
            if (event.key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey) {
                event.preventDefault();
                await send(event.key);
                input.value = '';
                lastInputValue = '';
            }
        });
        input?.addEventListener('paste', async (event) => {
            event.stopPropagation();
            event.preventDefault();
            await send(event.clipboardData?.getData('text') || '');
            input.value = '';
            lastInputValue = '';
        });
        input?.addEventListener('input', async () => {
            const value = input.value || '';
            const delta = value.startsWith(lastInputValue) ? value.slice(lastInputValue.length) : value;
            lastInputValue = value;
            if (delta) await send(delta);
            input.value = '';
            lastInputValue = '';
        });
        input?.addEventListener('terminal-input', async (event) => {
            event.stopPropagation();
            await send(event.detail?.data || '');
            input.value = '';
            lastInputValue = '';
        });
    }

    function render(payload, renderCtx) {
        ctx = renderCtx;
        output = ctx.appBody.querySelector('#terminalOutput');
        input = ctx.appBody.querySelector('#terminalInput');
        surface = ctx.appBody.querySelector('#terminalSurface');
        offset = 0;
        buffer = '';
        ctx.appBody.querySelector('#terminalNotice')?.classList.toggle(
            'visible',
            payload?.terminal?.keyboard_present === false
        );

        if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }

        bindKeyboard();
        post('start', { offset: 0, ...terminalSize() }).finally(() => {
            pollTimer = setInterval(poll, 300);
            window.setTimeout(focusInput, 50);
        });
    }

    return { render };
})();
