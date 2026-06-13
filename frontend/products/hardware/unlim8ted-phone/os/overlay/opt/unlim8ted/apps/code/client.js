window.Unlim8tedAppClients = window.Unlim8tedAppClients || {};
window.Unlim8tedAppClients.code = (() => {
    let currentCtx = null;

    async function sendAction(action, payload = {}) {
        if (!currentCtx) return null;
        const response = await currentCtx.requestJson('/api/apps/code/action', {
            method: 'POST',
            body: JSON.stringify({ action, payload })
        });
        if (response?.app) {
            currentCtx.payload = response.app;
            render(response.app, currentCtx);
            currentCtx.rememberRecentApp?.('code', response.app);
        }
        if (response?.system) currentCtx.syncSystemState?.();
        return response;
    }

    function entryIcon(entry) {
        if (entry.kind === 'nav') return '..';
        if (entry.kind === 'dir') return 'DIR';
        return entry.editable ? '</>' : 'BIN';
    }

    function entriesMarkup(entries) {
        if (!entries.length) {
            return '<div class="code-selection">This workspace is empty. Create a project or add a file to get started.</div>';
        }
        return entries.map((entry) => `
            <button type="button" class="code-entry ${entry.selected ? 'is-selected' : ''}" data-code-action="${currentCtx.escapeHtml(entry.action || '')}" data-code-value="${currentCtx.escapeHtml(entry.value || '')}">
                <div class="code-entry-icon">${currentCtx.escapeHtml(entryIcon(entry))}</div>
                <div>
                    <div class="code-entry-name">${currentCtx.escapeHtml(entry.name || '')}</div>
                    <div class="code-entry-meta">${currentCtx.escapeHtml(entry.meta || '')}</div>
                </div>
                <div class="code-entry-badge">${currentCtx.escapeHtml(entry.kind === 'dir' ? 'Open' : entry.editable ? 'Edit' : 'View')}</div>
            </button>
        `).join('');
    }

    function lineNumberMarkup(value) {
        const lines = Math.max(1, String(value || '').split('\n').length);
        return Array.from({ length: lines }, (_item, index) => String(index + 1)).join('\n');
    }

    function templateOptionsMarkup(templates) {
        return ['<option value="">Blank file</option>'].concat(
            (templates || []).map((template) => `<option value="${currentCtx.escapeHtml(template.id || '')}">${currentCtx.escapeHtml(template.label || '')} starter</option>`)
        ).join('');
    }

    function syncLineNumbers() {
        const editor = currentCtx.appBody.querySelector('#codeEditorInput');
        const numbers = currentCtx.appBody.querySelector('#codeLineNumbers');
        if (!editor || !numbers) return;
        numbers.textContent = lineNumberMarkup(editor.value);
        numbers.scrollTop = editor.scrollTop;
    }

    function bindEvents(payload) {
        currentCtx.appBody.querySelectorAll('[data-code-action]').forEach((button) => {
            button.addEventListener('click', () => {
                sendAction(button.dataset.codeAction || '', { value: button.dataset.codeValue || '' });
            });
        });

        currentCtx.appBody.querySelector('#codeProjectForm')?.addEventListener('submit', (event) => {
            event.preventDefault();
            const form = new FormData(event.currentTarget);
            sendAction('create_project', { name: String(form.get('name') || '') });
            event.currentTarget.reset();
        });

        currentCtx.appBody.querySelector('#codeFolderForm')?.addEventListener('submit', (event) => {
            event.preventDefault();
            const form = new FormData(event.currentTarget);
            sendAction('create_folder', { name: String(form.get('name') || '') });
            event.currentTarget.reset();
        });

        currentCtx.appBody.querySelector('#codeFileForm')?.addEventListener('submit', (event) => {
            event.preventDefault();
            const form = new FormData(event.currentTarget);
            sendAction('create_file', {
                name: String(form.get('name') || ''),
                template: String(form.get('template') || '')
            });
            event.currentTarget.reset();
            const select = currentCtx.appBody.querySelector('#codeTemplateSelect');
            if (select) select.value = '';
        });

        currentCtx.appBody.querySelector('#codeRenameForm')?.addEventListener('submit', (event) => {
            event.preventDefault();
            const form = new FormData(event.currentTarget);
            sendAction('rename_entry', {
                value: payload?.selected_path || '',
                name: String(form.get('name') || '')
            });
        });

        currentCtx.appBody.querySelector('#codeDeleteBtn')?.addEventListener('click', () => {
            if (!payload?.selected_path) return;
            sendAction('delete_entry', { value: payload.selected_path });
        });

        const editor = currentCtx.appBody.querySelector('#codeEditorInput');
        editor?.addEventListener('input', syncLineNumbers);
        editor?.addEventListener('scroll', syncLineNumbers);

        currentCtx.appBody.querySelector('#codeSaveBtn')?.addEventListener('click', () => {
            if (!payload?.selected_path || !payload?.selected_editable) return;
            sendAction('save_file', {
                value: payload.selected_path,
                body: editor?.value || ''
            });
        });
    }

    async function render(payload, ctx) {
        currentCtx = ctx;
        currentCtx.payload = payload || {};

        const pathLabel = ctx.appBody.querySelector('#codePathLabel');
        const status = ctx.appBody.querySelector('#codeStatus');
        const stats = ctx.appBody.querySelector('#codeStats');
        const entries = ctx.appBody.querySelector('#codeEntryList');
        const editorTitle = ctx.appBody.querySelector('#codeEditorTitle');
        const languageChip = ctx.appBody.querySelector('#codeLanguageChip');
        const editor = ctx.appBody.querySelector('#codeEditorInput');
        const editorStatus = ctx.appBody.querySelector('#codeEditorStatus');
        const renameInput = ctx.appBody.querySelector('#codeRenameInput');
        const selectionMeta = ctx.appBody.querySelector('#codeSelectionMeta');
        const templateSelect = ctx.appBody.querySelector('#codeTemplateSelect');
        const saveBtn = ctx.appBody.querySelector('#codeSaveBtn');
        const deleteBtn = ctx.appBody.querySelector('#codeDeleteBtn');

        if (pathLabel) pathLabel.textContent = payload?.path_label || 'Projects';
        if (status) status.textContent = payload?.notice || payload?.subtitle || 'Build small projects directly on the device.';
        if (stats) {
            const values = payload?.stats || {};
            stats.innerHTML = [
                `<div class="code-chip">${ctx.escapeHtml(String(values.folders || 0))} folders</div>`,
                `<div class="code-chip">${ctx.escapeHtml(String(values.files || 0))} files</div>`,
                `<div class="code-chip">${ctx.escapeHtml(String(values.code_files || 0))} source</div>`
            ].join('');
        }
        if (entries) entries.innerHTML = entriesMarkup(payload?.entries || []);
        if (editorTitle) editorTitle.textContent = payload?.editor?.title || 'No file selected';
        if (languageChip) languageChip.textContent = payload?.editor?.language || 'Text';
        if (editor) {
            editor.value = payload?.editor?.body || '';
            editor.readOnly = Boolean(payload?.editor?.read_only);
        }
        if (editorStatus) editorStatus.textContent = payload?.editor?.status || 'Ready.';
        if (renameInput) renameInput.value = payload?.selected_name || '';
        if (selectionMeta) {
            selectionMeta.textContent = payload?.selected_relative_path
                ? `${payload.selected_relative_path} (${payload.selected_kind || 'file'})`
                : 'No file selected.';
        }
        if (templateSelect) templateSelect.innerHTML = templateOptionsMarkup(payload?.templates || []);
        if (saveBtn) saveBtn.disabled = !payload?.selected_editable;
        if (deleteBtn) deleteBtn.disabled = !payload?.selected_path;

        syncLineNumbers();
        bindEvents(payload || {});
    }

    return { render };
})();
