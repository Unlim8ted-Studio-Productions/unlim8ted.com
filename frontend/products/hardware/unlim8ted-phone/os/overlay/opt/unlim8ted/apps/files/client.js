window.Unlim8tedAppClients = window.Unlim8tedAppClients || {};
window.Unlim8tedAppClients.files = (() => {
    let currentCtx = null;
    let dialogMode = '';

    async function sendAction(action, payload = {}) {
        if (!currentCtx) return null;
        const response = await currentCtx.requestJson('/api/apps/files/action', {
            method: 'POST',
            body: JSON.stringify({ action, payload })
        });
        if (response?.app) {
            currentCtx.payload = response.app;
            render(response.app, currentCtx);
            currentCtx.rememberRecentApp?.('files', response.app);
        }
        if (response?.system) currentCtx.syncSystemState?.();
        return response;
    }

    function fileType(entry) {
        if (entry.kind === 'dir') return 'folder';
        if (entry.kind === 'root') return 'root';
        if (entry.kind === 'nav') return 'up';
        const ext = String(entry.extension || '').replace('.', '').slice(0, 4);
        return ext || 'file';
    }

    function sortedFilteredEntries(payload) {
        const query = String(payload?.query || '').toLowerCase();
        const entries = [...(payload?.entries || [])].filter((entry) => {
            if (!query) return true;
            return String(entry.name || '').toLowerCase().includes(query);
        });
        const rank = (entry) => entry.kind === 'nav' ? 0 : entry.kind === 'root' ? 1 : entry.kind === 'dir' ? 2 : 3;
        const sort = payload?.sort || 'name';
        entries.sort((a, b) => {
            const group = rank(a) - rank(b);
            if (group) return group;
            if (sort === 'date') return String(b.modified_at || '').localeCompare(String(a.modified_at || ''));
            if (sort === 'size') return Number(b.size || 0) - Number(a.size || 0);
            if (sort === 'type') return String(a.extension || a.kind || '').localeCompare(String(b.extension || b.kind || ''));
            return String(a.name || '').localeCompare(String(b.name || ''));
        });
        return entries;
    }

    function rootsMarkup(payload) {
        return (payload?.roots || []).map((root) => `
            <button type="button" class="files-root-chip ${root.path === payload.path ? 'active' : ''}" data-files-action="open_path" data-files-value="${currentCtx.escapeHtml(root.path || '')}">
                ${currentCtx.escapeHtml(root.label || root.path || '')}
            </button>
        `).join('');
    }

    function entriesMarkup(payload) {
        const entries = sortedFilteredEntries(payload);
        if (!entries.length) return '<div class="files-empty">No files match this view.</div>';
        return entries.map((entry) => {
            const type = fileType(entry);
            const iconClass = entry.kind === 'dir' || entry.kind === 'root' ? 'folder' : '';
            return `
                <button type="button" class="files-entry ${entry.selected ? 'selected' : ''}" data-files-action="${currentCtx.escapeHtml(entry.action || '')}" data-files-value="${currentCtx.escapeHtml(entry.value || '')}">
                    <div class="files-icon ${iconClass}">${currentCtx.escapeHtml(type.toUpperCase())}</div>
                    <div>
                        <div class="files-name">${currentCtx.escapeHtml(entry.name || '')}</div>
                        <div class="files-meta">${currentCtx.escapeHtml(entry.meta || entry.description || '')}</div>
                    </div>
                    <div class="files-more">${entry.kind === 'file' ? 'i' : '>'}</div>
                </button>
            `;
        }).join('');
    }

    function previewMarkup(preview) {
        const kind = preview?.kind || 'empty';
        if (kind === 'text') {
            return `
                <form id="filesSaveForm">
                    <textarea class="files-editor-textarea" name="body">${currentCtx.escapeHtml(preview.body || '')}</textarea>
                    <div class="files-dialog-actions" style="margin-top:10px;">
                        <button class="files-mode-btn" type="submit">Save</button>
                    </div>
                </form>
            `;
        }
        if (kind === 'image') {
            return `<img class="files-image-preview" src="${currentCtx.escapeHtml(preview.url || '')}" alt="${currentCtx.escapeHtml(preview.title || 'Preview')}" />`;
        }
        if (kind === 'dir') {
            return `<div class="files-empty">${currentCtx.escapeHtml(preview.body || 'Folder')}</div>`;
        }
        if (kind === 'binary') {
            return `<div class="files-empty">${currentCtx.escapeHtml(preview.body || 'No preview available.')}</div>`;
        }
        return '<div class="files-empty">Tap a file or folder to preview it.</div>';
    }

    function updateViewOptions(partial = {}) {
        const payload = currentCtx.payload || {};
        sendAction('set_view_options', {
            query: partial.query ?? payload.query ?? '',
            sort: partial.sort ?? payload.sort ?? 'name',
            view_mode: partial.view_mode ?? payload.view_mode ?? 'list'
        });
    }

    function openDialog(mode, title, name = '', body = '') {
        dialogMode = mode;
        const dialog = currentCtx.appBody.querySelector('#filesDialog');
        currentCtx.appBody.querySelector('#filesDialogTitle').textContent = title;
        currentCtx.appBody.querySelector('#filesDialogName').value = name;
        currentCtx.appBody.querySelector('#filesDialogBody').value = body;
        currentCtx.appBody.querySelector('#filesDialogBody').style.display = mode === 'create_file' ? '' : 'none';
        dialog?.showModal?.();
    }

    function bindEvents(payload) {
        currentCtx.appBody.querySelectorAll('[data-files-action]').forEach((button) => {
            button.addEventListener('click', () => sendAction(button.dataset.filesAction || '', { value: button.dataset.filesValue || '' }));
        });

        currentCtx.appBody.querySelector('#filesRefreshBtn')?.addEventListener('click', () => sendAction('open_path', { value: payload?.path || '' }));
        currentCtx.appBody.querySelector('#filesModeBtn')?.addEventListener('click', () => {
            updateViewOptions({ view_mode: payload?.view_mode === 'grid' ? 'list' : 'grid' });
        });
        currentCtx.appBody.querySelector('#filesSortSelect')?.addEventListener('change', (event) => updateViewOptions({ sort: event.target.value }));
        currentCtx.appBody.querySelector('#filesSearchForm')?.addEventListener('submit', (event) => {
            event.preventDefault();
            updateViewOptions({ query: String(new FormData(event.currentTarget).get('query') || '') });
        });

        currentCtx.appBody.querySelector('#filesNewFolderBtn')?.addEventListener('click', () => openDialog('create_folder', 'New Folder'));
        currentCtx.appBody.querySelector('#filesNewFileBtn')?.addEventListener('click', () => openDialog('create_file', 'New Text File'));
        currentCtx.appBody.querySelector('#filesRenameBtn')?.addEventListener('click', () => {
            if (!payload?.selected_path) return;
            openDialog('rename_path', 'Rename', payload.selected_name || '');
        });
        currentCtx.appBody.querySelector('#filesOpenBtn')?.addEventListener('click', () => {
            if (payload?.selected_kind === 'dir') sendAction('open_path', { value: payload.selected_path });
        });
        currentCtx.appBody.querySelector('#filesDeleteBtn')?.addEventListener('click', () => {
            if (payload?.selected_path) sendAction('delete_file', { value: payload.selected_path });
        });
        currentCtx.appBody.querySelector('#filesClosePreviewBtn')?.addEventListener('click', () => {
            currentCtx.appBody.querySelector('#filesPreviewSheet')?.classList.remove('visible');
        });
        currentCtx.appBody.querySelector('#filesDialogForm')?.addEventListener('submit', (event) => {
            if (event.submitter?.value !== 'ok') return;
            event.preventDefault();
            const form = new FormData(event.currentTarget);
            const name = String(form.get('name') || '');
            const body = String(form.get('body') || '');
            if (dialogMode === 'rename_path') sendAction('rename_path', { value: payload?.selected_path || '', name });
            if (dialogMode === 'create_folder') sendAction('create_folder', { name });
            if (dialogMode === 'create_file') sendAction('create_file', { name, body });
            currentCtx.appBody.querySelector('#filesDialog')?.close?.();
        });
        currentCtx.appBody.querySelector('#filesSaveForm')?.addEventListener('submit', (event) => {
            event.preventDefault();
            sendAction('save_file', {
                value: payload?.selected_path || '',
                body: String(new FormData(event.currentTarget).get('body') || '')
            });
        });
    }

    async function render(payload, ctx) {
        currentCtx = ctx;
        currentCtx.payload = payload || {};

        const list = currentCtx.appBody.querySelector('#filesEntryList');
        const roots = currentCtx.appBody.querySelector('#filesRootStrip');
        const sheet = currentCtx.appBody.querySelector('#filesPreviewSheet');
        const preview = currentCtx.appBody.querySelector('#filesPreviewPanel');
        const downloadLink = currentCtx.appBody.querySelector('#filesDownloadLink');

        currentCtx.appBody.querySelector('#filesPathLabel').textContent = payload?.path_label || 'Files';
        currentCtx.appBody.querySelector('#filesSearchInput').value = payload?.query || '';
        currentCtx.appBody.querySelector('#filesSortSelect').value = payload?.sort || 'name';
        currentCtx.appBody.querySelector('#filesModeBtn').textContent = payload?.view_mode === 'grid' ? 'List' : 'Grid';
        currentCtx.appBody.querySelector('#filesSelectedName').textContent = payload?.selected_name || 'No item selected';
        currentCtx.appBody.querySelector('#filesSelectedMeta').textContent = payload?.selected_path || payload?.notice || payload?.subtitle || 'Select an item for actions.';

        if (roots) roots.innerHTML = rootsMarkup(payload || {});
        if (list) {
            list.classList.toggle('grid', payload?.view_mode === 'grid');
            list.innerHTML = entriesMarkup(payload || {});
        }
        if (preview) preview.innerHTML = previewMarkup(payload?.preview || {});
        if (sheet) sheet.classList.toggle('visible', Boolean(payload?.selected_path));

        const isFile = payload?.selected_kind === 'file' && payload?.selected_path;
        const isDir = payload?.selected_kind === 'dir';
        currentCtx.appBody.querySelector('#filesOpenBtn').disabled = !isDir;
        currentCtx.appBody.querySelector('#filesRenameBtn').disabled = !payload?.selected_path;
        currentCtx.appBody.querySelector('#filesDeleteBtn').disabled = !payload?.selected_path;
        if (downloadLink) {
            downloadLink.style.display = isFile ? 'inline-grid' : 'none';
            downloadLink.href = isFile ? `/api/apps/files/item?path=${encodeURIComponent(payload.selected_path)}` : '#';
            downloadLink.download = payload?.selected_name || 'download';
        }

        bindEvents(payload || {});
    }

    return { render };
})();
