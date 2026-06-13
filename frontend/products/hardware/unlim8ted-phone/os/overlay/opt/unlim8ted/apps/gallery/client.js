window.Unlim8tedAppClients = window.Unlim8tedAppClients || {};
window.Unlim8tedAppClients.gallery = (() => {
    let currentCtx = null;

    async function sendAction(action, payload = {}) {
        if (!currentCtx) return null;
        const response = await currentCtx.requestJson('/api/apps/gallery/action', {
            method: 'POST',
            body: JSON.stringify({ action, payload })
        });
        if (response?.app) {
            currentCtx.payload = response.app;
            render(response.app, currentCtx);
            currentCtx.rememberRecentApp?.('gallery', response.app);
        }
        if (response?.system) currentCtx.syncSystemState?.();
        return response;
    }

    function stageMarkup(selected) {
        if (!selected) {
            return '<div class="gallery-stage-empty"><div class="files-form-title">No captures yet</div><div>Open Camera and take a photo to populate the gallery.</div></div>';
        }
        return `<img class="gallery-stage-image" src="${currentCtx.escapeHtml(selected.url || '')}" alt="${currentCtx.escapeHtml(selected.name || 'Capture')}" />`;
    }

    function detailsMarkup(selected) {
        if (!selected) {
            return '<div class="gallery-stage-empty"><div>Select an image to inspect it here.</div></div>';
        }
        return `
            <div class="files-form-title">${currentCtx.escapeHtml(selected.name || '')}</div>
            <div class="gallery-detail-row">
                <div class="gallery-detail-label">Captured</div>
                <div class="gallery-detail-value">${currentCtx.escapeHtml(selected.created_label || '')}</div>
            </div>
            <div class="gallery-detail-row">
                <div class="gallery-detail-label">Size</div>
                <div class="gallery-detail-value">${currentCtx.escapeHtml(selected.size_label || '')}</div>
            </div>
            <div class="gallery-detail-row">
                <div class="gallery-detail-label">Path</div>
                <div class="gallery-detail-value">${currentCtx.escapeHtml(selected.url || '')}</div>
            </div>
            <button class="gallery-delete" id="galleryDeleteBtn" type="button">Delete Capture</button>
        `;
    }

    function gridMarkup(items, selectedName) {
        if (!items?.length) {
            return '<div class="gallery-stage-empty" style="padding:18px;"><div>No captures saved yet.</div></div>';
        }
        return items.map((item) => `
            <button type="button" class="gallery-thumb ${item.name === selectedName ? 'selected' : ''}" data-gallery-select="${currentCtx.escapeHtml(item.name || '')}">
                <img src="${currentCtx.escapeHtml(item.url || '')}" alt="${currentCtx.escapeHtml(item.name || '')}" />
                <div>
                    <div class="gallery-thumb-title">${currentCtx.escapeHtml(item.name || '')}</div>
                    <div class="gallery-thumb-meta">${currentCtx.escapeHtml(item.created_label || '')}</div>
                    <div class="gallery-thumb-meta">${currentCtx.escapeHtml(item.size_label || '')}</div>
                </div>
            </button>
        `).join('');
    }

    function bindEvents(selected) {
        currentCtx.appBody.querySelectorAll('[data-gallery-select]').forEach((button) => {
            button.addEventListener('click', () => sendAction('select_capture', { value: button.dataset.gallerySelect || '' }));
        });
        currentCtx.appBody.querySelector('#galleryDeleteBtn')?.addEventListener('click', () => {
            if (!selected?.name) return;
            sendAction('delete_capture', { value: selected.name });
        });
    }

    async function render(payload, ctx) {
        currentCtx = ctx;
        currentCtx.payload = payload || {};
        const gallery = payload?.gallery || {};
        const items = gallery.items || [];
        const selected = gallery.selected || null;

        const subline = currentCtx.appBody.querySelector('#gallerySubline');
        const status = currentCtx.appBody.querySelector('#galleryStatus');
        const stage = currentCtx.appBody.querySelector('#galleryStage');
        const details = currentCtx.appBody.querySelector('#galleryDetails');
        const grid = currentCtx.appBody.querySelector('#galleryGrid');

        if (subline) subline.textContent = payload?.subtitle || `${items.length} captures saved locally`;
        if (status) status.textContent = gallery.notice || (selected ? `Viewing ${selected.name}` : 'Browse recent photos and manage saved captures.');
        if (stage) stage.innerHTML = stageMarkup(selected);
        if (details) details.innerHTML = detailsMarkup(selected);
        if (grid) grid.innerHTML = gridMarkup(items, selected?.name || '');

        bindEvents(selected);
    }

    return { render };
})();
