(() => { 
  'use strict';

  /************************************************************
   * Global Namespace (single, sealed)
   ************************************************************/
  if (window.TryOn) return;

  const TryOn = {};
  Object.defineProperty(window, 'TryOn', {
    value: TryOn,
    writable: false,
    configurable: false
  });

  /************************************************************
   * Utilities
   ************************************************************/
  const uid = () => Math.random().toString(36).slice(2);

  const freeze = obj => Object.freeze(obj);

  /************************************************************
   * Shadow DOM Mount
   ************************************************************/
  function createShadowRoot(host) {
    const root = host.attachShadow({ mode: 'closed' });
    return root;
  }

  /************************************************************
   * Branding Enforcement
   ************************************************************/
  function createBranding() {
    const brand = document.createElement('div');
    brand.setAttribute('part', 'branding');
    brand.textContent = 'Powered by Unlim8ted Try-On';
    brand.style.cssText = `
      font-family: system-ui, sans-serif;
      font-size: 12px;
      opacity: 0.7;
      text-align: center;
      padding: 6px;
      user-select: none;
      pointer-events: none;
    `;
    return brand;
  }

  function enforceBranding(root, brand) {
    const observer = new MutationObserver(() => {
      if (!root.contains(brand)) {
        root.innerHTML = '';
        root.appendChild(createFailureState());
      }
    });

    observer.observe(root, {
      childList: true,
      subtree: true
    });
  }

  function createFailureState() {
    const fail = document.createElement('div');
    fail.textContent = 'Unlim8ted Try-On unavailable';
    fail.style.cssText = `
      padding: 24px;
      text-align: center;
      font-family: system-ui, sans-serif;
      opacity: 0.6;
    `;
    return fail;
  }

  /************************************************************
   * Core Shell Layout
   ************************************************************/
  function createShell({ garment, category, theme }) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
      width: 100%;
      max-width: 420px;
      border-radius: 12px;
      overflow: hidden;
      background: ${theme === 'dark' ? '#111' : '#fff'};
      box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    `;

    const stage = document.createElement('div');
    stage.style.cssText = `
      position: relative;
      width: 100%;
      aspect-ratio: 3 / 4;
      background: #eee;
    `;
    stage.textContent = 'Try-On Preview (pipeline coming online)';

    const branding = createBranding();

    wrapper.appendChild(stage);
    wrapper.appendChild(branding);

    return { wrapper, branding };
  }

  /************************************************************
   * Mount Logic
   ************************************************************/
  function mountOne(host, options = {}) {
    if (host.__tryonMounted) return;
    host.__tryonMounted = true;

    const shadow = createShadowRoot(host);

    const garment =
      options.garment || host.getAttribute('data-garment');
    const category =
      options.category || host.getAttribute('data-category') || 'top';
    const theme =
      options.theme || host.getAttribute('data-theme') || 'light';

    const { wrapper, branding } = createShell({
      garment,
      category,
      theme
    });

    shadow.appendChild(wrapper);
    enforceBranding(shadow, branding);
  }

  /************************************************************
   * Public API
   ************************************************************/
  TryOn.mount = function mount(config = {}) {
    const selector = config.selector || '[data-tryon]';
    const nodes = document.querySelectorAll(selector);

    nodes.forEach(node => mountOne(node, config));
  };

  freeze(TryOn);

  /************************************************************
   * Auto-mount on load
   ************************************************************/
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => TryOn.mount());
  } else {
    TryOn.mount();
  }
})();
