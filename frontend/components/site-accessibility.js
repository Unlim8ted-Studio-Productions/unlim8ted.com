(() => {
if (!document.getElementById("checkoutOverlay")) return;
// Shared focus handling for the existing checkout overlays, including nested
// confirmation dialogs. Restores the initiating control when a dialog closes.
let active = null;
let returnTo = null;
const focusable = dialog => [...dialog.querySelectorAll('button, a[href], input, select, textarea, [tabindex="0"]')].filter(el => !el.disabled && el.getClientRects().length);
function updateDialogs() {
  const overlays = [...document.querySelectorAll('.overlay.show')];
  const overlay = overlays.at(-1);
  const dialog = overlay?.querySelector('.modal');
  if (dialog === active) return;
  if (dialog) {
    if (!active) returnTo = document.activeElement;
    active = dialog;
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    const heading = dialog.querySelector('h2, h3');
    dialog.setAttribute('aria-label', heading?.textContent || 'Checkout');
    dialog.tabIndex = -1;
    (focusable(dialog)[0] || dialog).focus();
  } else {
    active = null;
    document.querySelectorAll('body > main, body > site-navbar').forEach(el => { el.inert = false; });
    returnTo?.focus();
    returnTo = null;
  }
  document.querySelectorAll('body > main, body > site-navbar').forEach(el => { el.inert = !!dialog; });
}
document.addEventListener('keydown', event => {
  if (!active) return;
  if (event.key === 'Escape') {
    const close = active.querySelector('.modalClose');
    if (close) { event.preventDefault(); close.click(); }
  }
  if (event.key === 'Tab') {
    const targets = focusable(active);
    const first = targets[0] || active;
    const last = targets.at(-1) || active;
    if (event.shiftKey && (document.activeElement === first || !active.contains(document.activeElement))) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && (document.activeElement === last || !active.contains(document.activeElement))) { event.preventDefault(); first.focus(); }
  }
});
new MutationObserver(updateDialogs).observe(document.body, { subtree: true, attributes: true, attributeFilter: ['class'] });
updateDialogs();

})();
