// /js/popup-checkout.js

export function buildSquareRedirectUrl(baseUrl, redirectTo) {
  const u = String(baseUrl || "");
  const join = u.includes("?") ? "&" : "?";
  return u + join + "redirect_url=" + encodeURIComponent(redirectTo);
}

/**
 * Opens a popup. While it's cross-origin, reading location throws (ignored).
 * When it returns to your origin, we capture the URL and close it.
 */
export function openCheckoutPopup(url, { onReturn, onClose } = {}) {
  const w = 520, h = 740;
  const left = Math.max(0, (window.screen.width - w) / 2);
  const top = Math.max(0, (window.screen.height - h) / 2);

  const popup = window.open(
    url,
    "unlim8tedCheckout",
    `width=${w},height=${h},left=${left},top=${top},resizable=yes,scrollbars=yes`
  );

  if (!popup) {
    onClose?.({ reason: "blocked" });
    return null;
  }

  const timer = setInterval(() => {
    try {
      if (popup.closed) {
        clearInterval(timer);
        onClose?.({ reason: "closed" });
        return;
      }

      // Only works once popup is back on your origin.
      const href = popup.location.href;

      // If it returned to your site, close and notify.
      if (href.startsWith(window.location.origin)) {
        clearInterval(timer);
        try { popup.close(); } catch {}
        onReturn?.(href);
      }
    } catch {
      // Still cross-origin (Square). Ignore.
    }
  }, 250);

  return popup;
}
