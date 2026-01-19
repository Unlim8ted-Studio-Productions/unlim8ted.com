/* =========================================================
           cart.js (ESM) — Unlim8ted custom checkout
           IMPORTANT UPDATE:
           - products.json varients[].id IS the Printful catalog variant id (printful_catalog_variant_id).
           - Cart item variantId is expected to be that same Printful variant id.
           - Worker expects items: { printfulVariantId, qty } where printfulVariantId is that id.
        ========================================================= */

        import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";
        import { collection, onSnapshot, doc, deleteDoc, updateDoc, serverTimestamp, getDocs } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";

        import { getFirebase } from "/components/firebase-init.js";
        const { auth, db } = getFirebase();

        /* =============================
           CONFIG
        ============================= */
        const WORKER_BASE = "https://api.unlim8ted.com";

        // Square IDs (public + safe on frontend)
        const SQUARE_APP_ID = "sq0idp-Nnvnru9L9hR3CwVKikShGA";
        const SQUARE_LOCATION_ID = "L4KPR2BE0PAA4";

        /* =============================
           URL mode: single-item checkout
           /cart?source=buy&product=theglitch
        ============================= */
        const QS = new URLSearchParams(location.search);
        const BUY_MODE = (QS.get("source") || "").toLowerCase() === "buy";
        const BUY_PRODUCT = String(QS.get("product") || "").trim();

        /* =============================
           Donation links
        ============================= */
        const SQUARE_DONATION = {
            one_time: {
                custom: "https://square.link/u/h4bY1vEF",
                "1": "https://square.link/u/bWytGN4p",
                "5": "https://square.link/u/4XBSzfLg",
                "10": "https://square.link/u/AvF6x51S",
            },
            weekly: { custom: "https://square.link/u/FkJZDx1R" },
            monthly: { custom: "https://square.link/u/ZK2wPwwU" },
            annual: { custom: "https://square.link/u/ypx1lQj1" },
        };

        /* =============================
           products.json cache/index
        ============================= */
        let _productsLoaded = false;
        let _productsLoading = null;
        let _productById = new Map();
        let _variantByKey = new Map(); // `${productId}::${variantId}` -> variant
        let _productsArray = [];

        function safeUrl(u) {
            const s = String(u || "").trim();
            if (!s) return "";
            if (s.startsWith("https://") || s.startsWith("http://") || s.startsWith("/")) return s;
            return "";
        }

        async function loadProducts() {
            if (_productsLoaded) return;
            if (_productsLoading) return _productsLoading;

            _productsLoading = (async () => {
                try {
                    const r = await fetch("/tools/data/products.json", { cache: "no-store" });
                    if (!r.ok) throw new Error("Failed to load products.json");
                    const data = await r.json();
                    const products = Array.isArray(data) ? data : (data?.products || []);
                    _productsArray = Array.isArray(products) ? products : [];
                    indexProducts(_productsArray);
                    _productsLoaded = true;
                } catch (e) {
                    console.warn("products.json load failed:", e);
                } finally {
                    _productsLoading = null;
                }
            })();

            return _productsLoading;
        }

        function indexProducts(products) {
            _productById = new Map();
            _variantByKey = new Map();

            for (const p of (products || [])) {
                // ✅ YOUR products.json uses "id" as the product key
                const pid = String(p.id ?? p.productId ?? p.printful_sync_id ?? "").trim();
                if (!pid) continue;

                _productById.set(pid, p);

                const vars = Array.isArray(p.varients) ? p.varients : (Array.isArray(p.variants) ? p.variants : []);
                for (const v of vars) {
                    // ✅ Variant id should be v.id (your rule: printful_catalog_variant_id is just varient.id)
                    const vid = String(v.id ?? v.printful_catalog_variant_id ?? v.variantId ?? "").trim();
                    if (!vid) continue;

                    _variantByKey.set(`${pid}::${vid}`, v);
                }
            }
        }

        function parsePrice(val) {
            if (val == null) return null;
            if (typeof val === "number") return Number.isFinite(val) ? val : null;
            const s = String(val).trim();
            if (!s) return null;
            const cleaned = s.replace(/[^0-9.\-]/g, "");
            if (!cleaned) return null;
            const n = Number(cleaned);
            return Number.isFinite(n) ? n : null;
        }

        function normalizePriceNumber(n) {
            if (!Number.isFinite(n)) return null;
            if (n >= 1000 && Math.abs(n - Math.round(n)) < 1e-9) return n / 100;
            return n;
        }

        const money = (n) => (normalizePriceNumber(Number(n)) ?? 0).toFixed(2);

        function escapeHtml(s) {
            return String(s).replace(/[&<>"']/g, (c) => ({
                "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
            }[c]));
        }

        /* =============================
           Page notice
        ============================= */
        function setNotice(kind, msgHtml) {
            const el = document.getElementById("notice");
            el.style.display = "block";
            el.className = "notice " + (kind || "");
            el.innerHTML = msgHtml;
        }

        function clearNotice() {
            const el = document.getElementById("notice");
            el.style.display = "none";
            el.className = "notice";
            el.innerHTML = "";
        }

        /* =============================
           Confirm modal (clear cart)
        ============================= */
        const modalOverlay = document.getElementById("modalOverlay");
        const modalTitle = document.getElementById("modalTitle");
        const modalBody = document.getElementById("modalBody");
        const modalClose = document.getElementById("modalClose");
        const modalCancel = document.getElementById("modalCancel");
        const modalConfirm = document.getElementById("modalConfirm");
        let _modalResolver = null;
        let _lastFocus = null;

        function openModal({ title = "Confirm", bodyHtml = "", confirmText = "Confirm", confirmClass = "btn btn-danger" } = {}) {
            modalTitle.textContent = title;
            modalBody.innerHTML = bodyHtml;
            modalConfirm.textContent = confirmText;
            modalConfirm.className = confirmClass;

            modalOverlay.classList.add("show");
            modalOverlay.setAttribute("aria-hidden", "false");
            _lastFocus = document.activeElement;
            setTimeout(() => modalConfirm.focus(), 0);

            return new Promise(resolve => { _modalResolver = resolve; });
        }

        function closeModal(result) {
            modalOverlay.classList.remove("show");
            modalOverlay.setAttribute("aria-hidden", "true");
            const r = _modalResolver;
            _modalResolver = null;
            if (typeof r === "function") r(result);
            if (_lastFocus && typeof _lastFocus.focus === "function") _lastFocus.focus();
        }

        modalClose.addEventListener("click", () => closeModal(false));
        modalCancel.addEventListener("click", () => closeModal(false));
        modalConfirm.addEventListener("click", () => closeModal(true));
        modalOverlay.addEventListener("click", (e) => { if (e.target === modalOverlay) closeModal(false); });
        document.addEventListener("keydown", (e) => {
            if (!modalOverlay.classList.contains("show")) return;
            if (e.key === "Escape") closeModal(false);
        });

        /* =============================
           Donation modal
        ============================= */
        const donateOverlay = document.getElementById("donateOverlay");
        const donateClose = document.getElementById("donateClose");
        const donateCancel = document.getElementById("donateCancel");
        const donateGo = document.getElementById("donateGo");
        const donateFreq = document.getElementById("donateFreq");
        const donateAmount = document.getElementById("donateAmount");
        let _donateLastFocus = null;

        function openDonateModal() {
            donateOverlay.classList.add("show");
            donateOverlay.setAttribute("aria-hidden", "false");
            _donateLastFocus = document.activeElement;
            setTimeout(() => donateGo.focus(), 0);
        }

        function closeDonateModal() {
            donateOverlay.classList.remove("show");
            donateOverlay.setAttribute("aria-hidden", "true");
            if (_donateLastFocus && typeof _donateLastFocus.focus === "function") _donateLastFocus.focus();
        }

        donateClose.addEventListener("click", closeDonateModal);
        donateCancel.addEventListener("click", closeDonateModal);
        donateOverlay.addEventListener("click", (e) => { if (e.target === donateOverlay) closeDonateModal(); });
        document.addEventListener("keydown", (e) => {
            if (!donateOverlay.classList.contains("show")) return;
            if (e.key === "Escape") closeDonateModal();
        });

        function openPopup(url, name = "popup", features = "width=520,height=760,noopener,noreferrer") {
            return window.open(url, name, features);
        }

        donateGo.addEventListener("click", () => {
            const freq = donateFreq.value;
            const amt = donateAmount.value;

            const url = SQUARE_DONATION?.[freq]?.[amt] || SQUARE_DONATION?.[freq]?.custom || "";
            if (!url) {
                closeDonateModal();
                setNotice("warn", `<strong>Donation link missing.</strong> Update SQUARE_DONATION.`);
                return;
            }

            const popup = openPopup(url, "squareDonate", "width=520,height=760,noopener,noreferrer");
            closeDonateModal();

            if (!popup) {
                setNotice("warn", `<strong>Popup blocked.</strong> Allow popups and try again.`);
                return;
            }

            setNotice("ok",
                `<div style="display:flex;gap:10px;align-items:flex-start">
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" style="width:18px;height:18px;margin-top:2px;">
            <path d="M20 6 9 17l-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <div><strong>Donation popup opened.</strong><br/>Thank you for supporting Unlim8ted.</div>
        </div>`
            );
        });

        /* =============================
           Expired modal
        ============================= */
        const expiredOverlay = document.getElementById("expiredOverlay");
        const expiredClose = document.getElementById("expiredClose");
        const expiredReload = document.getElementById("expiredReload");
        function showExpired() {
            try { closeCheckoutModal(); } catch { }
            expiredOverlay.classList.add("show");
            expiredOverlay.setAttribute("aria-hidden", "false");
            setTimeout(() => expiredReload.focus(), 0);
        }
        function closeExpired() {
            expiredOverlay.classList.remove("show");
            expiredOverlay.setAttribute("aria-hidden", "true");
        }
        expiredClose.addEventListener("click", () => { closeExpired(); location.reload(); });
        expiredReload.addEventListener("click", () => location.reload());
        expiredOverlay.addEventListener("click", (e) => { if (e.target === expiredOverlay) location.reload(); });
        document.addEventListener("keydown", (e) => {
            if (!expiredOverlay.classList.contains("show")) return;
            if (e.key === "Escape") location.reload();
        });

        /* =============================
           Checkout modal: state + step UI
        ============================= */
        const checkoutOverlay = document.getElementById("checkoutOverlay");
        const checkoutClose = document.getElementById("checkoutClose");
        const btnQuote = document.getElementById("btnQuote");
        const btnPay = document.getElementById("btnPay");

        const shipMethodEl = document.getElementById("ship_method");
        const payStatusEl = document.getElementById("payStatus");
        const quoteHintEl = document.getElementById("quoteHint");

        const qItemsEl = document.getElementById("q_items");
        const qSubtotalEl = document.getElementById("q_subtotal");
        const qShippingEl = document.getElementById("q_shipping");
        const qTotalEl = document.getElementById("q_total");
        const qExpWrap = document.getElementById("q_exp");
        const qExpTime = document.getElementById("q_exp_time");
        const qShipRow = document.getElementById("q_ship_row");
        const qTaxRow = document.getElementById("q_tax_row");
        const qTotalRow = document.getElementById("q_total_row");
        const qTaxEl = document.getElementById("q_tax");
        const cardContainer = document.getElementById("card-container");
        const payHintEl = document.getElementById("payHint");

        let _checkoutLastFocus = null;
        let _quote = null; // { quoteId, shippingOptions, subtotalCents }
        let _quoteExpiresAt = 0;
        let _quoteExpireTimer = null;

        let _squarePayments = null;
        let _squareCard = null;
        let _cardMounted = false;

        let _step = 1; // 1=Address+Shipping, 2=Payment

        let _progressEl = null;
        function ensureProgressUI() {
            if (_progressEl) return;
            const body = checkoutOverlay.querySelector(".modalBody");
            if (!body) return;

            const wrap = document.createElement("div");
            wrap.style.display = "flex";
            wrap.style.alignItems = "center";
            wrap.style.justifyContent = "space-between";
            wrap.style.gap = "10px";
            wrap.style.marginBottom = "12px";

            const left = document.createElement("div");
            left.style.fontWeight = "950";
            left.style.letterSpacing = ".2px";
            left.textContent = "Step 1 of 2";

            const bar = document.createElement("div");
            bar.style.flex = "1";
            bar.style.height = "10px";
            bar.style.borderRadius = "999px";
            bar.style.border = "1px solid rgba(255,255,255,.12)";
            bar.style.background = "rgba(255,255,255,.06)";
            bar.style.overflow = "hidden";

            const fill = document.createElement("div");
            fill.style.height = "100%";
            fill.style.width = "50%";
            fill.style.borderRadius = "999px";
            fill.style.background = "linear-gradient(90deg, rgba(184,107,255,1), rgba(99,214,255,1))";
            bar.appendChild(fill);

            const right = document.createElement("div");
            right.style.color = "rgba(233,231,255,.72)";
            right.style.fontSize = "12px";
            right.textContent = "Address → Payment";

            wrap.appendChild(left);
            wrap.appendChild(bar);
            wrap.appendChild(right);

            _progressEl = { wrap, left, fill };
            body.prepend(wrap);
        }

        function setStep(n) {
            _step = n;
            ensureProgressUI();
            if (_progressEl) {
                _progressEl.left.textContent = `Step ${n} of 2`;
                _progressEl.fill.style.width = n === 1 ? "50%" : "100%";
            }

            if (n === 1) {
                btnQuote.style.display = "inline-flex";
                btnQuote.textContent = "Next";
                btnPay.textContent = "Pay";
                btnPay.disabled = true;

                cardContainer.style.display = "none";
                payHintEl.textContent = "We’ll show secure payment after we calculate shipping.";
                quoteHintEl.textContent = "Enter your address, then click Next to calculate shipping.";
            } else {
                btnQuote.style.display = "none";
                btnPay.style.display = "inline-flex";
                btnPay.textContent = "Pay";
                cardContainer.style.display = "block";
                payHintEl.textContent = "Card details are handled by Square securely.";
                quoteHintEl.textContent = "Quote ready. Review totals and pay.";
                btnPay.disabled = false;
            }
        }

        function setPayStatus(kind, html) {
            payStatusEl.style.display = "block";
            payStatusEl.className = "notice " + (kind || "");
            payStatusEl.innerHTML = html;
        }
        function clearPayStatus() {
            payStatusEl.style.display = "none";
            payStatusEl.className = "notice";
            payStatusEl.innerHTML = "";
        }

        function openCheckoutModal() {
            ensureProgressUI();
            checkoutOverlay.classList.add("show");
            checkoutOverlay.setAttribute("aria-hidden", "false");
            _checkoutLastFocus = document.activeElement;
            setTimeout(() => btnQuote.focus(), 0);
        }

        function closeCheckoutModal() {
            checkoutOverlay.classList.remove("show");
            checkoutOverlay.setAttribute("aria-hidden", "true");
            if (_checkoutLastFocus && typeof _checkoutLastFocus.focus === "function") _checkoutLastFocus.focus();
        }

        checkoutClose.addEventListener("click", () => closeCheckoutModal());
        checkoutOverlay.addEventListener("click", (e) => { if (e.target === checkoutOverlay) closeCheckoutModal(); });
        document.addEventListener("keydown", (e) => {
            if (!checkoutOverlay.classList.contains("show")) return;
            if (e.key === "Escape") closeCheckoutModal();
        });

        function resetCheckoutState() {
            _quote = null;
            _quoteExpiresAt = 0;
            if (_quoteExpireTimer) clearTimeout(_quoteExpireTimer);
            _quoteExpireTimer = null;

            shipMethodEl.innerHTML = `<option value="">Shipping options will appear after Next…</option>`;
            shipMethodEl.disabled = true;

            qShipRow.style.display = "none";
            qTaxRow.style.display = "none";
            qTotalRow.style.display = "none";
            qTaxEl.textContent = "0.00";

            clearPayStatus();

            qItemsEl.textContent = "0";
            qSubtotalEl.textContent = "0.00";
            qShippingEl.textContent = "0.00";
            qTotalEl.textContent = "0.00";
            qExpWrap.style.display = "none";
            qExpTime.textContent = "";

            setStep(1);
        }

        function scheduleQuoteExpiry(expiresAtMs) {
            if (_quoteExpireTimer) clearTimeout(_quoteExpireTimer);
            const msLeft = Math.max(0, expiresAtMs - Date.now());
            _quoteExpireTimer = setTimeout(() => showExpired(), msLeft);
        }

        function formatTime(msEpoch) {
            const d = new Date(msEpoch);
            const hh = String(d.getHours()).padStart(2, "0");
            const mm = String(d.getMinutes()).padStart(2, "0");
            return `${hh}:${mm}`;
        }

        /* =============================
           Square card
        ============================= */
        async function ensureSquareCardMounted() {
            if (!window.Square) throw new Error("Square Web Payments SDK not loaded.");
            if (_cardMounted && _squareCard) return;

            _squarePayments = _squarePayments || window.Square.payments(SQUARE_APP_ID, SQUARE_LOCATION_ID);
            _squareCard = _squareCard || await _squarePayments.card();

            await _squareCard.attach("#card-container");
            _cardMounted = true;
        }

        /* =============================
           Catalog resolution helpers
        ============================= */
        function resolveProductImage(productId, variantId) {
            const pid = String(productId || "").trim();
            if (!pid) return "";
            const p = _productById.get(pid) || null;
            if (!p) return "";

            const vid = String(variantId || "").trim();
            if (vid) {
                const v = _variantByKey.get(`${pid}::${vid}`) || null;
                const vImg =
                    (v && Array.isArray(v.images) && v.images.length && safeUrl(v.images[0])) ||
                    safeUrl(v?.image) ||
                    safeUrl(v?.imageUrl) ||
                    safeUrl(v?.thumbnail) ||
                    "";
                if (vImg) return vImg;
            }

            const pImg =
                safeUrl(p.image) ||
                safeUrl(p.imageUrl) ||
                safeUrl(p.thumbnail) ||
                safeUrl(p.thumb) ||
                "";
            if (pImg) return pImg;

            if (Array.isArray(p.images) && p.images.length) {
                const first = p.images[0];
                if (typeof first === "string") return safeUrl(first);
                if (first && typeof first === "object") return safeUrl(first.url || first.src || first.imageUrl || first.image);
            }
            return "";
        }

        // UPDATED: printful_catalog_variant_id is just varient.id (and cart variantId matches it)
        // Resolve the Printful catalog/sync variant ID for a cart item
        function getPrintfulVariantIdForCartItem(it) {
            // 0) already explicit on cart item
            const direct = it?.printful_catalog_variant_id;
            if (direct != null && String(direct).trim() !== "") return String(direct).trim();

            const productId = String(it?.productId ?? "").trim();
            if (!productId) return "";

            const raw = String(it?.variantId ?? "").trim();
            if (!raw) return "";

            // normalize "#6956..." -> "6956..."
            const rawNoHash = raw.startsWith("#") ? raw.slice(1) : raw;

            const p = _productById.get(productId) || null;
            if (!p) return "";

            const vars = Array.isArray(p.varients) ? p.varients : (Array.isArray(p.variants) ? p.variants : []);
            if (!vars.length) return "";

            const v = vars.find(vr => {
                const id = String(vr?.id ?? "").trim();
                const idNoHash = id.startsWith("#") ? id.slice(1) : id;

                const cat = String(vr?.printful_cat_id ?? "").trim();
                const catNoHash = cat.startsWith("#") ? cat.slice(1) : cat;

                const pf = vr?.printful_catalog_variant_id != null ? String(vr.printful_catalog_variant_id).trim() : "";

                return (
                    raw === id || rawNoHash === idNoHash ||
                    raw === cat || rawNoHash === catNoHash ||
                    raw === pf
                );
            }) || null;

            if (!v) return "";

            const pfid = v?.printful_catalog_variant_id;
            if (pfid == null || String(pfid).trim() === "") return "";

            return String(pfid).trim(); // <-- this is what the worker wants
        }

        function resolveFromCatalog(item) {
            const it = { ...(item || {}) };
            const productId = String(it.productId || "").trim();
            const variantId = String(it.variantId || "").trim();

            const p = productId ? (_productById.get(productId) || null) : null;
            const v = (productId && variantId) ? (_variantByKey.get(`${productId}::${variantId}`) || null) : null;

            if (!String(it.title || it.name || "").trim()) {
                it.title = String(p?.name || p?.title || productId || it.id || "Item").trim();
            }

            if (!String(it.variantLabel || "").trim()) {
                it.variantLabel =
                    String(v?.variantLabel || v?.name || "").trim() ||
                    (Array.isArray(v?.optionParts) ? v.optionParts.join(" / ") : "");
            }

            const hasPrice =
                it.price !== null &&
                it.price !== undefined &&
                Number.isFinite(Number(it.price));

            if (!hasPrice) {
                const catalogPrice =
                    (v && Number.isFinite(Number(v.price)) ? Number(v.price) : null) ??
                    (p && Number.isFinite(Number(p.price)) ? Number(p.price) : null) ??
                    null;
                if (catalogPrice !== null) it.price = catalogPrice;
            }

            const hasImage = !!safeUrl(it.image || it.imageUrl || "");
            if (!hasImage) {
                const img = resolveProductImage(productId, variantId);
                if (img) it.image = img;
            }

            // Normalize printful_catalog_variant_id for checkout (optional)
            if (it.printful_catalog_variant_id == null) {
                const pfid = getPrintfulVariantIdForCartItem(it);
                if (pfid) it.printful_catalog_variant_id = pfid;
            }

            return it;
        }

        /* =============================
           Buy-mode helpers
        ============================= */
        function getProductByQueryKey(keyRaw) {
            const key = String(keyRaw || "").trim().toLowerCase();
            if (!key) return null;

            for (const [pid, p] of _productById.entries()) {
                if (String(pid).toLowerCase() === key) return p;
            }

            for (const p of (_productsArray || [])) {
                const candidates = [
                    p.id, p.productId, p.slug, p.handle, p.key, p.shortId,
                    p?.meta?.slug, p?.meta?.handle
                ].filter(Boolean).map(x => String(x).trim().toLowerCase());
                if (candidates.includes(key)) return p;
            }

            for (const p of (_productsArray || [])) {
                const name = String(p.name || p.title || "").trim().toLowerCase();
                if (name && name.replace(/\s+/g, "") === key.replace(/\s+/g, "")) return p;
            }

            return null;
        }

        function pickDefaultVariant(product) {
            const vars = Array.isArray(product?.varients) ? product.varients : (Array.isArray(product?.variants) ? product.variants : []);
            if (!vars.length) return null;
            for (const v of vars) {
                const vid = String(v?.id ?? v?.variantId ?? "").trim();
                if (vid) return v;
            }
            return vars[0] || null;
        }

        /* =============================
           Local cart helpers (signed-out)
        ============================= */
        function getLocalCart() {
            try {
                const raw = localStorage.getItem("unlim8ted-cart");
                const arr = raw ? JSON.parse(raw) : [];
                return Array.isArray(arr) ? arr : [];
            } catch { return []; }
        }

        function setLocalCart(items) {
            try {
                localStorage.setItem("unlim8ted-cart", JSON.stringify(items || []));
                window.dispatchEvent(new CustomEvent("cart-changed"));
            } catch { }
        }

        /* =============================
           Cart state + listeners
        ============================= */
        let currentUser = null;
        let cartItems = [];
        let unsubCart = null;

        async function emitCart(items) {
            await loadProducts();
            cartItems = (items || []).map(x => resolveFromCatalog(x));
            renderCart();
        }

        window.addEventListener("cart-changed", () => {
            if (BUY_MODE) return;
            if (!currentUser) emitCart(getLocalCart());
        });

        function startCartData() {
            onAuthStateChanged(auth, (user) => {
                currentUser = user || null;

                if (unsubCart) { unsubCart(); unsubCart = null; }
                if (BUY_MODE) return;

                if (!user) {
                    emitCart(getLocalCart());
                    return;
                }

                const cartRef = collection(db, "users", user.uid, "cartItems");
                unsubCart = onSnapshot(cartRef, (snap) => {
                    const raw = [];
                    snap.forEach(d => {
                        const data = d.data() || {};
                        raw.push({
                            id: d.id,
                            productId: data.productId ?? null,
                            // UPDATED: cart variantId should be the Printful catalog variant id
                            variantId: data.variantId ?? data.printful_catalog_variant_id ?? null,
                            printful_catalog_variant_id: data.printful_catalog_variant_id ?? null,
                            variantLabel: data.variantLabel ?? "",
                            title: data.title || data.name || data.productName || "",
                            price: (data.price ?? data.priceNum ?? null),
                            qty: Number.isFinite(data.qty) ? Number(data.qty) : 1,
                            image: data.image ?? data.imageUrl ?? null,
                            accessUrl: data.accessUrl ?? data.downloadUrl ?? data.url ?? data.link ?? null,
                        });
                    });
                    emitCart(raw);
                });
            });
        }

        /* =============================
           Cart mutations
        ============================= */
        async function setQty(id, qty) {
            qty = Math.max(1, qty | 0);

            if (BUY_MODE) {
                const idx = cartItems.findIndex(x => String(x.id) === String(id));
                if (idx >= 0) {
                    cartItems[idx].qty = qty;
                    renderCart();
                }
                return;
            }

            if (!currentUser?.uid) {
                const arr = getLocalCart();
                const idx = arr.findIndex(x => String(x.id) === String(id));
                if (idx >= 0) {
                    arr[idx].qty = qty;
                    setLocalCart(arr);
                }
                return;
            }

            await updateDoc(doc(db, "users", currentUser.uid, "cartItems", id), {
                qty,
                updatedAt: serverTimestamp(),
            });
        }

        async function removeItem(id) {
            if (BUY_MODE) {
                cartItems = cartItems.filter(x => String(x.id) !== String(id));
                renderCart();
                return;
            }

            if (!currentUser?.uid) {
                setLocalCart(getLocalCart().filter(x => String(x.id) !== String(id)));
                return;
            }
            await deleteDoc(doc(db, "users", currentUser.uid, "cartItems", id));
        }

        async function clearCart() {
            if (BUY_MODE) {
                cartItems = [];
                renderCart();
                return;
            }

            if (!currentUser?.uid) {
                setLocalCart([]);
                return;
            }
            const snap = await getDocs(collection(db, "users", currentUser.uid, "cartItems"));
            await Promise.all(snap.docs.map(d => deleteDoc(d.ref)));
        }

        /* =============================
           Checkout helpers
        ============================= */
        // "Paid" means: NOT a free-link item.
        // Free items are identified by having an accessUrl/downloadUrl/url/link.
        function getPaidItems(allItems) {
            return (allItems || [])
                .map(it => {
                    const qty = Math.max(1, Number(it.qty) || 1);
                    const accessUrl = safeUrl(it.accessUrl || it.downloadUrl || it.url || it.link);
                    return { ...it, qty, _isFree: !!accessUrl, accessUrl };
                })
                .filter(it => !it._isFree);
        }


        function findPaidMissingVariant(allItems) {
            const paid = getPaidItems(allItems);

            return paid
                .map(it => ({
                    title: String(it.title || it.name || "Item"),
                    productId: String(it.productId || "").trim(),
                    variantId: String(it.variantId || it.printful_catalog_variant_id || "").trim(),
                }))
                .filter(x => (!x.productId || !x.variantId));
        }


        function findPaidMissingPrintfulVariantId(paidItems) {
            const missing = [];
            for (const it of (paidItems || [])) {
                const pfid = getPrintfulVariantIdForCartItem(it);
                if (!pfid) missing.push(it);
            }
            return missing;
        }

        function getAddressForWorker() {
            const get = (id) => document.getElementById(id).value.trim();
            return {
                name: get("ship_full_name"),
                email: get("ship_email"),
                phone: get("ship_phone"),
                address1: get("ship_address1"),
                address2: get("ship_address2"),
                city: get("ship_city"),
                state: get("ship_state_code"),
                zip: get("ship_zip"),
                country: get("ship_country_code").toUpperCase(),
            };
        }

        function validateAddressForWorker(a) {
            const errs = [];
            const req = (k, label) => { if (!String(a[k] || "").trim()) errs.push(`${label} is required.`); };

            req("name", "Full name");
            req("email", "Email");
            req("address1", "Address line 1");
            req("city", "City");
            req("zip", "ZIP / Postal");
            req("country", "Country");

            if (a.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(a.email)) errs.push("Email looks invalid.");

            if (a.country === "US") {
                if (!/^[A-Z]{2}$/.test(String(a.state || "").toUpperCase())) errs.push("State must be 2 letters (US).");
                if (!/^\d{5}(-\d{4})?$/.test(String(a.zip || ""))) errs.push("ZIP must be 5 digits (or 5+4).");
            }

            if (a.phone) {
                const digits = a.phone.replace(/[^\d]/g, "");
                if (digits.length < 7) errs.push("Phone looks too short.");
            }

            return errs;
        }

        function centsToUsdString(cents) {
            const n = Number(cents || 0) / 100;
            return n.toFixed(2);
        }

        function computeSubtotalCents(items) {
            return (items || []).reduce((sum, it) => {
                const qty = Math.max(1, Number(it.qty) || 1);
                const price = normalizePriceNumber(parsePrice(it.price)) || 0;
                return sum + Math.round(price * 100) * qty;
            }, 0);
        }

        function buildWorkerItemsFromPaidCart(paidItems) {
            const out = [];
            for (const it of (paidItems || [])) {
                const pfid = getPrintfulVariantIdForCartItem(it);
                if (!pfid) throw new Error(`Missing Printful catalog variant id for ${it.productId || "?"} / ${it.variantId || "?"}`);
                out.push({
                    printfulVariantId: pfid,
                    qty: Math.max(1, Number(it.qty) || 1),
                });
            }
            return out;
        }



        /* =============================
           Worker calls
        ============================= */
        async function workerFetchJson(path, opts) {
            const res = await fetch(`${WORKER_BASE}${path}`, {
                mode: "cors",
                credentials: "omit",
                ...opts,
            });

            const txt = await res.text().catch(() => "");
            let j = {};
            try { j = txt ? JSON.parse(txt) : {}; } catch { j = {}; }

            if (!res.ok) {
                const msg = j?.error || j?.detail || txt || `HTTP ${res.status}`;
                const err = new Error(msg);
                err.status = res.status;
                err.body = j;
                throw err;
            }
            return j;
        }

        function workerQuote(payload) {
            return workerFetchJson("/quote", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
        }

        function workerPay(payload) {
            return workerFetchJson("/pay", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
        }

        function workerPaymentStatus(quoteId) {
            return workerFetchJson(`/payment-status?quoteId=${encodeURIComponent(quoteId)}`, { method: "GET" });
        }

        /* =============================
           Checkout flow
        ============================= */
        async function beginCheckoutFlow() {
            clearNotice();

            if (!cartItems.length) {
                setNotice("warn", `<strong>Your cart is empty.</strong> Add something first.`);
                return;
            }

            const paid = getPaidItems(cartItems);
            if (!paid.length) {
                setNotice("ok",
                    `<div style="display:flex;gap:10px;align-items:flex-start">
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" style="width:18px;height:18px;margin-top:2px;">
              <path d="M20 6 9 17l-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div><strong>No paid items.</strong><br/>Your free items are ready via their links.</div>
          </div>`
                );
                return;
            }

            const missingVariant = findPaidMissingVariant(cartItems);
            if (missingVariant.length) {
                setNotice("warn",
                    `<strong>Some paid items can’t be checked out yet.</strong><br/>
           These need productId + variantId (Printful catalog variant id) in your cart data/products.json:<br/>
           <div style="margin-top:6px;opacity:.9">${missingVariant.map(m => `• ${escapeHtml(m.title)}`).join("<br/>")}</div>`
                );
                return;
            }

            const missingPfid = findPaidMissingPrintfulVariantId(paid);
            if (missingPfid.length) {
                setNotice("warn",
                    `<strong>Missing Printful catalog variant id.</strong><br/>
           Make sure your cart item <code>variantId</code> is the Printful catalog variant id (products.json varient.id).<br/>
           <div style="margin-top:6px;opacity:.9">${missingPfid.map(m => `• ${escapeHtml(m.title)} (${escapeHtml(m.variantLabel || "")})`).join("<br/>")}</div>`
                );
                return;
            }

            resetCheckoutState();
            openCheckoutModal();

            const itemsCount = cartItems.reduce((s, it) => s + (Number(it.qty) || 1), 0);
            const subtotalCents = computeSubtotalCents(paid);
            qItemsEl.textContent = String(itemsCount);
            qSubtotalEl.textContent = centsToUsdString(subtotalCents);
            qShippingEl.textContent = "0.00";
            qTaxEl.textContent = "0.00";
            qTotalEl.textContent = "0.00";

            qShipRow.style.display = "none";
            qTaxRow.style.display = "none";
            qTotalRow.style.display = "none";

        }

        // Step 1: Next => quote
        btnQuote.addEventListener("click", async () => {
            clearPayStatus();

            const paid = getPaidItems(cartItems);
            if (!paid.length) {
                setPayStatus("warn", `<strong>No paid items.</strong> Nothing to checkout.`);
                return;
            }

            const address = getAddressForWorker();
            const errs = validateAddressForWorker(address);
            if (errs.length) {
                setPayStatus("warn", `<strong>Please fix:</strong><br/>${errs.map(e => `• ${escapeHtml(e)}`).join("<br/>")}`);
                return;
            }

            btnQuote.disabled = true;
            shipMethodEl.disabled = true;
            shipMethodEl.innerHTML = `<option value="">Calculating…</option>`;
            quoteHintEl.textContent = "Calculating shipping options…";

            try {
                const items = buildWorkerItemsFromPaidCart(paid);

                const resp = await workerQuote({ address, items });
                const quoteId = String(resp.quoteId || "").trim();
                const shippingOptions = Array.isArray(resp.shippingOptions) ? resp.shippingOptions : [];
                const subtotalCents = typeof resp.subtotal === "number" ? resp.subtotal : computeSubtotalCents(paid);

                if (!quoteId) throw new Error("Worker did not return quoteId.");
                if (!shippingOptions.length) throw new Error("No shipping options returned.");

                _quote = { quoteId, shippingOptions, subtotalCents };

                _quoteExpiresAt = Number(resp.expiresAt || 0) || (Date.now() + 10 * 60 * 1000);
                scheduleQuoteExpiry(_quoteExpiresAt);

                qExpWrap.style.display = "inline-flex";
                qExpTime.textContent = formatTime(_quoteExpiresAt);

                shipMethodEl.innerHTML = "";
                for (const opt of shippingOptions) {
                    const id = String(opt.id || "").trim();
                    const name = String(opt.name || "Shipping").trim();
                    const cost = Number(opt.cost || 0);
                    const label = `${name} — $${centsToUsdString(cost)}`;

                    const o = document.createElement("option");
                    o.value = id;
                    o.textContent = label;
                    shipMethodEl.appendChild(o);
                }

                shipMethodEl.disabled = false;

                const selected = shipMethodEl.value;
                const chosen = shippingOptions.find(x => String(x.id) === String(selected)) || shippingOptions[0];

                const shipCents = Number(chosen?.cost || 0);
                const taxCents = Number(chosen?.tax || 0);
                const totalCents = Number(chosen?.total || (subtotalCents + shipCents + taxCents));

                qSubtotalEl.textContent = centsToUsdString(subtotalCents);
                qShippingEl.textContent = centsToUsdString(shipCents);
                qTaxEl.textContent = centsToUsdString(taxCents);
                qTotalEl.textContent = centsToUsdString(totalCents);

                qShipRow.style.display = "";
                qTaxRow.style.display = "";
                qTotalRow.style.display = "";


                setPayStatus("ok",
                    `<div style="display:flex;gap:10px;align-items:flex-start">
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" style="width:18px;height:18px;margin-top:2px;">
              <path d="M20 6 9 17l-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div><strong>Shipping calculated.</strong><br/>Proceed to payment.</div>
          </div>`
                );

                setStep(2);
                await ensureSquareCardMounted();

            } catch (err) {
                console.error(err);
                shipMethodEl.innerHTML = `<option value="">Shipping options will appear after Next…</option>`;
                shipMethodEl.disabled = true;
                quoteHintEl.textContent = "Enter your address, then click Next to calculate shipping.";
                setPayStatus("bad", `<strong>Checkout error:</strong> ${escapeHtml(err?.message || String(err))}`);
            } finally {
                btnQuote.disabled = false;
            }
        });

        shipMethodEl.addEventListener("change", () => {
            if (!_quote) return;
            const id = shipMethodEl.value;
            const chosen = _quote.shippingOptions.find(x => String(x.id) === String(id));
            const shipCents = Number(chosen?.cost || 0);
            const taxCents = Number(chosen?.tax || 0);
            const totalCents = Number(chosen?.total || (_quote.subtotalCents + shipCents + taxCents));

            qShippingEl.textContent = centsToUsdString(shipCents);
            qTaxEl.textContent = centsToUsdString(taxCents);
            qTotalEl.textContent = centsToUsdString(totalCents);
        });


        // Step 2: Pay
        btnPay.addEventListener("click", async () => {
            clearPayStatus();

            if (!_quote?.quoteId) {
                setPayStatus("warn", `<strong>Start checkout again.</strong>`);
                setStep(1);
                return;
            }

            if (Date.now() >= _quoteExpiresAt) {
                showExpired();
                return;
            }

            const selectedShippingId = String(shipMethodEl.value || "").trim();
            if (!selectedShippingId) {
                setPayStatus("warn", `<strong>Select a shipping method.</strong>`);
                return;
            }

            btnPay.disabled = true;
            shipMethodEl.disabled = true;

            setPayStatus("warn", `<strong>Processing payment…</strong> Do not refresh.`);

            try {
                await ensureSquareCardMounted();

                const tokenizeResult = await _squareCard.tokenize();
                if (tokenizeResult.status !== "OK") {
                    const msg = tokenizeResult.errors?.[0]?.message || "Card tokenization failed.";
                    throw new Error(msg);
                }

                if (Date.now() >= _quoteExpiresAt) {
                    showExpired();
                    return;
                }

                await workerPay({
                    quoteId: _quote.quoteId,
                    selectedShippingId,
                    sourceId: tokenizeResult.token,
                });

                const start = Date.now();
                const hardStop = 60 * 1000;

                while (true) {
                    if (Date.now() >= _quoteExpiresAt) {
                        showExpired();
                        return;
                    }
                    if (Date.now() - start > hardStop) break;

                    const st = await workerPaymentStatus(_quote.quoteId);
                    const status = String(st.status || "").toLowerCase();

                    if (status === "paid" || status === "completed") {
  setPayStatus("ok", `...`);

  await openAlertModal({
    title: "Payment confirmed",
    kind: "ok",
    bodyHtml: `
      <strong>Success.</strong> Your payment was confirmed.<br/>
      Please check your email for your Square receipt.
    `,
    okText: "Done"
  });

  try { await clearCart(); } catch { }
  renderCart();
  setTimeout(() => closeCheckoutModal(), 150); // shorter since popup already informed them
  return;
}


                    if (status === "failed" || status === "canceled") {
                        setPayStatus("bad", `<strong>Payment failed.</strong> Please try again.`);
                        await openAlertModal({
  title: "Payment not completed",
  kind: "bad",
  bodyHtml: `
    <strong>Payment failed or was canceled.</strong><br/>
    Please double-check your card details and try again.
  `,
  okText: "Try again"
});

                        btnPay.disabled = false;
                        shipMethodEl.disabled = false;
                        return;
                    }

                    await new Promise(r => setTimeout(r, 1200));
                }

                setPayStatus("warn",
                    `<strong>Processing.</strong><br/>If you were charged, it will finalize shortly. You can keep this page open and retry Pay if needed.`
                );

            } catch (err) {
                console.error(err);
                setPayStatus("bad", `<strong>Payment error:</strong> ${escapeHtml(err?.message || String(err))}`);
                await openAlertModal({
  title: "Payment error",
  kind: "bad",
  bodyHtml: `
    <strong>We couldn’t process the payment.</strong><br/>
    ${escapeHtml(err?.message || String(err))}
  `,
  okText: "OK"
});

            } finally {
                if (!expiredOverlay.classList.contains("show")) {
                    btnPay.disabled = false;
                    shipMethodEl.disabled = false;
                }
            }
        });
/* =============================
   Alert modal (success / error popups)
   Uses existing #modalOverlay UI
============================= */
let _modalMode = "confirm"; // "confirm" | "alert"

function setModalButtons({ showCancel, confirmText, confirmClass } = {}) {
  modalCancel.style.display = showCancel ? "inline-flex" : "none";
  modalConfirm.textContent = confirmText || "OK";
  modalConfirm.className = confirmClass || "btn btn-primary";
}

// Alert-style modal: single OK button, resolves when closed
function openAlertModal({
  title = "Notice",
  bodyHtml = "",
  okText = "OK",
  kind = "ok", // ok | warn | bad
} = {}) {
  _modalMode = "alert";
  modalTitle.textContent = title;

  const icon = kind === "ok"
    ? `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true" style="width:18px;height:18px;margin-top:2px;flex:0 0 auto;">
         <path d="M20 6 9 17l-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
       </svg>`
    : kind === "warn"
    ? `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true" style="width:18px;height:18px;margin-top:2px;flex:0 0 auto;">
         <path d="M12 9v5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
         <path d="M12 17h.01" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
         <path d="M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
       </svg>`
    : `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true" style="width:18px;height:18px;margin-top:2px;flex:0 0 auto;">
         <path d="M12 9v4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
         <path d="M12 17h.01" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
         <path d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" stroke="currentColor" stroke-width="2"/>
       </svg>`;

  const tint =
    kind === "ok" ? "rgba(99,214,255,.14)"
    : kind === "warn" ? "rgba(255,209,102,.14)"
    : "rgba(255,77,77,.14)";

  modalBody.innerHTML = `
    <div style="display:flex;gap:10px;align-items:flex-start;padding:10px 2px;">
      <div style="color:rgba(255,255,255,.9)">${icon}</div>
      <div style="flex:1">
        <div style="border:1px solid rgba(255,255,255,.10);background:${tint};border-radius:14px;padding:10px 12px;">
          ${bodyHtml}
        </div>
      </div>
    </div>
  `;

  setModalButtons({
    showCancel: false,
    confirmText: okText,
    confirmClass: kind === "bad" ? "btn btn-danger" : "btn btn-primary",
  });

  modalOverlay.classList.add("show");
  modalOverlay.setAttribute("aria-hidden", "false");
  _lastFocus = document.activeElement;
  setTimeout(() => modalConfirm.focus(), 0);

  return new Promise(resolve => { _modalResolver = resolve; });
}

// Patch closeModal to behave differently for alert vs confirm
const _origCloseModal = closeModal;
closeModal = function (result) {
  modalOverlay.classList.remove("show");
  modalOverlay.setAttribute("aria-hidden", "true");

  const r = _modalResolver;
  _modalResolver = null;

  // restore confirm modal defaults so clear-cart confirm keeps working
  if (_modalMode === "alert") {
    _modalMode = "confirm";
    // restore default confirm UI
    setModalButtons({
      showCancel: true,
      confirmText: "Confirm",
      confirmClass: "btn btn-danger",
    });
  }

  if (typeof r === "function") r(result);
  if (_lastFocus && typeof _lastFocus.focus === "function") _lastFocus.focus();
};

// For alert mode: clicking OK should resolve
modalConfirm.addEventListener("click", () => {
  if (modalOverlay.classList.contains("show") && _modalMode === "alert") {
    closeModal(true);
  }
});

        /* =============================
           Render
        ============================= */
        function itemHTML(it) {
            const qty = Math.max(1, Number(it.qty) || 1);
            const price = normalizePriceNumber(parsePrice(it.price)) || 0;

            const title = escapeHtml(it.title || it.name || "Item");
            const variantLabel = String(it.variantLabel || "").trim();
            const variantLine = variantLabel
                ? `<div class="mini">Variant: <strong>${escapeHtml(variantLabel)}</strong></div>`
                : ``;

            const imgUrl = safeUrl(it.image || it.imageUrl || "");
            const img = imgUrl ? `<img src="${escapeHtml(imgUrl)}" alt="">` : "No image";

            const priceHtml = price > 0
                ? `$${money(price)}`
                : `<span class="free">FREE</span>`;

            return `
        <div class="item">
          <div class="thumb">${img}</div>

          <div class="meta">
            <div class="title">${title}</div>
            ${variantLine}

            <div class="row2">
              <div class="price">${priceHtml}</div>

              <div class="qty" aria-label="Quantity">
                <button type="button" data-dec="${escapeHtml(it.id)}" aria-label="Decrease quantity">−</button>
                <input type="text" inputmode="numeric" value="${qty}" data-qty="${escapeHtml(it.id)}" aria-label="Quantity value">
                <button type="button" data-inc="${escapeHtml(it.id)}" aria-label="Increase quantity">+</button>
              </div>
            </div>

            ${safeUrl(it.accessUrl) && price <= 0
                    ? `<div class="mini">Free access: <a href="${escapeHtml(it.accessUrl)}" target="_blank" rel="noopener" style="color:rgba(99,214,255,.95);text-decoration:none;font-weight:900;">Open link</a></div>`
                    : ``
                }
          </div>

          <div class="actions">
            <button class="iconbtn" type="button" data-remove="${escapeHtml(it.id)}" title="Remove" aria-label="Remove">
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" style="width:18px;height:18px;">
                <path d="M4 7h16" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
                <path d="M10 11v6" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
                <path d="M14 11v6" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
                <path d="M6 7l1 14h10l1-14" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
                <path d="M9 7V4h6v3" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
              </svg>
            </button>
          </div>
        </div>
      `;
        }

        function renderCart() {
            const list = document.getElementById("cartList");
            const empty = document.getElementById("cartEmpty");

            list.innerHTML = "";
            if (!cartItems.length) {
                empty.style.display = "block";
            } else {
                empty.style.display = "none";
                for (const it of cartItems) {
                    const d = document.createElement("div");
                    d.innerHTML = itemHTML(it);
                    list.appendChild(d.firstElementChild);
                }
            }

            const itemsCount = cartItems.reduce((s, it) => s + (Number(it.qty) || 1), 0);
            const subtotal = cartItems.reduce((s, it) => {
                const price = normalizePriceNumber(parsePrice(it.price)) || 0;
                return s + price * (Number(it.qty) || 1);
            }, 0);

            document.getElementById("itemsCount").textContent = String(itemsCount);
            document.getElementById("subtotal").textContent = money(subtotal);
            document.getElementById("total").textContent = money(subtotal);

            const checkoutBtn = document.getElementById("checkoutBtn");
            checkoutBtn.disabled = cartItems.length === 0;

            if (BUY_MODE) {
                const it = cartItems[0] || null;
                const price = it ? (normalizePriceNumber(parsePrice(it.price)) || 0) : 0;
                const url = it ? safeUrl(it.accessUrl) : "";
                checkoutBtn.textContent = (it && price <= 0 && url) ? "Get free item" : "Checkout";
            } else {
                checkoutBtn.textContent = "Checkout";
            }

            const clearBtn = document.getElementById("clearBtn");
            clearBtn.disabled = cartItems.length === 0 || BUY_MODE;
            clearBtn.style.display = BUY_MODE ? "none" : "inline-flex";
        }

        /* =============================
           UI events
        ============================= */
        document.addEventListener("click", async (e) => {
            const rm = e.target.closest("[data-remove]");
            if (rm) {
                const id = rm.getAttribute("data-remove");
                try { await removeItem(id); } catch (err) { console.error(err); }
                return;
            }

            const inc = e.target.closest("[data-inc]");
            if (inc) {
                const id = inc.getAttribute("data-inc");
                const input = document.querySelector(`[data-qty="${CSS.escape(id)}"]`);
                const current = Math.max(1, parseInt(input?.value || "1", 10) || 1);
                input.value = String(current + 1);
                try { await setQty(id, current + 1); } catch (err) { console.error(err); }
                return;
            }

            const dec = e.target.closest("[data-dec]");
            if (dec) {
                const id = dec.getAttribute("data-dec");
                const input = document.querySelector(`[data-qty="${CSS.escape(id)}"]`);
                const current = Math.max(1, parseInt(input?.value || "1", 10) || 1);
                const next = Math.max(1, current - 1);
                input.value = String(next);
                try { await setQty(id, next); } catch (err) { console.error(err); }
                return;
            }

            if (e.target.closest("#checkoutBtn")) {
                if (BUY_MODE && cartItems.length === 1) {
                    const it = cartItems[0];
                    const price = normalizePriceNumber(parsePrice(it.price)) || 0;
                    const url = safeUrl(it.accessUrl);

                    if (price <= 0 && url) {
                        setNotice("ok",
                            `<div style="display:flex;gap:10px;align-items:flex-start">
                <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" style="width:18px;height:18px;margin-top:2px;">
                  <path d="M20 6 9 17l-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <div><strong>Opening your free item…</strong></div>
              </div>`
                        );
                        window.open(url, "_blank", "noopener,noreferrer");
                        return;
                    }
                }

                await beginCheckoutFlow();
                return;
            }

            if (e.target.closest("#clearBtn")) {
                if (BUY_MODE) return;
                clearNotice();
                if (!cartItems.length) return;

                const ok = await openModal({
                    title: "Clear cart?",
                    bodyHtml: `This will remove <strong>all items</strong> from your cart. This can’t be undone.`,
                    confirmText: "Clear cart",
                    confirmClass: "btn btn-danger"
                });

                if (!ok) return;
                try { await clearCart(); } catch (err) { console.error(err); }
                return;
            }

            if (e.target.closest("#openDonate")) {
                openDonateModal();
                return;
            }
        });

        let t = null;
        document.addEventListener("input", (e) => {
            const inp = e.target.closest("[data-qty]");
            if (!inp) return;

            const id = inp.getAttribute("data-qty");
            let qty = parseInt(inp.value, 10);
            if (!Number.isFinite(qty) || qty < 1) qty = 1;

            clearTimeout(t);
            t = setTimeout(async () => {
                try { await setQty(id, qty); } catch (err) { console.error(err); }
            }, 300);
        });

        /* =============================
           BUY MODE bootstrap
        ============================= */
        async function startBuyMode(productKey) {
            await loadProducts();

            const p = getProductByQueryKey(productKey);
            if (!p) {
                cartItems = [];
                renderCart();
                setNotice("warn",
                    `<strong>Product not found:</strong> ${escapeHtml(productKey)}<br/>
           <a href="/products" style="color:rgba(99,214,255,.95);text-decoration:none;font-weight:900;">Go to products</a>`
                );
                return;
            }

            const pid = String(p.id ?? p.productId ?? "").trim();
            const v = pickDefaultVariant(p);

            // UPDATED: vid is Printful catalog variant id
            const vid = String(v?.id ?? v?.variantId ?? "").trim();

            const price = normalizePriceNumber(parsePrice(v?.price ?? p?.price)) || 0;
            const accessUrl = p.accessUrl ?? p.downloadUrl ?? p.url ?? p.link ?? null;

            if (price > 0 && !vid) {
                cartItems = [];
                renderCart();
                setNotice("warn",
                    `<strong>This product can’t be purchased yet.</strong><br/>
           Missing varient.id (Printful catalog variant id) in products.json for <strong>${escapeHtml(pid || productKey)}</strong>.`
                );
                return;
            }

            if (price <= 0 && !safeUrl(accessUrl)) {
                cartItems = [];
                renderCart();
                setNotice("warn",
                    `<strong>This free item is missing its link.</strong><br/>
           Add <code>accessUrl</code> (or <code>downloadUrl/url/link</code>) in products.json.`
                );
                return;
            }

            const image = resolveProductImage(pid, vid) || safeUrl(p.image || p.imageUrl || "");
            const title = String(p.name || p.title || pid || productKey || "Item").trim();
            const variantLabel = String(v?.name || "").trim();

            cartItems = [
                resolveFromCatalog({
                    id: `buy-${pid}-${vid || "free"}`,
                    productId: pid,
                    variantId: vid || null,
                    printful_catalog_variant_id: vid ? Number(vid) : null,
                    title,
                    variantLabel,
                    price,
                    qty: 1,
                    image: image || null,
                    accessUrl
                })
            ];

            renderCart();
            clearNotice();
        }

        /* =============================
           Start
        ============================= */
        const y = new Date().getFullYear();
        document.getElementById("footer-text").innerHTML =
            `&copy; 2019-${y} Unlim8ted Studio Productions. All rights reserved.`;

        if (BUY_MODE) {
            document.getElementById("pageTitle").textContent = "Checkout";
            document.getElementById("pageSub").textContent = "One item. Adjust quantity, then checkout.";
            document.getElementById("continueBtn").textContent = "Back to products";
            document.getElementById("continueBtn").setAttribute("href", "/products");
            document.title = "Unlim8ted - Checkout";
            document.getElementById("openDonate").style.display = "none";
        }

        loadProducts().catch(() => { });
        startCartData();

        if (BUY_MODE) {
            if (!BUY_PRODUCT) {
                setNotice("warn",
                    `<strong>Missing product.</strong> Use <code>?source=buy&amp;product=theglitch</code><br/>
           <a href="/products" style="color:rgba(99,214,255,.95);text-decoration:none;font-weight:900;">Go to products</a>`
                );
            } else {
                startBuyMode(BUY_PRODUCT).catch(err => {
                    console.error(err);
                    setNotice("warn", `<strong>Error:</strong> ${escapeHtml(err?.message || String(err))}`);
                });
            }
        } else {
            renderCart();
        }

        const cc = document.getElementById("ship_country_code");
        if (cc) cc.value = "US";

        btnQuote.textContent = "Next";
