/* =========================================================
   /js/cart.js  (NO inline; CSP-friendly)
   - Renders cart from Firebase (placeholder hooks)
   - Two-step checkout modal:
       Step 1: Address + quote + shipping selection
       Step 2: Square card field fills entire right panel
========================================================= */

import { getFirebase } from "/components/firebase-init.js";
import {
  collection, onSnapshot, doc, updateDoc, deleteDoc, getDocs
} from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";

const API_BASE = "https://api.unlim8ted.com";

const $ = (sel) => document.querySelector(sel);

const els = {
  cartList: $("#cartList"),
  cartCount: $("#cartCount"),
  checkoutBtn: $("#checkoutBtn"),
  clearCartBtn: $("#clearCartBtn"),
  summaryRows: $("#summaryRows"),
  summaryTotal: $("#summaryTotal"),

  checkoutOverlay: $("#checkoutOverlay"),
  checkoutCloseBtn: $("#checkoutCloseBtn"),
  backBtn: $("#backBtn"),
  nextBtn: $("#nextBtn"),

  confirmOverlay: $("#confirmOverlay"),
  confirmCancelBtn: $("#confirmCancelBtn"),
  confirmClearBtn: $("#confirmClearBtn"),

  // Address fields
  shipName: $("#shipName"),
  shipEmail: $("#shipEmail"),
  shipPhone: $("#shipPhone"),
  shipCountry: $("#shipCountry"),
  shipAddress1: $("#shipAddress1"),
  shipAddress2: $("#shipAddress2"),
  shipCity: $("#shipCity"),
  shipState: $("#shipState"),
  shipZip: $("#shipZip"),

  // Step 1 UI
  shipOptions: $("#shipOptions"),
  shipError: $("#shipError"),

  // Right panel summary
  summaryBox: $("#summaryBox"),
  modalSubtotal: $("#modalSubtotal"),
  modalShipping: $("#modalShipping"),
  modalTotal: $("#modalTotal"),

  // Step 2 UI
  payTotal: $("#payTotal"),
  payBtn: $("#payBtn"),
  payStatus: $("#payStatus"),
  cardContainer: $("#card-container"),
  cardErrors: $("#card-errors"),
  payLeftError: $("#payLeftError"),
};

const stepEls = {
  step1: document.querySelector('[data-step="1"]'),
  step2: document.querySelector('[data-step="2"]'),
  pane1: document.querySelector('[data-pane="1"]'),
  pane2: document.querySelector('[data-pane="2"]'),
  right1: document.querySelector('[data-right="1"]'),
  right2: document.querySelector('[data-right="2"]'),
};

const { auth, db } = getFirebase();

let currentUser = null;
let cartItems = [];
let productsIndex = null;

// Checkout state
let checkoutStep = 1;
let quoteId = null;
let shippingOptions = [];
let selectedShippingId = null;
let quoteSubtotalCents = 0;
let squarePayments = null;
let squareCard = null;

// =========================
// Products.json
// =========================
async function loadProductsJson() {
  if (productsIndex) return productsIndex;
  const res = await fetch("/tools/data/products.json", { cache: "no-store" });
  const json = await res.json();

  // Build a map from product/variant keys → printfulVariantId + price
  // Expect variants like: { printfulVariantId, price, variantLabel, ... }
  const productById = new Map();
  const variantByPrintful = new Map();

  const products = Array.isArray(json?.products) ? json.products : (Array.isArray(json) ? json : []);
  for (const p of products) {
    if (p?.id) productById.set(p.id, p);
    const vars = p?.variants || p?.varients || [];
    if (Array.isArray(vars)) {
      for (const v of vars) {
        const pfid = v?.printfulVariantId ?? v?.printful_variant_id ?? v?.variant_id;
        if (pfid != null) variantByPrintful.set(String(pfid), v);
      }
    }
  }

  productsIndex = { raw: json, productById, variantByPrintful };
  return productsIndex;
}

// =========================
// Cart (Firebase) — you already have this pattern
// =========================
function money(cents) {
  const n = (Number(cents) || 0) / 100;
  return n.toLocaleString(undefined, { style: "currency", currency: "USD" });
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[c]));
}

async function readCartOnce(uid) {
  const snap = await getDocs(collection(db, "users", uid, "cart"));
  const out = [];
  snap.forEach((d) => out.push({ id: d.id, ...d.data() }));
  return out;
}

function listenCart(uid) {
  return onSnapshot(collection(db, "users", uid, "cart"), (snap) => {
    const out = [];
    snap.forEach((d) => out.push({ id: d.id, ...d.data() }));
    cartItems = normalizeCart(out);
    renderCart();
  });
}

function normalizeCart(items) {
  // Your cart items should contain printfulVariantId + qty at minimum for physical items.
  // Example cart item:
  // { id, title, image, qty, kind:"physical", printfulVariantId:1234, price:1950 }
  return (items || []).map((it) => ({
    ...it,
    qty: Math.max(1, Number(it.qty) || 1),
  }));
}

function renderCart() {
  const count = cartItems.reduce((a, it) => a + (Number(it.qty) || 1), 0);
  els.cartCount.textContent = `${count} item${count === 1 ? "" : "s"}`;

  els.checkoutBtn.disabled = cartItems.length === 0;

  els.cartList.innerHTML = cartItems.length
    ? cartItems.map(renderItemCard).join("")
    : `<div class="muted">Your cart is empty.</div>`;

  // Summary (estimate)
  const estSubtotal = cartItems.reduce((a, it) => a + (Number(it.priceCents || it.price || 0) * (Number(it.qty) || 1)), 0);
  els.summaryRows.innerHTML = `
    <div class="row"><div>Subtotal (est.)</div><div class="val">${money(estSubtotal)}</div></div>
  `;
  els.summaryTotal.textContent = money(estSubtotal);

  // Hook buttons
  els.cartList.querySelectorAll("[data-action='qty']").forEach((inp) => {
    inp.addEventListener("change", onQtyChange);
  });
  els.cartList.querySelectorAll("[data-action='remove']").forEach((btn) => {
    btn.addEventListener("click", onRemove);
  });
}

function renderItemCard(it) {
  const img = it.image ? `<img src="${escapeHtml(it.image)}" alt="">` : `<div class="muted tiny">No image</div>`;
  const title = escapeHtml(it.title || it.name || "Item");
  const sub = it.variantLabel ? `<div class="sub">Variant: <strong>${escapeHtml(it.variantLabel)}</strong></div>` : "";
  const priceCents = Number(it.priceCents ?? it.price ?? 0);
  return `
    <div class="card" data-id="${escapeHtml(it.id)}">
      ${img}
      <div class="info">
        <div class="title">${title}</div>
        ${sub}
        <div class="sub">${escapeHtml(it.kind || "item")}</div>
      </div>
      <div class="actions">
        <input class="qty" data-action="qty" type="number" min="1" value="${Number(it.qty) || 1}">
        <div class="price">${money(priceCents * (Number(it.qty) || 1))}</div>
        <button class="btn" data-action="remove" type="button">Remove</button>
      </div>
    </div>
  `;
}

async function onQtyChange(e) {
  const card = e.target.closest(".card");
  if (!card || !currentUser) return;
  const id = card.getAttribute("data-id");
  const qty = Math.max(1, Number(e.target.value) || 1);
  await updateDoc(doc(db, "users", currentUser.uid, "cart", id), { qty });
}

async function onRemove(e) {
  const card = e.target.closest(".card");
  if (!card || !currentUser) return;
  const id = card.getAttribute("data-id");
  await deleteDoc(doc(db, "users", currentUser.uid, "cart", id));
}

// =========================
// Clear cart confirm modal
// =========================
function openConfirm() { els.confirmOverlay.hidden = false; }
function closeConfirm() { els.confirmOverlay.hidden = true; }

async function clearCartAll() {
  if (!currentUser) return;
  const items = await readCartOnce(currentUser.uid);
  await Promise.all(items.map((it) => deleteDoc(doc(db, "users", currentUser.uid, "cart", it.id))));
}

// =========================
// Checkout modal
// =========================
function openCheckout() {
  resetCheckoutState();
  els.checkoutOverlay.hidden = false;

  // Fill right-panel summary immediately
  renderModalSummary(null);

  // Pre-fill some fields if present in cart/user item
  // (leave as-is; you can wire profile later)
}

function closeCheckout() {
  els.checkoutOverlay.hidden = true;
  destroySquareCard();
}

function setStep(n) {
  checkoutStep = n;

  // left panes
  stepEls.pane1.hidden = n !== 1;
  stepEls.pane2.hidden = n !== 2;

  // right panes
  stepEls.right1.hidden = n !== 1;
  stepEls.right2.hidden = n !== 2;

  // header steps
  stepEls.step1.classList.toggle("is-active", n === 1);
  stepEls.step2.classList.toggle("is-active", n === 2);

  // footer buttons
  els.backBtn.hidden = n === 1;
  els.nextBtn.textContent = n === 1 ? "Next" : "Next";

  if (n === 2) {
    // Payment fills entire right panel (CSS already does this)
    initSquareIfNeeded().catch((err) => showErr(els.cardErrors, err.message || String(err)));
  }
}

function resetCheckoutState() {
  quoteId = null;
  shippingOptions = [];
  selectedShippingId = null;
  quoteSubtotalCents = 0;

  hideErr(els.shipError);
  hideErr(els.cardErrors);
  hideErr(els.payLeftError);

  els.shipOptions.innerHTML = `<div class="muted tiny">Enter your address and click Next to load shipping options.</div>`;
  els.modalShipping.textContent = "$—";
  els.modalTotal.textContent = "$—";
  els.payTotal.textContent = "$—";
  els.payStatus.textContent = "";

  destroySquareCard();
  setStep(1);
}

function getAddressFromInputs() {
  return {
    name: els.shipName.value.trim(),
    email: els.shipEmail.value.trim(),
    phone: els.shipPhone.value.trim(),
    address1: els.shipAddress1.value.trim(),
    address2: els.shipAddress2.value.trim(),
    city: els.shipCity.value.trim(),
    state: els.shipState.value.trim(),
    zip: els.shipZip.value.trim(),
    country: els.shipCountry.value.trim().toUpperCase(),
  };
}

function validateAddressClient(a) {
  const miss = [];
  if (!a.address1) miss.push("Address line 1");
  if (!a.city) miss.push("City");
  if (!a.zip) miss.push("ZIP / Postal");
  if (!a.country) miss.push("Country");
  if (a.country === "US" && !a.state) miss.push("State");
  if (!a.email || !/^\S+@\S+\.\S+$/.test(a.email)) miss.push("Valid email");
  return miss;
}

async function createQuote() {
  const address = getAddressFromInputs();
  const missing = validateAddressClient(address);
  if (missing.length) {
    showErr(els.shipError, `Please provide: ${missing.join(", ")}`);
    return null;
  }

  // Build items for the Worker: printfulVariantId + qty
  const items = cartItems
    .filter((it) => it.kind !== "free" && it.kind !== "digital")
    .map((it) => ({
      printfulVariantId: Number(it.printfulVariantId),
      qty: Math.max(1, Number(it.qty) || 1),
    }));

  if (!items.length) {
    showErr(els.shipError, "No shippable items in cart.");
    return null;
  }

  hideErr(els.shipError);
  els.nextBtn.disabled = true;
  els.nextBtn.textContent = "Quoting…";

  try {
    const res = await fetch(`${API_BASE}/quote`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ address, items }),
    });

    const data = await res.json().catch(() => null);

    if (!res.ok) {
      throw new Error(data?.error || `Quote failed (${res.status})`);
    }

    quoteId = data.quoteId;
    shippingOptions = Array.isArray(data.shippingOptions) ? data.shippingOptions : [];
    quoteSubtotalCents = Number(data.subtotal || 0);

    // auto-select first shipping option
    selectedShippingId = shippingOptions[0]?.id || null;

    renderShippingOptions();
    renderModalSummary({ subtotal: quoteSubtotalCents, shippingId: selectedShippingId });
    return data;
  } finally {
    els.nextBtn.disabled = false;
    els.nextBtn.textContent = "Next";
  }
}

function renderShippingOptions() {
  if (!shippingOptions.length) {
    els.shipOptions.innerHTML = `<div class="err">No shipping options available.</div>`;
    return;
  }

  els.shipOptions.innerHTML = shippingOptions.map((o) => {
    const sel = o.id === selectedShippingId ? "is-selected" : "";
    return `
      <div class="shipOpt ${sel}" data-ship-id="${escapeHtml(o.id)}" role="button" tabindex="0">
        <div>
          <div class="name">${escapeHtml(o.name || o.id)}</div>
          <div class="tiny muted">${escapeHtml(o.id)}</div>
        </div>
        <div class="cost">${money(Number(o.cost) || 0)}</div>
      </div>
    `;
  }).join("");

  els.shipOptions.querySelectorAll(".shipOpt").forEach((el) => {
    el.addEventListener("click", () => {
      selectedShippingId = el.getAttribute("data-ship-id");
      renderShippingOptions();
      renderModalSummary({ subtotal: quoteSubtotalCents, shippingId: selectedShippingId });
    });
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        el.click();
      }
    });
  });
}

function renderModalSummary(state) {
  // summary list
  els.summaryBox.innerHTML = cartItems.map((it) => {
    const title = escapeHtml(it.title || it.name || "Item");
    const qty = Math.max(1, Number(it.qty) || 1);
    return `<div class="row"><div>${title} <span class="tiny muted">× ${qty}</span></div><div class="val">${money((Number(it.priceCents ?? it.price ?? 0) * qty) || 0)}</div></div>`;
  }).join("");

  // totals (server subtotal + selected shipping)
  if (!state) {
    // unknown until quote
    const estSubtotal = cartItems.reduce((a, it) => a + (Number(it.priceCents ?? it.price ?? 0) * (Number(it.qty) || 1)), 0);
    els.modalSubtotal.textContent = money(estSubtotal);
    els.modalShipping.textContent = "$—";
    els.modalTotal.textContent = "$—";
    els.payTotal.textContent = "$—";
    return;
  }

  const subtotal = Number(state.subtotal || 0);
  const ship = shippingOptions.find((o) => o.id === state.shippingId);
  const shipCents = Number(ship?.cost || 0);
  const total = subtotal + shipCents;

  els.modalSubtotal.textContent = money(subtotal);
  els.modalShipping.textContent = money(shipCents);
  els.modalTotal.textContent = money(total);
  els.payTotal.textContent = money(total);
}

// =========================
// Square Web Payments SDK
// =========================
// You must set these to your actual Square Application ID + location.
// If you keep the Application ID server-side only, you’ll need an endpoint to fetch it.
// For now, set it here or load from a static JSON file.
const SQUARE_APP_ID = "sq0idp-Nnvnru9L9hR3CwVKikShGA";
const SQUARE_LOCATION_ID = "L4KPR2BE0PAA4";

async function initSquareIfNeeded() {
  if (!quoteId) {
    showErr(els.payLeftError, "Missing quote. Go Back and run shipping again.");
    return;
  }
  if (!selectedShippingId) {
    showErr(els.payLeftError, "Select a shipping method.");
    return;
  }

  if (!window.Square) {
    throw new Error("Square SDK not loaded. Check CSP script-src + squarecdn.");
  }

  if (squareCard) return;

  squarePayments = window.Square.payments(SQUARE_APP_ID, SQUARE_LOCATION_ID);
  squareCard = await squarePayments.card();
  await squareCard.attach("#card-container");
}

function destroySquareCard() {
  try { squareCard?.destroy?.(); } catch {}
  squareCard = null;
  squarePayments = null;
}

async function payNow() {
  hideErr(els.cardErrors);
  els.payStatus.textContent = "Tokenizing…";
  els.payBtn.disabled = true;

  try {
    if (!squareCard) await initSquareIfNeeded();

    const tokenResult = await squareCard.tokenize();
    if (tokenResult.status !== "OK") {
      throw new Error(tokenResult.errors?.[0]?.message || "Card tokenization failed");
    }

    const sourceId = tokenResult.token;

    els.payStatus.textContent = "Charging…";

    const res = await fetch(`${API_BASE}/pay`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        quoteId,
        selectedShippingId,
        sourceId,
      }),
    });

    const data = await res.json().catch(() => null);
    if (!res.ok) throw new Error(data?.error || `Payment failed (${res.status})`);

    if (data.status === "paid") {
      els.payStatus.textContent = "Paid ✅";
      // TODO: clear cart or remove only shippable items; this is your existing behavior decision.
      // Here we clear everything for simplicity:
      await clearCartAll();
      closeCheckout();
      return;
    }

    // If pending, poll
    els.payStatus.textContent = "Confirming…";
    await pollPaymentStatus(quoteId);
  } catch (err) {
    showErr(els.cardErrors, err.message || String(err));
    els.payStatus.textContent = "";
  } finally {
    els.payBtn.disabled = false;
  }
}

async function pollPaymentStatus(qid) {
  const deadline = Date.now() + 45 * 1000;

  while (Date.now() < deadline) {
    const res = await fetch(`${API_BASE}/payment-status?quoteId=${encodeURIComponent(qid)}`, {
      method: "GET",
      headers: { "Accept": "application/json" },
    });
    const data = await res.json().catch(() => null);
    const st = data?.status || "pending";

    if (st === "paid") {
      els.payStatus.textContent = "Paid ✅";
      await clearCartAll();
      closeCheckout();
      return;
    }
    if (st === "failed") {
      throw new Error("Payment failed");
    }
    await sleep(1200);
  }

  throw new Error("Payment confirmation timed out. If you were charged, contact support.");
}

// =========================
// UI helpers
// =========================
function showErr(el, msg) {
  if (!el) return;
  el.hidden = false;
  el.textContent = msg;
}
function hideErr(el) {
  if (!el) return;
  el.hidden = true;
  el.textContent = "";
}
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

// =========================
// Wire up events
// =========================
els.checkoutBtn.addEventListener("click", openCheckout);
els.checkoutCloseBtn.addEventListener("click", closeCheckout);

els.clearCartBtn.addEventListener("click", openConfirm);
els.confirmCancelBtn.addEventListener("click", closeConfirm);
els.confirmClearBtn.addEventListener("click", async () => {
  await clearCartAll();
  closeConfirm();
});

els.backBtn.addEventListener("click", () => setStep(1));

els.nextBtn.addEventListener("click", async () => {
  if (checkoutStep === 1) {
    const q = await createQuote();
    if (!q) return;

    // Must have options selected before moving on
    if (!selectedShippingId) {
      showErr(els.shipError, "Select a shipping method.");
      return;
    }

    setStep(2);
    return;
  }
});

els.payBtn.addEventListener("click", payNow);

// Close overlay if click outside modal
els.checkoutOverlay.addEventListener("click", (e) => {
  if (e.target === els.checkoutOverlay) closeCheckout();
});
els.confirmOverlay.addEventListener("click", (e) => {
  if (e.target === els.confirmOverlay) closeConfirm();
});

// =========================
// Init
// =========================
let unsubCart = null;

onAuthStateChanged(auth, async (user) => {
  currentUser = user || null;

  if (unsubCart) { unsubCart(); unsubCart = null; }

  if (!currentUser) {
    cartItems = [];
    renderCart();
    return;
  }

  await loadProductsJson().catch(() => null);
  unsubCart = listenCart(currentUser.uid);
});

// Initial render (before auth)
renderCart();
