// /js/cart-data.js
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";
import {
  collection, onSnapshot, doc, deleteDoc, updateDoc, serverTimestamp,
  addDoc, query, orderBy, getDocs
} from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";

import { getFirebase } from "/components/firebase-init.js";
const { auth, db } = getFirebase();

/* =============================
   Tiny event bus
============================= */
const listeners = new Map(); // event -> Set(fn)
function emit(event, payload) {
  const set = listeners.get(event);
  if (!set) return;
  for (const fn of set) {
    try { fn(payload); } catch (e) { console.error(e); }
  }
}
export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(fn);
  return () => listeners.get(event)?.delete(fn);
}

/* =============================
   Local storage helpers
============================= */
export function getLocalCart() {
  try {
    const raw = localStorage.getItem("unlim8ted-cart");
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch { return []; }
}
export function setLocalCart(items) {
  try {
    localStorage.setItem("unlim8ted-cart", JSON.stringify(items || []));
    window.dispatchEvent(new CustomEvent("cart-changed"));
  } catch {}
}
export function getLocalPurchases() {
  try {
    const raw = localStorage.getItem("unlim8ted-purchases");
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch { return []; }
}
export function addLocalPurchase(row) {
  try {
    const arr = getLocalPurchases();
    arr.unshift(row);
    localStorage.setItem("unlim8ted-purchases", JSON.stringify(arr.slice(0, 200)));
  } catch {}
}

/* =============================
   products.json index
============================= */
let _productsLoaded = false;
let _productsLoading = null;
let _productById = new Map();
let _variantByKey = new Map(); // `${productId}::${variantId}` -> variant

function safeUrl(u) {
  const s = String(u || "").trim();
  if (!s) return "";
  if (s.startsWith("https://") || s.startsWith("http://") || s.startsWith("/")) return s;
  return "";
}

export async function loadProducts() {
  if (_productsLoaded) return;
  if (_productsLoading) return _productsLoading;

  _productsLoading = (async () => {
    try {
      const r = await fetch("https://unlim8ted.com/tools/data/products.json", { cache: "no-store" });
      if (!r.ok) throw new Error("Failed to load products.json");
      const data = await r.json();
      const products = Array.isArray(data) ? data : (data?.products || []);
      indexProducts(products);
      _productsLoaded = true;
    } catch (e) {
      console.warn("cart-data products.json load failed:", e);
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
    const pid = String(p.id ?? p.productId ?? "").trim();
    if (!pid) continue;
    _productById.set(pid, p);

    const vars = Array.isArray(p.varients) ? p.varients : (Array.isArray(p.variants) ? p.variants : []);
    for (const v of vars) {
      const vid = String(v.id ?? v.variantId ?? "").trim();
      if (!vid) continue;
      _variantByKey.set(`${pid}::${vid}`, v);
    }
  }
}

export function getProduct(pid) {
  return _productById.get(String(pid || "").trim()) || null;
}
export function getVariant(pid, vid) {
  const k = `${String(pid || "").trim()}::${String(vid || "").trim()}`;
  return _variantByKey.get(k) || null;
}

export function resolveProductImage(productId, variantId) {
  const pid = String(productId || "").trim();
  if (!pid) return "";
  const p = _productById.get(pid) || null;
  if (!p) return "";

  const vid = String(variantId || "").trim();
  if (vid) {
    const v = _variantByKey.get(`${pid}::${vid}`) || null;
    const vImg =
      (v && Array.isArray(v.images) && v.images.length && safeUrl(v.images[0])) ||
      safeUrl(v?.image) || safeUrl(v?.imageUrl) || safeUrl(v?.thumbnail) || "";
    if (vImg) return vImg;
  }

  const pImg = safeUrl(p.image) || safeUrl(p.imageUrl) || safeUrl(p.thumbnail) || safeUrl(p.thumb) || "";
  if (pImg) return pImg;

  if (Array.isArray(p.images) && p.images.length) {
    const first = p.images[0];
    if (typeof first === "string") return safeUrl(first);
    if (first && typeof first === "object") return safeUrl(first.url || first.src || first.imageUrl || first.image);
  }
  return "";
}

/* =============================
   Firestore mutations
============================= */
export async function setQty(user, id, qty) {
  qty = Math.max(1, qty | 0);
  const uid = user?.uid || null;

  if (!uid) {
    const arr = getLocalCart();
    const idx = arr.findIndex(x => String(x.id) === String(id));
    if (idx >= 0) {
      arr[idx].qty = qty;
      setLocalCart(arr);
    }
    return;
  }

  await updateDoc(doc(db, "users", uid, "cartItems", id), {
    qty,
    updatedAt: serverTimestamp(),
  });
  window.dispatchEvent(new CustomEvent("cart-changed"));
}

export async function removeItem(user, id) {
  const uid = user?.uid || null;
  if (!uid) {
    const arr = getLocalCart().filter(x => String(x.id) !== String(id));
    setLocalCart(arr);
    return;
  }
  await deleteDoc(doc(db, "users", uid, "cartItems", id));
  window.dispatchEvent(new CustomEvent("cart-changed"));
}

export async function clearEntireCart(user) {
  const uid = user?.uid || null;
  if (!uid) {
    setLocalCart([]);
    return;
  }
  const snap = await getDocs(collection(db, "users", uid, "cartItems"));
  await Promise.all(snap.docs.map(d => deleteDoc(d.ref)));
  window.dispatchEvent(new CustomEvent("cart-changed"));
}

export async function writePurchase(user, payload) {
  const uid = user?.uid || null;
  if (!uid) return;
  try {
    await addDoc(collection(db, "users", uid, "purchases"), {
      ...payload,
      createdAt: serverTimestamp(),
    });
  } catch (err) {
    console.error("writePurchase failed", err);
  }
}

/* =============================
   Start listeners
============================= */
let unsubCart = null;
let unsubPurchases = null;

export function startCartData() {
  onAuthStateChanged(auth, async (user) => {
    emit("auth", { user: user || null });

    if (unsubCart) { unsubCart(); unsubCart = null; }
    if (unsubPurchases) { unsubPurchases(); unsubPurchases = null; }

    if (!user) {
      emit("cart", { user: null, items: getLocalCart() });
      emit("purchases", { user: null, rows: getLocalPurchases() });
      return;
    }

    unsubCart = onSnapshot(collection(db, "users", user.uid, "cartItems"), (snap) => {
      const raw = [];
      snap.forEach(d => raw.push({ id: d.id, ...d.data() }));
      emit("cart", { user, items: raw });
    });

    unsubPurchases = onSnapshot(
      query(collection(db, "users", user.uid, "purchases"), orderBy("createdAt", "desc")),
      (snap) => {
        const rows = [];
        snap.forEach(d => rows.push(d.data() || {}));
        emit("purchases", { user, rows });
      }
    );
  });

  window.addEventListener("cart-changed", () => {
    emit("cart", { user: null, items: getLocalCart() });
  });
}
