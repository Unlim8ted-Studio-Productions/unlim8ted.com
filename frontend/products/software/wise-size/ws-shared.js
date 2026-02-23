// /products/software/wise-size/dashboard/ws-shared.js
import { getFirebase } from "/components/firebase-init.js";
import { onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import {
  doc, getDoc, setDoc, updateDoc,
  collection, query, where, orderBy, limit, getDocs, addDoc,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

export const PATHS = {
  DASHBOARD_BASE: "/products/software/wise-size/dashboard/",
  DASHBOARD_HOME: "/products/software/wise-size/dashboard/index.html",
  SETUP: "/products/software/wise-size/dashboard/setup.html",
  PRODUCTS: "/products/software/wise-size/dashboard/products/index.html",
  USAGE: "/products/software/wise-size/dashboard/usage/index.html",
  SETTINGS: "/products/software/wise-size/dashboard/settings/index.html",
  DOCS: "/products/software/wise-size/docs/index.html",
  SIGNIN: "/sign-in",
};

export const WS = {
  ROOT_COLLECTION: "wise-size",
  ROOT_DOC: "app",
};

export function getApp() {
  const { auth, db } = getFirebase();
  if (!auth || !db) throw new Error("firebase-init.js must provide auth + db via getFirebase()");
  return { auth, db };
}

/**
 * Correct Firestore refs (fixes your FirebaseError):
 * /wise-size/app/{subcollection}/{docId?}
 */
export function wsDoc(db, subcollection, docId) {
  return doc(db, WS.ROOT_COLLECTION, WS.ROOT_DOC, subcollection, docId);
}
export function wsCol(db, subcollection) {
  return collection(db, WS.ROOT_COLLECTION, WS.ROOT_DOC, subcollection);
}

export function redirectToSignIn() {
  // redirect back to the exact page user tried to access
  const here = window.location.pathname + window.location.search + window.location.hash;
  const redirect = encodeURIComponent(here);
  window.location.href = `${PATHS.SIGNIN}?redirect=${redirect}`;
}

export function merchantReady(merchantDoc) {
  const nameOk = !!String(merchantDoc?.name || "").trim();
  const websiteOk = !!String(merchantDoc?.website || "").trim();
  const verifiedOk = merchantDoc?.websiteVerified === true;
  return { nameOk, websiteOk, verifiedOk, ready: nameOk && websiteOk && verifiedOk };
}

export function isValidHttpUrl(u) {
  try {
    const x = new URL(u);
    return x.protocol === "http:" || x.protocol === "https:";
  } catch {
    return false;
  }
}

export function toastify(containerEl, title, detail = "", ms = 2400) {
  if (!containerEl) return;
  const el = document.createElement("div");
  el.style.border = "1px solid rgba(255,255,255,.14)";
  el.style.background = "rgba(15,17,22,.96)";
  el.style.borderRadius = "14px";
  el.style.padding = "10px 12px";
  el.style.boxShadow = "0 20px 70px rgba(0,0,0,.45)";
  el.innerHTML = `
    <div style="font-weight:950">${escapeHtml(title)}</div>
    ${detail ? `<div style="margin-top:4px;opacity:.75;font-size:12px;line-height:1.35">${escapeHtml(detail)}</div>` : ""}
  `;
  containerEl.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .2s"; }, Math.max(600, ms - 200));
  setTimeout(() => el.remove(), ms);
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[m]));
}

export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch {}
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
  return true;
}

/**
 * Gate any dashboard page:
 * - If not signed in => /sign-in?redirect=...
 * - If not merchant-ready => setup.html
 */
export function gateMerchantPage({ allowUnverified = false } = {}) {
  const { auth, db } = getApp();

  return new Promise((resolve) => {
    onAuthStateChanged(auth, async (user) => {
      if (!user) {
        redirectToSignIn();
        return;
      }

      const merchantRef = wsDoc(db, "merchants", user.uid);
      const snap = await getDoc(merchantRef);
      const merchant = snap.exists() ? snap.data() : null;

      // If no merchant doc or not ready, bounce to setup unless allowUnverified = true
      const st = merchantReady(merchant);
      const needsSetup = !merchant || (!allowUnverified && !st.ready);

      if (needsSetup) {
        // IMPORTANT: setup is a FILE, not /setup
        window.location.href = PATHS.SETUP;
        return;
      }

      resolve({ user, db, auth, merchantId: user.uid, merchant });
    });
  });
}

export async function doSignOut() {
  const { auth } = getApp();
  await signOut(auth);
  redirectToSignIn();
}

/**
 * Setup helpers (create merchant doc, update profile, set verified)
 * Must comply with your Firestore rules.
 */
export async function ensureMerchantDocExists(db, merchantId) {
  const ref = wsDoc(db, "merchants", merchantId);
  const snap = await getDoc(ref);
  if (snap.exists()) return snap.data();

  // Your rules for CREATE require:
  // ownerUid + name + website (required strings)
  // So we cannot create until user fills those.
  return null;
}

export async function createMerchantDoc(db, merchantId, { name, website }) {
  if (!String(name || "").trim()) throw new Error("Store name required");
  if (!String(website || "").trim()) throw new Error("Website required");
  if (!isValidHttpUrl(website)) throw new Error("Website must be http(s)");

  const ref = wsDoc(db, "merchants", merchantId);

  await setDoc(ref, {
    ownerUid: merchantId,
    name: String(name).trim(),
    website: String(website).trim().replace(/\/+$/, ""),
    features: {
      profileSaveEnabled: true,
      aiTryOnEnabled: false,
    },
    websiteVerified: false,
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  }, { merge: false });

  const snap = await getDoc(ref);
  return snap.data();
}

export async function updateMerchantProfile(db, merchantId, patch) {
  const ref = wsDoc(db, "merchants", merchantId);

  // Your UPDATE rules restrict changed keys to:
  // name, website, features, websiteVerified, updatedAt
  const update = { ...patch, updatedAt: serverTimestamp() };
  await updateDoc(ref, update);

  const snap = await getDoc(ref);
  return snap.data();
}

/**
 * Items & usage queries
 */
export async function listMyItems(db, merchantId, { limitN = 60 } = {}) {
  const qy = query(
    wsCol(db, "items"),
    where("merchantId", "==", merchantId),
    orderBy("createdAt", "desc"),
    limit(limitN)
  );
  const snap = await getDocs(qy);
  return snap.docs.map(d => ({ id: d.id, ...d.data() }));
}

export async function createItem(db, merchantId, item) {
  // Must match your itemShapeOk() keys & types.
  return await addDoc(wsCol(db, "items"), {
    merchantId,
    brand: String(item.brand || "").trim(),
    name: String(item.name || "").trim(),
    category: item.category,
    sizeChart: item.sizeChart,       // map
    measurement: item.measurement || { type: "body_recommended", unit: "cm", method: "merchant_entered", tolerances: {} },
    publishStatus: item.publishStatus || "submitted",
    verification: item.verification || { stage: "auto_schema", status: "pending", progress: 10 },
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  });
}

export async function listUsageEvents(db, merchantId, { limitN = 100 } = {}) {
  const qy = query(
    wsCol(db, "usageEvents"),
    where("merchantId", "==", merchantId),
    orderBy("ts", "desc"),
    limit(limitN)
  );
  const snap = await getDocs(qy);
  return snap.docs.map(d => ({ id: d.id, ...d.data() }));
}