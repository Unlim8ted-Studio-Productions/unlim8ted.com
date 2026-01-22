import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";
    import {
      collection, query, orderBy, onSnapshot, doc, setDoc, getDoc, deleteDoc, updateDoc,
      Timestamp
    } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";
    import { getFirebase } from "/components/firebase-init.js";

    const { auth, db } = getFirebase();
    const $ = (id) => document.getElementById(id);

    // --- DOM
    const hero = $("hero");
    const thumbs = $("thumbs");
    const pName = $("pName");
    const pDesc = $("pDesc");
    const pPrice = $("pPrice");
    const buyBtn = $("buyBtn");
    const addBtn = $("addBtn");

    const avgNum = $("avgNum");
    const avgLabel = $("avgLabel");
    const countLabel = $("countLabel");
    const avgStars = $("avgStars");
    const avgStarsBig = $("avgStarsBig");
    const bars = $("bars");

    const commentList = $("commentList");
    const commentForm = $("commentForm");
    const mustSignIn = $("mustSignIn");
    const ratingSel = $("ratingSel");
    const displayNameInput = $("displayName");
    const commentText = $("commentText");
    const formMsg = $("formMsg");
    const deleteMyReviewBtn = $("deleteMyReviewBtn");
    const postBtn = $("postBtn");

    const pageTitle = document.querySelector(".title");

    // ----------------------------
    // Utils
    // ----------------------------
    function starsText(value) {
      const v = Math.max(0, Math.min(5, value));
      let out = "";
      for (let i = 1; i <= 5; i++) out += (i <= Math.round(v)) ? "★" : "☆";
      return out;
    }

    function toMoney(n) {
      if (typeof n !== "number" || Number.isNaN(n)) return "$—";
      return "$" + n.toFixed(2);
    }

    function extractYouTubeID(url) {
      const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|embed)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
      const m = (url || "").match(regex);
      return m ? m[1] : null;
    }

    function toast(msg) {
      const t = document.createElement("div");
      t.textContent = msg;
      t.style.position = "fixed";
      t.style.right = "18px";
      t.style.bottom = "18px";
      t.style.padding = "12px 14px";
      t.style.borderRadius = "14px";
      t.style.background = "rgba(0,255,153,.92)";
      t.style.color = "#0b0b0f";
      t.style.fontWeight = "900";
      t.style.boxShadow = "0 12px 30px rgba(0,0,0,.35)";
      t.style.zIndex = "99999";
      document.body.appendChild(t);
      setTimeout(() => t.remove(), 1600);
    }

    function hasContent(v) { return typeof v === "string" && v.trim().length > 0; }

    function chevronSvg() {
      return `
        <svg class="chev" viewBox="0 0 24 24" fill="none"
          stroke="rgba(255,255,255,.75)" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M6 9l6 6 6-6"></path>
        </svg>
      `;
    }

    function removeInfoAccordions() {
      document.querySelectorAll("#infoAccordions").forEach(el => el.remove());
    }

    function renderInfoAccordions(item) {
      removeInfoAccordions();

      const detailsText = hasContent(item.details) ? item.details.trim() : "";
      const notesText = hasContent(item.notes) ? item.notes.trim() : "";

      if (!detailsText && !notesText) return;

      const wrap = document.createElement("div");
      wrap.id = "infoAccordions";
      wrap.className = "infoAccordions";

      if (detailsText) {
        const d = document.createElement("details");
        d.className = "infoDrop";
        d.innerHTML = `
          <summary>
            <span class="sumLeft">
              <span>Details</span>
              <span class="sumHint">More info</span>
            </span>
            ${chevronSvg()}
          </summary>
          <div class="content">${escapeHtml(detailsText)}</div>
        `;
        wrap.appendChild(d);
      }

      if (notesText) {
        const n = document.createElement("details");
        n.className = "infoDrop important";
        n.open = true;
        n.innerHTML = `
          <summary>
            <span class="sumLeft">
              <span>Notes</span>
              <span class="badgeImportant" aria-label="Important notes">IMPORTANT</span>
            </span>
            ${chevronSvg()}
          </summary>
          <div class="content">${escapeHtml(notesText)}</div>
        `;
        wrap.appendChild(n);
      }

      pDesc.insertAdjacentElement("afterend", wrap);
    }

    function notifyCartChanged() {
      window.dispatchEvent(new CustomEvent("cart-changed"));
    }

    function safeStr(v, max = 600) { return (v == null) ? "" : String(v).slice(0, max); }

    function escapeHtml(s) {
      return String(s ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[c]));
    }

    // ----------------------------
    // Media rendering
    // ----------------------------
    function setHeroMediaFromSource(itemName, source) {
      hero.innerHTML = "";

      if (!source) {
        hero.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:rgba(255,255,255,.55);">No media</div>`;
        return;
      }

      if (source.type === "img") {
        const img = document.createElement("img");
        img.src = source.src;
        img.alt = itemName || "Product";
        hero.appendChild(img);
        return;
      }

      const url = source.src || "";
      if (url.includes("youtube.com") || url.includes("youtu.be")) {
        const id = extractYouTubeID(url);
        const iframe = document.createElement("iframe");
        iframe.src = `https://www.youtube.com/embed/${id}`;
        iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
        iframe.allowFullscreen = true;
        hero.appendChild(iframe);
      } else {
        const v = document.createElement("video");
        v.controls = true;
        v.src = url;
        hero.appendChild(v);
      }
    }

    function buildThumbsFromSources(itemName, sources) {
      thumbs.innerHTML = "";

      const list = (sources || []).filter(Boolean).slice(0, 12);
      if (!list.length) {
        setHeroMediaFromSource(itemName, null);
        return;
      }

      setHeroMediaFromSource(itemName, list[0]);

      list.forEach(s => {
        const t = document.createElement("div");
        t.className = "thumb";

        if (s.type === "img") {
          const img = document.createElement("img");
          img.src = s.src;
          img.alt = itemName || "Product";
          t.appendChild(img);
        } else {
          t.innerHTML = `<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:rgba(255,255,255,.75);font-weight:800;">▶</div>`;
        }

        t.addEventListener("click", () => setHeroMediaFromSource(itemName, s));
        thumbs.appendChild(t);
      });
    }

    // ----------------------------
    // Products loading + normalization
    // ----------------------------
    let productsCache = null;
    async function loadProducts() {
      const r = await fetch("assets.unlim8ted.com/data/products.json", { cache: "no-store" });
      if (!r.ok) throw new Error("Failed to load assets.unlim8ted.com/data/products.json");
      return await r.json();
    }
    async function getProductsOnce() {
      if (productsCache) return productsCache;
      productsCache = await loadProducts();
      return productsCache;
    }

function normalizeItemForPage(raw) {
  const item = { ...raw };

  item.id = safeStr(item.id, 200);
  item.name = item.name ?? item.title ?? "";
  item.description = item.description ?? item.desc ?? "";
  item.details = item.details ?? "";
  item.notes = item.notes ?? "";

  item.productType = item["product-type"] || item.productType || item.type || "";
  item.productType = String(item.productType || "").toLowerCase();

  // images list (new schema uses item.images)
  item.image = item.image ?? item.main_image ?? "";
  item.images = Array.isArray(item.images) ? item.images : (Array.isArray(item.additional_images) ? item.additional_images : []);
  item.video = item.video ?? null;
  item.additional_videos = item.additional_videos ?? [];

  // variation types
  item.variation_types = Array.isArray(item.variation_types) ? item.variation_types : [];

  // variants (spelled varients in your JSON)
  item.varients = item.varients ?? item.variants ?? [];
  if (!Array.isArray(item.varients)) item.varients = [];

  // digital price fallback
  if (typeof item.price !== "number") {
    const p = Number(item.price);
    item.price = Number.isFinite(p) ? p : 0;
  }

  // normalize each variant to one consistent shape:
  // { id, label, parts[], price, image, available }
  item.varients = item.varients.map(v => {
    const vv = { ...v };
    vv.id = safeStr(vv.id || "", 200);

    vv.label = (vv.variantLabel ?? vv.name ?? vv.title ?? "").trim();
    vv.optionParts = Array.isArray(vv.optionParts) ? vv.optionParts.map(x => String(x).trim()) : [];

    // if optionParts missing, derive from label "Black, XS"
    if (!vv.optionParts.length && vv.label) {
      vv.optionParts = vv.label.split(",").map(x => x.trim()).filter(Boolean);
    }

    const pn = Number(vv.price);
    vv.price = Number.isFinite(pn) ? pn : 0;

    vv.currency = vv.currency || "USD";
    vv.image = vv.image || "";

    // availability boolean (default true if missing)
    vv.available = (typeof vv.available === "boolean") ? vv.available : true;

    return vv;
  });

  return item;
}

    // ----------------------------
    // Variant parsing
    // ----------------------------
function titleCaseWord(s) {
  const w = String(s || "").trim();
  if (!w) return "";
  return w.charAt(0).toUpperCase() + w.slice(1);
}

function buildVariantGroupsFromParts(varients, variationTypes = []) {
  const partsList = (varients || []).map(v => Array.isArray(v.optionParts) ? v.optionParts : []);
  const maxLen = Math.max(0, ...partsList.map(a => a.length));

  const groups = [];
  for (let i = 0; i < maxLen; i++) {
    const vals = new Set();
    for (const parts of partsList) if (parts[i]) vals.add(parts[i]);

    const vt = Array.isArray(variationTypes) ? variationTypes[i] : null;
    const label = vt ? titleCaseWord(vt) : `Option ${i + 1}`;

    groups.push({ index: i, label, values: Array.from(vals) });
  }
  return groups;
}

function findVariantByParts(varients, selectionParts) {
  const target = (selectionParts || []).map(x => String(x || "").trim());
  return (varients || []).find(v => {
    const parts = Array.isArray(v.optionParts) ? v.optionParts : [];
    if (parts.length !== target.length) return false;
    for (let i = 0; i < parts.length; i++) if (parts[i] !== target[i]) return false;
    return true;
  }) || null;
}

    // ----------------------------
    // CART: add/merge by product + variantId
    // ----------------------------
    async function addToCartMerged(payload) {
      const user = auth.currentUser;

      const clampInt = (n, min, max) =>
        Math.max(min, Math.min(max, Math.trunc(Number(n) || 0)));

      const safeStr = (s, maxLen) => {
        const v = String(s ?? "").trim();
        if (!v) return "";
        return v.length > maxLen ? v.slice(0, maxLen) : v;
      };

      const productId = safeStr(payload.productId, 200);
      const variantId = safeStr(payload.variantId, 200);
      const addQty = clampInt(payload.qty ?? 1, 1, 99);

      if (!productId) throw new Error("addToCart: missing productId");

      const itemId = variantId ? `${productId}__${variantId}` : productId;

      if (user) {
        const ref = doc(db, "users", user.uid, "cartItems", itemId);
        const snap = await getDoc(ref);
        const now = Timestamp.now();

        if (!snap.exists()) {
          const docData = {
            productId,
            ...(variantId ? { variantId } : {}),
            qty: addQty,
            createdAt: now,
            updatedAt: now,
          };
          await setDoc(ref, docData);
        } else {
          const prevQty = clampInt(snap.data()?.qty ?? 0, 0, 99);
          const nextQty = clampInt(prevQty + addQty, 1, 99);

          await updateDoc(ref, {
            ...(variantId ? { variantId } : {}),
            qty: nextQty,
            updatedAt: now,
          });
        }

        notifyCartChanged();
        return;
      }

      const key = "unlim8ted-cart";
      const arr = JSON.parse(localStorage.getItem(key) || "[]");
      const idx = arr.findIndex(x => x && (x.id === itemId || x._id === itemId));

      if (idx >= 0) {
        const prev = clampInt(arr[idx].qty ?? 0, 0, 99);
        arr[idx].qty = clampInt(prev + addQty, 1, 99);
        arr[idx].updatedAt = Date.now();
      } else {
        arr.push({
          id: itemId,
          productId,
          variantId: variantId || null,
          qty: addQty,
          createdAt: Date.now(),
          updatedAt: Date.now(),
        });
      }

      localStorage.setItem(key, JSON.stringify(arr));
      notifyCartChanged();
    }

    // ----------------------------
    // Reviews helpers
    // ----------------------------
    function renderBars(counts, total) {
      bars.innerHTML = "";
      for (let star = 5; star >= 1; star--) {
        const n = counts[star] || 0;
        const pct = total > 0 ? (n / total) * 100 : 0;

        const row = document.createElement("div");
        row.className = "barRow";

        const left = document.createElement("div");
        left.textContent = `${star} star`;

        const track = document.createElement("div");
        track.className = "barTrack";

        const fill = document.createElement("div");
        fill.className = "barFill";
        fill.style.width = `${pct.toFixed(1)}%`;

        const right = document.createElement("div");
        right.style.textAlign = "right";
        right.textContent = String(n);

        track.appendChild(fill);
        row.appendChild(left);
        row.appendChild(track);
        row.appendChild(right);
        bars.appendChild(row);
      }
    }

    function renderComments(docs, currentUid) {
      commentList.innerHTML = "";

      if (!docs.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No reviews yet. Be the first.";
        commentList.appendChild(empty);
        return;
      }

      docs.forEach(d => {
        const data = d.data();
        const rating = Number(data.rating || 0);
        const text = (data.text || "").trim();
        const who = (data.displayName || "").trim() || "Anonymous";

        let when = "";
        const ts = data.createdAt;
        if (ts && typeof ts.toDate === "function") when = ts.toDate().toLocaleString();

        const el = document.createElement("div");
        el.className = "comment";

        const top = document.createElement("div");
        top.className = "commentTop";

        const whoEl = document.createElement("div");
        whoEl.className = "who";
        whoEl.style.color = "#ffcc66";
        whoEl.textContent = `${who} • ${starsText(rating)}`;

        if (currentUid && d.id === currentUid) {
          const badge = document.createElement("span");
          badge.className = "mineBadge";
          badge.textContent = "Yours";
          whoEl.appendChild(badge);
        }

        const whenEl = document.createElement("div");
        whenEl.className = "when";
        whenEl.textContent = when;

        const p = document.createElement("p");
        p.className = "commentText";
        p.textContent = text;

        top.appendChild(whoEl);
        top.appendChild(whenEl);

        el.appendChild(top);
        el.appendChild(p);

        commentList.appendChild(el);
      });
    }

    function setFormBusy(busy) {
      postBtn.disabled = busy;
      deleteMyReviewBtn.disabled = busy;
      ratingSel.disabled = busy;
      displayNameInput.disabled = busy;
      commentText.disabled = busy;
    }

    function showFormMessage(msg, isError = false) {
      formMsg.textContent = msg || "";
      formMsg.style.color = isError ? "rgba(255,120,120,.95)" : "rgba(255,255,255,.65)";
    }

    // ----------------------------
    // Variant UI + availability gating
    // ----------------------------
    function removeVariantUI() {
      document.querySelectorAll("#variantWrap, [data-variant-ui='1']").forEach(el => el.remove());
    }

    function setBuyEnabled(enabled, hrefWhenEnabled) {
      if (enabled) {
        buyBtn.setAttribute("aria-disabled", "false");
        buyBtn.style.pointerEvents = "";
        buyBtn.style.opacity = "";
        buyBtn.href = hrefWhenEnabled || "#";
      } else {
        buyBtn.setAttribute("aria-disabled", "true");
        buyBtn.style.pointerEvents = "none";
        buyBtn.style.opacity = ".55";
        buyBtn.href = "#";
      }
    }

function renderVariantUI(item) {
  removeVariantUI();

  const variants = Array.isArray(item.varients) ? item.varients : [];
  const groups = buildVariantGroupsFromParts(variants, item.variation_types || []);

  const wrap = document.createElement("div");
  wrap.id = "variantWrap";
  wrap.setAttribute("data-variant-ui", "1");
  wrap.style.marginTop = "12px";
  wrap.style.display = "grid";
  wrap.style.gap = "10px";

  wrap.innerHTML = `
    <div style="display:grid;gap:10px;grid-template-columns: 1fr 1fr;">
      ${groups.map(g => `
        <div>
          <label style="font-size:12px;color:rgba(255,255,255,.70);display:block;margin-bottom:6px;">${escapeHtml(g.label)}</label>
          <select data-variant-group="${g.index}" style="width:100%;padding:10px 12px;border-radius:12px;border:1px solid rgba(0,255,153,.30);background:rgba(16,16,23,.88);color:rgba(255,255,255,.92);outline:none;"></select>
        </div>
      `).join("")}
    </div>

    <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;">
      <div style="color:rgba(255,255,255,.62);font-size:13px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <span>Selected: <span id="selectedVariantLabel" style="color:rgba(255,255,255,.92);font-weight:800;"></span></span>
        <span id="variantAvailPill" class="pill" style="display:none;"></span>
      </div>
      <div style="color:rgba(0,255,153,.95);font-weight:900;font-size:16px;text-shadow:0 0 16px rgba(0,255,153,.22);">
        <span id="variantPrice">$—</span>
      </div>
    </div>
  `;

  pDesc.insertAdjacentElement("afterend", wrap);

  const buyHref = `/cart?source=buy&product=${encodeURIComponent(item.id)}`;
  const selects = Array.from(wrap.querySelectorAll("select[data-variant-group]"));

  // Fill dropdowns
  selects.forEach(sel => {
    const idx = Number(sel.getAttribute("data-variant-group"));
    const g = groups.find(x => x.index === idx);
    sel.innerHTML = "";
    (g?.values || []).forEach(val => {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = val;
      sel.appendChild(opt);
    });
    if ((g?.values || []).length) sel.value = g.values[0];
  });

  const selectedVariantLabelEl = wrap.querySelector("#selectedVariantLabel");
  const variantPriceEl = wrap.querySelector("#variantPrice");
  const variantAvailPill = wrap.querySelector("#variantAvailPill");

  const getSelectionParts = () => selects.map(s => s.value).filter(Boolean);

  function matchesExcept(parts, selected, skipIndex) {
    for (let i = 0; i < selected.length; i++) {
      if (i === skipIndex) continue;
      if (selected[i] && parts[i] !== selected[i]) return false;
    }
    return true;
  }

  function recomputeOptionAvailability() {
    const selected = getSelectionParts();

    selects.forEach((sel) => {
      const groupIndex = Number(sel.getAttribute("data-variant-group"));
      Array.from(sel.options).forEach((opt) => {
        const val = opt.value;

        const possibleInStock = variants.some(v => {
          const parts = Array.isArray(v.optionParts) ? v.optionParts : [];
          if (parts[groupIndex] !== val) return false;
          if (!matchesExcept(parts, selected, groupIndex)) return false;
          return v.available !== false; // boolean only
        });

        opt.disabled = !possibleInStock;
        opt.textContent = possibleInStock ? val : `${val} — Out of stock`;
      });
    });
  }

  function setAvailabilityUI(isAvail, hasAvailField) {
    if (!hasAvailField) {
      variantAvailPill.style.display = "none";
      return;
    }
    variantAvailPill.style.display = "inline-flex";
    variantAvailPill.className = `pill ${isAvail ? "ok" : "bad"}`;
    variantAvailPill.textContent = isAvail ? "In stock" : "Out of stock";
  }

  function syncVariant() {
    const parts = getSelectionParts();
    recomputeOptionAvailability();

    const vv = findVariantByParts(variants, parts);

    if (!vv) {
      selectedVariantLabelEl.textContent = parts.join(", ");
      variantPriceEl.textContent = "$—";
      pPrice.textContent = "$—";

      setAvailabilityUI(false, true);
      addBtn.disabled = true;
      setBuyEnabled(false, buyHref);
      addBtn.onclick = null;
      return;
    }

    const label = vv.label || vv.variantLabel || parts.join(", ");
    selectedVariantLabelEl.textContent = label;

    const priceNum = Number(vv.price);
    variantPriceEl.textContent = Number.isFinite(priceNum) ? toMoney(priceNum) : "$—";
    pPrice.textContent = Number.isFinite(priceNum) ? toMoney(priceNum) : "$—";

    // media: prefer variant image, else product images
    const sources = [];
    if (vv.image) sources.push({ type: "img", src: vv.image });
    if (!sources.length) {
      if (item.image) sources.push({ type: "img", src: item.image });
      (item.images || []).forEach(src => sources.push({ type: "img", src }));
      if (item.video) sources.push({ type: "video", src: item.video });
      (item.additional_videos || []).forEach(src => sources.push({ type: "video", src }));
    }
    buildThumbsFromSources(item.name, sources);

    const hasAvailField = (typeof vv.available === "boolean");
    const isAvail = (vv.available !== false);

    setAvailabilityUI(isAvail, hasAvailField);

    addBtn.disabled = !isAvail;
    setBuyEnabled(isAvail, buyHref);

    addBtn.onclick = async () => {
      if (!isAvail) {
        toast("This variant is out of stock");
        return;
      }
      try {
        await addToCartMerged({
          productId: item.id,
          ...(vv?.id ? { variantId: vv.id } : {}),
          qty: 1
        });
        toast("Added to cart");
      } catch (e) {
        console.error("addToCartMerged (variant) error:", e?.code, e?.message, e);
        toast("Could not add to cart");
      }
    };
  }

  selects.forEach(sel => sel.addEventListener("change", syncVariant));

  // initial
  syncVariant();

  return { wrap, selects, syncVariant };
}

    // ----------------------------
    // Page render
    // ----------------------------
    function getProductIdFromUrl() {
      return (window.location.hash || "").replace("#", "").trim();
    }

    function resetUI() {
      removeVariantUI();
      removeInfoAccordions();

      pageTitle.textContent = "Product";
      pName.textContent = "Loading…";
      pDesc.textContent = "";
      pPrice.textContent = "$—";

      setBuyEnabled(true, "#"); // default, gets set later
      buyBtn.removeAttribute("target");

      hero.innerHTML = "";
      thumbs.innerHTML = "";

      addBtn.style.display = "inline-flex";
      addBtn.disabled = true;
      addBtn.onclick = null;
    }

    let lastRenderedPid = null;
    let currentUid = null;
    let myReviewCache = null;
    let unsubAuth = null;
    let unsubComments = null;

    async function renderCurrentProduct(force = false) {
      const pid = getProductIdFromUrl();
      if (!force && pid === lastRenderedPid) return;
      lastRenderedPid = pid;

      resetUI();

      if (unsubAuth) { unsubAuth(); unsubAuth = null; }
      if (unsubComments) { unsubComments(); unsubComments = null; }

      if (!pid) {
        pageTitle.textContent = "Product Not Found";
        pName.textContent = "No product selected";
        pDesc.textContent = "Open a product using a URL like /product#some-id";
        setBuyEnabled(false);
        return;
      }

      const products = await getProductsOnce();
      const raw = products.find((p) => String(p.id) === String(pid));
      const item = raw ? normalizeItemForPage(raw) : null;

      if (!item) {
        pageTitle.textContent = "Product Not Found";
        pName.textContent = "Product not found";
        pDesc.textContent = "That product ID doesn't exist.";
        setBuyEnabled(false);
        return;
      }

      pageTitle.textContent = item.name || "Product";
      pName.textContent = item.name || "Untitled";
      pDesc.textContent = item.description || "";
      renderInfoAccordions(item);

      const isPhysical =
        item.productType === "physical" ||
        (Array.isArray(item.varients) && item.varients.length > 0);

      const buyHref = `/cart?source=buy&product=${encodeURIComponent(item.id)}`;

      if (isPhysical) {
        pPrice.textContent = "$—";
        setBuyEnabled(false, buyHref); // enabled only when a selected variant is available
        addBtn.style.display = "inline-flex";
        addBtn.disabled = true;
        renderVariantUI(item);
      } else {
        pPrice.textContent = toMoney(Number(item.price || 0));
        setBuyEnabled(true, buyHref);

        const sources = [];
        if (item.image) sources.push({ type: "img", src: item.image });
        (item.additional_images || []).forEach((src) => sources.push({ type: "img", src }));
        if (item.video) sources.push({ type: "video", src: item.video });
        (item.additional_videos || []).forEach((src) => sources.push({ type: "video", src }));
        buildThumbsFromSources(item.name, sources);

        addBtn.disabled = false;
        addBtn.onclick = async () => {
          try {
            await addToCartMerged({
              productId: item.id,
              qty: 1,
              variantId: "",
            });
            toast("Added to cart");
          } catch (e) {
            console.error("addToCartMerged error:", e?.code, e?.message, e);
            toast("Could not add to cart");
          }
        };
      }

      // -----------------------------------
      // COMMENTS LIVE QUERY
      // products/{productId}/comments
      // -----------------------------------
      const commentsRef = collection(db, "products", String(item.id), "comments");
      const commentsQ = query(commentsRef, orderBy("createdAt", "desc"));

      unsubAuth = onAuthStateChanged(auth, async (user) => {
        currentUid = user ? user.uid : null;

        if (user) {
          mustSignIn.style.display = "none";
          commentForm.style.display = "grid";
          displayNameInput.value = user.displayName || "";

          try {
            const myRef = doc(db, "products", String(item.id), "comments", user.uid);
            const mySnap = await getDoc(myRef);

            if (mySnap.exists()) {
              myReviewCache = mySnap.data();
              ratingSel.value = String(myReviewCache.rating || "5");
              displayNameInput.value = (myReviewCache.displayName || "").trim();
              commentText.value = (myReviewCache.text || "").trim();
              deleteMyReviewBtn.style.display = "inline-block";
              showFormMessage("Editing your review.");
            } else {
              myReviewCache = null;
              ratingSel.value = "5";
              displayNameInput.value = "";
              commentText.value = "";
              deleteMyReviewBtn.style.display = "none";
              showFormMessage("");
            }
          } catch (e) {
            console.error("load my review error:", e?.code, e?.message, e);
            myReviewCache = null;
            deleteMyReviewBtn.style.display = "none";
            showFormMessage("");
          }

        } else {
          myReviewCache = null;
          commentForm.style.display = "none";
          mustSignIn.style.display = "block";
          deleteMyReviewBtn.style.display = "none";
          showFormMessage("");
        }
      });

      unsubComments = onSnapshot(
        commentsQ,
        (snap) => {
          const docs = snap.docs;
          const total = docs.length;
          let sum = 0;
          const counts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };

          for (const d of docs) {
            const r = Number(d.data().rating || 0);
            if (r >= 1 && r <= 5) {
              sum += r;
              counts[r] = (counts[r] || 0) + 1;
            }
          }

          const avg = total ? sum / total : 0;
          avgNum.textContent = total ? avg.toFixed(1) : "—";
          avgLabel.textContent = total ? "Average rating" : "No ratings yet";
          countLabel.textContent = `${total} review${total === 1 ? "" : "s"}`;
          avgStars.textContent = starsText(avg);
          avgStarsBig.textContent = starsText(avg);

          renderBars(counts, total);
          renderComments(docs, currentUid);
        },
        (err) => {
          console.error("Comments listener error:", err?.code, err?.message, err);
          commentList.innerHTML = `<div class="empty">Reviews unavailable.</div>`;
        }
      );

      commentForm.onsubmit = async (e) => {
        e.preventDefault();

        const user = auth.currentUser;
        if (!user) return;

        const rating = Math.trunc(parseInt(ratingSel.value, 10) || 0);
        const text = commentText.value.trim();
        const rawName = displayNameInput.value.trim();
        const finalName = rawName ? rawName : "Anonymous";

        if (!text) { showFormMessage("Please write a comment.", true); return; }
        if (text.length > 2000) { showFormMessage("Comment is too long (max 2000 chars).", true); return; }
        if (rating < 1 || rating > 5) { showFormMessage("Please select a rating 1–5.", true); return; }
        if (finalName.length > 80) { showFormMessage("Name too long (max 80 chars).", true); return; }

        showFormMessage("Saving…");
        setFormBusy(true);

        try {
          const myRef = doc(db, "products", String(item.id), "comments", user.uid);
          const existing = await getDoc(myRef);

          if (existing.exists()) {
            await setDoc(myRef, { rating, text, displayName: finalName }, { merge: true });
          } else {
            await setDoc(myRef, {
              userId: user.uid,
              rating,
              text,
              displayName: finalName,
              createdAt: Timestamp.now(),
            });
          }

          const snap = await getDoc(myRef);
          myReviewCache = snap.exists() ? snap.data() : null;

          deleteMyReviewBtn.style.display = myReviewCache ? "inline-block" : "none";
          showFormMessage("Saved");
          setTimeout(() => showFormMessage(myReviewCache ? "Editing your review." : ""), 900);
        } catch (err) {
          console.error("save review error:", err?.code, err?.message, err);
          showFormMessage("Failed to save review.", true);
        } finally {
          setFormBusy(false);
        }
      };

      deleteMyReviewBtn.onclick = async () => {
        const user = auth.currentUser;
        if (!user) return;
        if (!confirm("Delete your review?")) return;

        showFormMessage("Deleting…");
        setFormBusy(true);

        try {
          const myRef = doc(db, "products", String(item.id), "comments", user.uid);
          await deleteDoc(myRef);

          myReviewCache = null;
          ratingSel.value = "5";
          displayNameInput.value = "";
          commentText.value = "";
          deleteMyReviewBtn.style.display = "none";
          showFormMessage("Deleted");
          setTimeout(() => showFormMessage(""), 900);
        } catch (err) {
          console.error("delete review error:", err?.code, err?.message, err);
          showFormMessage("Failed to delete.", true);
        } finally {
          setFormBusy(false);
        }
      };
    }

    await renderCurrentProduct(true);

    window.addEventListener("hashchange", () => {
      renderCurrentProduct(true).catch(console.error);
    });

    window.addEventListener("pageshow", () => {
      renderCurrentProduct(false).catch(console.error);
    });

    const y = new Date().getFullYear();
    $("footerText").innerHTML = `&copy; 2019-${y} Unlim8ted Studio Productions. All rights reserved.`;
