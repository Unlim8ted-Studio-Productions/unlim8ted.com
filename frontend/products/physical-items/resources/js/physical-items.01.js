const grid = document.getElementById("grid");
    const searchEl = document.getElementById("search");
    const filterEl = document.getElementById("filter");

    function esc(s) {
      return String(s ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[c]));
    }

    function productHref(id) {
      return `/products/product#${encodeURIComponent(id)}`;
    }

    function stockFromVariants(item) {
      const vs = Array.isArray(item?.varients) ? item.varients : null;
      if (!vs || vs.length === 0) return { key: "in", label: "In stock" };

      let total = 0, avail = 0;
      for (const v of vs) {
        total++;
        if (v?.available !== false) avail++;
      }
      if (avail === 0) return { key: "out", label: "Out of stock" };
      if (avail === total) return { key: "in", label: "In stock" };
      return { key: "partial", label: "Partial stock" };
    }

    function categoryGuessFromTags(p) {
      const tags = Array.isArray(p?.tags) ? p.tags.map(String) : [];
      const t = tags.join(" ").toLowerCase();
      if (t.includes("hood")) return "hoodies";
      if (t.includes("shirt") || t.includes("tee")) return "t-shirts";
      if (t.includes("sticker")) return "stickers";
      if (t.includes("cap") || t.includes("beanie") || t.includes("hat") || t.includes("slides") || t.includes("windbreaker") || t.includes("bag")) return "accessories";
      if (t.includes("mug") || t.includes("glass") || t.includes("coaster") || t.includes("notebook") || t.includes("puzzle") || t.includes("home")) return "home";
      return "accessories";
    }

    async function loadCatalog() {
      const r = await fetch("assets.unlim8ted.com/data/products.json", { cache: "no-store" });
      if (!r.ok) throw new Error(`Failed to load assets.unlim8ted.com/data/products.json (${r.status})`);
      const data = await r.json();
      const list = Array.isArray(data) ? data : (Array.isArray(data?.products) ? data.products : []);
      return list;
    }

    function passesFilters(p) {
      const q = (searchEl.value || "").trim().toLowerCase();
      const f = filterEl.value;

      const title = String(p?.name ?? p?.title ?? "").toLowerCase();
      const desc = String(p?.description ?? p?.desc ?? "").toLowerCase();
      const cat = String(p?.category ?? categoryGuessFromTags(p)).toLowerCase();

      const hay = `${title} ${desc} ${cat}`.trim();
      if (q && !hay.includes(q)) return false;
      if (f !== "all" && cat !== f) return false;
      return true;
    }

    function bestImage(p) {
      const img =
        p?.image ||
        p?.imageUrl ||
        p?.thumbnail ||
        p?.thumb ||
        p?.images?.[0] ||
        p?.media?.[0]?.src ||
        "";
      return String(img || "");
    }

    // ---- NEW: min/max price helpers (from varients[].price primarily) ----
    function toNumPrice(v) {
      if (v == null) return null;
      if (typeof v === "number" && Number.isFinite(v)) return v;
      const s = String(v).trim();
      if (!s) return null;
      // strip currency symbols and commas, keep digits/.- 
      const cleaned = s.replace(/[^0-9.\-]/g, "");
      const n = Number(cleaned);
      return Number.isFinite(n) ? n : null;
    }

    function formatMoney(n, currency = "USD") {
      try {
        return new Intl.NumberFormat(undefined, { style: "currency", currency }).format(n);
      } catch {
        // fallback
        return `$${n.toFixed(2)}`;
      }
    }

    function priceRangeFromProduct(p) {
      const vs = Array.isArray(p?.varients) ? p.varients : [];
      const prices = [];
      let currency = String(p?.currency || "") || "";

      for (const v of vs) {
        const n = toNumPrice(v?.price ?? v?.amount ?? v?.cost);
        if (n != null) prices.push(n);
        if (!currency && v?.currency) currency = String(v.currency);
      }

      // fallback to product-level price fields if variants don't have prices
      if (prices.length === 0) {
        const n = toNumPrice(p?.price);
        if (n != null) prices.push(n);
      }

      if (!currency) currency = "USD";
      if (prices.length === 0) return { min: null, max: null, currency };

      let min = prices[0], max = prices[0];
      for (const n of prices) {
        if (n < min) min = n;
        if (n > max) max = n;
      }
      return { min, max, currency };
    }

    function bestPriceLabel(p) {
      // Prefer computed min/max from variants
      const pr = priceRangeFromProduct(p);
      if (pr.min != null && pr.max != null) {
        if (Math.abs(pr.max - pr.min) < 0.000001) return formatMoney(pr.min, pr.currency);
        return `${formatMoney(pr.min, pr.currency)} – ${formatMoney(pr.max, pr.currency)}`;
      }

      // Fallback to whatever string fields you might have had before
      const pl = p?.priceLabel || p?.price_range || p?.priceRange || p?.price || "";
      return pl ? String(pl) : "";
    }
    // ---- /NEW ----

    function render(products) {
      grid.innerHTML = "";

      const shown = products
        .filter(p => String(p?.["product-type"] || p?.productType || "").toLowerCase() === "physical")
        .map(p => ({
          id: String(p?.id || ""),
          title: String(p?.name ?? p?.title ?? "Item"),
          desc: String(p?.description ?? p?.desc ?? ""),
          category: String(p?.category ?? categoryGuessFromTags(p)),
          image: bestImage(p),
          priceLabel: bestPriceLabel(p), // now min–max
          varients: Array.isArray(p?.varients) ? p.varients : null
        }))
        .filter(p => p.id)
        .filter(passesFilters);

      if (!shown.length) {
        grid.innerHTML = `<div class="empty"><strong>No results.</strong><div style="margin-top:8px;color:rgba(255,255,255,.68)">Try a different search or category.</div></div>`;
        return;
      }

      for (const p of shown) {
        const stock = stockFromVariants(p);
        const href = productHref(p.id);

        const card = document.createElement("a");
        card.className = "card";
        card.href = href;
        card.setAttribute("aria-label", `View ${p.title}`);

        const img = p.image ? `<img src="${esc(p.image)}" alt="${esc(p.title)}">` : "";

        card.innerHTML = `
          <div class="thumb">
            ${img}
            <div class="badges">
              <div class="badge stock-${esc(stock.key)}">
                <span class="dot" aria-hidden="true"></span>
                ${esc(stock.label)}
              </div>
              <div class="badge cat">${esc(p.category)}</div>
            </div>
          </div>

          <div class="info">
            <h3 class="title">${esc(p.title)}</h3>
            <p class="desc">${esc(p.desc)}</p>
            <div class="meta">
              <div class="price">${esc(p.priceLabel)}</div>
                        <div class="actions">
            <span class="btn primary" role="button" tabindex="-1">
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V9l-6-6zm0 2.5L18.5 10H14V5.5zM8 13h8v2H8v-2zm0 4h8v2H8v-2z"/></svg>
              View product
            </span>
          </div>
            </div>
          </div>
        `;

        grid.appendChild(card);
      }
    }

    (async () => {
      try {
        const products = await loadCatalog();
        const onChange = () => render(products);

        searchEl.addEventListener("input", onChange);
        filterEl.addEventListener("change", onChange);

        render(products);

        const y = new Date().getFullYear();
        document.getElementById("footer-text").innerHTML =
          `&copy; 2019-${y} Unlim8ted Studio Productions. All rights reserved.`;
      } catch (e) {
        console.error(e);
        grid.innerHTML = `<div class="empty"><strong>Could not load products.</strong><div style="margin-top:8px;color:rgba(255,255,255,.68)">Please refresh and try again.</div></div>`;
      }
    })();
