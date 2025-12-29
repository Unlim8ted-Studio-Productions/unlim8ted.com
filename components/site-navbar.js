import { initializeApp, getApps } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-app.js";
import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";
import { getFirestore, collection, onSnapshot, query } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";

class SiteNavbar extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });

    this.unsubCart = null;
    this.currentUser = null;
    this.cartItems = [];

    this._resizeObs = null;
    this._onWinResize = null;

    this._isMobile = false;
    this._focusTrapHandler = null;

    // products.json cache/index
    this._productsLoaded = false;
    this._productsLoading = null;
    this._productById = new Map();
    this._variantByKey = new Map(); // `${productId}::${variantId}` -> variant
  }

  connectedCallback() {
    const base = this.getAttribute("base") || "";
    const cartHref = this.getAttribute("cart-href") || `${base}/cart`;
    const signInHref = this.getAttribute("signin-href") || "https://unlim8ted.com/sign-in";
    const profileHref = this.getAttribute("profile-href") || "https://unlim8ted.com/profile";

    const firebaseConfig = {
      apiKey: "AIzaSyC8rw6kaFhJ2taebKRKKEA7iLqBvak_Dbc",
      authDomain: "unlim8ted-db.firebaseapp.com",
      projectId: "unlim8ted-db",
      storageBucket: "unlim8ted-db.appspot.com",
      messagingSenderId: "1059428499872",
      appId: "1:1059428499872:web:855308683718237de6e4c5",
    };

    const app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
    const auth = getAuth(app);
    const db = getFirestore(app);

    this.shadowRoot.innerHTML = `
      <style>
        :host{
          display:block;
          --nav-h: 56px;
          --maxw: 1180px;

          /* “Modern” palette hooks (safe defaults) */
          --nav-bg: rgba(10, 8, 18, .55);
          --nav-bg2: rgba(10, 8, 18, .30);
          --nav-stroke: rgba(255,255,255,.10);
          --nav-ink: rgba(233,231,255,.92);
          --nav-muted: rgba(233,231,255,.72);
          --nav-accent: rgba(184,107,255,.95);
          --nav-accent2: rgba(103,213,255,.85);
          --nav-hover: rgba(255,255,255,.07);
          --nav-radius: 16px;
        }

        .navbar{
          position: fixed;
          top:0; left:0; right:0;
          z-index: 9999;
          height: var(--nav-h);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          background: linear-gradient(180deg, var(--nav-bg), var(--nav-bg2));
          border-bottom: 1px solid var(--nav-stroke);
        }

        .navbar-header{
          width:100%;
          max-width: var(--maxw);
          margin: 0 auto;
          position:relative;
          height: var(--nav-h);
          display:flex;
          align-items:center;
          justify-content:center; /* keep links centered */
          padding: 0 12px;
        }

        .brand{
          position:absolute;
          left: 12px;
          top: 50%;
          transform: translateY(-50%);
          display:flex;
          align-items:center;
          gap:10px;
          text-decoration:none;
          color: var(--nav-ink);
          font-weight: 900;
          letter-spacing: .04em;
          user-select:none;
        }

        .brand-dot{
          width:10px;height:10px;border-radius:50%;
          background: linear-gradient(135deg, var(--nav-accent), var(--nav-accent2));
          box-shadow: 0 0 16px rgba(184,107,255,.35);
        }

        .navbar-toggle{
          display:none;
          position:absolute;
          left: 12px;
          top: 8px;
          padding: 10px 12px;
          border-radius: 14px;
          border: 1px solid var(--nav-stroke);
          background: rgba(255,255,255,.06);
          color: var(--nav-ink);
          cursor:pointer;
          font-size:18px;
          line-height: 1;
        }
        .navbar-toggle:hover{ background: var(--nav-hover); }

        /* Center links */
        ul#links{
          list-style:none;
          padding:0;
          margin:0;
          display:flex;
          justify-content:center;
          align-items:center;
          gap: 6px;
        }

        li{ position:relative; }

        li a{
          display:inline-flex;
          align-items:center;
          gap:8px;
          color: var(--nav-ink);
          padding: 10px 12px;
          text-decoration:none;
          border-radius: 14px;
          border: 1px solid transparent;
          white-space:nowrap;
          font-weight: 700;
          transition: transform .18s ease, background .18s ease, border-color .18s ease;
        }
        li a:hover{
          background: var(--nav-hover);
          border-color: rgba(180,130,255,.18);
          transform: translateY(-1px);
        }

        .active{
          border-color: rgba(184,107,255,.35) !important;
          background: rgba(184,107,255,.10) !important;
        }

        /* More dropdown */
        .dropdown{ position:relative; }
        .dropdown-content{
          display:none;
          position:absolute;
          left: 0;
          top: calc(100% + 8px);
          min-width: 210px;
          padding: 8px;
          border-radius: 18px;
          background: rgba(10,8,18,.92);
          border: 1px solid var(--nav-stroke);
          box-shadow: 0 18px 60px rgba(0,0,0,.55);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          z-index: 10001;
        }
        .dropdown-content a{
          display:flex;
          padding: 10px 12px;
          border-radius: 14px;
          border: 1px solid transparent;
        }
        .dropdown-content a:hover{
          background: var(--nav-hover);
          border-color: rgba(255,255,255,.10);
          transform:none;
        }

        /* hover on desktop, click on mobile */
        .dropdown:hover .dropdown-content{ display:block; }
        .dropdown.open .dropdown-content{ display:block; }

        /* Right icons */
        .right-icons{
          position:absolute;
          right:12px;
          top:50%;
          transform: translateY(-50%);
          display:flex;
          align-items:center;
          gap:8px;
        }

        .icon-btn{
          display:flex;
          align-items:center;
          justify-content:center;
          width:44px;
          height:44px;
          border-radius: 14px;
          border: 1px solid var(--nav-stroke);
          background: rgba(255,255,255,.05);
          cursor:pointer;
          color: var(--nav-ink);
          padding:0;
          transition: transform .18s ease, background .18s ease, border-color .18s ease;
        }
        .icon-btn:hover{
          transform: translateY(-1px);
          background: var(--nav-hover);
          border-color: rgba(180,130,255,.18);
        }

        .icon{ width:22px; height:22px; display:block; }

        .cart-wrap{ position:relative; }
        .badge{
          position:absolute;
          top: 6px;
          right: 6px;
          min-width: 18px;
          height: 18px;
          padding: 0 6px;
          border-radius: 999px;
          background: linear-gradient(135deg, var(--nav-accent), var(--nav-accent2));
          color: rgba(10,8,18,.95);
          font-size: 12px;
          font-weight: 900;
          display:flex;
          align-items:center;
          justify-content:center;
          line-height: 18px;
          box-shadow: 0 10px 22px rgba(0,0,0,.45);
          transform: translate(20%, -20%);
        }
        .badge.hidden{ display:none; }

        .menu{
          display:none;
          position:absolute;
          right:0;
          top: 54px;
          min-width: 280px;
          border-radius: 18px;
          overflow:hidden;
          border: 1px solid rgba(255,255,255,.10);
          background: rgba(10,8,18,.92);
          box-shadow: 0 18px 70px rgba(0,0,0,.58);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          z-index: 10002;
        }
        .menu.open{ display:block; }

        .menu .meta{
          padding: 10px 14px;
          font-size: 12px;
          color: var(--nav-muted);
          border-bottom: 1px solid rgba(255,255,255,.08);
          letter-spacing: .08em;
          text-transform: uppercase;
        }

        .menu a, .menu button{
          width:100%;
          border:none;
          background:none;
          color: var(--nav-ink);
          padding: 12px 14px;
          text-decoration:none;
          display:block;
          text-align:left;
          cursor:pointer;
          font-size: 14px;
          font-weight: 700;
        }
        .menu a:hover, .menu button:hover{ background: var(--nav-hover); }

        .cart-items{ max-height: 320px; overflow:auto; }
        .cart-item{
          display:flex;
          gap:10px;
          padding: 10px 14px;
          border-bottom:1px solid rgba(255,255,255,.06);
        }
        .cart-item:last-child{ border-bottom:none; }

        .thumb{
          width:44px; height:44px;
          border-radius: 14px;
          background: rgba(255,255,255,.07);
          overflow:hidden;
          flex:0 0 auto;
          display:flex;
          align-items:center;
          justify-content:center;
          color: var(--nav-muted);
          font-size: 12px;
          border: 1px solid rgba(255,255,255,.08);
        }
        .thumb img{ width:100%; height:100%; object-fit:cover; display:block; }

        .ci-main{ flex:1; min-width:0; }
        .ci-title{
          font-size: 13px;
          color: var(--nav-ink);
          white-space:nowrap;
          overflow:hidden;
          text-overflow:ellipsis;
          font-weight: 800;
        }
        .ci-sub{
          margin-top:4px;
          font-size:12px;
          color: var(--nav-muted);
          display:flex;
          justify-content:space-between;
          gap:10px;
        }

        .cart-actions{
          border-top: 1px solid rgba(255,255,255,.08);
          display:flex;
        }
        .cart-actions a{
          flex:1;
          text-align:center;
          padding: 12px 10px;
          font-weight: 900;
        }

        /* Mobile */
        @media (max-width: 820px){
          :host{ --nav-h: 56px; }

          .brand{ display:none; } /* keeps layout clean on mobile */

          .navbar-header{
            justify-content:flex-start;
            align-items:stretch;
            height:auto;
          }

          .navbar-toggle{ display:block; }

          ul#links{
            display:none;
            flex-direction:column;
            align-items:stretch;
            width: 100%;
            padding: 64px 10px 10px;
            background: rgba(10,8,18,.88);
            border-top: 1px solid rgba(255,255,255,.08);
          }
          ul#links.show{ display:flex; }

          .right-icons{
            position:fixed;
            right:12px;
            top:6px;
            transform:none;
            z-index: 10010;
          }

          .menu{
            right:12px;
            top: 56px;
            min-width: 92vw;
            max-width: 380px;
          }

          /* disable hover dropdown; click toggles .open */
          .dropdown:hover .dropdown-content{ display:none; }
        }

        /* reduced motion */
        @media (prefers-reduced-motion: reduce){
          *{ transition:none !important; animation:none !important; }
        }
      </style>

      <nav class="navbar" role="navigation" aria-label="Primary">
        <div class="navbar-header">
          <a class="brand" href="${base}/" aria-label="Unlim8ted Home">
            <span class="brand-dot" aria-hidden="true"></span>
            <span>Unlim8ted</span>
          </a>

          <button class="navbar-toggle" id="toggleBtn" aria-label="Toggle menu" aria-expanded="false">☰</button>

          <ul id="links" role="menubar">
            <li><a href="${base}/" role="menuitem">Home</a></li>
            <li><a href="${base}/products" role="menuitem">Products</a></li>
            <li><a href="${base}/help" role="menuitem">Help</a></li>
            <li><a href="${base}/about" role="menuitem">About</a></li>

            <li class="dropdown" id="moreDropdown">
              <a href="javascript:void(0)" id="moreBtn" role="menuitem" aria-haspopup="true" aria-expanded="false">
                More <span aria-hidden="true">▾</span>
              </a>
              <div class="dropdown-content" id="moreMenu" role="menu" aria-label="More links">
                <a href="${base}/portfolio" role="menuitem">Portfolio</a>
                <a href="${base}/contact" role="menuitem">Contact</a>
                <a href="${base}/blog" role="menuitem">Blog</a>
              </div>
            </li>

            <li>
              <a href="${base}/old" role="menuitem"
                style="font-weight:900;border:1px solid rgba(255,255,255,.10);background:rgba(255,255,255,.06);">
                Old site
              </a>
            </li>
          </ul>

          <div class="right-icons" aria-label="Account controls">
            <div class="cart-wrap">
              <button class="icon-btn" id="cartBtn" type="button" title="Cart" aria-label="Cart" aria-expanded="false">
                ${this.cartSvg()}
                <span id="cartBadge" class="badge hidden">0</span>
              </button>

              <div class="menu" id="cartMenu" aria-label="Cart menu">
                <div class="meta" id="cartMeta">Cart</div>
                <div class="cart-items" id="cartItems"></div>
                <div class="cart-actions">
                  <a href="${cartHref}">View cart</a>
                </div>
              </div>
            </div>

            <div class="profile-wrap" style="position:relative;">
              <button class="icon-btn" id="profileBtn" type="button" title="Account" aria-label="Account" aria-expanded="false">
                ${this.userSvg()}
              </button>

              <div class="menu" id="profileMenu" role="menu" aria-label="Account menu">
                <div class="meta" id="menuMeta">Not signed in</div>
                <a id="menuPrimary" href="${signInHref}" role="menuitem">Sign In</a>
                <button id="menuSignOut" type="button" role="menuitem" style="display:none;">Sign out</button>
              </div>
            </div>
          </div>
        </div>
      </nav>
    `;

    // ---- spacer below fixed nav ----
    this.syncSpacer();

    // ---- state ----
    this._isMobile = window.matchMedia("(max-width: 820px)").matches;

    // Preload products.json (cart display will re-render when ready)
    this.loadProducts().then(() => {
      this.renderCartMenu();
    }).catch(() => {});

    // Mobile toggle
    const toggleBtn = this.shadowRoot.getElementById("toggleBtn");
    const links = this.shadowRoot.getElementById("links");
    toggleBtn.addEventListener("click", () => {
      const open = links.classList.toggle("show");
      toggleBtn.setAttribute("aria-expanded", open ? "true" : "false");
      this.ensureSpacer();
    });

    // “More” dropdown click on mobile
    const moreDropdown = this.shadowRoot.getElementById("moreDropdown");
    const moreBtn = this.shadowRoot.getElementById("moreBtn");
    moreBtn.addEventListener("click", (e) => {
      if (!window.matchMedia("(max-width: 820px)").matches) return; // desktop hover handles it
      e.preventDefault();
      const open = moreDropdown.classList.toggle("open");
      moreBtn.setAttribute("aria-expanded", open ? "true" : "false");
    });

    // Active link styling
    const path = window.location.pathname.replace(/\/$/, "");
    this.shadowRoot.querySelectorAll("a[href]").forEach((a) => {
      try {
        const href = new URL(a.getAttribute("href"), window.location.origin).pathname.replace(/\/$/, "");
        if (href === path) a.classList.add("active");
      } catch {}
    });

    // Menus
    const cartBtn = this.shadowRoot.getElementById("cartBtn");
    const cartMenu = this.shadowRoot.getElementById("cartMenu");
    const profileBtn = this.shadowRoot.getElementById("profileBtn");
    const profileMenu = this.shadowRoot.getElementById("profileMenu");

    const setExpanded = (btn, isOpen) => btn?.setAttribute("aria-expanded", isOpen ? "true" : "false");

    const closeMenus = () => {
      cartMenu.classList.remove("open");
      profileMenu.classList.remove("open");
      setExpanded(cartBtn, false);
      setExpanded(profileBtn, false);
    };

    // Cart toggle
    cartBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      profileMenu.classList.remove("open");
      setExpanded(profileBtn, false);
      const open = cartMenu.classList.toggle("open");
      setExpanded(cartBtn, open);
    });

    // Profile toggle
    profileBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      cartMenu.classList.remove("open");
      setExpanded(cartBtn, false);
      const open = profileMenu.classList.toggle("open");
      setExpanded(profileBtn, open);
    });

    // Outside click / ESC close
    const onEsc = (e) => {
      if (e.key === "Escape") closeMenus();
    };

    window.addEventListener("click", closeMenus);
    document.addEventListener("keydown", onEsc);
    cartMenu.addEventListener("click", (e) => e.stopPropagation());
    profileMenu.addEventListener("click", (e) => e.stopPropagation());

    // Sign out
    this.shadowRoot.getElementById("menuSignOut").addEventListener("click", async () => {
      try {
        await signOut(auth);
        closeMenus();
      } catch (e) {
        console.error("Sign out failed:", e);
      }
    });

    // Auth state
    onAuthStateChanged(auth, (user) => {
      this.currentUser = user || null;
      this.updateAccountMenu({ user, signInHref, profileHref });
      this.bindCartListener({ db, user });
    });

    // Logged-out cart initial
    this.cartItems = this.getLocalCartItems();
    this.renderCartMenu();
    this.updateCartBadge(this.countCart(this.cartItems));
  }

  disconnectedCallback() {
    if (this.unsubCart) this.unsubCart();
    this.unsubCart = null;

    if (this._resizeObs) this._resizeObs.disconnect();
    this._resizeObs = null;

    if (this._onWinResize) window.removeEventListener("resize", this._onWinResize);
    this._onWinResize = null;
  }

  // ===== products.json =====
  safeUrl(u) {
    const s = String(u || "").trim();
    if (!s) return "";
    if (s.startsWith("https://") || s.startsWith("http://") || s.startsWith("/")) return s;
    return "";
  }

  async loadProducts() {
    if (this._productsLoaded) return;
    if (this._productsLoading) return this._productsLoading;

    this._productsLoading = (async () => {
      try {
        const r = await fetch("https://unlim8ted.com/tools/data/products.json", { cache: "no-store" });
        if (!r.ok) throw new Error("Failed to load products.json");
        const data = await r.json();
        const products = Array.isArray(data) ? data : (data?.products || []);
        this.indexProducts(products);
        this._productsLoaded = true;
      } catch (e) {
        console.warn("Navbar products.json load failed:", e);
      } finally {
        this._productsLoading = null;
      }
    })();

    return this._productsLoading;
  }

  indexProducts(products) {
    this._productById = new Map();
    this._variantByKey = new Map();

    for (const p of (products || [])) {
      const pid = String(p.id ?? p.productId ?? "").trim();
      if (!pid) continue;

      this._productById.set(pid, p);

      // your json uses "varients" (typo); support both
      const vars = Array.isArray(p.varients) ? p.varients : (Array.isArray(p.variants) ? p.variants : []);
      for (const v of vars) {
        const vid = String(v.id ?? v.variantId ?? "").trim();
        if (!vid) continue;
        this._variantByKey.set(`${pid}::${vid}`, v);
      }
    }
  }

  resolveCartDisplay(it) {
    const productId = String(it.productId || "").trim();
    const variantId = String(it.variantId || "").trim();

    const p = productId ? (this._productById.get(productId) || null) : null;
    const v = (productId && variantId) ? (this._variantByKey.get(`${productId}::${variantId}`) || null) : null;

    const title =
      String(it.title || it.name || "").trim() ||
      String(p?.name || p?.title || productId || it.id || "Item").trim();

    const variantLabel =
      String(it.variantLabel || "").trim() ||
      String(v?.name || "").trim();

    const price =
      (Number.isFinite(Number(it.price)) ? Number(it.price) : null) ??
      (v && Number.isFinite(Number(v.price)) ? Number(v.price) : null) ??
      null;

    let image = this.safeUrl(it.image || it.imageUrl || "");
    if (!image && Array.isArray(v?.images) && v.images.length) {
      image = this.safeUrl(v.images[0]);
    }
    if (!image) image = this.safeUrl(p?.image || "");

    return { title, variantLabel, price, image };
  }

  // ----- spacer logic -----
  ensureSpacer() {
    let spacer = this.nextElementSibling;
    if (!spacer || !spacer.classList.contains("site-navbar-spacer")) {
      spacer = document.createElement("div");
      spacer.className = "site-navbar-spacer";
      this.insertAdjacentElement("afterend", spacer);
    }
    spacer.style.height = this.getNavbarHeight() + "px";
  }

  getNavbarHeight() {
    const nav = this.shadowRoot?.querySelector(".navbar");
    if (!nav) return 56;
    const rect = nav.getBoundingClientRect();
    return Math.max(44, Math.round(rect.height || 56));
  }

  syncSpacer() {
    this.ensureSpacer();

    if (this._resizeObs) this._resizeObs.disconnect();
    const nav = this.shadowRoot?.querySelector(".navbar");
    if (nav && "ResizeObserver" in window) {
      this._resizeObs = new ResizeObserver(() => this.ensureSpacer());
      this._resizeObs.observe(nav);
    }

    this._onWinResize = () => this.ensureSpacer();
    window.addEventListener("resize", this._onWinResize, { passive: true });
  }

  // ----- account menu -----
  updateAccountMenu({ user, signInHref, profileHref }) {
    const meta = this.shadowRoot.getElementById("menuMeta");
    const primary = this.shadowRoot.getElementById("menuPrimary");
    const signOutBtn = this.shadowRoot.getElementById("menuSignOut");

    if (!user) {
      meta.textContent = "Not signed in";
      primary.textContent = "Sign In";
      primary.href = signInHref;
      signOutBtn.style.display = "none";
      return;
    }

    meta.textContent = user.email || "Signed in";
    primary.textContent = "Profile";
    primary.href = profileHref;
    signOutBtn.style.display = "block";
  }

  // ----- cart listener -----
  bindCartListener({ db, user }) {
    if (this.unsubCart) this.unsubCart();
    this.unsubCart = null;

    if (!user) {
      this.cartItems = this.getLocalCartItems();
      this.loadProducts().then(() => this.renderCartMenu()).catch(() => {});
      this.renderCartMenu();
      this.updateCartBadge(this.countCart(this.cartItems));
      return;
    }

    const itemsRef = collection(db, "users", user.uid, "cartItems");
    const q = query(itemsRef);

    this.unsubCart = onSnapshot(
      q,
      async (snap) => {
        const items = [];
        snap.forEach((d) => {
          const data = d.data() || {};
          items.push({
            id: d.id,
            productId: data.productId ?? null,
            variantId: data.variantId ?? null,
            qty: Number.isFinite(data.qty) ? Number(data.qty) : 1,
          });
        });

        this.cartItems = items;
        await this.loadProducts();
        this.renderCartMenu();
        this.updateCartBadge(this.countCart(items));
      },
      (err) => {
        console.error("Cart listener error:", err);
        this.cartItems = this.getLocalCartItems();
        this.loadProducts().then(() => this.renderCartMenu()).catch(() => {});
        this.renderCartMenu();
        this.updateCartBadge(this.countCart(this.cartItems));
      }
    );
  }

  countCart(items) {
    return (items || []).reduce((sum, it) => sum + (Number(it.qty) || 1), 0);
  }

  updateCartBadge(count) {
    const badge = this.shadowRoot.getElementById("cartBadge");
    if (!badge) return;
    const n = Math.max(0, Number(count) || 0);
    badge.textContent = n > 99 ? "99+" : String(n);
    badge.classList.toggle("hidden", n === 0);
  }

  renderCartMenu() {
    const meta = this.shadowRoot.getElementById("cartMeta");
    const list = this.shadowRoot.getElementById("cartItems");
    if (!meta || !list) return;

    const items = this.cartItems || [];
    const totalCount = this.countCart(items);

    meta.textContent = totalCount ? `Cart (${totalCount})` : "Cart (empty)";

    if (!items.length) {
      list.innerHTML = `
        <div style="padding:12px 14px; color:rgba(233,231,255,.72); font-size:13px;">
          Your cart is empty.
        </div>
      `;
      return;
    }

    const show = items.slice(0, 5);
    list.innerHTML = show
      .map((it) => {
        const r = this.resolveCartDisplay(it);
        const qty = Number(it.qty) || 1;

        const subLeft = r.variantLabel ? `Qty: ${qty} • ${this.escapeHtml(r.variantLabel)}` : `Qty: ${qty}`;
        const subRight = (r.price != null) ? `$${this.escapeHtml(String(r.price))}` : "";

        return `
          <div class="cart-item">
            <div class="thumb">
              ${r.image ? `<img src="${this.escapeHtml(r.image)}" alt="">` : "Item"}
            </div>
            <div class="ci-main">
              <div class="ci-title">${this.escapeHtml(r.title)}</div>
              <div class="ci-sub">
                <span>${subLeft}</span>
                <span>${subRight}</span>
              </div>
            </div>
          </div>
        `;
      })
      .join("");

    if (items.length > show.length) {
      list.innerHTML += `
        <div style="padding:10px 14px; font-size:12px; color:rgba(233,231,255,.72); border-top:1px solid rgba(255,255,255,.06);">
          + ${items.length - show.length} more item(s)
        </div>
      `;
    }
  }

  getLocalCartItems() {
    try {
      const raw = localStorage.getItem("unlim8ted-cart");
      if (!raw) return [];
      const arr = JSON.parse(raw);
      if (!Array.isArray(arr)) return [];

      return arr.map((it, idx) => ({
        id: it.id || it._id || `local-${idx}`,
        productId: it.productId || it.pid || null,
        variantId: it.variantId || it.vid || null,
        qty: Number(it.qty) || 1,

        // legacy optional fields (if present, ok)
        title: it.title || it.name || it.productName || "",
        variantLabel: it.variantLabel || "",
        price: it.price ?? null,
        image: it.image ?? it.imageUrl ?? null,
      }));
    } catch {
      return [];
    }
  }

  escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));
  }

  cartSvg() {
    return `
      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="9" cy="21" r="1"></circle>
        <circle cx="20" cy="21" r="1"></circle>
        <path d="M1 1h4l2.7 13.4a2 2 0 0 0 2 1.6h9.7a2 2 0 0 0 2-1.6L23 6H6"></path>
      </svg>
    `;
  }

  userSvg() {
    return `
      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M20 21a8 8 0 0 0-16 0"></path>
        <circle cx="12" cy="7" r="4"></circle>
      </svg>
    `;
  }
}

customElements.define("site-navbar", SiteNavbar);
