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

    // https://assets.unlim8ted.com/data/products.json cache/index
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

      .brand-logo{
  width:26px;
  height:26px;
  display:block;
  flex:0 0 auto;
  filter: drop-shadow(0 0 14px rgba(184,107,255,.25));
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
        }/* Keep your existing positioning */
.dropdown-content{
  top: calc(100% + 8px);
}

/* Add an invisible hover bridge filling the gap */
.dropdown::after{
  content:"";
  position:absolute;
  left: 0;
  right: 0;
  top: 100%;
  height: 12px;          /* >= your gap (8px) */
  background: transparent;
}

/* Also keep it open when hovering the menu itself */
.dropdown:hover .dropdown-content,
.dropdown:focus-within .dropdown-content{
  display:block;
}

      </style>

      <nav class="navbar" role="navigation" aria-label="Primary">
        <div class="navbar-header">
       <a class="brand" href="${base}/" aria-label="Unlim8ted Home">
  <svg class="brand-logo" viewBox="0 0 960 720" fill="none" stroke="none" stroke-linecap="square" stroke-miterlimit="10" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
    <clipPath id="p.0">
      <path d="m0 0l960 0l0 720l-960 0l0 -720z" clip-rule="nonzero"></path>
    </clipPath>
    <g clip-path="url(#p.0)">
      <path fill="#000000" fill-opacity="0.0" d="m0 0l960 0l0 720l-960 0z" fill-rule="evenodd"></path>

      <path fill="#f22632" d="m502.70752 74.00787l1.7294617 1.9808502l-171.06903 153.4367l-63.916504 -54.968704l113.68451 -99.98256z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.1994750656167978" stroke-linejoin="round" stroke-linecap="butt" d="m502.70752 74.00787l1.7294617 1.9808502l-171.06903 153.4367l-63.916504 -54.968704l113.68451 -99.98256z" fill-rule="evenodd"></path>

      <path fill="#febd29" d="m613.6493 175.20947l-61.49231 54.221115l-109.47186 -96.60121l62.181396 -56.327393z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.1994750656167978" stroke-linejoin="round" stroke-linecap="butt" d="m613.6493 175.20947l-61.49231 54.221115l-109.47186 -96.60121l62.181396 -56.327393z" fill-rule="evenodd"></path>

      <path fill="#2a47aa" d="m519.24677 341.0839l96.66144 86.29483l-57.24939 49.61026l-95.90024 -85.46564z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m519.24677 341.0839l96.66144 86.29483l-57.24939 49.61026l-95.90024 -85.46564z" fill-rule="evenodd"></path>

      <path fill="#69ba40" d="m613.6493 176.00478l0.07550049 108.24129l-109.4718 96.48282l-60.699524 -54.47946z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.1994750656167978" stroke-linejoin="round" stroke-linecap="butt" d="m613.6493 176.00478l0.07550049 108.24129l-109.4718 96.48282l-60.699524 -54.47946z" fill-rule="evenodd"></path>

      <path fill="#ef3da7" d="m615.6588 427.50406l-0.37530518 111.92465l-45.398926 40.016296l-129.62732 1.4278564z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m615.6588 427.50406l-0.37530518 111.92465l-45.398926 40.016296l-129.62732 1.4278564z" fill-rule="evenodd"></path>

      <path fill="#2a46aa" d="m569.32025 580.18134l-66.8504 58.009827l-62.299194 -56.554993z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m569.32025 580.18134l-66.8504 58.009827l-62.299194 -56.554993z" fill-rule="evenodd"></path>

      <path fill="#211f83" d="m502.084 639.04425l-127.03937 -0.89801025l64.76376 -56.65381z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m502.084 639.04425l-127.03937 -0.89801025l64.76376 -56.65381z" fill-rule="evenodd"></path>

      <path fill="#8031a6" d="m373.72177 638.9666l-112.66403 -100.28021l-0.43569946 -113.576996l178.93701 155.87729z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m373.72177 638.9666l-112.66403 -100.28021l-0.43569946 -113.576996l178.93701 155.87729z" fill-rule="evenodd"></path>

      <path fill="#fea226" d="m320.99475 477.1177l95.50131 -82.75659l-59.275604 -53.1904l-96.375305 83.741425z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m320.99475 477.1177l95.50131 -82.75659l-59.275604 -53.1904l-96.375305 83.741425z" fill-rule="evenodd"></path>

      <path fill="#2746ab" d="m359.01837 341.86877l55.989502 1.0452881l-26.641876 26.007477z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m359.01837 341.86877l55.989502 1.0452881l-26.641876 26.007477z" fill-rule="evenodd"></path>

      <path fill="#0e2767" d="m415.90817 342.4668l50.362183 45.123108l-3.8110046 3.918396l2.312317 2.2810059l-48.624664 -0.35620117l-27.776886 -25.162842z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m415.90817 342.4668l50.362183 45.123108l-3.8110046 3.918396l2.312317 2.2810059l-48.624664 -0.35620117l-27.776886 -25.162842z" fill-rule="evenodd"></path>

      <path fill="#176030" d="m415.00787 349.85483l28.407196 -25.051636l61.01953 54.375183l-27.981598 25.037415z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m415.00787 349.85483l28.407196 -25.051636l61.01953 54.375183l-27.981598 25.037415z" fill-rule="evenodd"></path>

      <path fill="#f4388c" d="m414.09113 351.08453l-68.079315 -0.59664917l-76.18378 -67.51694l-0.3765869 -108.62448l173.80118 150.51797z" fill-rule="evenodd"></path>
      <path stroke="#000000" stroke-width="1.1994750656167978" stroke-linejoin="round" stroke-linecap="butt" d="m414.09113 351.08453l-68.079315 -0.59664917l-76.18378 -67.51694l-0.3765869 -108.62448l173.80118 150.51797z" fill-rule="evenodd"></path>
    </g>
  </svg>

  <span>Unlim8ted</span>
</a>


          <button class="navbar-toggle" id="toggleBtn" aria-label="Toggle menu" aria-expanded="false">☰</button>

          <ul id="links" role="menubar">
            <li><a href="${base}/" role="menuitem">Home</a></li>
            <li><a href="${base}/products" role="menuitem">Products</a></li>
            <li><a href="${base}/contact" role="menuitem">Contact</a></li>
            <li><a href="${base}/about" role="menuitem">About</a></li>

            <li class="dropdown" id="moreDropdown">
              <a href="javascript:void(0)" id="moreBtn" role="menuitem" aria-haspopup="true" aria-expanded="false">
                More <span aria-hidden="true">▾</span>
              </a>
              <div class="dropdown-content" id="moreMenu" role="menu" aria-label="More links">
                <a href="${base}/portfolio" role="menuitem">Portfolio</a>
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

    // Preload https://assets.unlim8ted.com/data/products.json (cart display will re-render when ready)
    this.loadProducts().then(() => {
      this.renderCartMenu();
    }).catch(() => { });

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
      } catch { }
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

  // ===== https://assets.unlim8ted.com/data/products.json =====
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
        const r = await fetch("https://assets.unlim8ted.com/data/products.json", { cache: "no-store" });
        if (!r.ok) throw new Error("Failed to load https://assets.unlim8ted.com/data/products.json");
        const data = await r.json();
        const products = Array.isArray(data) ? data : (data?.products || []);
        this.indexProducts(products);
        this._productsLoaded = true;
      } catch (e) {
        console.warn("Navbar https://assets.unlim8ted.com/data/products.json load failed:", e);
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

  // --- Price resolution (cart item wins, then variant, then product) ---
  const parsedItem = this.parsePriceAny(it.price);
  const parsedVar  = this.parsePriceAny(v?.price);
  const parsedProd = this.parsePriceAny(p?.price);

  // currency preference: item -> variant -> product -> USD
  const currency =
    (parsedItem?.currency) ||
    (v?.currency) ||
    (parsedVar?.currency) ||
    (p?.currency) ||
    (parsedProd?.currency) ||
    "USD";

  // pick best source
  const picked = parsedItem || parsedVar || parsedProd;

  // Sometimes your https://assets.unlim8ted.com/data/products.json uses dollars (major units) like 23.0
  // Sometimes Square returns cents (minor units) like 2300
  // Heuristic: if picked is from objects (isMinor=true) treat as cents; otherwise treat as dollars.
  const priceAmount = picked ? picked.amount : null;
  const priceIsMinor = picked ? !!picked.isMinor : false;

  let image = this.safeUrl(it.image || it.imageUrl || "");
  if (!image && Array.isArray(v?.images) && v.images.length) image = this.safeUrl(v.images[0]);
  if (!image) image = this.safeUrl(v?.image || ""); // <- add support for v.image (your json uses image)
  if (!image) image = this.safeUrl(p?.image || "");

  return {
    title,
    variantLabel,
    priceAmount,
    priceCurrency: currency,
    priceIsMinor,
    image
  };
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
      this.loadProducts().then(() => this.renderCartMenu()).catch(() => { });
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
        this.loadProducts().then(() => this.renderCartMenu()).catch(() => { });
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
parsePriceAny(val) {
  // Handles: 23, "23", "$23.00", {amount: 2300}, {amount_money:{amount:2300,currency:"USD"}}
  if (val == null) return null;

  // Square-like money objects
  if (typeof val === "object") {
    if (val.amount_money && typeof val.amount_money.amount === "number") {
      return { amount: val.amount_money.amount, currency: val.amount_money.currency || "USD", isMinor: true };
    }
    if (typeof val.amount === "number") {
      return { amount: val.amount, currency: val.currency || "USD", isMinor: true };
    }
    // sometimes { money: { amount, currency } }
    if (val.money && typeof val.money.amount === "number") {
      return { amount: val.money.amount, currency: val.money.currency || "USD", isMinor: true };
    }
  }

  // Numeric
  if (typeof val === "number" && Number.isFinite(val)) {
    return { amount: val, currency: "USD", isMinor: false };
  }

  // String like "$23.00" or "23.00"
  if (typeof val === "string") {
    const s = val.trim();
    if (!s) return null;
    const n = Number(s.replace(/[^0-9.\-]/g, ""));
    if (Number.isFinite(n)) return { amount: n, currency: "USD", isMinor: false };
  }

  return null;
}

formatMoney(amount, currency = "USD", isMinor = false) {
  // If isMinor, amount is cents (USD minor units). Convert to major units.
  let major = Number(amount);
  if (!Number.isFinite(major)) return "";
  if (isMinor) major = major / 100;

  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
      maximumFractionDigits: 2,
    }).format(major);
  } catch {
    // fallback if Intl currency fails
    return `$${major.toFixed(2)}`;
  }
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
        const subRight =
  (r.priceAmount != null)
    ? this.escapeHtml(this.formatMoney(r.priceAmount, r.priceCurrency, r.priceIsMinor))
    : "";


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
