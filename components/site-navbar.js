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
  }

  connectedCallback() {
    const base = this.getAttribute("base") || "";
    const cartHref = this.getAttribute("cart-href") || `${base}/cart`;
    const signInHref = this.getAttribute("signin-href") || "https://unlim8ted.com/sign-in";
    const profileHref = this.getAttribute("profile-href") || "https://unlim8ted.com/profile";

    // Firebase config (same as your sign-in page)
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
        :host { display:block; }

        /* Navbar shell */
        .navbar{
          position: fixed;
          top:0; left:0; right:0;
          display:flex;
          justify-content:center;
          background-color:#333333d0;
          z-index:3;
        }

        /* This container is the key: centered links, right icons, without shifting */
        .navbar-header{
          width:100%;
          max-width:1100px;
          position:relative;
          display:flex;
          justify-content:center; /* center nav list */
          align-items:center;
          height: 52px;
        }

        .navbar-toggle{
          display:none;
          background:#333;
          color:#fff;
          padding:10px 14px;
          border:none;
          cursor:pointer;
          font-size:18px;
          position:absolute;
          top:6px;
          left:10px;
          border-radius:8px;
        }

        /* Center links */
        ul#links{
          list-style:none;
          padding:0;
          margin:0;
          display:flex;
          justify-content:center;
          align-items:center;
          gap:0;
        }

        li a{
          display:block;
          color:#fff;
          padding:14px 18px;
          text-decoration:none;
          text-align:center;
          white-space:nowrap;
        }
        li a:hover{
          background:#15131f;
          color:rgb(51,207,103);
        }

        /* More dropdown (hover) */
        .dropdown{ position:relative; }
        .dropdown-content{
          display:none;
          position:absolute;
          left:0;
          background:#251f1f;
          box-shadow:0 8px 16px rgba(0,0,0,.2);
          min-width:180px;
          z-index:5;
        }
        .dropdown-content a{
          color:#fff;
          padding:12px 16px;
          display:block;
          text-align:left;
        }
        .dropdown-content a:hover{ background:#2d2849; }
        .dropdown:hover .dropdown-content{ display:block; }

        /* Right-side icons (ABSOLUTE so they don't affect centering) */
        .right-icons{
          position:absolute;
          right:10px;
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
          width:42px;
          height:42px;
          border-radius:12px;
          border:none;
          background:transparent;
          cursor:pointer;
          color:white;
          padding:0;
        }
        .icon-btn:hover{
          background:#15131f;
          color:rgb(51,207,103);
        }

        .icon{
          width:22px;
          height:22px;
          display:block;
        }

        /* Cart badge */
        .cart-wrap{
          position:relative;
        }
        .badge{
          position:absolute;
          top:6px;
          right:6px;
          min-width:18px;
          height:18px;
          padding:0 5px;
          border-radius:999px;
          background:rgba(0,255,153,.95);
          color:#15131f;
          font-size:12px;
          font-weight:700;
          display:flex;
          align-items:center;
          justify-content:center;
          line-height:18px;
          transform: translate(20%, -20%);
          box-shadow: 0 2px 8px rgba(0,0,0,.35);
        }
        .badge.hidden{ display:none; }

        /* Menus (cart + profile) */
        .menu{
          display:none;
          position:absolute;
          right:0;
          top:48px;
          background:#251f1f;
          box-shadow:0 8px 16px rgba(0,0,0,.2);
          min-width:260px;
          border-radius:12px;
          overflow:hidden;
          z-index:10;
        }
        .menu.open{ display:block; }

        .menu .meta{
          padding:10px 14px;
          font-size:12px;
          color:rgba(255,255,255,.75);
          border-bottom:1px solid rgba(255,255,255,.08);
        }
        .menu a, .menu button{
          width:100%;
          border:none;
          background:none;
          color:#fff;
          padding:12px 14px;
          text-decoration:none;
          display:block;
          text-align:left;
          cursor:pointer;
          font-size:14px;
        }
        .menu a:hover, .menu button:hover{ background:#2d2849; }

        /* Cart item list */
        .cart-items{
          max-height: 320px;
          overflow:auto;
        }
        .cart-item{
          display:flex;
          gap:10px;
          padding:10px 14px;
          border-bottom:1px solid rgba(255,255,255,.06);
        }
        .cart-item:last-child{ border-bottom:none; }

        .thumb{
          width:42px; height:42px;
          border-radius:10px;
          background:rgba(255,255,255,.08);
          flex:0 0 auto;
          overflow:hidden;
          display:flex;
          align-items:center;
          justify-content:center;
          color:rgba(255,255,255,.65);
          font-size:12px;
        }
        .thumb img{ width:100%; height:100%; object-fit:cover; display:block; }

        .ci-main{ flex:1; min-width:0; }
        .ci-title{
          font-size:13px;
          color:#fff;
          white-space:nowrap;
          overflow:hidden;
          text-overflow:ellipsis;
        }
        .ci-sub{
          margin-top:4px;
          font-size:12px;
          color:rgba(255,255,255,.7);
          display:flex;
          justify-content:space-between;
          gap:10px;
        }

        .cart-actions{
          border-top:1px solid rgba(255,255,255,.08);
          display:flex;
        }
        .cart-actions a{
          flex:1;
          text-align:center;
          padding:12px 10px;
        }

        @media (max-width:768px){
          .navbar-header{
            justify-content:flex-start;
            height:auto;
            align-items:flex-start;
          }
          .navbar-toggle{ display:block; }

          ul#links{
            flex-direction:column;
            align-items:stretch;
            display:none;
            background-color:#333333f2;
            padding-top:48px;
            width:100%;
          }
          ul#links.show{ display:flex; }

          .right-icons{
            position:fixed;
            right:10px;
            top:6px;
            transform:none;
            z-index:11;
          }

          .menu{
            right:10px;
            top:52px;
            min-width: 92vw;
            max-width: 360px;
          }

          .dropdown-content{ position:relative; box-shadow:none; }
        }
      </style>

      <nav class="navbar">
        <div class="navbar-header">
          <button class="navbar-toggle" id="toggleBtn">☰</button>

          <ul id="links">
            <li><a href="${base}/">Home</a></li>
            <li><a href="${base}/products">Products</a></li>
            <li><a href="${base}/help">Help</a></li>
            <li><a href="${base}/about">About</a></li>

            <li class="dropdown">
              <a href="javascript:void(0)">More</a>
              <div class="dropdown-content">
                <a href="${base}/portfolio">Portfolio</a>
                <a href="${base}/contact">Contact</a>
                <a href="${base}/blog">Blog</a>
                <a href="${base}/live-chat">Live Chat</a>
                <a href="${base}/live-game">Live Chat and Game</a>
                <a href="${base}/puzzle-squares">Puzzle Squares</a>
              </div>
            </li>

            <li><a href="${base}/old"><button style="border:none;border-radius:10px;padding:8px 12px;cursor:pointer;">Old site</button></a></li>
          </ul>

          <!-- Right icons (don’t affect centering) -->
          <div class="right-icons">
            <!-- Cart -->
            <div class="cart-wrap">
              <button class="icon-btn" id="cartBtn" type="button" title="Cart" aria-label="Cart">
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

            <!-- Profile -->
            <div class="profile-wrap" style="position:relative;">
              <button class="icon-btn" id="profileBtn" type="button" title="Account" aria-label="Account">
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

    // Mobile toggle
    this.shadowRoot.getElementById("toggleBtn").addEventListener("click", () => {
      this.shadowRoot.getElementById("links").classList.toggle("show");
    });

    // Active link outline
    const path = window.location.pathname.replace(/\/$/, "");
    this.shadowRoot.querySelectorAll('a[href]').forEach(a => {
      try {
        const href = new URL(a.getAttribute("href"), window.location.origin).pathname.replace(/\/$/, "");
        if (href === path) a.style.outline = "2px solid rgba(0,255,153,.5)";
      } catch {}
    });

    const cartBtn = this.shadowRoot.getElementById("cartBtn");
    const cartMenu = this.shadowRoot.getElementById("cartMenu");
    const profileBtn = this.shadowRoot.getElementById("profileBtn");
    const profileMenu = this.shadowRoot.getElementById("profileMenu");

    const closeMenus = () => {
      cartMenu.classList.remove("open");
      profileMenu.classList.remove("open");
    };

    // Cart dropdown toggle
    cartBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      profileMenu.classList.remove("open");
      cartMenu.classList.toggle("open");
    });

    // Profile dropdown toggle
    profileBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      cartMenu.classList.remove("open");
      profileMenu.classList.toggle("open");
    });

    // Click outside closes
    window.addEventListener("click", closeMenus);
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

    // Logged-out cart fallback initially
    this.cartItems = this.getLocalCartItems();
    this.renderCartMenu();
    this.updateCartBadge(this.countCart(this.cartItems));
  }

  disconnectedCallback() {
    if (this.unsubCart) this.unsubCart();
    this.unsubCart = null;
  }

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

  bindCartListener({ db, user }) {
    if (this.unsubCart) this.unsubCart();
    this.unsubCart = null;

    if (!user) {
      this.cartItems = this.getLocalCartItems();
      this.renderCartMenu();
      this.updateCartBadge(this.countCart(this.cartItems));
      return;
    }

    // Default schema: users/{uid}/cartItems (one doc per item)
    const itemsRef = collection(db, "users", user.uid, "cartItems");
    const q = query(itemsRef);

    this.unsubCart = onSnapshot(q, (snap) => {
      const items = [];
      snap.forEach((d) => {
        const data = d.data() || {};
        items.push({
          id: d.id,
          title: data.title || data.name || data.productName || d.id,
          qty: Number.isFinite(data.qty) ? Number(data.qty) : 1,
          price: data.price ?? null,
          image: data.image ?? data.imageUrl ?? null,
        });
      });

      this.cartItems = items;
      this.renderCartMenu();
      this.updateCartBadge(this.countCart(items));
    }, (err) => {
      console.error("Cart listener error:", err);
      this.cartItems = this.getLocalCartItems();
      this.renderCartMenu();
      this.updateCartBadge(this.countCart(this.cartItems));
    });
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
        <div style="padding:12px 14px; color:rgba(255,255,255,.75); font-size:13px;">
          Your cart is empty.
        </div>
      `;
      return;
    }

    const show = items.slice(0, 5); // show first 5 items
    list.innerHTML = show.map(it => `
      <div class="cart-item">
        <div class="thumb">
          ${it.image ? `<img src="${this.escapeHtml(it.image)}" alt="">` : "Item"}
        </div>
        <div class="ci-main">
          <div class="ci-title">${this.escapeHtml(it.title)}</div>
          <div class="ci-sub">
            <span>Qty: ${Number(it.qty) || 1}</span>
            ${it.price != null ? `<span>$${this.escapeHtml(String(it.price))}</span>` : `<span></span>`}
          </div>
        </div>
      </div>
    `).join("");

    if (items.length > show.length) {
      list.innerHTML += `
        <div style="padding:10px 14px; font-size:12px; color:rgba(255,255,255,.7); border-top:1px solid rgba(255,255,255,.06);">
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
        id: it.id || `local-${idx}`,
        title: it.title || it.name || it.productName || `Item ${idx + 1}`,
        qty: Number(it.qty) || 1,
        price: it.price ?? null,
        image: it.image ?? it.imageUrl ?? null,
      }));
    } catch {
      return [];
    }
  }

  escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
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
