import { getFirebase } from "/components/firebase-init.js";
    import { onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";
    import { collection, onSnapshot, query } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";

    const { auth, db } = getFirebase();


    const cartBtn = document.getElementById("cartBtn");
    const cartMenu = document.getElementById("cartMenu");
    const cartBadge = document.getElementById("cartBadge");
    const cartMeta = document.getElementById("cartMeta");
    const cartItemsEl = document.getElementById("cartItems");

    const profileBtn = document.getElementById("profileBtn");
    const profileMenu = document.getElementById("profileMenu");
    const menuMeta = document.getElementById("menuMeta");
    const menuPrimary = document.getElementById("menuPrimary");
    const menuSignOut = document.getElementById("menuSignOut");

    let unsubCart = null;
    let cartItems = [];

    const closeMenus = () => {
      cartMenu?.classList.remove("open");
      profileMenu?.classList.remove("open");
    };

    cartBtn?.addEventListener("click", (e) => {
      e.stopPropagation();
      profileMenu?.classList.remove("open");
      cartMenu?.classList.toggle("open");
    });

    profileBtn?.addEventListener("click", (e) => {
      e.stopPropagation();
      cartMenu?.classList.remove("open");
      profileMenu?.classList.toggle("open");
    });

    window.addEventListener("click", closeMenus);
    cartMenu?.addEventListener("click", (e) => e.stopPropagation());
    profileMenu?.addEventListener("click", (e) => e.stopPropagation());

    menuSignOut?.addEventListener("click", async () => {
      try {
        await signOut(auth);
        closeMenus();
      } catch (e) {
        console.error("Sign out failed:", e);
      }
    });

    const escapeHtml = (s) =>
      String(s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[c]));

    const countCart = (items) => (items || []).reduce((sum, it) => sum + (Number(it.qty) || 1), 0);

    const updateBadge = (count) => {
      if (!cartBadge) return;
      const n = Math.max(0, Number(count) || 0);
      cartBadge.textContent = n > 99 ? "99+" : String(n);
      cartBadge.classList.toggle("hidden", n === 0);
    };

    const getLocalCartItems = () => {
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
    };

    const renderCartMenu = () => {
      if (!cartMeta || !cartItemsEl) return;

      const totalCount = countCart(cartItems);
      cartMeta.textContent = totalCount ? `Cart (${totalCount})` : "Cart (empty)";

      if (!cartItems.length) {
        cartItemsEl.innerHTML = `
        <div style="padding:12px 14px; color:rgba(255,255,255,.75); font-size:13px;">
          Your cart is empty.
        </div>
      `;
        return;
      }

      const show = cartItems.slice(0, 5);
      cartItemsEl.innerHTML = show.map((it) => `
      <div class="cart-item">
        <div class="thumb">
          ${it.image ? `<img src="${escapeHtml(it.image)}" alt="">` : "Item"}
        </div>
        <div class="ci-main">
          <div class="ci-title">${escapeHtml(it.title)}</div>
          <div class="ci-sub">
            <span>Qty: ${Number(it.qty) || 1}</span>
            ${it.price != null ? `<span>$${escapeHtml(String(it.price))}</span>` : `<span></span>`}
          </div>
        </div>
      </div>
    `).join("");

      if (cartItems.length > show.length) {
        cartItemsEl.innerHTML += `
        <div style="padding:10px 14px; font-size:12px; color:rgba(255,255,255,.7); border-top:1px solid rgba(255,255,255,.06);">
          + ${cartItems.length - show.length} more item(s)
        </div>
      `;
      }
    };

    const bindCartListener = (user) => {
      if (unsubCart) unsubCart();
      unsubCart = null;

      if (!user) {
        cartItems = getLocalCartItems();
        renderCartMenu();
        updateBadge(countCart(cartItems));
        return;
      }

      const itemsRef = collection(db, "users", user.uid, "cartItems");
      const q = query(itemsRef);

      unsubCart = onSnapshot(
        q,
        (snap) => {
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
          cartItems = items;
          renderCartMenu();
          updateBadge(countCart(items));
        },
        (err) => {
          console.error("Cart listener error:", err);
          cartItems = getLocalCartItems();
          renderCartMenu();
          updateBadge(countCart(cartItems));
        }
      );
    };

    const updateAccountMenu = (user) => {
      if (!menuMeta || !menuPrimary || !menuSignOut) return;

      if (!user) {
        menuMeta.textContent = "Not signed in";
        menuPrimary.textContent = "Sign In";
        menuPrimary.href = "/sign-in";
        menuSignOut.style.display = "none";
        return;
      }

      menuMeta.textContent = user.email || "Signed in";
      menuPrimary.textContent = "Profile";
      menuPrimary.href = "/profile";
      menuSignOut.style.display = "block";
    };

    // Initial render (logged-out fallback)
    cartItems = getLocalCartItems();
    renderCartMenu();
    updateBadge(countCart(cartItems));

    onAuthStateChanged(auth, (user) => {
      updateAccountMenu(user || null);
      bindCartListener(user || null);
    });
