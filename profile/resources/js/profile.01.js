import { onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";
    import { doc, getDoc, setDoc } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";
    import { getFirebase } from "/components/firebase-init.js";

    const { auth, db } = getFirebase();
    const $ = (id) => document.getElementById(id);

    // MUST MATCH YOUR RULES
    const MAX_USERNAME_LEN = 80;
    const MAX_PIC_LEN = 600;

    const SIGNIN_URL = "https://unlim8ted.com/sign-in";
    const PROFILE_DEFAULT = "https://via.placeholder.com/110";

    const displayNameEl = $("displayName");
    const emailEl = $("email");
    const profilePicEl = $("profilePic");
    const usernameEl = $("username");
    const photoUrlEl = $("photoUrl");
    const statusEl = $("status");
    const nameHintEl = $("nameHint");
    const urlHintEl = $("urlHint");

    function setStatus(msg, isError = false) {
      statusEl.textContent = msg || "";
      statusEl.style.color = isError ? "rgba(255,120,120,.95)" : "rgba(255,255,255,.75)";
    }

    function basicSanitize(s) {
      return (s || "").trim().replace(/\s+/g, " ");
    }

    function clampText(s, max) {
      s = (s || "").trim();
      return s.length > max ? s.slice(0, max) : s;
    }

    function isValidHttpUrl(url) {
      if (!url) return true; // optional
      if (url.length > MAX_PIC_LEN) return false;
      try {
        const u = new URL(url);
        return u.protocol === "https:" || u.protocol === "http:";
      } catch {
        return false;
      }
    }

    function updateHints() {
      nameHintEl.textContent = `${(usernameEl.value || "").length}/${MAX_USERNAME_LEN}`;
      urlHintEl.textContent = `${(photoUrlEl.value || "").length}/${MAX_PIC_LEN}`;
    }

    usernameEl.addEventListener("input", () => {
      usernameEl.value = clampText(basicSanitize(usernameEl.value), MAX_USERNAME_LEN);
      updateHints();
    });

    photoUrlEl.addEventListener("input", () => {
      photoUrlEl.value = clampText(photoUrlEl.value.trim(), MAX_PIC_LEN);
      updateHints();
      const url = photoUrlEl.value.trim();
      if (url && isValidHttpUrl(url)) profilePicEl.src = url;
      if (!url) profilePicEl.src = PROFILE_DEFAULT;
    });

    function fillUI(user, data) {
      const username = (data?.username || "").toString().trim();
      const name = (data?.name || "").toString().trim();

      // Display name preference: username -> name -> auth displayName -> fallback
      const title = username || name || user.displayName || "User";

      // Email should be read-only in your rules, so just show auth email
      const email = user.email || data?.email || "—";

      // Picture preference: profilePicture -> auth photo -> placeholder
      const pic = (data?.profilePicture || user.photoURL || PROFILE_DEFAULT).toString().trim();

      displayNameEl.textContent = title;
      emailEl.textContent = email;

      usernameEl.value = username; // input edits `username` field
      photoUrlEl.value = (data?.profilePicture || "").toString();

      profilePicEl.src = pic || PROFILE_DEFAULT;
      updateHints();
    }

    async function ensureUserDocExists(user) {
      const ref = doc(db, "users", user.uid);
      const snap = await getDoc(ref);

      // If no doc yet, CREATE it using ONLY fields allowed by your CREATE rule:
      // hasOnly(["name","email","profilePicture"])
      if (!snap.exists()) {
        const payload = {
          name: clampText(basicSanitize(user.displayName || ""), 80),
          email: (user.email || "").trim().slice(0, 254),
          profilePicture: clampText((user.photoURL || "").trim(), MAX_PIC_LEN),
        };

        // Create (no merge) so Firestore evaluates it as a CREATE.
        await setDoc(ref, payload);
      }

      // Now read fresh
      const snap2 = await getDoc(ref);
      return snap2.data() || {};
    }

    onAuthStateChanged(auth, async (user) => {
      if (!user) {
        window.location.href = SIGNIN_URL;
        return;
      }

      try {
        setStatus("Loading profile…");
        const data = await ensureUserDocExists(user);
        fillUI(user, data);
        setStatus("");
      } catch (e) {
        console.error("Profile load error:", e);
        setStatus("Could not load profile (permissions/rules mismatch).", true);
      }
    });

    $("saveBtn").addEventListener("click", async () => {
      const user = auth.currentUser;
      if (!user) return;

      const username = clampText(basicSanitize(usernameEl.value), MAX_USERNAME_LEN);
      const profilePicture = clampText(photoUrlEl.value.trim(), MAX_PIC_LEN);

      if (!isValidHttpUrl(profilePicture)) {
        setStatus("Profile photo URL must be http(s) and within limits.", true);
        return;
      }

      try {
        setStatus("Saving…");

        // IMPORTANT:
        // Your current UPDATE rule only allows ["username","profilePicture"],
        // so ONLY write those fields here. Do NOT write name/email on update.
        await setDoc(
          doc(db, "users", user.uid),
          {
            username: username || "",
            profilePicture: profilePicture || "",
          },
          { merge: true }
        );

        const snap = await getDoc(doc(db, "users", user.uid));
        fillUI(user, snap.data());
        setStatus("Saved");
        setTimeout(() => setStatus(""), 1200);
      } catch (e) {
        console.error("Profile save error:", e);
        setStatus("Failed to save profile (permissions/rules mismatch).", true);
      }
    });

    $("productsBtn").addEventListener("click", () => {
      window.location.href = "https://unlim8ted.com/products";
    });

    $("logoutBtn").addEventListener("click", async () => {
      try {
        await signOut(auth);
        window.location.href = "https://unlim8ted.com";
      } catch (e) {
        console.error(e);
        setStatus("Logout failed.", true);
      }
    });

    // Footer year
    const y = new Date().getFullYear();
    $("footerText").innerHTML = `&copy; 2019-${y} Unlim8ted Studio Productions. All rights reserved.`;
