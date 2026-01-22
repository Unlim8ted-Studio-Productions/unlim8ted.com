import { getFirebase } from "/components/firebase-init.js";
        import {
            doc, setDoc, serverTimestamp
        } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";

        const { db, auth } = getFirebase();

        const form = document.getElementById("newsletterForm");
        const emailEl = document.getElementById("newsletterEmail");
        const msgEl = document.getElementById("newsletterMsg");
        const honeypotEl = document.getElementById("website");

        const ICONS = {
            success: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`,
            error: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 9v5" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
      <path d="M12 17h.01" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
      <path d="M10.3 4.7l-7 12.1A2 2 0 0 0 5 20h14a2 2 0 0 0 1.7-3.2l-7-12.1a2 2 0 0 0-3.4 0Z"
        stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
    </svg>`,
            info: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 8h.01" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
      <path d="M11 12h1v6h1" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10Z"
        stroke="currentColor" stroke-width="2"/>
    </svg>`
        };

        function setMsg(kind, text) {
            msgEl.innerHTML = `${ICONS[kind] || ""}<div>${text}</div>`;
        }

        function isValidEmail(email) {
            return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
        }

        async function sha256Hex(str) {
            const data = new TextEncoder().encode(str);
            const hash = await crypto.subtle.digest("SHA-256", data);
            return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, "0")).join("");
        }

        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            msgEl.textContent = "";

            // Honeypot = bot
            if (honeypotEl.value && honeypotEl.value.trim().length) return;

            const email = (emailEl.value || "").trim().toLowerCase();
            if (!isValidEmail(email)) {
                setMsg("error", "<strong>Invalid email.</strong> Please check and try again.");
                return;
            }

            try {
                setMsg("info", "Adding you to the list…");

                // DEDUPE: stable doc id derived from email (keeps emails out of the URL path)
                const id = await sha256Hex(email);

                await setDoc(doc(db, "newsletter_signups", id), {
                    email,
                    createdAt: serverTimestamp(),     // first time set; subsequent merges keep it unless overwritten
                    updatedAt: serverTimestamp(),     // you can keep this to track resubscribes
                    source: location.pathname,
                    uid: auth?.currentUser?.uid || null
                }, { merge: true });

                emailEl.value = "";
                setMsg("success", "<strong>Subscribed.</strong> You’re on the list.");
            } catch (err) {
                console.error(err);
                setMsg("error", "<strong>Couldn’t subscribe.</strong> Please try again in a moment.");
            }
        });
