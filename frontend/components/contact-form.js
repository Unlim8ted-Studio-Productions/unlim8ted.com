    const form = document.getElementById("contactForm");
    const btn = document.getElementById("submitBtn");
    const status = document.getElementById("status");
    const buttonLabel = btn.textContent.trim();

    const ICONS = {
      ok: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>`,
      err: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
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

    const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    function setStatus(kind, html) {
      status.className = `status ${kind || ""}`;
      status.innerHTML = html ? `${ICONS[kind] || ""}<div>${html}</div>` : "";
    }

    let cooldownUntil = 0;
    let submitting = false;
    let requestId = null;
    let lastPayload = null;
    function startCooldown(ms) {
      cooldownUntil = Date.now() + ms;
      btn.disabled = true;

      const base = buttonLabel;
      const t = setInterval(() => {
        const left = cooldownUntil - Date.now();
        if (left <= 0) {
          clearInterval(t);
          btn.disabled = false;
          btn.innerHTML = `${ICONS.info.replace('aria-hidden="true"', 'aria-hidden="true" style="display:none"')}${base}`;
          btn.textContent = base;
          return;
        }
        btn.textContent = `Submit (${Math.ceil(left / 1000)}s)`;
      }, 250);
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      for (const field of ["name", "email", "message"]) document.getElementById(field).removeAttribute("aria-invalid");
      // honeypot
      const hp = (document.getElementById("website").value || "").trim();
      if (hp) return;

      if (submitting || Date.now() < cooldownUntil) return;

      const name = (document.getElementById("name").value || "").trim();
      const email = (document.getElementById("email").value || "").trim().toLowerCase();
      const message = (document.getElementById("message").value || "").trim();

      if (name.length < 2 || name.length > 80) {
        document.getElementById("name").setAttribute("aria-invalid", "true");
        document.getElementById("name").focus();
        setStatus("err", "<strong>Name required.</strong> Please enter your name.");
        return;
      }
      if (email.length > 254 || !EMAIL_RE.test(email)) {
        document.getElementById("email").setAttribute("aria-invalid", "true");
        document.getElementById("email").focus();
        setStatus("err", "<strong>Invalid email.</strong> Please check your address.");
        return;
      }
      if (message.length < 5 || message.length > 5000) {
        document.getElementById("message").setAttribute("aria-invalid", "true");
        document.getElementById("message").focus();
        setStatus("err", "<strong>Message required.</strong> Please add a bit more detail.");
        return;
      }

      submitting = true;
      try {
        setStatus("info", "Sending…");
        btn.disabled = true;
        btn.textContent = "Sending…";

        const source = location.pathname.replace(/\/index\.html$/, '/');
        const payload = { name, email, message, source };
        const fingerprint = JSON.stringify(payload);
        if (!requestId || fingerprint !== lastPayload) { requestId = crypto.randomUUID(); lastPayload = fingerprint; }
        const response = await fetch('https://api.unlim8ted.com/contact', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...payload, requestId }),
          signal: AbortSignal.timeout(20000)
        });
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error('Support submission was not accepted');
        requestId = null; lastPayload = null;

        form.reset();
        setStatus("ok", "<strong>Message received.</strong> We'll reply at the email address you provided.");
        startCooldown(8000);
      } catch (err) {
        console.error(err);
        setStatus("err", "<strong>Couldn’t send.</strong> Please try again in a moment.");
        startCooldown(4000);
      } finally {
        submitting = false;
      }
    });



