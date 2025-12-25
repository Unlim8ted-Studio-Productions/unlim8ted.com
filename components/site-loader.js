class SiteLoader extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
  }

  connectedCallback() {
    const text = this.getAttribute("text") || "Loading...";
    const bg = this.getAttribute("bg") || "black";
    const fadeMs = Number(this.getAttribute("fade-ms") || 4000);

    this.shadowRoot.innerHTML = `
      <style>
        :host { display: contents; }

        @media only screen and (max-width: 767px) {
          html { overflow-x: hidden; width: 100%; }
        }

        .no-js-message {
          background: #2c4762;
          color: red;
          font-size: 20px;
          text-align: center;
          padding: 20px;
          z-index: 2147483647;
          position: fixed;
          width: 100%;
          top: 0;
          left: 0;
        }

        .loader {
          position: fixed;
          background-color: ${bg};
          opacity: 1;
          height: 100%;
          width: 100%;
          top: 0;
          left: 0;
          z-index: 999999988;
          pointer-events: none;
          display: flex;
        }

        .loaderr-container {
          display: flex;
          justify-content: center;
          align-items: center;
          flex-direction: column;
          height: 100vh;
          width: 100%;
        }

        .loaderr {
          width: 100px;
          height: 100px;
          border-radius: 50%;
          border: 8px solid transparent;
          border-top: 8px solid #3498db;
          border-right: 8px solid #e74c3c;
          border-bottom: 8px solid #f1c40f;
          border-left: 8px solid #9b59b6;
          animation: spin 1.5s linear infinite;
          box-shadow:
            0 0 15px rgba(52, 152, 219, 0.7),
            0 0 15px rgba(231, 76, 60, 0.7),
            0 0 15px rgba(241, 196, 15, 0.7),
            0 0 15px rgba(155, 89, 182, 0.7);
          position: relative;
        }

        .loaderr:before {
          content: '';
          position: absolute;
          inset: 0;
          border-radius: 50%;
          border: 8px solid transparent;
          border-top: 8px solid rgba(52, 152, 219, 0.7);
          border-right: 8px solid rgba(231, 76, 60, 0.7);
          border-bottom: 8px solid rgba(241, 196, 15, 0.7);
          border-left: 8px solid rgba(155, 89, 182, 0.7);
          animation: spin-reverse 1.5s linear infinite;
        }

        .loading-text {
          margin-top: 20px;
          font-size: 18px;
          color: #3498db;
          font-family: Arial, sans-serif;
          letter-spacing: 2px;
          animation: pulse 1.5s infinite ease-in-out;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes spin-reverse { to { transform: rotate(-360deg); } }

        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }

        @keyframes load-out {
          from { opacity: 1; }
          to { opacity: 0; }
        }
      </style>

      <noscript>
        <div class="no-js-message">
          Please enable JavaScript to use this website properly.
        </div>
      </noscript>

      <div class="loader" part="overlay">
        <div class="loaderr-container">
          <div class="loaderr"></div>
          <div class="loading-text">${text}</div>
        </div>
      </div>
    `;

    const overlay = this.shadowRoot.querySelector(".loader");
    const spinner = this.shadowRoot.querySelector(".loaderr");
    const label = this.shadowRoot.querySelector(".loading-text");

    // Public methods
    this.show = () => {
      overlay.style.display = "flex";
      overlay.style.opacity = "1";
    };

    this.hide = () => {
      // Fade everything out like your original
      spinner.style.animation = "load-out 1.5s forwards, spin 2s linear infinite";
      label.style.animation = "load-out 1.5s forwards";
      overlay.style.animation = `load-out ${Math.max(1, fadeMs / 1000)}s forwards`;

      // Remove from layout after fade
      window.setTimeout(() => {
        overlay.style.display = "none";
      }, fadeMs);
    };

    // Auto-hide on full page load (same as your original behavior)
    window.addEventListener("load", () => this.hide(), { once: true });
  }
}

customElements.define("site-loader", SiteLoader);
