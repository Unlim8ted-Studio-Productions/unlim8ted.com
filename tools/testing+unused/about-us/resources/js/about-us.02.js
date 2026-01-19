(function () {
      const prefersReduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      const root = document.documentElement;
      const film = document.getElementById("scrollFilm");
      const sigil = document.getElementById("sigil");

      const beat1 = document.getElementById("beat1");
      const panel1 = document.getElementById("panel1");
      const beat2 = document.getElementById("beat2");
      const panel2 = document.getElementById("panel2");

      const hsection = document.getElementById("hsection");
      const htrack = document.getElementById("htrack");

      const endSlate = document.getElementById("endSlate");

      const svg = document.getElementById("sigilSvg");
      const eightPath = document.getElementById("eightPath");
      const eightCore = document.getElementById("eightCore");
      const eightFire = document.getElementById("eightFire");

      // Mirror the same path into core + fire
      eightCore.setAttribute("d", eightPath.getAttribute("d"));
      eightFire.setAttribute("d", eightPath.getAttribute("d"));

      // Path drawing setup
      const pathLen = eightPath.getTotalLength();
      [eightPath, eightCore, eightFire].forEach(p => {
        p.style.strokeDasharray = pathLen.toFixed(2);
        p.style.strokeDashoffset = pathLen.toFixed(2);
      });

      const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
      const lerp = (a, b, t) => a + (b - a) * t;
      const smoothstep = (t) => t * t * (3 - 2 * t);
      const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

      function range(t, a, b) {
        return clamp((t - a) / (b - a));
      }
      function setVar(name, val) {
        root.style.setProperty(name, val);
      }

      // Convert SVG point to viewport coords (centered sigil)
      function svgPointToViewport(pt) {
        const rect = svg.getBoundingClientRect();
        const vb = svg.viewBox.baseVal;
        const x = rect.left + ((pt.x - vb.x) / vb.width) * rect.width;
        const y = rect.top + ((pt.y - vb.y) / vb.height) * rect.height;
        return { x, y };
      }

      // Horizontal track metrics
      let hMaxShift = 0; // px (positive number)
      function measureH() {
        const wrap = hsection?.querySelector(".hwrap");
        if (!wrap || !htrack) return;
        const wrapW = wrap.getBoundingClientRect().width;
        const trackW = htrack.scrollWidth;
        hMaxShift = Math.max(0, trackW - wrapW);
      }

      // Update loop
      let ticking = false;
      function onScroll() {
        if (!ticking) {
          ticking = true;
          requestAnimationFrame(update);
        }
      } let hShiftSmooth = 0;          // smoothed horizontal px
      let dashSmooth = pathLen;      // smoothed dashoffset
      let igniteSmooth = 0;          // smoothed ignite
      let pathVisSmooth = 0;         // smoothed visibility

      // tweak knobs (smaller = slower)
      const H_FOLLOW = .045;      // sideways scroll “lag”
      const SIG_FOLLOW = 0.045;      // sigil draw “lag”

      function update() {
        ticking = false;

        const maxScroll = film.offsetHeight - window.innerHeight;
        const y = window.scrollY || window.pageYOffset || 0;
        const p = maxScroll > 0 ? clamp(y / maxScroll) : 0;

        // subtle drifting background
        setVar("--bgx", (Math.sin(p * Math.PI * 2) * 20).toFixed(2));
        setVar("--bgy", (Math.cos(p * Math.PI * 2) * 14).toFixed(2));
        setVar("--p", p.toFixed(5));

        const vibe = prefersReduced ? 0 : (Math.sin(p * Math.PI * 2) * 2.2);
        setVar("--vibe", vibe.toFixed(3));

        // Segment map:
        // 0.00 - 0.18 : Hero
        // 0.14 - 0.55 : Offerings
        // 0.52 - 0.78 : Community
        // 0.60 - 0.74 : Sideways products (NEW)
        // 0.74 - 1.00 : Sigil + ignite + end slate rising

        const heroOn = easeOutCubic(1 - range(p, 0.00, 0.18));
        setVar("--heroOn", clamp(heroOn).toFixed(4));

        const beat1On = smoothstep(range(p, 0.10, 0.26)) * (1 - smoothstep(range(p, 0.42, 0.56)));
        const panel1On = smoothstep(range(p, 0.16, 0.30)) * (1 - smoothstep(range(p, 0.48, 0.60)));
        beat1.style.setProperty("--beatOn", beat1On.toFixed(4));
        panel1.style.setProperty("--panelOn", panel1On.toFixed(4));

        const beat2On = smoothstep(range(p, 0.48, 0.62)) * (1 - smoothstep(range(p, 0.70, 0.82)));
        const panel2On = smoothstep(range(p, 0.54, 0.66)) * (1 - smoothstep(range(p, 0.74, 0.86)));
        beat2.style.setProperty("--beatOn", beat2On.toFixed(4));
        panel2.style.setProperty("--panelOn", panel2On.toFixed(4));

        const liftInA = 0.70;  // lift starts (later)
        const liftInB = 0.78;  // lift ends

        const hLiftTD = smoothstep(range(p, liftInA, liftInB));

        // Compute target delta for centering hsection (guard if null)
        let deltaY = 0;
        if (hsection) {
          const hRect = hsection.getBoundingClientRect();
          const hTargetY = (window.innerHeight * 0.5) - (hRect.height * 0.5);
          const hStartY = (window.innerHeight * 0.80); // because style top:80vh
          deltaY = hTargetY - hStartY;

          // hsection lift
          const hY = lerp(0, deltaY, hLiftTD);
          setVar("--hY", hY.toFixed(2) + "px");
        } else {
          setVar("--hY", "0px");
        }

        // panel2 follows (slightly reduced)
        if (panel2) {
          const panel2Lift = lerp(0, deltaY * 0.6, hLiftTD);
          panel2.style.transform = `translateX(-50%) translate3d(0, ${panel2Lift.toFixed(2)}px, 0)`;
        }

        // beat2 follows (make sure your CSS uses --beatY in its transform)
        if (beat2) {
          const beat2Lift = lerp(0, deltaY * 0.45, hLiftTD);
          beat2.style.setProperty("--beatY", beat2Lift.toFixed(2) + "px");
        }

        const sideStart = 0.78;   // sideways begins
        const sideEnd = 0.88;   // sideways ends EARLIER (more time after)
        const gapA = 0.88;   // gap start
        const gapB = 0.915;  // gap end (this is your “pause”)

        const sideT = smoothstep(range(p, sideStart, sideEnd));

        // keep hsection on during sideways, then fade out across the gap
        const hFadeIn = smoothstep(range(p, sideStart - 0.03, sideStart + 0.02));
        const hFadeOut = smoothstep(range(p, gapA + 0.01, gapB)); // fades out during the gap
        const hOn = clamp(hFadeIn * (1 - hFadeOut), 0, 1);

        setVar("--hOn", hOn.toFixed(4));
        if (hsection) hsection.style.setProperty("--hOn", hOn.toFixed(4));

        // horizontal shift only during sideways
        const shiftTarget = -sideT * hMaxShift;
        hShiftSmooth = lerp(hShiftSmooth, shiftTarget, H_FOLLOW);
        setVar("--hShift", hShiftSmooth.toFixed(2) + "px");

        // --- SIGIL starts AFTER the gap ---
        const sigilOnT = smoothstep(range(p, gapB, 0.96));
        const drawTraw = smoothstep(range(p, gapB + 0.015, 0.975));
        const igniteT = smoothstep(range(p, 0.975, 1.008));

        // smoothing (keep your existing smoothing code)
        pathVisSmooth = lerp(pathVisSmooth, sigilOnT, SIG_FOLLOW);
        igniteSmooth = lerp(igniteSmooth, igniteT, SIG_FOLLOW);
        const dashTarget = (1 - drawTraw) * pathLen;
        dashSmooth = lerp(dashSmooth, dashTarget, SIG_FOLLOW);

        setVar("--pathVis", pathVisSmooth.toFixed(4));
        setVar("--ignite", igniteSmooth.toFixed(4));
        setVar("--pathDraw", (1 - dashSmooth / pathLen).toFixed(4));
        [eightPath, eightCore, eightFire].forEach(pth => pth.style.strokeDashoffset = dashSmooth.toFixed(2));
        sigil.dataset.ignite = igniteSmooth > 0.35 ? "1" : "0";

        /* END: starts later + lifts later + expands higher */
        const endOn = smoothstep(range(p, 0.970, 0.993));
        const endLiftT = smoothstep(range(p, 0.985, 1.000));
        const endLift = lerp(0, -260, endLiftT);

        setVar("--endOn", endOn.toFixed(4));
        setVar("--endLift", endLift.toFixed(2) + "px");
        endSlate.style.setProperty("--endOn", endOn.toFixed(4));

        const expandT = smoothstep(range(p, 0.982, 1.000));
        const endExpand = lerp(0, 820, expandT);   // ✅ higher expansion
        setVar("--endExpand", endExpand.toFixed(2) + "px");
      }

      // Footer year
      const year = new Date().getFullYear();
      const c = document.getElementById("copyright");
      if (c) c.textContent = `© 2019–${year} Unlim8ted Studios. All rights reserved.`;

      // Init
      measureH();
      update();

      window.addEventListener("scroll", onScroll, { passive: true });
      window.addEventListener("resize", () => { measureH(); requestAnimationFrame(update); }, { passive: true });

      (() => {
        const prefersReduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (prefersReduced) return; // respect accessibility

        const film = document.getElementById("scrollFilm"); // your tall container
        let virtualY = window.scrollY;
        let currentY = window.scrollY;
        let isAnimating = false;
        let lastTouchY = 0;

        const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

        function maxScroll() {
          return Math.max(0, film.offsetHeight - innerHeight);
        }

        function start() {
          if (isAnimating) return;
          isAnimating = true;

          const step = () => {
            const diff = virtualY - currentY;
            currentY += diff * 0.045;; // easing feel (0.08–0.18 range) / speed

            // Keep real scroll synced (so address bar / history behaves)
            window.scrollTo(0, currentY);

            // Tell your animation system to update based on currentY:
            // progress p = currentY / maxScroll
            const p = maxScroll() ? clamp(currentY / maxScroll(), 0, 1) : 0;
            window.__setScrollProgress?.(p); // we’ll wire this in below

            if (Math.abs(diff) > 0.5) {
              requestAnimationFrame(step);
            } else {
              isAnimating = false;
            }
          };
          requestAnimationFrame(step);
        }

        function onWheel(e) {
          e.preventDefault();
          virtualY = clamp(virtualY + e.deltaY, 0, maxScroll());
          start();
        }

        function onTouchStart(e) {
          lastTouchY = e.touches[0].clientY;
        }

        function onTouchMove(e) {
          e.preventDefault();
          const touchY = e.touches[0].clientY;
          const deltaY = lastTouchY - touchY;
          lastTouchY = touchY;

          virtualY = clamp(virtualY + deltaY, 0, maxScroll());
          start();
        }

        // IMPORTANT: passive:false so preventDefault works
        addEventListener("wheel", onWheel, { passive: false });
        addEventListener("touchstart", onTouchStart, { passive: true });
        addEventListener("touchmove", onTouchMove, { passive: false });

        addEventListener("resize", () => {
          virtualY = clamp(virtualY, 0, maxScroll());
          currentY = clamp(currentY, 0, maxScroll());
        });

        // Initialize
        const p0 = maxScroll() ? clamp(window.scrollY / maxScroll(), 0, 1) : 0;
        window.__setScrollProgress?.(p0);
      })();

    })();
