// Footer year
    (function(){
      const y = new Date().getFullYear();
      const el = document.getElementById("footer-text");
      if (el) el.textContent = `© 2019-${y} Unlim8ted Studio Productions. All rights reserved.`;
    })();

    // Notes persistence (nice UX upgrade)
    (function(){
      const key = "unlim8ted:puzzleSquare:notes";
      const ta = document.getElementById("notes");
      const saveBtn = document.getElementById("saveNotes");
      const clearBtn = document.getElementById("clearNotes");

      if (!ta) return;

      ta.value = localStorage.getItem(key) || "";

      saveBtn?.addEventListener("click", () => {
        localStorage.setItem(key, ta.value);
        saveBtn.textContent = "Saved ✓";
        setTimeout(()=> saveBtn.textContent = "Save", 900);
      });

      clearBtn?.addEventListener("click", () => {
        ta.value = "";
        localStorage.removeItem(key);
      });
    })();
