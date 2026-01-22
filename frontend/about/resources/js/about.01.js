function toggleMenu() {
      document.getElementById("navbarLinks").classList.toggle("show");
    }

    // Mobile dropdown open/close + click-outside
    function toggleDropdown(e) {
      e.stopPropagation();
      const dd = document.getElementById("moreDropdown");
      dd.classList.toggle("open");
    }

    window.addEventListener("click", () => {
      const dd = document.getElementById("moreDropdown");
      if (dd) dd.classList.remove("open");
    });

    document.addEventListener("DOMContentLoaded", function () {
      const currentYear = new Date().getFullYear();
      const footerText = `&copy; 2019-${currentYear} Unlim8ted Studio Productions. All rights reserved.`;
      const el = document.getElementById("footer-text");
      if (el) el.innerHTML = footerText;
    });
