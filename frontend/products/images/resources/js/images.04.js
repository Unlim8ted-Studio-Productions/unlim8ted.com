document.addEventListener("DOMContentLoaded", function () {
        const currentYear = new Date().getFullYear();
        const footerText = `&copy; 2019-${currentYear} Unlim8ted Studio Productions. All rights reserved.`;
        document.getElementById("footer-text").innerHTML = footerText;
      });
