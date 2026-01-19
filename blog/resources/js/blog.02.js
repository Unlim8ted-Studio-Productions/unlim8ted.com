document.addEventListener("DOMContentLoaded", function () {
            const currentYear = new Date().getFullYear();
            document.getElementById("footer-text").innerHTML =
                `&copy; 2019-${currentYear} Unlim8ted Studio Productions. All rights reserved.`;
        });
