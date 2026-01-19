document.addEventListener("DOMContentLoaded", () => {
      const portfolioItems = document.querySelectorAll(".portfolio-item");

      portfolioItems.forEach(item => {
        item.addEventListener("mouseover", () => {
          item.style.boxShadow = `0 0 30px #00ffc6, 0 0 60px rgba(0, 255, 198, 0.5)`;
        });

        item.addEventListener("mouseout", () => {
          item.style.boxShadow = `0 0 10px rgba(0, 0, 0, 0.5)`;
        });
      });
    });
