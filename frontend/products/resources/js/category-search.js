// Search and category filters for the static product listings.
(() => {
    const normalize = value => String(value || "").normalize("NFKD")
        .replace(/[\u0300-\u036f]/g, "").toLowerCase().trim();
    document.querySelectorAll(".search-bar").forEach(bar => {
        const input = bar.querySelector('input[type="text"], input[type="search"]');
        const select = bar.querySelector("select");
        const main = bar.closest("main");
        if (!input || !main) return;
        const items = Array.from(main.querySelectorAll(".item-category > .item-box, .scrollyCard"));
        if (!items.length) return;
        if (!input.hasAttribute("aria-label")) input.setAttribute("aria-label", input.placeholder || "Search products");
        if (select && !select.hasAttribute("aria-label")) select.setAttribute("aria-label", "Filter products by category");
        const status = document.createElement("p");
        status.setAttribute("role", "status");
        bar.after(status);
        const apply = () => {
            const words = normalize(input.value).split(/\s+/).filter(Boolean);
            const category = normalize(select?.value || "all");
            let count = 0;
            for (const item of items) {
                const categories = normalize(item.dataset.category || "scrolly");
                const text = normalize(item.textContent + " " + categories);
                const matchesCategory = category === "all" || categories === category || categories.split(/\s+/).includes(category);
                const visible = matchesCategory && words.every(word => text.includes(word));
                // Explicit display wins over the legacy cards' display rules.
                item.hidden = !visible;
                if (visible) item.style.removeProperty("display"); else item.style.setProperty("display", "none", "important");
                if (visible) count++;
            }
            status.textContent = count ? `${count} product${count === 1 ? "" : "s"} found.` : "No products match your search. Try another search or category.";
        };
        input.addEventListener("input", apply);
        select?.addEventListener("change", apply);
        apply();
    });
})();

