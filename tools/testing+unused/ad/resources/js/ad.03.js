// JavaScript to add 'active' class when scrolling close to the product
    document.addEventListener("scroll", function () {
        const products = document.querySelectorAll('.product');
        products.forEach(product => {
            const rect = product.getBoundingClientRect();
            const inViewport = rect.top >= 0 && rect.bottom <= window.innerHeight;

            if (inViewport) {
                product.classList.add('active');
            } else {
                product.classList.remove('active');
            }
        });
    });
