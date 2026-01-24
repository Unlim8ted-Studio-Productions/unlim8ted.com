fetch('https://assets.unlim8ted.com/data/products.json')
  .then(response => {
    if (!response.ok) throw new Error('Failed to load products.json');
    return response.json();
  })
  .then(products => {
    const gallery = document.getElementById('gallery');
    if (!gallery) return;

    products.forEach(product => {
      if (product['product-type'] !== 'image') return;

      const src = product.image || product.file;
      if (!src) return;

      const img = document.createElement('img');
      img.src = src;
      img.alt = product.name || product.id || 'Image';

      img.loading = 'lazy';        // ✅ performance
      img.decoding = 'async';      // ✅ performance

      gallery.appendChild(img);
    });
  })
  .catch(err => {
    console.error('Gallery load error:', err);
  });
