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

      const link = document.createElement('a');
      link.href = `/products/product#${encodeURIComponent(product.id || '')}`;
      link.className = 'gallery-link';

      const card = document.createElement('article');
      card.className = 'gallery-item';

      const media = document.createElement('div');
      media.className = 'gallery-media';

      const img = document.createElement('img');
      img.src = src;
      img.alt = product.name || product.id || 'Image';
      img.loading = 'lazy';
      img.decoding = 'async';

      const copy = document.createElement('div');
      copy.className = 'gallery-copy';

      const title = document.createElement('h2');
      title.textContent = product.name || product.id || 'Image';

      const desc = document.createElement('p');
      const raw = String(product.description || '').replace(/\s+/g, ' ').trim();
      desc.textContent = raw
        ? raw.slice(0, 120) + (raw.length > 120 ? '...' : '')
        : 'Open the product page for the full image details.';

      media.appendChild(img);
      copy.appendChild(title);
      copy.appendChild(desc);
      card.appendChild(media);
      card.appendChild(copy);
      link.appendChild(card);
      gallery.appendChild(link);
    });
  })
  .catch(err => {
    console.error('Gallery load error:', err);
  });
