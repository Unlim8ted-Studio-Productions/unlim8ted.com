fetch('https://unlim8ted.com/tools/data/images.json')
      .then(response => response.json())
      .then(images => {
        const gallery = document.getElementById('gallery');
        images.forEach(filename => {
          const img = document.createElement('img');
          img.src = `https://unlim8ted.com/images/Unlim8tedImages/${filename}`;
          img.alt = filename;
          gallery.appendChild(img);
        });
      });
