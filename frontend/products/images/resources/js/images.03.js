fetch('assets.unlim8ted.com/dataassets.unlim8ted.com/images.json')
      .then(response => response.json())
      .then(images => {
        const gallery = document.getElementById('gallery');
        images.forEach(filename => {
          const img = document.createElement('img');
          img.src = `assets.unlim8ted.com/images/Unlim8tedImages/${filename}`;
          img.alt = filename;
          gallery.appendChild(img);
        });
      });
