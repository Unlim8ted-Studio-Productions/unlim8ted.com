// This function will execute after the full page has finished loading
    window.onload = function () {
      // Trigger fade-out animation for the loader
      document.querySelector('.loaderr').style.animation = 'load-out 1.5s forwards, spin 2s linear infinite';
      document.querySelector('.loaderr').style.webkitAnimation = 'load-out 1.5s forwards, spin 2s linear infinite';

      document.querySelector('.loading-text').style.animation = 'load-out 1.5s forwards, spin 2s linear infinite';
      document.querySelector('.loading-text').style.webkitAnimation = 'load-out 1.5s forwards, spin 2s linear infinite';

      document.querySelector('.loader').style.animation = 'load-out 4s forwards';
      document.querySelector('.loader').style.webkitAnimation = 'load-out 4s forwards';

    }
