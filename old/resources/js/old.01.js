// This function will execute after the full page has finished loading
    window.onload = function () {
      // Trigger fade-out animation for the loader spinner
      const loaderSpinner = document.querySelector('.loaderr');
      const loadingText = document.querySelector('.loading-text');
      const loaderContainer = document.querySelector('.loader');

      // Apply animations
      loaderSpinner.style.animation = 'load-out 1.5s forwards, spin 2s linear infinite';
      loaderSpinner.style.webkitAnimation = 'load-out 1.5s forwards, spin 2s linear infinite';

      loadingText.style.animation = 'load-out 1.5s forwards, spin 2s linear infinite';
      loadingText.style.webkitAnimation = 'load-out 1.5s forwards, spin 2s linear infinite';

      loaderContainer.style.animation = 'load-out 4s forwards';
      loaderContainer.style.webkitAnimation = 'load-out 4s forwards';


    };
