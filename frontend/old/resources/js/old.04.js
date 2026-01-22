// Select the elements
    const homeContainer = document.getElementById("home");
    const firsthero = document.getElementById("firstherocontent");
    const scrollingContainer = document.getElementById("moving-scrolling-container");
    const footer = document.getElementById("homefooter");

    // Track virtual scroll positions
    let virtualScrollPosition = 0;
    let currentScrollPositionHome = 0;
    let currentScrollPositionScrolling = 0;
    let isAnimating = false;
    let lastTouchY = 0; // For touch event tracking
    const arrow = document.querySelector('.neon-arrow');

    // Function to update the transformation of both containers
    const updateTransformOnScroll = (deltaY, speedHome, speedScrolling) => {
          // Remove the neon arrow when the user scrolls
      arrow.style.animation="fade 1.5s linear forwards"
      // Update the virtual scroll position
      virtualScrollPosition += deltaY;

      // Footer's effective position is offsetTop minus the virtual scroll position
      const contentHeight = Math.max(
        homeContainer.offsetHeight,
        scrollingContainer.offsetHeight,
        footer.offsetTop + footer.offsetHeight
      );

      // Calculate the maximum scrollable position
      maxScroll = Math.max(0, contentHeight - window.innerHeight);
      console.log(maxScroll);

      // Clamp the virtual scroll position
      virtualScrollPosition = Math.max(0, Math.min(virtualScrollPosition, maxScroll));

      // Start animations for both containers if not already animating
      if (!isAnimating) {
        smoothScroll(speedHome, speedScrolling);
      }
    };

    // Smooth scrolling function
    const smoothScroll = (speedHome, speedScrolling) => {
      isAnimating = true;

      const step = () => {
        const difference = virtualScrollPosition - currentScrollPositionHome;

        // Calculate movement for home container
        const movementHome = difference * 0.1; // Easing factor
        currentScrollPositionHome += movementHome;

        // Calculate movement for scrolling container (faster)
        const movementScrolling = (virtualScrollPosition - currentScrollPositionHome) * 0.1;
        currentScrollPositionScrolling += movementScrolling;

        // Apply transforms
        firsthero.style.transform = `scale(${100 - currentScrollPositionHome / 10}%)`
        homeContainer.style.transform = `translateY(-${currentScrollPositionHome}px)`;
        scrollingContainer.style.transform = `translateY(-${currentScrollPositionScrolling}px)`;

        // Stop animation if close to target
        if (Math.abs(difference) > 0.5) {
          requestAnimationFrame(step);
        } else {
          isAnimating = false; // Stop animating
        }
      };

      requestAnimationFrame(step);
    };

    // Attach the wheel event listener for desktop with { passive: false }
    window.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault(); // Prevent default scroll behavior
        updateTransformOnScroll(event.deltaY, 0.02, 0.07); // Speed: Home = 1x, Scrolling Container = 1.5x
      },
      { passive: false }
    );

    // Handle touch events for mobile
    window.addEventListener("touchstart", (event) => {
      // Record the initial touch position
      lastTouchY = event.touches[0].clientY;
    });

    window.addEventListener("touchmove", (event) => {
      event.preventDefault(); // Prevent default scroll behavior

      // Calculate the deltaY (vertical movement) from touch
      const touchY = event.touches[0].clientY;
      const deltaY = lastTouchY - touchY;
      lastTouchY = touchY;

      // Update transform based on touch movement
      updateTransformOnScroll(deltaY, 0.02, 0.07); // Speed: Home = 1x, Scrolling Container = 1.5x
    }, { passive: false });
