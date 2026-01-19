function smoothScroll(duration) {
        let start = null;
        const step = timestamp => {
            if (!start) start = timestamp;
            const progress = timestamp - start;
            const position = progress / duration; // Calculate progress percentage
            const y = position * (document.body.scrollHeight - window.innerHeight); // Target scroll position
            window.scrollTo(0, y); // Perform the scroll
            if (progress < duration) { // Continue scrolling
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    window.addEventListener('keydown', function(event) {
        if (event.keyCode === 32) { // 32 is the key code for the spacebar
            event.preventDefault(); // Prevent the default spacebar action (scroll down)
            smoothScroll(1500); // Adjust duration in milliseconds (10000ms = 10 seconds)
        }
    });
