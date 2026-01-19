// Function to replace placeholders with product details
  function replacePlaceholders() {
    // Replace simple placeholders
    document.body.innerHTML = document.body.innerHTML.replace(/@product url@/g, window.location.hash.replace("#", ""));
  }
  replacePlaceholders();
