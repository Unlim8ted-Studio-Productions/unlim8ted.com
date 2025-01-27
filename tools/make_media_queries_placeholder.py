from bs4 import BeautifulSoup
import re

# Define the HTML content
html_content = """
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <!-- Primary Meta Tags -->
  <title>Unlim8ted - Home</title>
  <meta name="title" content="Unlim8ted" />
  <meta name="description"
    content="Unlim8ted Studio Productions offers innovative entertainment experiences, including films, music, books, games, software, and AI products." />

  <!-- Website / General -->
  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://unlim8ted.com" />
  <meta property="og:title" content="Unlim8ted" />
  <meta property="og:description"
    content="Unlim8ted Studio Productions offers innovative entertainment experiences, including films, music, books, games, software, and AI products." />
  <meta property="og:image" content="https://unlim8ted.com/images/logo/logo.png" />
  <meta property="og:site_name" content="Unlim8ted Studio Productions" />

  <!-- GitHub -->
  <meta property="og:profile" content="GitHub" />
  <meta property="og:url" content="https://github.com/Unlim8ted-Studio-Productions" />
  <meta property="og:image" content="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" />
  <meta property="og:description" content="Check out Unlim8ted Studio Productions projects on GitHub!" />

  <!-- YouTube -->
  <meta property="og:profile" content="YouTube" />
  <meta property="og:url" content="https://www.youtube.com/@unlim8tedstudioproductions" />
  <meta property="og:image" content="https://www.youtube.com/img/desktop/yt_1200.png" />
  <meta property="og:description" content="Watch videos from Unlim8ted Studio Productions on YouTube." />

  <!-- Reddit 
    <meta property="og:profile" content="Reddit" />
    <meta property="og:url" content="https://www.reddit.com/user/Unlim8tedStudios/" />
    <meta property="og:image" content="https://www.redditinc.com/assets/images/site/reddit-logo.png" />
    <meta property="og:description" content="Join the conversation on Reddit with Unlim8ted Studios!" />-->

  <!-- Optional Meta Tags for other platforms -->
  <meta name="keywords"
    content="films, music, books, games, software, AI products, entertainment, unlim8ted, Unlim8ted Studio Productions" />
  <meta name="author" content="Unlim8ted Studio Productions" />
  <script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Organization",
  "name": "Unlim8ted Studio Productions",
  "url": "https://unlim8ted.com",
  "logo": "https://unlim8ted.com/images/logo/logo.png",
  "description": "Unlim8ted Studio Productions is a dynamic entertainment company creating innovative experiences across films, games, music, books, software, and AI products.",
  "foundingDate": "2019",
  "founder": {
    "@type": "Person",
    "name": "Anonymous Programmer"
  },
  "contactPoint": [
        {
      "@type": "ContactPoint",
      "email": "contact@unlim8ted.com",
      "contactType": "General Inquiry",
      "description": "Contact for general inquiries about Unlim8ted Studio Productions."
    },
        {
      "@type": "ContactPoint",
      "email": "support@unlim8ted.com",
      "contactType": "Customer Support",
      "description": "Contact for customer support and assistance."
    }
  ],
  "sameAs": [
    "https://www.youtube.com/@unlim8tedstudioproductions",
    "https://github.com/Unlim8ted-Studio-Productions",
    "https://www.reddit.com/r/Unlim8ted/",
    "https://discord.gg/Unlim8ted"
  ],
  "department": [
    {
      "@type": "Movie",
      "name": "The Life of a Meatball",
      "description": "An animated film that follows the adventures of a meatball, exploring themes of identity and belonging in a humorous and thought-provoking way.",
      "image": "https://unlim8ted.com/images/products/films/The Life of a Meatball.jpg",
      "datePublished": "2024-01-05",
      "director": {
        "@type": "Person",
        "name": "Anonymous"
      },
      "voiceActors": [
        {
          "@type": "Person",
          "name": "Anonymous"
        },
        {
          "@type": "Person",
          "name": "Anonymous"
        }
      ],
      "genre": "Animation, Comedy",
      "url": "https://m.youtube.com/watch?v=FiDwWp7ZteE"
    },
    {
      "@type": "VideoGame",
      "name": "Square Pixels",
      "description": "A 2D pixelated sandbox adventure inspired by Terraria, offering a rich world filled with exploration, crafting, and action-packed adventures.",
      "image": "https://unlim8ted.com/images/products/software/Screenshot 2023-09-21 181742(1).png",
      "datePublished": "2025-06-15",
      "operatingSystem": [
        "Windows",
        "Linux",
        "macOS"
      ],
      "applicationCategory": "Game",
      "genre": "Adventure, Sandbox",
      "url": "https://unlim8ted.com/square-pixels",
      "offers": {
        "@type": "Offer",
        "price": "19.99",
        "priceCurrency": "USD",
        "availability": "https://schema.org/PreOrder",
        "url": "https://store.steampowered.com/app/123456/Square_Pixels/"
      }
    }
  ]
}
</script>


  <!--______________________________START OF LOADING AND NOSCRIPT____________________________________-->
  <style media="screen" type="text/css">
    .loader {
      position: fixed;
      background-color: black;
      opacity: 1;
      height: 100%;
      width: 100%;
      top: 0;
      left: 0%;
      z-index: 999999988;
      pointer-events: none;
    }

    .loaderr-container {
      display: flex;
      justify-content: center;
      align-items: center;
      flex-direction: column;
      /* To stack spinner and text vertically */
      height: 100vh;
      /* Full viewport height to ensure vertical centering */
    }

    .loaderr {
      width: 100px;
      height: 100px;
      border-radius: 50%;
      border: 8px solid transparent;
      border-top: 8px solid #3498db;
      border-right: 8px solid #e74c3c;
      border-bottom: 8px solid #f1c40f;
      border-left: 8px solid #9b59b6;
      -webkit-animation: spin 1.5s linear infinite;
      animation: spin 1.5s linear infinite;
      box-shadow: 0 0 15px rgba(52, 152, 219, 0.7), 0 0 15px rgba(231, 76, 60, 0.7), 0 0 15px rgba(241, 196, 15, 0.7), 0 0 15px rgba(155, 89, 182, 0.7);
      position: relative;
    }

    .loaderr:before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      border-radius: 50%;
      border: 8px solid transparent;
      border-top: 8px solid rgba(52, 152, 219, 0.7);
      border-right: 8px solid rgba(231, 76, 60, 0.7);
      border-bottom: 8px solid rgba(241, 196, 15, 0.7);
      border-left: 8px solid rgba(155, 89, 182, 0.7);
      -webkit-animation: spin-reverse 1.5s linear infinite;
      animation: spin-reverse 1.5s linear infinite;
    }

    @-webkit-keyframes spin {
      0% {
        -webkit-transform: rotate(0deg);
      }

      100% {
        -webkit-transform: rotate(360deg);
      }
    }

    @keyframes spin {
      0% {
        transform: rotate(0deg);
      }

      100% {
        transform: rotate(360deg);
      }
    }

    @-webkit-keyframes spin-reverse {
      0% {
        -webkit-transform: rotate(360deg);
      }

      100% {
        -webkit-transform: rotate(0deg);
      }
    }

    @keyframes spin-reverse {
      0% {
        transform: rotate(360deg);
      }

      100% {
        transform: rotate(0deg);
      }
    }

    /* Loading text animation */
    .loading-text {
      margin-top: 20px;
      font-size: 18px;
      color: #3498db;
      font-family: Arial, sans-serif;
      letter-spacing: 2px;
      animation: pulse 1.5s infinite ease-in-out;
    }

    @keyframes pulse {
      0% {
        opacity: 1;
      }

      50% {
        opacity: 0.5;
      }

      100% {
        opacity: 1;
      }
    }


    @-webkit-keyframes load-out {
      from {
        opacity: 1;
      }

      to {
        opacity: 0;
      }
    }

    @keyframes load-out {
      from {
        opacity: 1;
      }

      to {
        opacity: 0;
      }
    }
  </style>
  <!-- Styles for the No-JavaScript message -->
  <style>
    .no-js-message {
      background: #2c4762;
      color: red;
      font-size: 20px;
      text-align: center;
      padding: 20px;
      z-index: 999999999999999999999999999999999999999999999;
      position: fixed;
      width: 100%;
      top: 0;
    }
  </style>

  <!-- No-JavaScript message -->
  <noscript>
    <div class="no-js-message">
      Please enable JavaScript to use this website properly.
    </div>
  </noscript>
  <div class="loader">
    <div class="loaderr-container">
      <div class="loaderr"></div>
      <div class="loading-text">Loading...</div>
    </div>
  </div>
  <script>
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

  </script>
  <!--______________________________END OF LOADING AND NOSCRIPT____________________________________-->

  <style>
    .content-section {
      display: none;
    }
  </style>
  <link rel="icon" href="favicon.ico" type="image/x-icon">

  <link rel="stylesheet" href="styles.css">
  <style>
    a.button {
      color: #8b00ff;
      text-decoration: none;
      font-weight: bold;
      background: black;
      border: 2px solid #8b00ff;
      padding: 15px 25px;
      font-size: 18px;
      border-radius: 8px;
      display: inline-block;
      margin-top: 20px;
    }

    body {
      font-family: Arial, sans-serif;
      background-color: #f4f4f4;
      color: #333;
      margin: 0;
      padding: 0;
    }

    .content-section {
      padding: 50px 20px;
      margin: 20px auto;
      max-width: 900px;
      background-color: #262c34;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
      border-radius: 8px;
    }

    h1 {
      font-size: 2.5em;
      margin-bottom: 20px;
      color: #ffffff;
    }

    h2 {
      font-size: 2rem;
      margin-top: 20px;
      color: #555;
      border-radius: 2rem;
    }

    p {
      line-height: 1.6;
    }

    a {
      color: #007bff;
      text-decoration: none;
    }

    a:hover {
      text-decoration: underline;
    }

    .portfolio-grid,
    .blog-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
      gap: 20px;
    }

    .portfolio-item,
    .blog-post {
      padding: 20px;
      background-color: #241e45;
      border: 1px solid #ddd;
      border-radius: 8px;
      transition: box-shadow 0.3s;
    }

    .portfolio-item:hover,
    .blog-post:hover {
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    form {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    label {
      font-weight: bold;
    }

    input,
    textarea {
      padding: 10px;
      font-size: 1em;
      border: 1px solid #ddd;
      border-radius: 4px;
    }

    button {
      padding: 10px;
      font-size: 1em;
      background-color: #000000;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background-color 0.3s;
    }

    button:hover {
      background-color: rgb(57, 200, 145);
    }

    /* Additional styling for responsiveness */
    @media (max-width: 768px) {
      h1 {
        font-size: 2em;
      }

      h2 {
        font-size: 1.5em;
      }

      .content-section {
        padding: 30px 15px;
      }

      button {
        padding: 8px;
        font-size: 0.9em;
      }
    }
  </style>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
  <style>
    .fa {
      padding: 20px;
      font-size: 30px;
      width: 70px;
      height: 70px;
      text-align: center;
      text-decoration: none;
      margin: 5px 2px;
      border-radius: 50%;
    }

    .fa:hover {
      opacity: 0.7;
    }

    .fa-facebook {
      background: #3B5998;
      color: white;
    }

    .fa-twitter {
      background: #55ACEE;
      color: white;
    }

    .fa-google {
      background: #dd4b39;
      color: white;
    }

    .fa-linkedin {
      background: #007bb5;
      color: white;
    }

    .fa-youtube {
      background: #bb0000;
      color: white;
    }

    .fa-instagram {
      background: #125688;
      color: white;
    }

    .fa-pinterest {
      background: #cb2027;
      color: white;
    }

    .fa-snapchat-ghost {
      background: #fffc00;
      color: white;
      text-shadow: -1px 0 black, 0 1px black, 1px 0 black, 0 -1px black;
    }

    .fa-skype {
      background: #00aff0;
      color: white;
    }

    .fa-android {
      background: #a4c639;
      color: white;
    }

    .fa-dribbble {
      background: #ea4c89;
      color: white;
    }

    .fa-vimeo {
      background: #45bbff;
      color: white;
    }

    .fa-tumblr {
      background: #2c4762;
      color: white;
    }

    .fa-vine {
      background: #00b489;
      color: white;
    }

    .fa-foursquare {
      background: #45bbff;
      color: white;
    }

    .fa-stumbleupon {
      background: #eb4924;
      color: white;
    }

    .fa-flickr {
      background: #f40083;
      color: white;
    }

    .fa-yahoo {
      background: #430297;
      color: white;
    }

    .fa-soundcloud {
      background: #ff5500;
      color: white;
    }

    .fa-reddit {
      background: #ff5700;
      color: white;
    }

    .fa-rss {
      background: #ff6600;
      color: white;
    }

    .fa-github {
      background: black;
      color: white;
    }
  </style>
</head>

<body>
  <style>
    .dropdown-content {
      display: none;
      position: absolute;
      background-color: #f9f9f9;
      box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
    }

    .dropdown:hover .dropdown-content {
      display: block;
    }

    .dropdown-content a {
      color: rgb(255, 255, 255);
      text-decoration: none;
      padding: 12px 16px;
      display: block;
    }

    .dropdown-content a:hover {
      background-color: #2d2849;
    }

    .navbar-toggle {
      display: none;
      background-color: #333;
      color: white;
      padding: 14px 20px;
      border: none;
      cursor: pointer;
      font-size: 18px;
    }

    /* Responsive styles */
    @media (max-width: 768px) {
      .nav-links {
        flex-direction: column;
        display: none;
      }

      .nav-links.show {
        display: flex;
      }

      .navbar-toggle {
        display: block;
      }
    }
  </style>
  <script>
    function toggleMenu() {
      const navLinks = document.getElementById("navbarLinks");
      navLinks.classList.toggle("show");
    }
  </script>
  <nav class="navbar">
    <div class="navbar-header">
      <button class="navbar-toggle" style="position:absolute; top:5px; left:1%" onclick="toggleMenu()">☰</button>
      <ul class="nav-links" id="navbarLinks">
        <li class="nav-item"><a href="/">Home</a></li>
        <li class="nav-item"><a href="products">Products</a></li>
        <li class="nav-item"><a href="help">Help</a></li>
        <li class="nav-item"><a href="about">About</a></li>
        <li class="nav-item dropdown">
          <a href="javascript:void(0)" class="dropbtn">More</a>
          <div class="dropdown-content">
            <a href="portfolio">Portfolio</a>
            <a href="contact">Contact</a>
            <a href="blog">Blog</a>
            <a href="live-chat">Live Chat</a>
            <a href="live-game">Live Chat and Game</a>
            <a href="puzzle-squares">Puzzle Squares</a>
          </div>
        </li>
        <a href="old"><button>Old site</button></a>
      </ul>
    </div>
  </nav>
  <style>
    /* logo */
    .logo {
      position: absolute;
      z-index: 1;
      top: 5%;
      /* Adjust top position relative to viewport height */
      left: 5%;
      /* Adjust left position relative to viewport width */
      display: block;
    }

    .logo img {
      height: 10vw;
      /* Adjust height relative to viewport width */
      max-height: 140px;
      /* Set a maximum height to prevent the logo from getting too large on larger screens */
      width: auto;
      /* Maintain aspect ratio */
    }

    /* Media query for small devices */
    @media (max-width: 600px) {
      .logo img {
        height: 20vw;
        /* Larger logo for smaller devices */
      }
    }

    /* Default (light mode) styles are already defined */

    /* Dark Mode Styles */
    body {
      background-color: #121212;
      color: #fff;
    }

    body a {
      color: #00ffcc;
      background-color: #28292f;
    }

    body a:hover {
      color: #ff00ff;
      background-color: #000;
    }

    body .content-section {
      background-color: #1a1a1a;
      box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);
    }

    body .button {
      background: #00ffcc;
      border: 2px solid #ff00ff;
      color: #121212;
    }

    body .button:hover {
      background: #ff00ff;
      border: 2px solid #00ffcc;
    }

    body .navbar {
      background-color: #1a1a1a;
    }

    body .spinner-choice {
      background-color: #1a1a1a;
      color: #00ffcc;
    }

    body .loading-text {
      color: #ff00ff;
    }

    body .fa {
      background: #1a1a1a;
      color: #00ffcc;
    }

    body .fa:hover {
      color: #ff00ff;
    }

    /* Base Styles */
    body {
      font-family: Arial, sans-serif;
      color: white;
      background-color: #000;
      margin: 0;
      overflow-x: hidden;
    }

    .background {
      position: absolute;
      width: 100%;
      height: 100%;
      overflow: hidden;
      z-index: -1;
    }

    .background span {
      position: absolute;
      width: 50px;
      height: 50px;
      background: linear-gradient(45deg, #00ff99, #ff0077);
      opacity: 0.7;
      filter: blur(8px);
      animation: float 6s infinite ease-in-out;
      border-radius: 50%;
    }

    @keyframes float {

      0%,
      100% {
        transform: translateY(0) translateX(0);
      }

      50% {
        transform: translateY(-40px) translateX(20px);
      }
    }

    /* Hero Section */
    .hero {
      height: 100vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      background: linear-gradient(135deg, #1a1a1a, #000000);
      padding: 0 1rem;
    }

    .hero h1 {
      font-size: 4rem;
      text-shadow: 0 0 20px #00ffcc, 0 0 30px #ff00ff;
      margin-bottom: 1rem;
      animation: glow 2s infinite alternate;
    }

    .hero p {
      font-size: 1.2rem;
      margin-bottom: 2rem;
    }

    .hero button {
      padding: 1rem 2rem;
      font-size: 1rem;
      font-weight: bold;
      background: #00ffcc;
      color: #000;
      border: none;
      cursor: pointer;
      border-radius: 5px;
      transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .hero button:hover {
      transform: scale(1.1);
      box-shadow: 0 0 20px #00ffcc;
    }

    /* Scrollable Sections */
    .section {
      height: 100vh;
      padding: 3rem 1rem;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      opacity: 0;
      transform: translateY(50px);
      transition: opacity 0.6s ease, transform 0.6s ease;
    }

    .section.visible {
      opacity: 1;
      transform: translateY(0);
    }

    .section h2 {
      font-size: 2.5rem;
      margin-bottom: 1rem;
    }

    .section p {
      max-width: 600px;
      font-size: 1rem;
      color: #ffffff;
      margin-bottom: 2rem;
    }

    .section img {
      max-width: 80%;
      margin: 2rem 0;
      border-radius: 10px;
      box-shadow: 0 5px 20px rgba(0, 0, 0, 0.5);
    }

    /* Gallery Section */
    #image-track {
      display: flex;
      gap: 4vmin;
      transform: translate(0%, -50%);
      user-select: none;
    }

    #image-track>.image {
      width: 40vmin;
      height: auto;
      object-fit: cover;
      object-position: 100% center;
    }

    /* Footer */
    .footer {
      text-align: center;
      background: rgba(0, 0, 0, 0.8);
      color: #aaa;
      padding: 1.5rem;
      font-size: 0.9rem;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
      .hero h1 {
        font-size: 2.5rem;
      }

      .hero p {
        font-size: 1rem;
      }

      .hero button {
        font-size: 0.9rem;
        padding: 0.8rem 1.5rem;
      }

      .section h2 {
        font-size: 2rem;
      }

      .section p {
        font-size: 0.9rem;
      }

      #image-track>.image {
        width: 60vmin;
      }

      .footer {
        font-size: 0.8rem;
        padding: 1rem;
      }
    }

    @media (max-width: 480px) {
      .hero h1 {
        font-size: 2rem;
      }

      .hero p {
        font-size: 0.8rem;
      }

      .hero button {
        font-size: 0.8rem;
        padding: 0.6rem 1rem;
      }

      .section h2 {
        font-size: 1.5rem;
      }

      .section p {
        font-size: 0.75rem;
      }

      #image-track>.image {
        width: 70vmin;
      }
    }

    #section3 {
      padding: 2rem 0.5rem;
      /* Reduced padding */
      transform: scale(0.8);
      /* Scales the entire section to 50% */
      transform-origin: top center;
      /* Keeps the scaling centered */
    }

    #section3 .content {
      display: flex;
      align-items: center;
      justify-content: space-between;
      max-width: 400px;
      /* Reduced max width */
      margin: 0 auto;
      gap: 0.75rem;
      /* Smaller gap */
    }

    #section3 .text-slide {
      flex: 1;
      opacity: 0;
      transform: translateX(-25px);
      /* Reduced translation */
      transition: opacity 0.4s ease, transform 0.4s ease;
    }

    #section3 .image-slide {
      flex: 1;
      opacity: 0;
      transform: translateX(25px);
      /* Reduced translation */
      transition: opacity 0.4s ease, transform 0.4s ease;
    }

    #section3.visible .text-slide {
      opacity: 1;
      transform: translateX(0);
    }

    #section3.visible .image-slide {
      opacity: 1;
      transform: translateX(0);
    }

    #section3 .text-slide h3 {
      font-size: 1rem;
      /* Smaller font size */
      margin-bottom: 0.25rem;
    }

    #section3 .text-slide p {
      font-size: 0.75rem;
      /* Smaller font size */
    }

    #section3 .image-slide img {
      max-width: 100%;
      border-radius: 4px;
      /* Reduced radius */
      box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4);
      /* Reduced shadow */
    }

    @media (max-width: 768px) {
      #section3 .content {
        flex-direction: column;
        text-align: center;
      }

      #section3 .text-slide,
      #section3 .image-slide {
        transform: translateX(0);
      }
    }

    /* Section 4 Infinite Scrolling Images */
    #section4 {
      position: relative;
      width: 100%;
      height: 60vh;
      /* Adjust the height */
      background-color: #111;
      overflow: hidden;
      /* Hide overflow */
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 2;
    }

    .slider-container-partners {
      display: flex;
      align-items: center;
      animation: slide 10s linear infinite;
      /* Continuous sliding */
      gap: 2rem;
      /* Space between images */
    }

    .slider-container img {
      height: 100%;
      /* Ensure full height */
      border-radius: 10px;
      box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
      /* Shadow effect */
      transition: transform 0.3s ease;
    }

    .slider-container-partners img:hover {
      transform: scale(1.1);
      /* Slight zoom on hover */
    }

    @keyframes slide {
      0% {
        transform: translateX(0);
        /* Start position */
      }

      100% {
        transform: translateX(-153%);
        /* Slide to the left */
      }
    }

    /* Duplicate the container for seamless looping */
    .slider-container-partners img {
      height: 20%;
      /* Ensure full height */
      border-radius: 10px;
      box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
      /* Shadow effect */
      transition: transform 0.3s ease;
      top: -10rem;
      position: relative;
    }

    @media (max-width: 768px) {
      #section4 {
        height: 40vh;
        /* Adjust for smaller screens */
      }

      .slider-container-partners img {
        width: 150px;
        /* Smaller images */
      }
    }

    body {
      font-family: Arial, sans-serif;
      color: white;
      background-color: #000;
      overflow-x: hidden;
    }

    .background {
      position: absolute;
      width: 100%;
      height: 100%;
      overflow: hidden;
      z-index: -1;
    }

    .background span {
      position: absolute;
      width: 50px;
      height: 50px;
      background: linear-gradient(45deg, #00ff99, #ff0077);
      opacity: 0.7;
      filter: blur(8px);
      animation: float 6s infinite ease-in-out;
      border-radius: 50%;
    }

    @keyframes float {

      0%,
      100% {
        transform: translateY(0) translateX(0);
      }

      50% {
        transform: translateY(-40px) translateX(20px);
      }
    }

    /* Section 4 Styling */
    #final {
      padding-top: 10rem;
      background-color: #00000000;
      background: #00000000;
      width: 100%;
      z-index: -1;
    }

    /* Final Section Styling */
    #final {
      position: sticky;
      top: 50%;
      transform: translateY(-50%);
      background:
        linear-gradient(to bottom, rgb(26, 26, 26) 80%, rgb(0, 0, 0) 100%),
        /* Vertical gradient */
        linear-gradient(135deg, #1a1a1a, #000000);
      /* Diagonal gradient */
      background-blend-mode: multiply;
      /* Blend the two gradients */
      text-align: center;
      z-index: 10;
    }

    #section4 {
      position: relative;
      width: 100%;
      height: 75vh;
      /* Double the viewport height for scrolling animation */
      background-color: #111;
      display: flex;
      justify-content: center;
      align-items: center;
    }

    .overlay-text {
      position: fixed;
      bottom: 35%;
      /* Adjust placement */
      left: 50%;
      transform: translateX(-50%);
      text-align: center;
      color: #00ffcc;
      font-size: 1.5rem;
      text-shadow: 0 0 10px #00ffcc, 0 0 20px #ff00ff;
      z-index: -1;
    }

    .overlay-text .highlight {
      color: #ff0077;
      font-weight: bold;
      text-shadow: 0 0 15px #ff0077, 0 0 25px #00ffcc;
    }
  </style>


  <a class="cart-btn" href="cart" style="font-size: larger; position: fixed; top: 30px">🛍️</a>
  <div id="home">


    <!-- Hero Section -->
    <div class="hero">
      <div class="background">
        <span style="top: 20%; left: 10%; animation-delay: 0s;"></span>
        <span style="top: 40%; left: 70%; animation-delay: 2s;"></span>
        <span style="top: 70%; left: 30%; animation-delay: 4s;"></span>
        <span style="top: 50%; left: 90%; animation-delay: 1.5s;"></span>
        <span style="top: 80%; left: 50%; animation-delay: 3s;"></span>
      </div>
      <h1 style="color:#bcfaf7;">Welcome to Unlim8ted</h1>
      <p>Innovative entertainment at your fingertips.</p>
      <a href="/sign-in"><button>Get Started</button></a>
    </div>


    <div id="moving-scrolling-container"
      style="transform: translateY(0rem); background: linear-gradient(rgb(26, 26, 26) 79%, rgb(0, 0, 0) 80%);">
      <!-- Section 1 -->
      <div class="section" id="section1">
        <h2>Explore the Future</h2>
        <p>Discover groundbreaking innovations that redefine the future of entertainment.</p>
        <img src="https://unlim8ted.com/images/home/path to future.png" alt="Image of Path to future">
      </div>

      <!-- Section 2 -->
      <div class="section" id="section2">
        <h2>Gallery</h2>
        <p>Take a look at some of our projects and achievements.</p>
        <div id="image-track" style="margin-top: 14rem;" data-mouse-down-at="0" data-prev-percentage="0">
          <img class="image" src="" draggable="false" />
          <img class="image" src="https://unlim8ted.com/images/products/films/The Life of a Meatball.jpg"
            draggable="false" />
          <img class="image" src="https://unlim8ted.com/images/products/films/TheGlitchCover.png" draggable="false" />
          <img class="image" src="https://unlim8ted.com/images/products/software/Screenshot 2023-09-21 181742(1).png"
            draggable="false" />
          <img class="image" src="https://unlim8ted.com/images/products/clothes/sfesehjkl_-removebg-preview.png"
            draggable="false" />
          <img class="image" src="https://unlim8ted.com/images/" draggable="false" />
          <img class="image" src="https://unlim8ted.com/images/" draggable="false" />
          <img class="image" src="https://unlim8ted.com/images/" draggable="false" />
        </div>
      </div>
      <!-- Section 3 -->
      <div class="section" id="section3">
        <div class="content">
          <div class="text-slide" id="text-slide-left">
            <h3>Innovate with Us</h3>
            <p>Explore our latest partners shaping the entertainment industry:</p>
          </div>
          <div class="image-slide" id="image-slide-right">
            <img src="https://unlim8ted.com/images/home/sunset.png" alt="Image of sunset" />
          </div>
        </div>
      </div>
      <div class="section" id="section4">
        <div class="slider-container-partners">
          <img src="images/logo/logoipsum-289.svg" alt="partner-logo">
          <img src="images/logo/logoipsum-331.svg" alt="partner-logo">
          <img src="images/logo/Zipper Line Logo.png" alt="partner-logo">
          <img src="images/logo/logoipsum-338.svg" alt="partner-logo">
          <img src="images/logo/Bnacintech Industries logo.png" alt="partner-logo">
          <img src="images/logo/logoipsum-346.svg" alt="partner-logo">
          <img src="images/logo/logoipsum-317.svg" alt="partner-logo">
          <img src="images/logo/logoipsum-339.svg" alt="partner-logo">
          <!--duplicates for seamless loop:-->
          <img src="images/logo/logoipsum-289.svg" alt="partner-logo">
          <img src="images/logo/logoipsum-331.svg" alt="partner-logo">
          <img src="images/logo/Zipper Line Logo.png" alt="partner-logo">
          <img src="images/logo/logoipsum-338.svg" alt="partner-logo">
          <img src="images/logo/Bnacintech Industries logo.png" alt="partner-logo">
          <img src="images/logo/logoipsum-346.svg" alt="partner-logo">
          <img src="images/logo/logoipsum-317.svg" alt="partner-logo">
          <img src="images/logo/logoipsum-339.svg" alt="partner-logo">
        </div>

      </div>

      <!-- Final Section -->
      <div class="hero" id="final">
        <div class="background" style="position:absolute; top:-10rem;">
          <span style="top: 20%; left: 10%; animation-delay: 0s;"></span>
          <span style="top: 40%; left: 70%; animation-delay: 2s;"></span>
          <span style="top: 70%; left: 30%; animation-delay: 4s;"></span>
          <span style="top: 50%; left: 90%; animation-delay: 1.5s;"></span>
          <span style="top: 80%; left: 50%; animation-delay: 3s;"></span>
        </div>
        <h1 id="neon-bar" style="color:#bcfaf7;position:relative;top:-13rem;">So are you ready to start using Unlim8ted?
        </h1>
        <a href="/sign-in" style="position:relative;top:-13rem;"><button>Get Started</button></a>
        <!-- Footer -->
        <div class="footer" style="width:110%; bottom:-10rem" id="homefooter">
          <p>© 2025 Unlim8ted Studio Productions. All rights reserved.</p>
        </div>

      </div>
    </div>


    <!-- Overlay text -->
    <div class="overlay-text">
      <p>Even our pages are <span class="highlight">Unlim8ted</span>...</p>
    </div>
  </div>

  <script>
    // Add the third section to the observer
    const section3 = document.getElementById("section3");
    observer.observe(section3);
  </script>
  <script>
    // Select the container element you want to control
    const parentDiv = document.getElementById("moving-scrolling-container");
    // Function to handle scrolling and update transform property
    const updateTransformOnScroll = () => {
      const scrollY = window.scrollY; // Get the current scroll position
      const translateY = -scrollY / 10; // Calculate transform value, adjust the divisor for speed

      // Apply the transform to the div
      parentDiv.style.transform = `translateY(${translateY}rem)`;
    };

    // Attach the scroll event listener
    window.addEventListener("scroll", updateTransformOnScroll);
  </script>
  <!-- JavaScript for Scroll Animations -->
  <script>
    const sections = document.querySelectorAll('.section');

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, { threshold: 0.2 });

    sections.forEach((section) => observer.observe(section));
  </script>

  <script>const track = document.getElementById("image-track");

    const handleOnDown = e => track.dataset.mouseDownAt = e.clientX;

    const handleOnUp = () => {
      track.dataset.mouseDownAt = "0";
      track.dataset.prevPercentage = track.dataset.percentage;
    }

    const handleOnMove = e => {
      if (track.dataset.mouseDownAt === "0") return;

      const mouseDelta = parseFloat(track.dataset.mouseDownAt) - e.clientX,
        maxDelta = window.innerWidth / 2;

      const percentage = (mouseDelta / maxDelta) * -100,
        nextPercentageUnconstrained = parseFloat(track.dataset.prevPercentage) + percentage,
        nextPercentage = Math.max(Math.min(nextPercentageUnconstrained, 0), -100);

      track.dataset.percentage = nextPercentage;

      track.animate({
        transform: `translate(${nextPercentage}%, -50%)`
      }, { duration: 1200, fill: "forwards" });

      for (const image of track.getElementsByClassName("image")) {
        image.animate({
          objectPosition: `${100 + nextPercentage}% center`
        }, { duration: 1200, fill: "forwards" });
      }
    }


    window.onmousedown = e => handleOnDown(e);

    window.ontouchstart = e => handleOnDown(e.touches[0]);

    window.onmouseup = e => handleOnUp(e);

    window.ontouchend = e => handleOnUp(e.touches[0]);

    window.onmousemove = e => handleOnMove(e);

    window.ontouchmove = e => handleOnMove(e.touches[0]);
  </script>
<style>
  /* Base styles are already defined. These media queries refine responsiveness. */

  /* Small devices (max-width: 480px) */
  @media (max-width: 480px) {
    .hero h1 {
      font-size: 2rem;
    }

    .hero p {
      font-size: 0.9rem;
    }

    .hero button {
      font-size: 0.8rem;
      padding: 0.6rem 1rem;
    }

    .section h2 {
      font-size: 1.5rem;
    }

    .section p {
      font-size: 0.75rem;
    }

    .section img {
      width: 90%;
    }

    #image-track .image {
      width: 80vmin;
    }

    .navbar-toggle {
      font-size: 1.5rem;
      padding: 10px;
    }

    .nav-links {
      font-size: 0.9rem;
    }

    .logo img {
      height: 18vw;
    }
  }

  /* Medium devices (max-width: 768px) */
  @media (max-width: 768px) {
    .hero h1 {
      font-size: 2.5rem;
    }

    .hero p {
      font-size: 1rem;
    }

    .hero button {
      font-size: 0.9rem;
      padding: 0.8rem 1.5rem;
    }

    .section h2 {
      font-size: 2rem;
    }

    .section p {
      font-size: 0.9rem;
    }

    #image-track .image {
      width: 60vmin;
    }

    .footer {
      font-size: 0.8rem;
      padding: 1rem;
    }

    .logo img {
      height: 15vw;
    }

    .nav-links {
      flex-direction: column;
      align-items: center;
    }

    .dropdown-content {
      position: static;
      width: 100%;
    }
  }

  /* Large devices (max-width: 1200px) */
  @media (max-width: 1200px) {
    .hero h1 {
      font-size: 3rem;
    }

    .hero p {
      font-size: 1.1rem;
    }

    .hero button {
      font-size: 1rem;
      padding: 1rem 2rem;
    }

    .section h2 {
      font-size: 2.2rem;
    }

    .section p {
      font-size: 1rem;
    }

    #image-track .image {
      width: 50vmin;
    }

    .logo img {
      height: 12vw;
    }
  }

 /* Media Queries for Specific Classes and Elements */
html {
  overflow-x: hidden;
}

</style>
</html>
"""

# Define the media query breakpoints
breakpoints = {
    "xl": "(min-width: 1200px)",          # Extra large devices
    "lg": "(min-width: 992px) and (max-width: 1199.98px)",  # Large devices
    "md": "(min-width: 768px) and (max-width: 991.98px)",   # Medium devices
    "sm": "(min-width: 576px) and (max-width: 767.98px)",   # Small devices
    "xs": "(max-width: 575.98px)"         # Extra small devices
}

# Parse the HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Extract unique tags
tags = set([tag.name for tag in soup.find_all(True)])

# Extract unique classes
classes = set(
    cls for tag in soup.find_all(True) if tag.get("class") for cls in tag.get("class")
)

# Extract unique IDs
ids = set(tag.get("id") for tag in soup.find_all(True) if tag.get("id"))

# Prepare CSS content
css_content = []

# Add media queries for each type
for name, query in breakpoints.items():
    css_content.append(f"/* {name.upper()} Media Query: {query} */")
    css_content.append(f"@media {query} {{")
    
    # Add tags
    for tag in tags:
        css_content.append(f"  {tag} {{}}")
    
    # Add classes
    for cls in classes:
        css_content.append(f"  .{cls} {{}}")
    
    # Add IDs
    for id_name in ids:
        css_content.append(f"  #{id_name} {{}}")
    
    css_content.append("}\n")

# Write CSS to a file
output_file = "responsive.txt"
with open(output_file, "w") as f:
    f.write("\n".join(css_content))

print(f"CSS file with media queries created: {output_file}")
