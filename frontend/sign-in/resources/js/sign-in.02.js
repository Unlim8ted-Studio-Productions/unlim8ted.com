import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-app.js";
    import {
      getAuth,
      GoogleAuthProvider,
      signInWithPopup,
      signInWithEmailAndPassword,
      createUserWithEmailAndPassword,
      setPersistence,
      browserLocalPersistence,
      browserSessionPersistence,
      onAuthStateChanged,
    } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";
    import { getFirestore, doc, setDoc, serverTimestamp } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";

    const firebaseConfig = {
      apiKey: "AIzaSyC8rw6kaFhJ2taebKRKKEA7iLqBvak_Dbc",
      authDomain: "unlim8ted-db.firebaseapp.com",
      projectId: "unlim8ted-db",
      storageBucket: "unlim8ted-db.appspot.com",
      messagingSenderId: "1059428499872",
      appId: "1:1059428499872:web:855308683718237de6e4c5",
    };

    const app = initializeApp(firebaseConfig);
    const auth = getAuth(app);
    const db = getFirestore(app);

    const redirectToProfile = () => (window.location.href = "https://unlim8ted.com/profile");

    // If already signed in, bounce to profile (optional but nice)
    onAuthStateChanged(auth, (user) => {
      if (user) redirectToProfile();
    });

    async function addUserToFirestore(user){
      const userRef = doc(db, "users", user.uid);
      await setDoc(userRef, {
        email: user.email,
        name: user.displayName || null,
        createdAt: serverTimestamp(),
      }, { merge: true });
    }

    async function applyPersistenceFromCheckbox(){
      const remember = document.getElementById("rememberMe")?.checked ?? true;
      await setPersistence(auth, remember ? browserLocalPersistence : browserSessionPersistence);
    }

    async function handleGoogleSignIn(){
      const provider = new GoogleAuthProvider();
      try{
        await applyPersistenceFromCheckbox();
        const result = await signInWithPopup(auth, provider);
        await addUserToFirestore(result.user);
        redirectToProfile();
      }catch(err){
        console.error("Google Sign-In error:", err);
        alert("Failed to sign in with Google.");
      }
    }

    async function handleEmailSignIn(email, password){
      await applyPersistenceFromCheckbox();
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      await addUserToFirestore(userCredential.user);
      redirectToProfile();
    }

    async function handleAccountCreation(email, password){
      await applyPersistenceFromCheckbox();
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      await addUserToFirestore(userCredential.user);
      redirectToProfile();
    }

    let isCreateMode = false;

    function setMode(createMode){
      isCreateMode = createMode;

      document.getElementById("form-mode").textContent = createMode ? "Create Account" : "Sign In";
      document.getElementById("form-submit").textContent = createMode ? "Create Account" : "Sign In";
      document.getElementById("toggle-text").textContent = createMode
        ? "Already have an account? Sign In."
        : "Don't have an account? Create one.";
    }

    function toggleMode(){
      setMode(!isCreateMode);
    }

    window.addEventListener("DOMContentLoaded", () => {
      document.getElementById("google-signin-btn").addEventListener("click", handleGoogleSignIn);
      document.getElementById("toggle-text").addEventListener("click", toggleMode);

      // Use the FORM submit event (more reliable than button click)
      document.getElementById("email-signin-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const email = document.getElementById("email").value.trim();
        const password = document.getElementById("password").value;

        try{
          if (isCreateMode) {
            await handleAccountCreation(email, password);
          } else {
            await handleEmailSignIn(email, password);
          }
        }catch(err){
          console.error("Auth error:", err);
          alert(isCreateMode
            ? "Failed to create account. Try again."
            : "Sign-in failed. Check your email and password."
          );
        }
      });

      setMode(false);

      // Footer year
      const y = new Date().getFullYear();
      document.getElementById("footer-text").innerHTML =
        `&copy; 2019-${y} Unlim8ted Studio Productions. All rights reserved.`;
    });
