import { initializeApp, getApps } from
  "https://www.gstatic.com/firebasejs/9.22.2/firebase-app.js";
import { getAuth } from
  "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";
import { getFirestore } from
  "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";
import {
  initializeAppCheck,
  ReCaptchaV3Provider
} from
  "https://www.gstatic.com/firebasejs/9.22.2/firebase-app-check.js";

const firebaseConfig = {
  apiKey: "AIzaSyC8rw6kaFhJ2taebKRKKEA7iLqBvak_Dbc",
  authDomain: "unlim8ted-db.firebaseapp.com",
  projectId: "unlim8ted-db",
  storageBucket: "unlim8ted-db.appspot.com",
  messagingSenderId: "1059428499872",
  appId: "1:1059428499872:web:855308683718237de6e4c5",
};

// 🔹 Create (or reuse) the app ONCE
const app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);

// 🔹 Initialize App Check ONCE, immediately after app creation
initializeAppCheck(app, {
  provider: new ReCaptchaV3Provider(
    "6LcuUjosAAAAAGpB2d7rGDV7RsADBtFpJA_6uACt"
  ),
  isTokenAutoRefreshEnabled: true
});

// 🔹 Create services AFTER App Check
const auth = getAuth(app);
const db = getFirestore(app);

// 🔹 Export a stable accessor
export function getFirebase() {
  return { app, auth, db };
}