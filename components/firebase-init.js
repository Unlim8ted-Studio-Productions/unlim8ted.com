import { initializeApp, getApps } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyC8rw6kaFhJ2taebKRKKEA7iLqBvak_Dbc",
  authDomain: "unlim8ted-db.firebaseapp.com",
  projectId: "unlim8ted-db",
  storageBucket: "unlim8ted-db.appspot.com",
  messagingSenderId: "1059428499872",
  appId: "1:1059428499872:web:855308683718237de6e4c5",
};

export function getFirebase() {
  const app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
  return {
    app,
    auth: getAuth(app),
    db: getFirestore(app),
  };
}
