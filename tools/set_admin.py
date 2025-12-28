import firebase_admin
from firebase_admin import credentials, auth

# Initialize the Firebase Admin SDK
cred = credentials.Certificate(r"")
firebase_admin.initialize_app(cred)


# Function to set a user as admin
def set_admin(uid):
    try:
        # Set the custom claim "admin" for the specified user
        auth.set_custom_user_claims(uid, {"admin": True})
        print(f"User {uid} has been granted admin privileges.")
    except Exception as e:
        print(f"Error setting admin: {e}")


user_uids = ["", "", ""]
for id in user_uids:
    set_admin(id)

users = auth.list_users().users
for user in users:
    if user.custom_claims and user.custom_claims.get("admin"):
        print(f"Admin: {user.uid}")
