# Help and contact email notifications

Help and Contact use the same form handler and save to `contact_messages`. A Firebase function queues a notification and sends it to `SUPPORT_NOTIFICATION_TO`, which defaults to `support@unlim8ted.com`. The visitor's email is the Reply-To address. Messages are plain text; visitors cannot select the destination, attach files, or create notification jobs.

The sender uses your existing SMTP provider. It requires TLS on port 465 or 587. Credentials stay in Firebase Secret Manager, never in the browser, public assets, or a committed environment file. The sender must be an address your provider allows you to send from.

Successful jobs are recorded and skipped on duplicate Firestore events. Failed sends remain queued with a retry time and an error code. A scheduled function processes due jobs every ten minutes and recovers worker crashes after a two-minute lease. The default limit is 100 sending attempts per hour, shared across both forms; excess jobs wait for the next hour. Only the destination you configure receives notifications. There is no automated reply to visitors or newsletter notification in this change.

## Enable it

These changes are prepared locally; email delivery has not been enabled or tested against a real mailbox. Firebase deployment requires an authenticated project, billing that supports Cloud Functions and Cloud Scheduler, and permission to use Secret Manager. Do not put SMTP credentials in this chat.

1. Commit the changes inside the `backend` submodule and update its commit in the website repository.
2. Install dependencies with `npm ci --prefix backend/functions --ignore-scripts`.
3. Copy `backend/functions/.env.example` to `backend/functions/.env.unlim8ted-db`. Set the real `SMTP_HOST`, port, allowed `NOTIFICATION_FROM`, destination, and optional hourly limit. This environment file is ignored by Git.
4. Run the following from the website repository. The secret commands prompt securely for their values.

```text
firebase functions:secrets:set SMTP_USERNAME --project unlim8ted-db
firebase functions:secrets:set SMTP_PASSWORD --project unlim8ted-db
firebase deploy --config firebase.json --only firestore:indexes,firestore:rules,functions:notifications --project unlim8ted-db
```

Use the actual Firestore database region in `backend/functions/index.js` if it differs from `us-central1`. Keep any existing production Firestore indexes when merging the new composite index configuration.

Deploy the frontend after the functions and index are ready. Submit one Help request and one Contact message using a controlled test address, confirm both arrive in the intended inbox, and confirm Reply-To uses that address. This is the first live delivery test; local tests never send email.

## Check delivery

In Firebase Console, inspect `email_notifications`. Each job links to its `contactId` and shows `pending`, `sent`, or `discarded`. `lastErrorCode` helps diagnose sending failures without storing credentials or duplicating the message body. For pending jobs, check SMTP credentials, the sender's permissions, TLS, provider sending limits, and that the scheduled retry function is running.

The form's confirmation means Firebase received the submission. It does not claim an email has already reached the inbox. The original message stays saved even if the provider is unavailable. Do not delete pending job records to retry them; the scheduler already retries them.

SMTP cannot guarantee exactly-once delivery if a connection or worker fails after the provider accepts the message but before the successful state is saved. The stable Message-ID helps identify or thread those rare duplicates. The persistent lease and completion record prevent routine duplicate events from sending multiple emails.

Disabling the notification functions stops new sends but leaves saved messages and jobs available for support. To change SMTP credentials, set the secrets again and redeploy the notification functions.

The implementation uses Firebase's [Firestore triggers](https://firebase.google.com/docs/functions/firestore-events) and [secret parameters](https://firebase.google.com/docs/functions/config-env), with [Nodemailer's SMTP transport](https://nodemailer.com/smtp).
