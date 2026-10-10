# Cloudflare Workers + Resend notifications

Help and Contact submit to `https://api.unlim8ted.com/contact`. The API Worker saves each valid message in the `SUPPORT_DO` Durable Object before confirming receipt. Its alarm sends a plain-text email through Resend's HTTP API. Failed attempts reuse the same idempotency key. There is no Firebase Functions or Firestore dependency for sending.

## Production setup

The Cloudflare Worker is named `api`. Its connected backend repository uses `api/` as the build root and `npx wrangler deploy` as the deploy command. `backend/api/wrangler.jsonc` is the live config for that build. It retains the existing `MEMORY` KV namespace, the `api.unlim8ted.com` custom domain, and dashboard runtime variables via `keep_vars: true`. Its first Git deployment registers `PAYMENT_DO`, `STOCK_DO`, and `SUPPORT_DO` as SQLite Durable Objects.

The committed config also defines these public runtime variables:

```text
NOTIFICATION_FROM=notifications@unlim8ted.com
SUPPORT_NOTIFICATION_TO=support@unlim8ted.com
NOTIFICATION_DAILY_LIMIT=90
NOTIFICATION_ORIGINS=https://unlim8ted.com,https://www.unlim8ted.com
```

Verify `unlim8ted.com` as a sender in Resend. The destination can remain your Zoho inbox. The daily limit counts attempts, including retries.

The screenshot confirms `notifications_key` exists under the Worker's **runtime** Variables and Secrets. Check that it holds a new Resend API key; replace the key exposed in chat. Build variables on **Settings > Builds** are unavailable to the running Worker. Never place the key in source code or `vars`.

Push the backend repository commit to `main` to trigger the connected Cloudflare build. Check the build log, confirm all four bindings (`MEMORY` and the three Durable Objects), then test `/contact`. For a manual deploy, run `npx.cmd wrangler deploy` from `backend/api` after authenticating Wrangler. Deploy the Worker before publishing the updated form handler; a missing binding or secret returns HTTP 503 and leaves the message in the form for retry.

The backend is a local Git submodule of the website repository. Push its commit first; then update and publish the parent repository pointer.

## Testing and recovery

Submit one Help request and one Contact message. Check your inbox, Resend's Emails page, and the Reply-To address. Browser confirmation means Cloudflare saved the submission. `sent` means Resend accepted it; Resend's email events confirm delivery or bounce outcomes.

Each job is saved under the storage key `job` in its per-request Durable Object. Inspect Cloudflare's Durable Object storage with administrative access. No public endpoint exposes messages. Provider failures are saved as codes such as `RESEND_401`; the key and provider response body are not persisted.

Limits allow ten submissions per ten minutes per IP and a shared daily sending-attempt budget. Jobs waiting for quota remain pending. Ambiguous provider attempts older than 23 hours become `needs_review`, before Resend's 24-hour idempotency window expires. Check Resend's logs before manually resending. Successful jobs remain durable; choose a retention policy before deleting them.

```powershell
node --test backend/api/tests/support-notifications.test.js
```

Tests use fake provider requests. Live delivery was not enabled in this session.

References: [Resend Send Email API](https://resend.com/docs/api-reference/emails/send-email), [Resend idempotency](https://resend.com/docs/dashboard/emails/idempotency-keys), [Cloudflare alarms](https://developers.cloudflare.com/durable-objects/api/alarms/).
