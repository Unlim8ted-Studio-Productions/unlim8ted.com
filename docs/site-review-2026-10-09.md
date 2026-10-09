# Site repair and release notes — October 9, 2026

## What changed

Checkout now routes each quote through a SQLite Durable Object. The exact Square request and its idempotency key are saved before charging, alongside a recovery alarm in a storage transaction. Concurrent submissions are serialized. Payment and fulfillment records do not expire; quotes and polling caches still have their short TTLs. Pending transactions are checked with Square, and charged orders retry Printful with increasing delays. Fulfillment first looks up the quote's external ID to recover a successful create whose response was lost. Existing drafts are confirmed rather than counted as fulfilled.

The Firestore rules deny all client purchase writes, merchant self-verification, and self-publishing or verification of merchandise. Profile updates use `affectedKeys`, so adding or deleting protected fields is also denied. Verified merchants cannot change their website while retaining its proof. Changing a verified domain needs trusted administrative code to reset verification first.

Website verification uses DNS TXT records only. HTTP-file verification was disabled rather than retaining an arbitrary URL fetch that could reach internal destinations or follow redirects. A successful check updates the merchant through a service account and a Firestore update-time precondition. The browser no longer writes `websiteVerified`.

The download endpoint requires authentication, a server-verified purchase, and a completed, unrefunded Square transaction with matching order, merchant, currency, and sufficient amount. Files come from a private R2 binding. Legacy client-created purchase records are rejected. There are currently no paid digital items in the catalog; this is a guarded delivery endpoint, not a new digital checkout product.

The phone browser accepts location messages only from its current iframe, from the same origin, with bounded HTTP(S) URLs and a valid schema. API CORS no longer reflects arbitrary origins.

Public-page repairs include error images, asset slashes, broken internal links, metadata, main landmarks, skip links, checkout labels, focus handling, native WiseSize controls, and reduced-motion CSS. Browser checks also drove fixes for mobile overflow, navigation semantics, filter labels, the phone's status icon, keyboard-accessible chess squares, and contrast.

Text to Speech now uses browser speech synthesis with voice selection, stop controls, and error feedback. Text Translator is an honest development page excluded from indexing. WiseSize's duplicate imports and mixed Firebase SDK versions were repaired. The wheel accepts either catalog shape, handles empty catalogs, and finishes even if `transitionend` never fires. Newsletter writes now match the authorized schema and confirmation follows the completed write.

The homepage features projects earlier, adds direct actions and status labels, and shows real counter values without JavaScript. Its main 52 KB stylesheet was extracted into a cacheable file, offscreen images are lazy-loaded, and ONNX Runtime loads only when an AI session needs it. Music visuals use the audio clock and a consistent time window across screen sizes, with reduced-motion support. TimeCat's Unity demo loads when opened rather than on every visit.

About, Help, and the development notes were rewritten with concrete projects. Square Pixels consistently uses its plural name and separates the checklist's implemented work from unfinished and planned features. Software listings explain their purpose; repository README files remain the authority for unverified installation requirements. Podcasts now have real playback buttons, file-derived durations, and an unavailable label for the empty Gusian Language audio file. Scrollys explain their interaction and variable runtime. Physical Items explains purchasing rather than exposing inventory fields. Privacy names the actual checkout and storage services; Terms identifies prototype availability and checkout expectations. Dates are fixed editorial dates rather than today's date on every visit.

The publishing generator validates IDs, escapes JSON for script contexts, refreshes its own generated pages, preserves hand-authored pages, validates sitemap targets, excludes noindex routes, and deduplicates repeated runs. Shared category styles were consolidated with compatibility URLs. The public staging step excludes the unused 32.8 MB Detective Edward editing directory. The supplied private PDF was never copied into the repository or published; public mentions use only the upcoming-film fact provided in the request.

## Validation

- 17 backend tests: duplicate requests, lost payment/fulfillment responses, restarts, pending-payment reconciliation, provider outages, durable retention, expired quotes, quantity bounds, missing bindings, SSRF restrictions, cross-origin requests, and download authorization.
- Nine Firebase emulator tests: authorized and unauthorized reads/writes, purchases, protected fields, merchant verification and publication, carts, newsletter payloads, account creation, sign-in, sign-out, password reset, and invalidated sessions.
- Browser smoke and axe checks on 22 routes at 360, 768, and 1440 pixels, plus a guest cart reload/quantity test, checkout/donation focus tests, wheel interaction, and speech-field validation. App Check is mocked and production writes are blocked in these tests.
- Structural HTML validation and JavaScript syntax checking on the staged site. The final reports are under `tools/reports/`.
- Offline published-link/asset audit: zero findings. This covers HTML attributes and locally mirrored assets; it is not a crawl of every remote repository, video, or dynamically computed URL.
- Running the sitemap generator twice produces the same sitemap hash.

The dependency audit reports no production dependency advisories. Seven development-tool advisories remain in the Firebase CLI dependency tree; they are not shipped in the public website. Do not apply the audit tool's suggested breaking downgrades without testing the emulator and CLI.

## Build and release

The backend directory is a separate Git submodule. Its edits and new files must be committed in that repository, then its commit must be recorded in the parent repository. A frontend-only commit will not deliver the payment or rules fixes. CI checks out the submodule; private repositories require a `BACKEND_READ_TOKEN` with read access to the parent and backend repositories. Deployment is now gated on validation.

Local prerequisites: Node 24, Python 3.12, PowerShell, Java 21 for the Firebase emulators, and Playwright Chromium. The local Java download used during testing is ignored by Git.

```text
npm ci --ignore-scripts
npm test
npm run test:rules
python tools/prepare_publish.py --output .site-preview
powershell -NoProfile -ExecutionPolicy Bypass -File tools/generate_product_pages_and_sitemap.ps1 -FrontendDir .site-preview
python tools/audit_site.py --frontend .site-preview --output tools/reports/site-audit.json --strict
npm run test:html
npm run test:syntax
npx playwright install chromium
npm run test:browser
```

On Linux CI, invoke the generator with `pwsh` instead of Windows PowerShell.

Before enabling checkout, merge the `PAYMENT_DO` binding and the appended `new_sqlite_classes` migration from `backend/api/wrangler.example.jsonc` into the actual Worker configuration. Keep existing namespace IDs, `STOCK_DO`, and all existing migration history. The Worker deliberately returns 503 for payments if `PAYMENT_DO` is absent. Preserve the existing `MEMORY` and `VERIFY_KV` bindings, provider secrets, GitHub catalog credentials, and tax configuration.

Add `FIREBASE_SERVICE_ACCOUNT` as a Worker secret containing the service-account JSON, and set `FIREBASE_PROJECT_ID` and `FIREBASE_WEB_API_KEY` to the intended project. Grant only the IAM permissions needed for merchant verification. Set `ALLOWED_ORIGIN` to the comma-separated approved storefront origins; a wildcard is not accepted.

Deploy the root `firebase.json` rules configuration separately from the existing `auth/firebase.json` hosting configuration. The website's existing hosting workflow does not deploy Firestore rules. Before releasing, compare the active rules with `backend/rules/database.rules`, then run the same allow/deny cases against a controlled production test account.

For paid downloads, configure the private `DOWNLOADS` R2 binding and a server-only `DIGITAL_DOWNLOADS` map of item IDs to `{key, priceCents}`. Trusted purchase creation must set `verificationVersion: 1`, `verifiedAt`, `itemId`, `orderId`, and `transactionId` after checking payment and ownership. Do not expose paid file URLs through public catalog data. Existing forged or unverified purchase records need review; do not convert them into entitlements automatically.

Drain the legacy checkout before switching Workers, and review recent Square transactions independently. Old KV records that already expired cannot be reconstructed from this checkout code. New requests refuse to recharge quotes with surviving legacy payment records.

## Recovery and rollback

For a processing order, use its quote reference to query `/payment-status`. `status: paid` and `fulfillment: retrying` means the charge succeeded but fulfillment needs recovery. Alarms continue checking with increasing delays, capped at six hours. `needsReview` becomes true after 20 attempts; configure operational alerting for this and the Worker recovery logs. There is no new email notification service.

Before manually submitting an order, check Square's payment and Printful's external-ID lookup. Never generate a new Square key to recover the same quote. For a permanent supplier problem, resolve fulfillment or refund the verified payment through the provider and record the decision. Refunds are an operator action; this code does not issue them automatically.

Rollback the frontend independently. Retain the hardened rules, payment object class, bindings, stored orders, and alarms when rolling back the Worker; the previous KV-only payment implementation is unsafe for retries. Disable `/pay` rather than returning to that implementation. Keep durable-order storage and provider receipts available for reconciliation.

API behavior follows [Square's payment idempotency contract](https://developer.squareup.com/reference/square/payments-api/CreatePayment), [Cloudflare's durable storage transactions and alarms](https://developers.cloudflare.com/durable-objects/api/sqlite-storage-api/), and [Printful's external-ID order lookup](https://developers.printful.com/docs/).

## Work requiring production access or further evidence

Production was not deployed or authenticated in this session. The Firebase CLI had no configured account; there was no production Worker configuration or credential connection to inspect. Live Square/Printful charges, existing lost orders, live stock, and deployed authorization remain unverified. Browser smoke checks are not a substitute for a sandbox provider checkout and a monitored production release.

Search Console coverage, production robots/sitemap delivery, alternate-host and trailing-slash redirects, social preview caching, mobile field Core Web Vitals, low-end-device profiling, and live external downloads need follow-up with the deployed site and relevant accounts. No GitHub issues were closed.

Manual screen-reader testing, touch-device gameplay, full game mechanics, 200%/400% browser zoom, real Google sign-in, email delivery, token expiry timing, and every application's installation have not been certified. The narrow browser checks must not be described as a complete accessibility audit.

Do not invent supported software versions, release dates, film runtimes, streaming-platform URLs, or episode publication dates. Several entries have no source for these details. Terms still needs the owner's decisions on governing jurisdiction, dispute procedures, refund/return rules, and account closure. Privacy still needs an agreed retention schedule and confirmed analytics practices. The source wording is not a verified legal policy.

The duplicate-file report records remaining exact matches. Firmware and embed copies were retained because they have separate delivery contexts; remaining unused legacy scripts can be removed once their external references are established. The site's inline scripts prevent a strict nonce-free CSP today; a restrictive global policy needs a separate tested migration. Static caching headers on GitHub Pages and provider configuration were not changed.
