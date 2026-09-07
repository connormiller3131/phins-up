/**
 * Serves docs/ via the ASSETS binding, but forces a real no-cache header on
 * every response -- confirmed the docs/_headers convention (a classic
 * Cloudflare Pages feature) is NOT honored by this newer Workers Assets
 * platform: a real response from phinsup.net came back with the platform's
 * own default `Cache-Control: public, max-age=0, must-revalidate` and
 * `cf-cache-status: HIT`, serving a stale build a full day old. Setting the
 * header explicitly in a Worker script is the reliable way to control this,
 * since it doesn't depend on whichever convention file the serving platform
 * happens to support this month. This site regenerates twice daily, so any
 * caching here is actively wrong.
 *
 * Also serves index.html for any path that isn't a real static asset (e.g.
 * /mlb/monday) -- this is a single-page app with client-side routing
 * (dashboard_live.html's applyRouteFromLocation), so a direct or
 * bookmarked deep link needs the actual page content back, not a real 404,
 * for the page's own JS to read the URL and switch to the right view. This
 * site has no other real static assets to accidentally shadow (everything
 * is inlined into the one HTML file), so a blanket "any 404 -> index.html"
 * fallback is safe here.
 *
 * Third job: serve the gated payload (player props and everything derived
 * from them) from KV at /api/gated. That data is deliberately not baked
 * into docs/index.html, because docs/ is committed to a public repo that
 * GitHub Pages serves in its own right -- inlining it would publish it no
 * matter what any check here did. Routing it through the Worker gives the
 * paid tiers a single door to lock.
 *
 * Fourth job: accounts and billing. Clerk session verification backs
 * /api/whoami; Stripe checkout, the billing portal and the subscription
 * webhook decide which of the two payloads above a visitor receives.
 */

const GATED_KEY = "current";
// The free-account half: the same shape, but carrying only the props the pick
// cards actually reference. Two keys rather than one payload sliced per
// request, because slicing would mean parsing several megabytes of JSON on
// every call; this way the right one is chosen and streamed untouched.
const ACCOUNT_KEY = "current:account";

// Clerk's Frontend API for the PRODUCTION instance, decoded from the
// publishable key (pk_live_<base64 of "clerk.phinsup.net$">). It is also
// the `iss` every session token must carry.
const CLERK_ISSUER = "https://clerk.phinsup.net";
const CLERK_JWKS_URL = `${CLERK_ISSUER}/.well-known/jwks.json`;

// Enforced. The rollout ran in two steps on purpose: /api/whoami shipped
// first with this false and was confirmed returning signedIn:true for a real
// signed-in session against the live Worker, before anything depended on it.
// identify() is the same code path in both endpoints, so that check exercised
// exactly what now guards the payload. Setting this back to false unlocks the
// site for everyone again without touching any other logic.
const GATE_ENFORCED = true;

// --- Clerk session tokens ---------------------------------------------------
// Verified against Clerk's public JWKS, so no secret key is involved and none
// needs to live in this Worker. The token arrives in an Authorization header
// rather than a cookie: Clerk development instances hand out sessions on a
// *.clerk.accounts.dev domain, and a header sidesteps every cross-domain
// cookie question in both dev and production.

let jwksCache = null;
let jwksFetchedAt = 0;
const JWKS_TTL_MS = 10 * 60 * 1000;

async function fetchJwks() {
  const r = await fetch(CLERK_JWKS_URL, { cf: { cacheTtl: 300 } });
  if (!r.ok) throw new Error(`jwks ${r.status}`);
  jwksCache = await r.json();
  jwksFetchedAt = Date.now();
  return jwksCache;
}

async function jwkForKid(kid) {
  if (!jwksCache || Date.now() - jwksFetchedAt > JWKS_TTL_MS) await fetchJwks();
  let jwk = (jwksCache.keys || []).find((k) => k.kid === kid);
  // A kid we have never seen usually means Clerk rotated signing keys, so
  // refetch once before rejecting rather than failing everyone until the TTL
  // happens to lapse.
  if (!jwk) {
    await fetchJwks();
    jwk = (jwksCache.keys || []).find((k) => k.kid === kid);
  }
  return jwk || null;
}

function b64urlToBytes(s) {
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(s.length / 4) * 4, "="));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function b64urlToJson(s) {
  return JSON.parse(new TextDecoder().decode(b64urlToBytes(s)));
}

/**
 * Returns the token's claims when every check passes, otherwise a {error}
 * describing the first failure. Checks, and why each one is here:
 *   - alg pinned to RS256. Accepting whatever the header asks for is the
 *     classic JWT break: "none" skips signing entirely, and HS256 lets an
 *     attacker sign with the public key as the shared secret.
 *   - signature verified against the JWKS key named by kid.
 *   - exp, so an old token cannot be replayed forever.
 *   - nbf, for tokens minted slightly ahead of us.
 *   - iss must be this exact Clerk instance, so a valid token from some
 *     other Clerk application is not accepted here.
 *   - sub must exist; it is the user id everything downstream keys on.
 * A small clock skew allowance keeps a correct token from being rejected
 * because two machines disagree by a second.
 */
async function verifyClerkToken(token) {
  const parts = (token || "").split(".");
  if (parts.length !== 3) return { error: "malformed token" };
  const [h, p, sig] = parts;

  let header, claims;
  try {
    header = b64urlToJson(h);
    claims = b64urlToJson(p);
  } catch {
    return { error: "undecodable token" };
  }

  if (header.alg !== "RS256") return { error: `unexpected alg ${header.alg}` };
  if (!header.kid) return { error: "no kid" };

  let jwk;
  try {
    jwk = await jwkForKid(header.kid);
  } catch (e) {
    return { error: `jwks unavailable: ${e.message}` };
  }
  if (!jwk) return { error: "unknown signing key" };

  const key = await crypto.subtle.importKey(
    "jwk",
    { kty: jwk.kty, n: jwk.n, e: jwk.e, alg: "RS256", ext: true },
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["verify"],
  );
  const ok = await crypto.subtle.verify(
    "RSASSA-PKCS1-v1_5",
    key,
    b64urlToBytes(sig),
    new TextEncoder().encode(`${h}.${p}`),
  );
  if (!ok) return { error: "bad signature" };

  const now = Math.floor(Date.now() / 1000);
  const SKEW = 5;
  if (typeof claims.exp !== "number" || claims.exp + SKEW < now) return { error: "expired" };
  if (typeof claims.nbf === "number" && claims.nbf - SKEW > now) return { error: "not yet valid" };
  if (claims.iss !== CLERK_ISSUER) return { error: "wrong issuer" };
  if (!claims.sub) return { error: "no subject" };

  return { claims };
}

function bearerToken(request) {
  const auth = request.headers.get("Authorization") || "";
  return auth.startsWith("Bearer ") ? auth.slice(7).trim() : null;
}

async function identify(request) {
  const token = bearerToken(request);
  if (!token) return { signedIn: false, reason: "no token" };
  const res = await verifyClerkToken(token);
  if (res.error) return { signedIn: false, reason: res.error };
  return { signedIn: true, userId: res.claims.sub, expiresAt: res.claims.exp };
}

const NO_STORE = { "Cache-Control": "no-store" };

// --- Stripe -----------------------------------------------------------------
// Entitlements live in KV, not D1: an entitlement is one key-value lookup per
// user, not a relational query, and the KV namespace already exists. Keys are
// prefixed so they cannot collide with the gated payload's own "current" key.
//   ent:<clerk user id>   -> the subscription record below
//   cust:<stripe cust id> -> clerk user id, a reverse lookup used only as a
//                            fallback when a webhook arrives without metadata
const STRIPE_API = "https://api.stripe.com/v1";
// LIVE mode price ids. These are mode-scoped: the test-mode ids that came
// before them do not exist here, and vice versa, so this file and the
// STRIPE_SECRET_KEY secret have to change in the same deploy or checkout
// fails with "No such price".
const PRICE_IDS = {
  monthly: "price_1UCpqQHo8gAhueTWLhfT4Oaa",
  yearly: "price_1UCpqSHo8gAhueTW8mvqrfLY",
};
// Stripe statuses that actually grant access. past_due is deliberately absent:
// a failed renewal should lose access, and Stripe retries for days before
// giving up, which would otherwise be days of unpaid access.
const ACTIVE_STATUSES = new Set(["active", "trialing"]);

// Accounts with permanent access and no Stripe subscription behind them --
// the owner, and any comp given out later. Deliberately code, not a KV
// record: reconcile treats Stripe as the source of truth and would revoke a
// hand-written entitlement within ten minutes for having no subscription.
//
// Safe to commit to a public repo. A Clerk user id identifies an account but
// does not authenticate one; using it still requires a session token signed
// by Clerk, which nobody else can mint.
const COMP_USER_IDS = new Set([
  "user_3IwWOrT9zXWv6gaUIdJKRSkFecU",   // Connor - site owner
  "user_3Iyh6kdAQ7CMH8rso3SOXZGEcoz",   // comp - friend, 2026 season
]);

// Comp can also be granted WITHOUT a deploy by writing a KV key named
// `comp:<clerk user id>` (any value -- the name is the grant; a short note
// like "friend, 2026 season" is useful for remembering why later). Checked
// before any entitlement lookup, so it is unaffected by reconcile: there is
// no subscription behind it and none is expected.
//
// Revoking is deleting the key. Both are one action in the Cloudflare
// dashboard, which is the point -- comping a friend should not require a
// code change.
async function hasComp(env, userId) {
  if (!userId) return false;
  if (COMP_USER_IDS.has(userId)) return true;
  if (!env.GATED) return false;
  return (await env.GATED.get(`comp:${userId}`)) !== null;
}

function formEncode(pairs) {
  return pairs.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");
}

/**
 * Retrieval. This exists because stripeApi is a POST helper, and POSTing to
 * /v1/subscriptions/<id> is not a read -- it is an UPDATE with an empty body.
 * It returns 200 and the right object, so it looked correct in the code and
 * in the logs, but it means every "fetch the subscription" was writing to a
 * subscription, including ones scheduled to cancel. Reads use GET.
 */
async function stripeGet(env, path) {
  if (!env.STRIPE_SECRET_KEY) throw new Error("Stripe is not configured yet");
  const r = await fetch(`${STRIPE_API}${path}`, {
    headers: { Authorization: `Bearer ${env.STRIPE_SECRET_KEY}` },
  });
  const body = await r.json();
  if (!r.ok) {
    const err = new Error(body?.error?.message || `stripe ${r.status}`);
    // Callers need to tell "this no longer exists" from "Stripe is down".
    err.status = r.status;
    err.code = body?.error?.code || null;
    throw err;
  }
  return body;
}

async function stripeApi(env, path, pairs) {
  // Explicit, so a missing secret reads as "Stripe is not configured" rather
  // than surfacing as an opaque 401 from Stripe's own API.
  if (!env.STRIPE_SECRET_KEY) throw new Error("Stripe is not configured yet");
  const r = await fetch(`${STRIPE_API}${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.STRIPE_SECRET_KEY}`,
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: formEncode(pairs),
  });
  const body = await r.json();
  if (!r.ok) {
    const err = new Error(body?.error?.message || `stripe ${r.status}`);
    err.status = r.status;
    err.code = body?.error?.code || null;
    throw err;
  }
  return body;
}

async function entitlementFor(env, userId) {
  if (!env.GATED || !userId) return null;
  return await env.GATED.get(`ent:${userId}`, { type: "json" });
}

function isPaid(ent) {
  return !!(ent && ACTIVE_STATUSES.has(ent.status));
}

// Constant-time comparison. A plain === on a signature leaks, through timing,
// how many leading bytes an attacker guessed right, which is enough to forge
// one byte at a time.
function timingSafeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

async function hmacHex(secret, message) {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Verifies Stripe's webhook signature over the RAW body. This has to be the
 * exact bytes Stripe sent -- parsing and re-serialising the JSON first would
 * change key order or spacing and the signature would never match.
 * Also enforces a timestamp tolerance, without which a captured webhook could
 * be replayed back at us forever with a signature that still verifies.
 */
async function stripeEventFrom(request, env) {
  const header = request.headers.get("Stripe-Signature") || "";
  const raw = await request.text();
  const parts = Object.fromEntries(
    header.split(",").map((kv) => kv.split("=").map((s) => s.trim())),
  );
  if (!parts.t || !parts.v1) return { error: "malformed signature header" };

  const age = Math.abs(Math.floor(Date.now() / 1000) - Number(parts.t));
  if (!Number.isFinite(age) || age > 300) return { error: "timestamp outside tolerance" };

  const expected = await hmacHex(env.STRIPE_WEBHOOK_SECRET, `${parts.t}.${raw}`);
  if (!timingSafeEqual(expected, parts.v1)) return { error: "bad signature" };

  try {
    return { event: JSON.parse(raw) };
  } catch {
    return { error: "unparseable body" };
  }
}

// Reports what the gate would decide, without gating anything. Exists so the
// token path can be proven against the real deployed Worker before
// GATE_ENFORCED is flipped.
// Webhooks are best effort. A dropped delivery, an event type not subscribed,
// a handler deployed after the fact -- any of these leave the stored record
// permanently disagreeing with Stripe, and nothing ever corrects it because
// nothing else writes. So rather than trusting webhooks alone, this
// reconciles against Stripe: whenever a record is missing a known field, or
// is simply older than RECONCILE_AFTER_MS, it is re-read and rewritten.
// Stripe is the source of truth for billing; KV is a cache of it.
//
// Cost is bounded to roughly one API call per active subscriber per interval,
// since the rewrite refreshes updatedAt.
const RECONCILE_AFTER_MS = 10 * 60 * 1000;

// Fields whose absence means the record predates the code that writes them,
// and so should be refreshed immediately rather than waiting out the cooldown.
// Add to this when adding a field, or existing records keep the old shape
// until the timer happens to lapse -- which is exactly what happened with
// cancelAt: one reconcile ran, refreshed updatedAt, and then the cooldown
// suppressed every later attempt, so a corrected reader never got to run.
const RECONCILE_FIELDS = ["cancelAtPeriodEnd", "currentPeriodEnd", "cancelAt"];

function needsReconcile(ent) {
  if (!ent || !ent.subscriptionId) return false;
  if (RECONCILE_FIELDS.some((f) => ent[f] === undefined)) return true;
  return Date.now() - (ent.updatedAt || 0) > RECONCILE_AFTER_MS;
}

async function healEntitlement(env, userId, ent) {
  if (!needsReconcile(ent)) return ent;
  try {
    const sub = await stripeGet(env, `/subscriptions/${ent.subscriptionId}`);
    const c = cancellationOf(sub);
    const healed = {
      ...ent,
      plan: planForPrice(sub.items?.data?.[0]?.price?.id),
      status: sub.status,
      currentPeriodEnd: periodEndOf(sub),
      cancelAtPeriodEnd: c.cancelling,
      cancelAt: c.cancelAt,
    };
    await writeEntitlement(env, userId, healed);
    return healed;
  } catch (e) {
    // A subscription Stripe says does not exist is not a transient failure,
    // and must not keep granting access. This matters most at the test-to-live
    // switch: every entitlement written against a test subscription would
    // otherwise 404 forever and stay "paid" on the strength of a record
    // pointing at something that no longer exists.
    if (e.status === 404 || e.code === "resource_missing") {
      const dead = { ...ent, status: "canceled", cancelAtPeriodEnd: false,
                     reconcileError: "subscription not found" };
      await writeEntitlement(env, userId, dead);
      return dead;
    }
    // Anything else -- Stripe down, a network blip -- leaves the record alone
    // rather than downgrading a paying customer over a transient error.
    return { ...ent, reconcileError: String(e.message || e).slice(0, 120) };
  }
}

async function serveWhoami(request, env) {
  const who = await identify(request);
  let ent = who.signedIn ? await entitlementFor(env, who.userId) : null;
  if (ent) ent = await healEntitlement(env, who.userId, ent);
  const comp = who.signedIn ? await hasComp(env, who.userId) : false;
  return Response.json(
    {
      ...who,
      enforced: GATE_ENFORCED,
      // So the page can say "US only" up front instead of showing buy
      // buttons that are going to 403 the moment they are clicked.
      country: (request.cf && request.cf.country) || null,
      canSubscribe: !request.cf || !request.cf.country || request.cf.country === "US",
      plan: comp ? "comp" : (ent?.plan || "free"),
      subscriptionStatus: ent?.status || null,
      paid: comp || isPaid(ent),
      currentPeriodEnd: ent?.currentPeriodEnd || null,
      cancelAtPeriodEnd: !!ent?.cancelAtPeriodEnd,
      subscriptionId: ent?.subscriptionId || null,
      cancelAt: ent?.cancelAt || null,
      reconciledAt: ent?.updatedAt || null,
      reconcileError: ent?.reconcileError || null,
    },
    { headers: NO_STORE },
  );
}

// Starts a subscription. The Clerk user id rides along in two places:
// client_reference_id, which comes back on checkout.session.completed, and
// subscription metadata, which is the only one of the two that later
// customer.subscription.* events carry -- without it a cancellation months
// from now could not be matched to an account.
async function serveCheckout(request, env) {
  const who = await identify(request);
  if (!who.signedIn) {
    return Response.json({ error: "sign in first" }, { status: 401, headers: NO_STORE });
  }
  const plan = new URL(request.url).searchParams.get("plan");
  const price = PRICE_IDS[plan];
  if (!price) {
    return Response.json({ error: "unknown plan" }, { status: 400, headers: NO_STORE });
  }

  // Sales are limited to the US for now. This is a tax decision, not a
  // product one: a non-EU seller owes VAT on digital services to EU consumers
  // from the FIRST sale, with no threshold and no grace period, which would
  // mean an OSS registration the day one European subscribes. US
  // economic-nexus thresholds, by contrast, are high enough to be a long way
  // off. Cloudflare resolves the country at the edge, so this costs nothing.
  //
  // Deliberately at checkout only: existing subscribers who travel keep their
  // access, because the obligation attaches to where a sale is made, not to
  // where the page is later read from.
  const country = request.cf && request.cf.country;
  if (country && country !== "US") {
    return Response.json(
      { error: "Subscriptions are currently available in the US only.", country },
      { status: 403, headers: NO_STORE },
    );
  }

  const origin = new URL(request.url).origin;
  const existing = await entitlementFor(env, who.userId);
  const pairs = [
    ["mode", "subscription"],
    ["line_items[0][price]", price],
    ["line_items[0][quantity]", "1"],
    ["client_reference_id", who.userId],
    ["subscription_data[metadata][clerk_user_id]", who.userId],
    ["success_url", `${origin}/?checkout=success`],
    ["cancel_url", `${origin}/?checkout=cancelled`],
  ];
  // Stripe rejects a session asking for automatic tax when Stripe Tax has not
  // been activated on the account, which fails the whole checkout rather than
  // degrading. So this is a variable, not a constant: the code should not
  // assume an external service is configured. Set STRIPE_AUTOMATIC_TAX=true
  // (a plain variable, not a secret) once Stripe Tax is switched on, which
  // has to happen before launch anyway -- the Terms promise tax is added at
  // checkout, and that promise is only true with this enabled.
  if (env.STRIPE_AUTOMATIC_TAX === "true") {
    pairs.push(["automatic_tax[enabled]", "true"]);
  }
  // Reuse the Stripe customer if this account has subscribed before, so a
  // resubscribe does not create a second customer with its own billing history.
  if (existing?.customerId) {
    pairs.push(["customer", existing.customerId], ["customer_update[address]", "auto"]);
  }

  try {
    const session = await stripeApi(env, "/checkout/sessions", pairs);
    return Response.json({ url: session.url }, { headers: NO_STORE });
  } catch (e) {
    // A stored customer id Stripe does not recognise must not block a sale.
    // It happens whenever the record outlives the customer -- most obviously
    // at the test-to-live switch, where every stored customer is a test one
    // and checkout died with "No such customer". Drop the reuse and let
    // Stripe create a fresh customer, then forget the dead id so the next
    // attempt does not repeat the round trip.
    const missing = e.status === 404 || e.code === "resource_missing" ||
                    /No such customer/i.test(e.message || "");
    if (missing && existing?.customerId) {
      await writeEntitlement(env, who.userId, { ...existing, customerId: null });
      const retry = pairs.filter(([k]) => k !== "customer" && k !== "customer_update[address]");
      try {
        const session = await stripeApi(env, "/checkout/sessions", retry);
        return Response.json({ url: session.url }, { headers: NO_STORE });
      } catch (e2) {
        return Response.json({ error: e2.message }, { status: 502, headers: NO_STORE });
      }
    }
    return Response.json({ error: e.message }, { status: 502, headers: NO_STORE });
  }
}

// Stripe's hosted billing portal. This is what actually delivers the
// cancellation the Terms promise ("cancel any time from your account page"),
// rather than us building a billing UI.
async function servePortal(request, env) {
  const who = await identify(request);
  if (!who.signedIn) {
    return Response.json({ error: "sign in first" }, { status: 401, headers: NO_STORE });
  }
  const ent = await entitlementFor(env, who.userId);
  if (!ent?.customerId) {
    return Response.json({ error: "no subscription to manage" }, { status: 404, headers: NO_STORE });
  }
  try {
    const session = await stripeApi(env, "/billing_portal/sessions", [
      ["customer", ent.customerId],
      ["return_url", new URL(request.url).origin + "/"],
    ]);
    return Response.json({ url: session.url }, { headers: NO_STORE });
  } catch (e) {
    return Response.json({ error: e.message }, { status: 502, headers: NO_STORE });
  }
}

async function writeEntitlement(env, userId, record) {
  await env.GATED.put(`ent:${userId}`, JSON.stringify({ ...record, updatedAt: Date.now() }));
  if (record.customerId) await env.GATED.put(`cust:${record.customerId}`, userId);
}

// Resolves which account a subscription belongs to. Metadata is the primary
// route; the customer reverse-lookup covers a subscription created outside
// this checkout flow (a Stripe dashboard edit, say) that never got metadata.
async function userIdForSubscription(env, sub) {
  const fromMeta = sub?.metadata?.clerk_user_id;
  if (fromMeta) return fromMeta;
  if (sub?.customer) return await env.GATED.get(`cust:${sub.customer}`);
  return null;
}

function planForPrice(priceId) {
  return Object.keys(PRICE_IDS).find((k) => PRICE_IDS[k] === priceId) || "unknown";
}

// Stripe moved current_period_end OFF the subscription and onto its items in
// the 2025-era API versions, and this account's webhook is pinned to
// 2026-08-26.dahlia. Reading only the old location silently yields null,
// which then renders as a 1970 date on the "ends <date>" badge. Checks both.
function periodEndOf(sub) {
  return sub?.current_period_end || sub?.items?.data?.[0]?.current_period_end || null;
}

/**
 * Whether this subscription is scheduled to end, and when.
 *
 * Stripe's older shape was a cancel_at_period_end boolean; newer API versions
 * (this account is pinned to 2026-08-26.dahlia) express the same thing as a
 * cancel_at timestamp. Reading only the boolean returned undefined, which
 * !!-ed to false, so a subscription the dashboard clearly showed as
 * "Cancels Oct 6" was reported as not cancelling at all.
 *
 * Both are read, boolean first, so this is correct on either version rather
 * than trading one wrong assumption for another.
 */
function cancellationOf(sub) {
  const at = sub?.cancel_at || null;
  // EITHER signal means it is ending. An earlier version preferred the
  // boolean when present, which was wrong in the one combination that
  // actually occurs here: Stripe returns cancel_at set to the end date AND
  // cancel_at_period_end false, so trusting the boolean reported a
  // subscription cancelling on Oct 6 as not cancelling at all. The two
  // shapes were tested separately and that combination never was, which is
  // why the test passed while the bug was live.
  const cancelling = !!sub?.cancel_at_period_end || !!at;
  return { cancelling, cancelAt: at || (cancelling ? periodEndOf(sub) : null) };
}

async function serveStripeWebhook(request, env) {
  if (!env.STRIPE_WEBHOOK_SECRET || !env.GATED) {
    return Response.json({ error: "not configured" }, { status: 503, headers: NO_STORE });
  }
  const { event, error } = await stripeEventFrom(request, env);
  // 400, never 200: a signature failure means this did not come from Stripe,
  // and answering 200 would tell a prober their forgery was accepted.
  if (error) return Response.json({ error }, { status: 400, headers: NO_STORE });

  const obj = event.data?.object || {};
  // What the handler actually did, returned in the 200 body. Stripe shows the
  // response in its delivery log, which makes this the only practical way to
  // see inside a Worker that has no readable logs. It exists because a bare
  // {received:true} was returned even when nothing was written -- a paid
  // customer got no access and the delivery log said 200, which is the most
  // misleading thing it could have said.
  let action = `ignored ${event.type}`;
  try {
    if (event.type === "checkout.session.completed") {
      let userId = obj.client_reference_id;
      // client_reference_id is set at checkout, but a session can arrive
      // without it. The subscription's own metadata carries the same id and
      // is the more durable of the two, so fall back to it rather than
      // dropping a sale on the floor.
      if (!userId && obj.subscription) {
        try {
          const s0 = await stripeGet(env, `/subscriptions/${obj.subscription}`);
          userId = s0?.metadata?.clerk_user_id || null;
        } catch { /* handled by the report below */ }
      }
      if (!userId) action = "skipped: no clerk user id on session or subscription";
      else if (!obj.subscription) action = `skipped: session ${obj.id} has no subscription`;
      if (userId && obj.subscription) {
        // The session carries no price or period, so read the subscription it
        // just created rather than guessing at either.
        const sub = await stripeGet(env, `/subscriptions/${obj.subscription}`);
        const c = cancellationOf(sub);
        await writeEntitlement(env, userId, {
          plan: planForPrice(sub.items?.data?.[0]?.price?.id),
          status: sub.status,
          currentPeriodEnd: periodEndOf(sub),
          cancelAtPeriodEnd: c.cancelling,
          cancelAt: c.cancelAt,
          customerId: sub.customer,
          subscriptionId: sub.id,
        });
        action = `wrote entitlement for ${userId} (${sub.status})`;
      }
    } else if (event.type.startsWith("customer.subscription.")) {
      const userId = await userIdForSubscription(env, obj);
      if (!userId) action = `skipped: no clerk user id for subscription ${obj.id}`;
      if (userId) {
        await writeEntitlement(env, userId, {
          plan: planForPrice(obj.items?.data?.[0]?.price?.id),
          // A deleted subscription reports its last status, which can still
          // read "active"; force it to canceled so access actually ends.
          status: event.type.endsWith(".deleted") ? "canceled" : obj.status,
          currentPeriodEnd: periodEndOf(obj),
          cancelAtPeriodEnd: cancellationOf(obj).cancelling,
          cancelAt: cancellationOf(obj).cancelAt,
          // Stripe cancels at period end by DEFAULT, which leaves status
          // "active" until the period expires -- correct behaviour, and what
          // the Terms promise, but indistinguishable from a live subscription
          // unless this flag travels. Without it the page says "paid" to
          // someone who just cancelled and looks like the cancellation failed.
          customerId: obj.customer,
          subscriptionId: obj.id,
        });
        action = `updated entitlement for ${userId} (${obj.status})`;
      }
    }
  } catch (e) {
    // 500 makes Stripe retry with backoff, which is what we want for a
    // transient failure -- swallowing it would silently lose the entitlement.
    return Response.json({ error: e.message }, { status: 500, headers: NO_STORE });
  }
  return Response.json({ received: true, action }, { headers: NO_STORE });
}

async function serveGated(request, env) {
  const who = await identify(request);
  if (GATE_ENFORCED && !who.signedIn) {
    return Response.json(
      { error: "sign in to see projections", signedIn: false, reason: who.reason },
      { status: 401, headers: NO_STORE },
    );
  }

  // The KV binding is added to wrangler.toml only once the namespace really
  // exists (see the note there: a placeholder id makes `wrangler deploy`
  // fail outright, which would take the twice-daily refresh down with it).
  // Until then this answers 503 and the page falls back to free content
  // rather than erroring.
  if (!env.GATED) {
    return Response.json({ error: "gated store not configured" }, { status: 503, headers: NO_STORE });
  }
  // Streamed rather than buffered: the payload is several megabytes and
  // there is no reason to hold all of it in the isolate to hand it straight
  // back.
  // Subscribers get the full props; a free account gets the reduced payload,
  // which reproduces every pick card exactly (verified per build by
  // pipeline/nfl/verify_account_split.mjs) while withholding the prop tables.
  const ent = who.signedIn ? await entitlementFor(env, who.userId) : null;
  const paid = (await hasComp(env, who.userId)) || isPaid(ent);
  const key = paid ? GATED_KEY : ACCOUNT_KEY;
  let body = await env.GATED.get(key, { type: "stream" });
  // A missing account payload must not silently fall back to the paid one --
  // that would hand every free account the full board. Fail closed.
  if (body === null) {
    return Response.json(
      { error: `no ${paid ? "paid" : "account"} payload published yet` },
      { status: 503, headers: NO_STORE },
    );
  }
  return new Response(body, {
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "X-Phinsup-Tier": paid ? "paid" : "account",
      // Never let a shared cache hold this. Once the response depends on who
      // is asking, a cached copy would be a straightforward way to serve one
      // subscriber's entitlement to everybody.
      ...NO_STORE,
    },
  });
}

// ---- crawler-facing routing ---------------------------------------------
// This Worker owns routing, because the asset layer cannot tell the three
// cases apart. wrangler.toml sets not_found_handling = "none" so a miss is
// a real 404 here rather than something that has to be detected later.
//
// Three kinds of path arrive:
//   1. A real file  -- /og.png, /sitemap.xml, /results/*.json
//   2. A pre-rendered page -- /nfl/2026/week-1/dolphins-vs-raiders, stored
//      as <path>/index.html by build_game_pages.py
//   3. A client-side route of the single-page app -- /nfl/week1, /mlb/<date>
// Everything else is a 404 and has to say so. It used to say 200 and hand
// back 414 KB of app: /robots.txt, /sitemap.xml and every typo'd, stale or
// scraped URL alike. To a crawler that is a soft 404 -- an unbounded space
// of URLs that all claim to exist and all return the same page -- and it is
// one of the few things Google will actively demote a site for.
//
// APP_ROUTES is case 3, and is the same route table as
// applyRouteFromLocation() in dashboard_live.html seen from the server:
// adding a route there means adding it here, or the new one 404s on refresh.
const APP_ROUTES = /^\/(?:(?:nfl|mlb|nhl)(?:\/(?:season|week\d{1,2}|\d{4}-\d{2}-\d{2}))?)?\/?$/;

// A path with an extension is asking for a real file, and a missing one is
// simply a 404 -- never the app, and never worth a directory-index retry.
const FILE_PATH = /\.[a-z0-9]{2,6}$/i;

function notFound() {
  return new Response(
    `<!doctype html><meta charset="utf-8"><meta name="robots" content="noindex">` +
    `<title>Not found - Phins Up</title>` +
    `<body style="background:#0E1A1C;color:#EDEFF2;font:14px system-ui;padding:48px">` +
    `<h1 style="font-size:20px">Not found</h1>` +
    `<p>No page at this address. <a style="color:#00C2B8" href="/">Go to Phins Up</a>.</p>`,
    { status: 404, headers: { "Content-Type": "text/html; charset=utf-8", ...NO_STORE } });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/gated") return serveGated(request, env);
    if (url.pathname === "/api/whoami") return serveWhoami(request, env);
    if (url.pathname === "/api/stripe-webhook") return serveStripeWebhook(request, env);
    if (url.pathname === "/api/checkout") return serveCheckout(request, env);
    if (url.pathname === "/api/portal") return servePortal(request, env);

    const isFile = FILE_PATH.test(url.pathname);
    let response = await env.ASSETS.fetch(request);

    // The asset layer does NOT report a miss as a 404. It answers an
    // unmatched path with a 307 to "/", and a directory missing its trailing
    // slash with a 307 to the slashed form. Passing either through is wrong:
    // the first stranded every deep link (/nfl/week1 -> "/" , week lost --
    // measured against the live site, not assumed), and the second makes the
    // canonical URL of a game page a redirect rather than a 200.
    //
    // So a redirect is treated exactly like a 404: "the asset layer could not
    // serve this", and routing is decided below instead.
    const assetMissed = response.status === 404 ||
      (response.status >= 300 && response.status < 400);

    if (assetMissed) {
      response = null;

      // 1. A pre-rendered page (build_game_pages.py), stored at
      //    <path>/index.html. Fetched directly rather than followed via the
      //    redirect, so the canonical URL is the one that returns 200.
      if (!isFile) {
        const idx = new URL(request.url);
        idx.pathname = url.pathname.replace(/\/+$/, "") + "/index.html";
        const viaIndex = await env.ASSETS.fetch(new Request(idx, request));
        if (viaIndex.status === 200) response = viaIndex;
      }

      // 2. A client-side route of the app: hand over the app and let its own
      //    router read the path.
      if (!response && APP_ROUTES.test(url.pathname)) {
        const indexUrl = new URL(request.url);
        indexUrl.pathname = "/index.html";
        response = await env.ASSETS.fetch(new Request(indexUrl, request));
      }

      // 3. Nothing owns this path.
      if (!response) return notFound();
    }
    const headers = new Headers(response.headers);
    headers.set("Cache-Control", "no-cache, no-store, must-revalidate");
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  },
};
