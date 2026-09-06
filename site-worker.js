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
const PRICE_IDS = {
  monthly: "price_1UCZq5QcsqR0UNig0qCoKtbi",
  yearly: "price_1UCZq5QcsqR0UNigp3LRNDZH",
};
// Stripe statuses that actually grant access. past_due is deliberately absent:
// a failed renewal should lose access, and Stripe retries for days before
// giving up, which would otherwise be days of unpaid access.
const ACTIVE_STATUSES = new Set(["active", "trialing"]);

function formEncode(pairs) {
  return pairs.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");
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
  if (!r.ok) throw new Error(body?.error?.message || `stripe ${r.status}`);
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
async function serveWhoami(request, env) {
  const who = await identify(request);
  const ent = who.signedIn ? await entitlementFor(env, who.userId) : null;
  return Response.json(
    {
      ...who,
      enforced: GATE_ENFORCED,
      // So the page can say "US only" up front instead of showing buy
      // buttons that are going to 403 the moment they are clicked.
      country: (request.cf && request.cf.country) || null,
      canSubscribe: !request.cf || !request.cf.country || request.cf.country === "US",
      plan: ent?.plan || "free",
      subscriptionStatus: ent?.status || null,
      paid: isPaid(ent),
      currentPeriodEnd: ent?.currentPeriodEnd || null,
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

async function serveStripeWebhook(request, env) {
  if (!env.STRIPE_WEBHOOK_SECRET || !env.GATED) {
    return Response.json({ error: "not configured" }, { status: 503, headers: NO_STORE });
  }
  const { event, error } = await stripeEventFrom(request, env);
  // 400, never 200: a signature failure means this did not come from Stripe,
  // and answering 200 would tell a prober their forgery was accepted.
  if (error) return Response.json({ error }, { status: 400, headers: NO_STORE });

  const obj = event.data?.object || {};
  try {
    if (event.type === "checkout.session.completed") {
      const userId = obj.client_reference_id;
      if (userId && obj.subscription) {
        // The session carries no price or period, so read the subscription it
        // just created rather than guessing at either.
        const sub = await stripeApi(env, `/subscriptions/${obj.subscription}`, []);
        await writeEntitlement(env, userId, {
          plan: planForPrice(sub.items?.data?.[0]?.price?.id),
          status: sub.status,
          currentPeriodEnd: sub.current_period_end || null,
          customerId: sub.customer,
          subscriptionId: sub.id,
        });
      }
    } else if (event.type.startsWith("customer.subscription.")) {
      const userId = await userIdForSubscription(env, obj);
      if (userId) {
        await writeEntitlement(env, userId, {
          plan: planForPrice(obj.items?.data?.[0]?.price?.id),
          // A deleted subscription reports its last status, which can still
          // read "active"; force it to canceled so access actually ends.
          status: event.type.endsWith(".deleted") ? "canceled" : obj.status,
          currentPeriodEnd: obj.current_period_end || null,
          customerId: obj.customer,
          subscriptionId: obj.id,
        });
      }
    }
  } catch (e) {
    // 500 makes Stripe retry with backoff, which is what we want for a
    // transient failure -- swallowing it would silently lose the entitlement.
    return Response.json({ error: e.message }, { status: 500, headers: NO_STORE });
  }
  return Response.json({ received: true }, { headers: NO_STORE });
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
  const paid = isPaid(ent);
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

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/gated") return serveGated(request, env);
    if (url.pathname === "/api/whoami") return serveWhoami(request, env);
    if (url.pathname === "/api/stripe-webhook") return serveStripeWebhook(request, env);
    if (url.pathname === "/api/checkout") return serveCheckout(request, env);
    if (url.pathname === "/api/portal") return servePortal(request, env);

    let response = await env.ASSETS.fetch(request);
    if (response.status === 404) {
      const indexUrl = new URL(request.url);
      indexUrl.pathname = "/index.html";
      response = await env.ASSETS.fetch(new Request(indexUrl, request));
    }
    const headers = new Headers(response.headers);
    headers.set("Cache-Control", "no-cache, no-store, must-revalidate");
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  },
};
