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
 * Fourth job: Clerk session verification, at /api/whoami. See GATE_ENFORCED
 * below for why the lock is not yet turned on.
 */

const GATED_KEY = "current";

// Clerk's Frontend API for this instance, decoded from the publishable key
// (pk_test_<base64 of "darling-toad-4960.clerk.accounts.dev$">). It is also
// the `iss` every session token must carry.
const CLERK_ISSUER = "https://darling-toad-4960.clerk.accounts.dev";
const CLERK_JWKS_URL = `${CLERK_ISSUER}/.well-known/jwks.json`;

// Two-step rollout. While false, /api/gated answers everyone exactly as it
// did before, and /api/whoami reports what verification WOULD have decided.
// That lets the token path be proven against the real deployed Worker before
// anything depends on it -- a verification bug flipped straight on would
// take every prop off the site for every visitor, which is precisely the
// failure this site already had once tonight.
const GATE_ENFORCED = false;

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

// Reports what the gate would decide, without gating anything. Exists so the
// token path can be proven against the real deployed Worker before
// GATE_ENFORCED is flipped.
async function serveWhoami(request) {
  const who = await identify(request);
  return Response.json({ ...who, enforced: GATE_ENFORCED }, { headers: NO_STORE });
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
  const body = await env.GATED.get(GATED_KEY, { type: "stream" });
  if (body === null) {
    return Response.json({ error: "no gated payload published yet" }, { status: 503, headers: NO_STORE });
  }
  return new Response(body, {
    headers: {
      "Content-Type": "application/json; charset=utf-8",
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
    if (url.pathname === "/api/whoami") return serveWhoami(request);

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
