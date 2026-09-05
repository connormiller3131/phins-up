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
 * paid tiers a single door to lock; the lock itself is Phase 1, so for now
 * this endpoint answers everyone, exactly as the inlined data did.
 */

const GATED_KEY = "current";

async function serveGated(env) {
  // The KV binding is added to wrangler.toml only once the namespace really
  // exists (see the note there: a placeholder id makes `wrangler deploy`
  // fail outright, which would take the twice-daily refresh down with it).
  // Until then this answers 503 and the page falls back to free content
  // rather than erroring.
  if (!env.GATED) {
    return Response.json({ error: "gated store not configured" }, { status: 503 });
  }
  // Streamed rather than buffered: the payload is several megabytes and
  // there is no reason to hold all of it in the isolate to hand it straight
  // back.
  const body = await env.GATED.get(GATED_KEY, { type: "stream" });
  if (body === null) {
    return Response.json({ error: "no gated payload published yet" }, { status: 503 });
  }
  return new Response(body, {
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      // Never let a shared cache hold this. Today it is public and the
      // header is merely consistent with the rest of the site; once Phase 1
      // makes the response depend on who is asking, a cached copy would be
      // a straightforward way to serve one subscriber's entitlement to
      // everybody.
      "Cache-Control": "no-store",
    },
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/gated") return serveGated(env);

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
