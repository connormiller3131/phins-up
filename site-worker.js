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
 */
export default {
  async fetch(request, env) {
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
