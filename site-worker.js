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
 */
export default {
  async fetch(request, env) {
    const response = await env.ASSETS.fetch(request);
    const headers = new Headers(response.headers);
    headers.set("Cache-Control", "no-cache, no-store, must-revalidate");
    // Temporary diagnostic -- an unmistakable marker no Cloudflare default
    // could produce, to prove definitively whether this script is even
    // running at all (multiple prior fixes deployed with zero visible
    // effect, need to isolate "worker not running" from "something else is
    // still caching regardless of what the worker sets"). Remove once the
    // caching issue is confirmed resolved.
    headers.set("X-Debug-Worker-Ran", "v1");
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  },
};
