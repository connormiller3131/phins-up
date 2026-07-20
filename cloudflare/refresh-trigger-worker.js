/**
 * Fires the "Refresh projections" GitHub Actions workflow on a reliable
 * schedule. GitHub's own `schedule:` cron trigger gets queued behind
 * manual/push-triggered runs on free-tier repos and can lag by hours (a 2am
 * run fired at 7:41am once, a noon run finished at 4:48pm) -- confirmed
 * `workflow_dispatch` doesn't have this problem, so this Worker's Cron
 * Trigger just calls that endpoint directly instead of relying on GitHub's
 * own schedule at all.
 *
 * Requires a Worker secret GH_PAT: a GitHub Personal Access Token (fine-
 * grained, scoped to this repo only, "Actions: write" permission) --
 * NEVER commit the actual token, only reference it as env.GH_PAT here.
 */
const OWNER = "connormiller3131";
const REPO = "phins-up";
const WORKFLOW_FILE = "refresh.yml";

export default {
  async scheduled(event, env, ctx) {
    const resp = await fetch(
      `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches`,
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${env.GH_PAT}`,
          "Accept": "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "phins-up-refresh-trigger-worker",
        },
        body: JSON.stringify({ ref: "main" }),
      }
    );
    if (!resp.ok) {
      console.error(`GitHub workflow_dispatch failed: ${resp.status} ${await resp.text()}`);
    } else {
      console.log(`Dispatched ${WORKFLOW_FILE} successfully (cron: ${event.cron})`);
    }
  },

  // Lets you hit the Worker's own URL to fire a dispatch manually too, for
  // testing this without waiting on a cron tick.
  async fetch(request, env, ctx) {
    await this.scheduled({ cron: "manual" }, env, ctx);
    return new Response("Dispatched refresh.yml");
  },
};
