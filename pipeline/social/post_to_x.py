"""Post model projections and results to X (@PhinsUpDotNet).

WHY AUTOMATE IT. The account is the credibility mechanism: predictions posted
publicly BEFORE games, and results posted after, including the losses. A human
doing that by hand misses days, and the days most likely to be missed are the
ones after a bad week -- which is exactly when skipping looks like hiding.

SAFE BY DEFAULT. This posts publicly under someone's real name, so:
  * DRY RUN unless POST_TO_X=true. It prints the exact text and sends nothing.
  * IDEMPOTENT via Cloudflare KV. A re-run, a retry or a double-trigger cannot
    post the same thing twice, which is the failure that would actually
    embarrass the account.
  * Results are only posted for games the pipeline has genuinely GRADED. No
    inferring an outcome from a score that might still be provisional.
  * FAILS CLOSED. Missing credentials means no post and a clear message, never
    a partial or malformed one.

OAUTH 1.0a, NOT 2.0, and signed here rather than by a library. The 1.0a
user-context tokens do not expire; OAuth 2.0's do, after ~2 hours, with a
refresh token that rotates on every use and would have to be persisted and
rewritten from CI. Signing is ~40 lines and avoids adding a dependency to a
pipeline that pip-installs on every run.
"""
import base64
import hashlib
import hmac
import json
import os
import pathlib
import secrets
import time
import urllib.parse

import requests

API_URL = "https://api.x.com/2/tweets"
SITE = "https://phinsup.net"
KV_NAMESPACE = "c1ca23642f584fd7a1f7f7971060134e"   # same id as wrangler.toml
DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs"

# X counts every link as this many characters regardless of real length.
TCO_LEN = 23
TWEET_MAX = 280


def _pe(s):
    """RFC 3986 percent-encoding. urllib's default leaves some reserved
    characters alone, and OAuth signatures are unforgiving about it."""
    return urllib.parse.quote(str(s), safe="-._~")


def oauth1_header(method, url, creds):
    """Sign a request with OAuth 1.0a HMAC-SHA1.

    The JSON body is deliberately NOT part of the signature base string --
    OAuth 1.0a only signs form-encoded bodies, and including a JSON one
    produces a signature X rejects with a 401 that says nothing useful.
    """
    params = {
        "oauth_consumer_key": creds["api_key"],
        "oauth_nonce": secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": creds["access_token"],
        "oauth_version": "1.0",
    }
    joined = "&".join("%s=%s" % (_pe(k), _pe(params[k])) for k in sorted(params))
    base = "&".join([method.upper(), _pe(url), _pe(joined)])
    key = "%s&%s" % (_pe(creds["api_secret"]), _pe(creds["access_secret"]))
    sig = base64.b64encode(
        hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()).decode()
    params["oauth_signature"] = sig
    return "OAuth " + ", ".join('%s="%s"' % (_pe(k), _pe(v))
                                for k, v in sorted(params.items()))


def tweet_length(text):
    """Length as X counts it: every URL costs TCO_LEN whatever its real size."""
    total = 0
    for word in text.split(" "):
        total += (TCO_LEN if word.startswith(("http://", "https://")) else len(word))
    return total + text.count(" ")


def credentials():
    # .strip() is not paranoia: a trailing newline pasted into a GitHub secret
    # is invisible in the UI, changes the signature, and surfaces only as a
    # bare 401 with no detail.
    keys = {
        "api_key": os.environ.get("X_API_KEY", "").strip(),
        "api_secret": os.environ.get("X_API_SECRET", "").strip(),
        "access_token": os.environ.get("X_ACCESS_TOKEN", "").strip(),
        "access_secret": os.environ.get("X_ACCESS_TOKEN_SECRET", "").strip(),
    }
    missing = [k for k, v in keys.items() if not v]
    return (None, missing) if missing else (keys, [])


# ---- idempotency ----------------------------------------------------------
# Cloudflare KV rather than a committed file: the workflow that posts does not
# commit anything, and a state file that is not committed would reset on every
# fresh checkout -- which is precisely how you double-post.

def _kv_url(key):
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    return ("https://api.cloudflare.com/client/v4/accounts/%s/storage/kv/"
            "namespaces/%s/values/%s" % (account, KV_NAMESPACE, urllib.parse.quote(key)))


def already_posted(key):
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token or not os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
        return False
    try:
        r = requests.get(_kv_url(key), timeout=20,
                         headers={"Authorization": "Bearer " + token})
        return r.status_code == 200
    except Exception as exc:
        # Fail CLOSED: if the ledger cannot be read we do not know whether this
        # already went out, and a duplicate post is worse than a missed one.
        print("  [kv] could not read %s (%s) -- treating as already posted" % (key, exc))
        return True


def mark_posted(key, tweet_id):
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        return
    try:
        requests.put(_kv_url(key), timeout=20,
                     headers={"Authorization": "Bearer " + token},
                     files={"value": (None, json.dumps({"id": tweet_id, "at": time.time()})),
                            "metadata": (None, "{}")})
    except Exception as exc:
        print("  [kv] WARNING: posted but could not record it (%s)" % exc)
        print("  [kv] re-running this job could post a duplicate")


# ---- composing ------------------------------------------------------------

def nickname(full_name, abbr):
    parts = (full_name or "").split()
    return parts[-1] if len(parts) > 1 else (abbr or "Team")


def compose_pregame(nfl):
    """The single game where the model disagrees most with the posted line.

    One game, not a list: a post that names one thing is readable and gets
    graded cleanly afterwards. A slate dump is noise nobody can hold you to.
    """
    week = int(nfl["current_week"])
    # PRIMETIME ONLY. Model win % is a paid feature for every other game, and
    # this module reads data/dashboard_published_nfl.json, which is written
    # BEFORE gated_payload.py lifts those numbers out. Without this filter the
    # account would publish, to everyone, the exact figure the site charges
    # for. TNF/SNF/MNF are the free sample, so they are the only safe ones.
    games = [g for g in nfl["weeks"][str(week)]["games"]
             if g.get("primetime")
             and g.get("elo_home_prob") is not None
             and g.get("market_home_prob") is not None
             and not g.get("already_played")]
    if not games:
        return None, None
    g = max(games, key=lambda x: abs(x["elo_home_prob"] - x["market_home_prob"]))
    eh, mh = g["elo_home_prob"], g["market_home_prob"]
    an = nickname(g.get("awayName"), g["awayAbbr"])
    hn = nickname(g.get("homeName"), g["homeAbbr"])

    if eh >= 0.5:
        side, ours, theirs = hn, eh, mh
    else:
        side, ours, theirs = an, 1 - eh, 1 - mh
    path = "/nfl/%s/week-%d/%s-vs-%s" % (nfl["season"], week, an.lower(), hn.lower())

    # This is the THIRD place the slug rule appears -- build_game_pages.py
    # generates it, dashboard_live.html reads it back off a stamped field, and
    # here it has to be recomputed because the published NFL json is written
    # before those stamps exist. A wrong guess would put a dead link in a
    # public post, so it is checked against the page that was actually
    # generated: a miss refuses to post rather than shipping a 404 to the
    # timeline, and says why.
    if not (DOCS / path.lstrip("/") / "index.html").exists():
        print("  REFUSING: composed link %s has no generated page -- the slug "
              "rule here has drifted from build_game_pages.py." % path)
        return None, None

    text = (
        "NFL Week %d. Where our model disagrees most with the market.\n\n"
        "%s vs %s\n"
        "Model: %s %.1f%%\n"
        "Market implies: %s %.1f%%\n\n"
        "Posted before kickoff. Graded after, win or lose.\n%s%s"
        % (week, an, hn, side, ours * 100, side, theirs * 100, SITE, path))
    return text, "x_posted:nfl-%s-wk%d-pregame" % (nfl["season"], week)


def compose_results(nfl, results_dir):
    """Only fires once every game of the week carries a graded result, so the
    record posted is the whole week rather than a partial one that would have
    to be corrected later."""
    season, week = nfl["season"], int(nfl["current_week"])
    graded, correct, mkt_n, mkt_correct = 0, 0, 0, 0
    total = len(nfl["weeks"][str(week)]["games"])
    for f in sorted(results_dir.glob("nfl_%s_wk%02d_*.json" % (season, week))):
        snap = json.loads(f.read_text(encoding="utf-8"))
        a = snap.get("actual")
        if not snap.get("graded") or not a:
            continue
        graded += 1
        correct += 1 if a.get("model_correct") else 0
        if a.get("market_pick"):
            mkt_n += 1
            mkt_correct += 1 if a.get("market_correct") else 0
    if not graded or graded < total:
        return None, None

    lines = ["NFL Week %d results.\n" % week,
             "The model called %d of %d winners." % (correct, graded)]
    if mkt_n:
        lines.append("The betting market called %d." % mkt_correct)
    lines.append("\nEvery prediction we have made, graded, including the "
                 "misses:\n%s/track-record" % SITE)
    return "\n".join(lines), "x_posted:nfl-%s-wk%d-results" % (season, week)


# ---- posting --------------------------------------------------------------

def post(text, idem_key, live):
    n = tweet_length(text)
    print("-" * 60)
    print(text)
    print("-" * 60)
    print("  length: %d/%d (links counted as %d)" % (n, TWEET_MAX, TCO_LEN))
    if n > TWEET_MAX:
        print("  REFUSING: over the limit")
        return 1
    if not live:
        print("  DRY RUN -- nothing sent. Set POST_TO_X=true to post for real.")
        return 0
    if already_posted(idem_key):
        print("  SKIP: already posted (%s)" % idem_key)
        return 0

    creds, missing = credentials()
    if not creds:
        print("  REFUSING: missing credentials: %s" % ", ".join(missing))
        return 1
    r = requests.post(API_URL, timeout=30, json={"text": text}, headers={
        "Authorization": oauth1_header("POST", API_URL, creds),
        "Content-Type": "application/json",
    })
    if r.status_code in (200, 201):
        tweet_id = (r.json().get("data") or {}).get("id")
        print("  POSTED: https://x.com/PhinsUpDotNet/status/%s" % tweet_id)
        mark_posted(idem_key, tweet_id)
        return 0
    print("  FAILED %d: %s" % (r.status_code, r.text[:600]))
    return 1


# ---- diagnostics ----------------------------------------------------------

def verify():
    """Check the credentials with a READ, so diagnosing auth never costs a
    public post.

    Tries both hosts on purpose. The OAuth 1.0a signature covers the request
    URL, so signing for one host and having the gateway validate against the
    other fails with exactly the bare "Unauthorized" this was written to
    diagnose -- and which of the two works is not something to guess at.
    """
    creds, missing = credentials()
    if not creds:
        print("MISSING credentials: %s" % ", ".join(missing))
        return 1
    for name, value in sorted(creds.items()):
        print("  %-14s len=%-3d starts=%s" % (name, len(value), value[:4]))

    ok = False
    for host in ("https://api.twitter.com", "https://api.x.com"):
        url = host + "/2/users/me"
        try:
            r = requests.get(url, timeout=30,
                             headers={"Authorization": oauth1_header("GET", url, creds)})
        except Exception as exc:
            print("  %-24s network error: %s" % (host, exc))
            continue
        if r.status_code == 200:
            who = (r.json().get("data") or {}).get("username")
            print("  %-24s OK -- authenticated as @%s" % (host, who))
            ok = True
        else:
            print("  %-24s %d %s" % (host, r.status_code, " ".join(r.text[:200].split())))
    return 0 if ok else 1
