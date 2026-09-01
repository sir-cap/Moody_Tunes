# Secrets — what they are and how to manage them

All secrets live in **Streamlit Cloud → your app → ⋮ (top right of the app card on
[share.streamlit.io](https://share.streamlit.io)) → Settings → Secrets**. They are edited as
TOML, one `KEY = "value"` per line. Changes take about a minute to propagate.

They are **not** in this repo, not in `.streamlit/secrets.toml` on disk in git (that file is
gitignored — see below), and not recoverable from GitHub. If you lose one, you have to
regenerate it at its source (Spotify/Cloudinary dashboards) and re-enter it here.

## The 7 secrets

| Key | What it's for | Where to get/regenerate it |
|---|---|---|
| `SPOTIPY_CLIENT_ID` | Identifies the Spotify app (MoodyTunes) to Spotify's API | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) → the app → Settings |
| `SPOTIPY_CLIENT_SECRET` | Authenticates the app itself (not a specific user) | Same dashboard page, "View client secret" / regenerate |
| `SPOTIFY_USER_ID` | The Spotify account that owns/receives the auto-created playlists | Your Spotify profile → the numeric ID in your profile URL, or `sp.current_user()["id"]` |
| `SPOTIFY_REFRESH_TOKEN` | Lets the app act as that user (create playlists) without them logging in every time | One-time manual OAuth flow — see `reference/spotify-reauth.md` if this needs redoing |
| `CLOUDINARY_CLOUD_NAME` | Which Cloudinary account/bucket photos get uploaded to | [console.cloudinary.com](https://cloudinary.com/console) → Dashboard, top of page |
| `CLOUDINARY_API_KEY` | Identifies the app to Cloudinary | Same dashboard |
| `CLOUDINARY_API_SECRET` | Authenticates the app to Cloudinary | Same dashboard → "click to reveal" / regenerate |

## How to tell it's a secrets problem (not a code bug)

The app crashes on startup or on first interaction with an error page whose traceback ends in
something like:

```
KeyError: 'st.secrets has no key "SOME_KEY". Did you forget to add it to secrets.toml...'
```

That means a key is missing or misspelled in Streamlit Cloud's Secrets box — go add/fix it
there, no code change needed. A `SpotifyOauthError` or `cv2.error` deeper in a traceback is a
different kind of problem (an actual expired/invalid credential, or a code bug) — see below.

## Debugging via the live logs

**Manage app** (bottom-right corner of the running app) opens a log panel with the build and
runtime logs. Caveat noticed while working on this: the panel can show a **stale/frozen**
snapshot — it stopped updating past a certain point in testing even across a fresh browser tab.
If it looks out of date (doesn't mention your most recent push, or shows an error you already
fixed), don't trust it as "current" — reproduce the issue live in the app instead and note the
exact error text shown there (it's usually enough on its own), or wait a few minutes and
reopen Manage app before concluding it's still broken.

## Files related to secrets (for context, not editing)

- `.streamlit/secrets.toml` exists locally on disk (for local testing) but is gitignored — it
  is **not** what the deployed app reads from. Streamlit Cloud has its own separate copy,
  entered through the Secrets UI above.
- Two files were previously committed to this **public** repo by mistake and contained real
  credentials: `secrets.toml` and `.cache` (a Spotify OAuth token cache). Both were removed
  from git tracking (`git rm --cached`) and are now properly gitignored. Their exposed values
  were rotated. If you ever add a new local secrets/cache file, double-check
  `git status` doesn't show it as staged before committing.
