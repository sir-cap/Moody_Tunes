# Re-authorizing the Spotify connection

`create_spotify_playlist()` needs a valid Spotify `refresh_token` for the account that owns
`SPOTIFY_USER_ID`, stored as the `SPOTIFY_REFRESH_TOKEN` secret. If it ever breaks again
(`spotipy.SpotifyOauthError` in the logs — the app itself degrades gracefully now and won't
crash, but playlist creation will silently stop working), redo this:

## Why this can break

Spotify refresh tokens can be invalidated if the app's authorization is revoked (from the
Spotify account's connected-apps settings), if the token goes unused for a very long time, or
if the app's client credentials are regenerated in the Spotify Developer Dashboard. There's no
way to detect this in advance — it just starts failing.

## How to get a fresh one (~5 min, needs the account owner's Spotify login)

1. Build the authorize URL (client_id/secret from `.streamlit/secrets.toml` or the Spotify
   Developer Dashboard):

   ```python
   from spotipy.oauth2 import SpotifyOAuth
   oauth = SpotifyOAuth(
       client_id="...",
       client_secret="...",
       redirect_uri="https://moodytunes.streamlit.app/callback",
       scope="playlist-modify-public",
   )
   print(oauth.get_authorize_url())
   ```

2. Open that URL in a browser logged into the Spotify account that should own the generated
   playlists. Log in / click Allow.

3. You'll land back on the live app — it won't do anything special with the redirect, but the
   browser's address bar will show a URL like
   `https://moodytunes.streamlit.app/callback?code=AQ...`. Copy that full URL.

4. Exchange the code for tokens:

   ```python
   code = oauth.parse_response_code(pasted_url)
   token_info = oauth.get_access_token(code, as_dict=True, check_cache=False)
   print(token_info["refresh_token"])
   ```

5. Update the `SPOTIFY_REFRESH_TOKEN` secret in the Streamlit Cloud app's Settings → Secrets
   (share.streamlit.io → the app → ⋮ → Settings → Secrets) with the new value. No code changes
   needed — `create_spotify_playlist()` reads it from `st.secrets` and seeds a
   `MemoryCacheHandler` with it on every call, so it doesn't depend on any file surviving a
   redeploy.

6. Sanity check before relying on it:

   ```python
   cache_handler = spotipy.cache_handler.MemoryCacheHandler(token_info={
       "refresh_token": new_refresh_token,
       "access_token": "", "expires_at": 0,
       "scope": "playlist-modify-public", "token_type": "Bearer",
   })
   sp = spotipy.Spotify(auth_manager=spotipy.oauth2.SpotifyOAuth(
       scope="playlist-modify-public", client_id=..., client_secret=...,
       redirect_uri="https://moodytunes.streamlit.app/callback",
       cache_handler=cache_handler,
   ))
   print(sp.current_user())  # should print the right account id/display name
   ```
