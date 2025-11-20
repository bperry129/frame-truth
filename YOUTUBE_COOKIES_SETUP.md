# YouTube Cookies Setup Guide

## Why Cookies Are Needed

YouTube has strict bot detection that blocks automated downloads. The **100% reliable solution** is to use authenticated cookies from your browser, which makes yt-dlp appear as a legitimate logged-in user.

## How to Export YouTube Cookies

### Method 1: Using Browser Extension (Easiest)

1. **Install a cookie exporter extension:**
   - Chrome/Edge: [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
   - Firefox: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

2. **Export cookies:**
   - Go to YouTube.com and make sure you're logged in
   - Click the extension icon
   - Click "Export" or "Download"
   - Save the file as `youtube_cookies.txt`

### Method 2: Using yt-dlp (Command Line)

```bash
# This will extract cookies from your browser
yt-dlp --cookies-from-browser firefox --cookies youtube_cookies.txt "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Replace `firefox` with your browser: `chrome`, `edge`, `safari`, `opera`, etc.

## How to Upload Cookies to Server

### Option 1: Via API (Postman/curl)

```bash
curl -X POST "https://your-server.com/api/admin/upload-cookies" \
  -H "X-Admin-User: admin" \
  -H "X-Admin-Pass: your-admin-password" \
  -F "file=@youtube_cookies.txt"
```

### Option 2: Manually on Server

If you have SSH access to your Railway/server:

```bash
# Create cookies directory
mkdir -p cookies

# Upload your cookies file
# (use scp, SFTP, or paste content directly)
nano cookies/youtube_cookies.txt

# Paste your cookies content and save
```

### Option 3: Via Railway CLI

```bash
# SSH into Railway container
railway shell

# Create directory and upload file
mkdir -p cookies
cat > cookies/youtube_cookies.txt
# Paste cookies content, then Ctrl+D to save
```

## Verify Cookies Are Working

Check cookies status:

```bash
curl -X GET "https://your-server.com/api/admin/cookies-status" \
  -H "X-Admin-User: admin" \
  -H "X-Admin-Pass: your-admin-password"
```

Expected response:
```json
{
  "cookies_configured": true,
  "cookies_path": "cookies/youtube_cookies.txt",
  "status": "YouTube downloads will work 100% reliably"
}
```

## How It Works

Once cookies are uploaded:

1. Server checks if `cookies/youtube_cookies.txt` exists
2. If yes, yt-dlp uses it automatically for all YouTube downloads
3. Downloads appear as requests from your authenticated YouTube account
4. **100% reliable** - no more bot detection errors!

## Security Notes

- **Keep cookies file secret!** It contains your YouTube session
- Cookies expire after ~6 months (YouTube session lifetime)
- When cookies expire, re-export and re-upload
- Only upload to trusted servers you control

## Troubleshooting

**Downloads still failing?**
- Cookies may have expired - re-export fresh ones
- Make sure you're logged into YouTube when exporting
- Try a different browser to export cookies

**How often to refresh?**
- Cookies last ~6 months
- Refresh when you start seeing bot detection errors again
