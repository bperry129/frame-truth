# 🚀 Vercel Deployment Guide for Frame Truth

## Why Vercel is Perfect for Frame Truth

✅ **Full-Stack Support** - Handles both React frontend and Python backend  
✅ **Serverless Functions** - Python backend runs as serverless functions  
✅ **Zero Configuration** - Auto-detects your setup  
✅ **Free Tier** - Generous free plan for personal projects  
✅ **Global CDN** - Fast worldwide performance  
✅ **GitHub Integration** - Auto-deploys on every push  

## 📋 Pre-Deployment Steps

### 1. Update Backend for Vercel
The backend needs a small modification for Vercel's serverless environment:

**Create `backend/api.py`** (Vercel entry point):
```python
from .server import app

# Vercel expects the app to be exported
def handler(request):
    return app(request)
```

### 2. Update Frontend API URLs
**Update `src/config.ts`** (create if doesn't exist):
```typescript
export const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api'  // Vercel will route /api/* to backend
  : 'http://localhost:8000';
```

## 🚀 Step-by-Step Deployment

### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Frame Truth ready for deployment"

# Add GitHub remote (replace with your repo URL)
git remote add origin https://github.com/yourusername/frame-truth.git

# Push to GitHub
git push -u origin main
```

### Step 2: Deploy to Vercel

1. **Go to [vercel.com](https://vercel.com)**
2. **Sign up/Login** with your GitHub account
3. **Import Project**:
   - Click "New Project"
   - Select your `frame-truth` repository
   - Vercel will auto-detect it's a Vite project

4. **Configure Build Settings**:
   - Framework Preset: **Vite**
   - Build Command: `npm run build`
   - Output Directory: `dist`
   - Install Command: `npm install`

5. **Add Environment Variables**:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ADMIN_USER=admin
   ADMIN_PASS=your_secure_password_here
   NODE_ENV=production
   ```

6. **Deploy!** 🎉

### Step 3: Verify Deployment

After deployment, Vercel will give you a URL like:
`https://frame-truth-abc123.vercel.app`

Test these endpoints:
- **Frontend**: `https://your-app.vercel.app`
- **Backend Health**: `https://your-app.vercel.app/api/docs`
- **Video Upload**: Try uploading a video through the UI

## 🔧 Configuration Files Explained

### `vercel.json`
```json
{
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",  // Builds React app
      "config": { "distDir": "dist" }
    },
    {
      "src": "backend/server.py",
      "use": "@vercel/python"  // Handles Python backend
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/backend/server.py"  // Route API calls to backend
    },
    {
      "src": "/(.*)",
      "dest": "/dist/$1"  // Serve React app for all other routes
    }
  ]
}
```

## 🔒 Environment Variables Setup

In Vercel Dashboard → Settings → Environment Variables:

| Variable | Value | Description |
|----------|-------|-------------|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | Your OpenRouter API key |
| `ADMIN_USER` | `admin` | Admin username |
| `ADMIN_PASS` | `your_password` | Secure admin password |
| `NODE_ENV` | `production` | Environment mode |

## 📊 Vercel Limits (Free Tier)

- **Bandwidth**: 100GB/month
- **Function Execution**: 100GB-hours/month
- **Function Duration**: 10 seconds max
- **Deployments**: Unlimited
- **Team Members**: 1 (you)

Perfect for personal projects and small-scale usage!

## 🔄 Continuous Deployment

Once set up, every time you push to GitHub:
1. Vercel automatically detects the push
2. Builds your app
3. Deploys the new version
4. Gives you a preview URL

## 🌐 Custom Domain (Optional)

1. **Buy a domain** (Namecheap, GoDaddy, etc.)
2. **In Vercel Dashboard**:
   - Go to Settings → Domains
   - Add your custom domain
   - Follow DNS setup instructions
3. **SSL Certificate**: Automatically provided by Vercel

## 🆘 Troubleshooting

### Common Issues:

**Build Failures:**
- Check Node.js version (Vercel uses Node 18 by default)
- Verify all dependencies are in `package.json`

**API Errors:**
- Check environment variables are set correctly
- Verify Python dependencies in `requirements.txt`

**CORS Issues:**
- Update CORS settings in `backend/server.py` to include your Vercel domain

**Function Timeouts:**
- Video processing might take time - consider optimizing or using async processing

### Getting Help:
- Check Vercel deployment logs
- Test API endpoints directly: `https://your-app.vercel.app/api/docs`
- Vercel has excellent documentation and community support

## 🎉 Success!

Once deployed, your Frame Truth app will be:
- ✅ Live at `https://your-app.vercel.app`
- ✅ Automatically updated on every GitHub push
- ✅ Globally distributed via Vercel's CDN
- ✅ Secured with HTTPS
- ✅ Ready for production use!

## 💡 Next Steps

1. **Test thoroughly** with different video types
2. **Monitor usage** in Vercel dashboard
3. **Set up custom domain** if desired
4. **Consider upgrading** to Pro plan if you exceed free limits
5. **Add monitoring** and analytics

Your Frame Truth application is now production-ready! 🚀
