# 🚀 Frame Truth Deployment Guide

## Overview
Frame Truth is a full-stack application with:
- **Frontend**: React + TypeScript (Vite)
- **Backend**: Python FastAPI + SQLite

## 🎯 Recommended Deployment Strategy

### Option 1: Split Deployment (Easiest)

#### Frontend → Netlify
1. **Push to GitHub** (if not already done)
2. **Connect to Netlify**:
   - Go to [netlify.com](https://netlify.com)
   - Click "New site from Git"
   - Connect your GitHub repo
   - Build settings:
     - Build command: `npm run build`
     - Publish directory: `dist`

3. **Environment Variables** (in Netlify dashboard):
   ```
   VITE_API_URL=https://your-backend-url.railway.app
   ```

#### Backend → Railway
1. **Go to [railway.app](https://railway.app)**
2. **Deploy from GitHub**:
   - Click "Deploy from GitHub repo"
   - Select your frame-truth repo
   - Railway will auto-detect Python

3. **Environment Variables** (in Railway dashboard):
   ```
   OPENROUTER_API_KEY=your_openrouter_key
   ADMIN_USER=admin
   ADMIN_PASS=your_secure_password
   PORT=8000
   ```

4. **Custom Start Command** (if needed):
   ```
   python backend/server.py
   ```

### Option 2: Vercel (Full-Stack)
1. **Go to [vercel.com](https://vercel.com)**
2. **Import your GitHub repo**
3. **Configure**:
   - Framework: Vite
   - Build command: `npm run build`
   - Output directory: `dist`

## 📋 Pre-Deployment Checklist

### 1. Update Backend for Production
The backend needs a small update for production deployment:

```python
# In backend/server.py, update the last lines:
if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 2. Update Frontend API URLs
Create a production environment file:

**`.env.production`**:
```
VITE_API_URL=https://your-backend-url.railway.app
```

### 3. Update CORS Settings
In `backend/server.py`, update CORS for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173", 
        "https://your-netlify-site.netlify.app",
        "https://your-custom-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🔧 Step-by-Step Deployment

### Step 1: Prepare Repository
```bash
# Make sure all files are committed
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### Step 2: Deploy Backend (Railway)
1. Go to [railway.app](https://railway.app)
2. Sign up/login with GitHub
3. Click "Deploy from GitHub repo"
4. Select your frame-truth repository
5. Railway will auto-detect it's a Python project
6. Add environment variables:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
   - `ADMIN_USER`: admin
   - `ADMIN_PASS`: Choose a secure password
   - `PORT`: 8000

7. Deploy! Railway will give you a URL like: `https://frame-truth-production.up.railway.app`

### Step 3: Deploy Frontend (Netlify)
1. Go to [netlify.com](https://netlify.com)
2. Click "New site from Git"
3. Connect GitHub and select your repo
4. Build settings:
   - Build command: `npm run build`
   - Publish directory: `dist`
5. Add environment variable:
   - `VITE_API_URL`: Your Railway backend URL
6. Deploy!

### Step 4: Update CORS
Once you have your Netlify URL, update the CORS settings in your backend code and redeploy.

## 🌐 Custom Domain (Optional)
- **Netlify**: Go to Domain settings → Add custom domain
- **Railway**: Go to Settings → Domains → Add custom domain

## 🔒 Security Notes
- Never commit `.env` files to Git
- Use strong passwords for admin access
- Consider adding rate limiting for production
- Monitor usage and costs

## 📊 Monitoring
- **Railway**: Built-in metrics and logs
- **Netlify**: Analytics and function logs
- **OpenRouter**: API usage dashboard

## 💰 Cost Estimates
- **Railway**: $5/month for hobby plan
- **Netlify**: Free tier available, $19/month for pro
- **OpenRouter**: Pay per API call (~$0.01-0.10 per analysis)

## 🆘 Troubleshooting

### Common Issues:
1. **CORS Errors**: Update allowed origins in backend
2. **API Key Issues**: Check environment variables
3. **Build Failures**: Check Node.js version (use 18+)
4. **Database Issues**: SQLite works fine for small scale

### Getting Help:
- Check Railway/Netlify logs
- Test API endpoints directly
- Verify environment variables are set

## 🎉 Success!
Once deployed, your Frame Truth app will be live and accessible worldwide!

**Frontend**: `https://your-site.netlify.app`
**Backend**: `https://your-backend.railway.app`
