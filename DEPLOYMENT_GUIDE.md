# 🚀 DocuBrain Deployment Guide

Complete step-by-step deployment instructions for DocuBrain on Render.com (Backend) and Vercel (Frontend).

---

## 📋 Prerequisites

Before starting deployment, ensure you have:
- ✅ GitHub account with DocuBrain repository
- ✅ MongoDB Atlas account and connection string
- ✅ Emergent Universal API key
- ✅ Render.com account
- ✅ Vercel account

---

## 🗄️ Part 1: MongoDB Atlas Setup

### Step 1: Create MongoDB Atlas Cluster (if not already done)
1. Go to [MongoDB Atlas](https://cloud.mongodb.com/)
2. Sign in or create account
3. Create new cluster (Free tier is sufficient for testing)
4. Configure network access (allow from anywhere: 0.0.0.0/0)
5. Create database user with read/write permissions

### Step 2: Get Connection String
1. In Atlas dashboard, click "Connect"
2. Choose "Connect your application"
3. Copy the connection string
4. Replace `<password>` with your database user password
5. Your connection string should look like:
   ```
   mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
   ```

---

## 🔧 Part 2: Backend Deployment on Render.com

### Step 1: Prepare Your Repository
1. Ensure your repository has the following structure:
   ```
   backend/
   ├── server.py
   ├── database.py
   ├── lightweight_embeddings.py
   ├── requirements.txt
   └── .env (don't commit this)
   ```

2. Create `render.yaml` in your repository root:
   ```yaml
   services:
     - type: web
       name: docubrain-backend
       env: python
       buildCommand: cd backend && pip install -r requirements.txt
       startCommand: cd backend && python server.py
       envVars:
         - key: MONGO_URL
           value: your_mongodb_connection_string
         - key: DATABASE_NAME
           value: askmydocs
         - key: EMERGENT_LLM_KEY
           value: sk-emergent-xxxxx
         - key: CORS_ORIGINS
           value: "*"
   ```

### Step 2: Deploy on Render.com
1. **Sign up/Login** to [Render.com](https://render.com/)

2. **Connect GitHub Repository**:
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select your DocuBrain repository

3. **Configure Service Settings**:
   ```
   Name: docubrain-backend
   Region: Choose closest to your users
   Branch: main (or your default branch)
   Runtime: Python 3
   Build Command: cd backend && pip install -r requirements.txt
   Start Command: cd backend && python server.py
   ```

4. **Add Environment Variables**:
   Click "Advanced" → "Add Environment Variable":
   ```
   MONGO_URL = mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
   DATABASE_NAME = askmydocs
   EMERGENT_LLM_KEY = sk-emergent-64dAeF36b167dF470D
   CORS_ORIGINS = *
   ```

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for deployment to complete (5-10 minutes)
   - Your backend will be available at: `https://your-app-name.onrender.com`

### Step 3: Test Backend Deployment
1. Visit `https://your-app-name.onrender.com/`
2. You should see: `{"message": "DocuBrain API is running! 🧠", ...}`
3. Test API documentation at: `https://your-app-name.onrender.com/docs`

---

## 🎨 Part 3: Frontend Deployment on Vercel

### Step 1: Prepare Frontend Configuration
1. Update `frontend/.env` with your backend URL:
   ```
   REACT_APP_BACKEND_URL=https://your-app-name.onrender.com
   ```

2. Ensure `frontend/package.json` has correct build scripts:
   ```json
   {
     "scripts": {
       "start": "craco start",
       "build": "craco build",
       "test": "craco test"
     }
   }
   ```

### Step 2: Deploy on Vercel
1. **Sign up/Login** to [Vercel](https://vercel.com/)

2. **Import Project**:
   - Click "New Project"
   - Import from GitHub (connect if needed)
   - Select your DocuBrain repository

3. **Configure Project Settings**:
   ```
   Project Name: docubrain-frontend
   Framework Preset: Create React App
   Root Directory: frontend
   Build Command: yarn build
   Output Directory: build
   Install Command: yarn install
   ```

4. **Add Environment Variables**:
   In project settings → Environment Variables:
   ```
   REACT_APP_BACKEND_URL = https://your-backend-name.onrender.com
   ```

5. **Deploy**:
   - Click "Deploy"
   - Wait for deployment (2-5 minutes)
   - Your frontend will be available at: `https://your-project-name.vercel.app`

### Step 3: Test Frontend Deployment
1. Visit your Vercel URL
2. You should see the DocuBrain login page
3. Test registration and login functionality
4. Try uploading a document and asking questions

---

## 🔗 Part 4: Connect Frontend and Backend

### Step 1: Update CORS Settings (if needed)
If you face CORS issues, update your backend `server.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["https://your-frontend-url.vercel.app"],  # Specific URL
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Step 2: Update Frontend Environment
In Vercel dashboard:
1. Go to Project Settings → Environment Variables
2. Update `REACT_APP_BACKEND_URL` if backend URL changed
3. Redeploy if needed

---

## 🔍 Part 5: Verification & Testing

### Backend Health Check
```bash
curl https://your-backend-name.onrender.com/
# Should return: {"message": "DocuBrain API is running! 🧠", ...}
```

### Frontend Functionality Test
1. **Registration**: Create new account
2. **Login**: Sign in with credentials
3. **Upload**: Try uploading a PDF file
4. **Query**: Ask questions about uploaded documents
5. **API Key**: Check if API key is displayed correctly

### External API Test
```bash
curl -X POST "https://your-backend-name.onrender.com/api/external/query" \
  -F "api_key=your-api-key" \
  -F "question=What is this document about?"
```

---

## ⚙️ Part 6: Custom Domain Setup (Optional)

### For Render.com (Backend)
1. Go to your service dashboard
2. Click "Settings" → "Custom Domains"
3. Add your domain (e.g., `api.yourdomain.com`)
4. Configure DNS records as instructed

### For Vercel (Frontend)
1. Go to Project Settings → Domains
2. Add your domain (e.g., `yourdomain.com`)
3. Configure DNS records as instructed
4. SSL certificates are automatically managed

---

## 📊 Part 7: Monitoring & Maintenance

### Render.com Monitoring
- **Logs**: Check service logs for errors
- **Metrics**: Monitor CPU, memory usage
- **Health Checks**: Set up health check endpoints
- **Alerts**: Configure failure notifications

### Vercel Monitoring
- **Analytics**: Enable Vercel Analytics
- **Performance**: Monitor Core Web Vitals
- **Deployment**: Track deployment status
- **Edge Network**: Monitor global CDN performance

### Database Monitoring (MongoDB Atlas)
- **Performance Advisor**: Review query performance
- **Real-time Metrics**: Monitor connections, operations
- **Alerts**: Set up alerts for performance issues
- **Backup**: Configure automated backups

---

## 🚨 Troubleshooting Common Issues

### Backend Issues

#### **Deployment Fails**
```bash
# Check build logs in Render dashboard
# Common fixes:
- Verify requirements.txt includes all dependencies
- Check Python version compatibility
- Ensure file paths are correct
```

#### **MongoDB Connection Error**
```bash
# Check connection string format
# Whitelist Render IP or use 0.0.0.0/0
# Verify database user permissions
```

#### **Environment Variables Not Loading**
```bash
# Verify variables are set in Render dashboard
# Check variable names match your code
# Restart service after adding variables
```

### Frontend Issues

#### **Build Fails**
```bash
# Check build logs in Vercel dashboard
# Common fixes:
- Verify package.json dependencies
- Check for TypeScript errors
- Ensure proper import statements
```

#### **API Connection Error**
```bash
# Verify REACT_APP_BACKEND_URL is correct
# Check CORS settings in backend
# Test backend API directly
```

#### **Environment Variables Not Working**
```bash
# Ensure variables start with REACT_APP_
# Redeploy after adding variables
# Check browser network tab for API calls
```

---

## 🔧 Advanced Configuration

### Performance Optimization

#### Backend (Render.com)
- **Instance Type**: Upgrade to higher tier for better performance
- **Health Checks**: Configure proper health check endpoints
- **Scaling**: Enable auto-scaling based on traffic

#### Frontend (Vercel)
- **Build Optimization**: Enable build cache
- **Edge Functions**: Use for API routes if needed
- **Image Optimization**: Use Vercel's image optimization

### Security Enhancements

#### Environment Security
- Use Render's encrypted environment variables
- Rotate API keys regularly
- Implement proper CORS policies
- Add rate limiting to APIs

#### Database Security
- Use MongoDB Atlas security features
- Enable audit logging
- Configure IP whitelisting
- Regular security updates

---

## 📝 Deployment Checklist

### Pre-Deployment
- [ ] MongoDB Atlas cluster configured
- [ ] Environment variables prepared
- [ ] Repository code updated and tested
- [ ] Dependencies updated in requirements.txt and package.json

### Backend Deployment
- [ ] Render.com account created
- [ ] Repository connected to Render
- [ ] Environment variables configured
- [ ] Service deployed successfully
- [ ] Health check endpoint responding
- [ ] API documentation accessible

### Frontend Deployment
- [ ] Vercel account created
- [ ] Repository connected to Vercel
- [ ] Build configuration correct
- [ ] Environment variables set
- [ ] Deployment successful
- [ ] Frontend accessible and functional

### Post-Deployment
- [ ] End-to-end functionality tested
- [ ] External API integration tested
- [ ] Performance monitoring setup
- [ ] Error logging configured
- [ ] Backup procedures in place
- [ ] Documentation updated with new URLs

---

## 📞 Support Resources

### Official Documentation
- [Render.com Docs](https://render.com/docs)
- [Vercel Docs](https://vercel.com/docs)
- [MongoDB Atlas Docs](https://docs.atlas.mongodb.com/)

### Community Support
- Render Community Forum
- Vercel Discord
- MongoDB Community Forum
- Stack Overflow (tag specific platform)

### Emergency Contacts
- Keep backend and frontend URLs documented
- Maintain list of environment variables
- Document database connection details
- Save API keys securely

---

**🎉 Congratulations!** Your DocuBrain application is now deployed and ready for production use!

For additional support or custom deployment requirements, refer to the individual platform documentation or community forums.