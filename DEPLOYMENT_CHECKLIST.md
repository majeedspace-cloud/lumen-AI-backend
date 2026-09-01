# Deployment Checklist for Next Version

## Current Status
- ✅ Lazy loading implemented for ML models
- ✅ Health check optimized (no model loading)
- ✅ Memory monitoring added
- ✅ Local testing successful (~470MB startup)
- ❌ FastAPI Cloud deployment failed (OOM during build)

## Next Version Planning

### Option A: Lightweight Version (Free Hosting Compatible)
**Changes Needed:**
- [ ] Replace `sentence-transformers` with OpenAI embeddings API
- [ ] Replace local reranker with API-based solution
- [ ] Remove heavy ML dependencies (torch, transformers)
- [ ] Use cloud-based LLM (already using Gemini - good!)
- [ ] Update requirements.txt
- [ ] Test with lighter dependencies

**Target Platforms:**
- FastAPI Cloud (free tier)
- Vercel
- Railway (free tier)
- Netlify

**Estimated Effort:** 2-3 days of refactoring

### Option B: Optimized ML Version (Paid Hosting)
**Changes Needed:**
- [ ] Add model quantization to reduce memory
- [ ] Implement request batching
- [ ] Add Redis for session storage (currently in-memory)
- [ ] Optimize FAISS index size
- [ ] Add model warmup endpoint
- [ ] Create proper Dockerfile

**Target Platforms:**
- Render ($7/month)
- Railway ($5/month)
- DigitalOcean ($5/month)
- AWS/GCP (free tier limits)

**Estimated Effort:** 1-2 days of optimization

### Option C: Microservices Architecture
**Changes Needed:**
- [ ] Split ML service from FastAPI backend
- [ ] Create separate ML inference service
- [ ] Add API gateway/routing
- [ ] Implement service communication
- [ ] Add monitoring between services

**Target Platforms:**
- ML service: GPU hosting (RunPod, Lambda Labs)
- API service: Standard hosting (Render, Railway)

**Estimated Effort:** 5-7 days of restructuring

## Pre-Deployment Checklist (Any Option)
- [ ] Complete feature development
- [ ] Add comprehensive error handling
- [ ] Implement logging and monitoring
- [ ] Add rate limiting (already done - good!)
- [ ] Security audit (API keys, CORS, etc.)
- [ ] Load testing
- [ ] Documentation update
- [ ] Environment variables documentation
- [ ] Backup/recovery plan

## Hosting Platform Comparison

### Free Options (Require Option A)
| Platform | RAM | CPU | Cost | ML Support |
|----------|-----|-----|------|------------|
| FastAPI Cloud | 500MB | 0.5 vCPU | Free | ❌ Heavy ML |
| Vercel | 1GB | 0.6 vCPU | Free | ❌ Heavy ML |
| Railway | 512MB | 0.5 vCPU | Free | ❌ Heavy ML |

### Low-Cost Options (Option B)
| Platform | RAM | CPU | Cost | ML Support |
|----------|-----|-----|------|------------|
| Render | 2GB | 1 vCPU | $7/mo | ✅ Good |
| Railway | 1GB | 0.5 vCPU | $5/mo | ✅ Adequate |
| DigitalOcean | 2GB | 1 vCPU | $6/mo | ✅ Good |

## Recommended Path
**For next version:** Start with Option A (lightweight) to get deployed free, then upgrade to Option B (optimized ML) when you need advanced features.

## Current Setup Documentation
- **Backend**: FastAPI with lazy-loaded ML models
- **ML Models**: sentence-transformers, CrossEncoder, Gemini API
- **Vector Store**: FAISS (in-memory)
- **Session Storage**: In-memory (consider Redis for production)
- **Frontend**: Separate (not currently integrated)
- **Deployment Target**: Was FastAPI Cloud (failed due to OOM)

## Environment Variables Needed
- GEMINI_API_KEY
- HF_TOKEN  
- TAVILY_API_KEY
- API_KEY (for your app)
- ALLOWED_ORIGINS
- SKIP_MODEL_LOADING (for testing)

## Next Steps
1. Decide on Option A, B, or C
2. Implement required changes
3. Test thoroughly locally
4. Choose hosting platform
5. Deploy and monitor
6. Gather user feedback
7. Iterate based on feedback