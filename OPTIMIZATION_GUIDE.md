# Optimization Guide for Future Deployment

## Current Optimizations Already Implemented ✅
- **Lazy Loading**: ML models load on-demand instead of at startup
- **Health Check**: Optimized to not trigger model loading
- **Memory Monitoring**: Added psutil for tracking memory usage
- **Thread-Safe Singletons**: Prevents duplicate model loading

## Additional Optimizations for Next Version

### 1. Model Quantization (Reduces memory by 50-70%)
```python
# Current: sentence-transformers loads full models
# Optimized: Use quantized models
from sentence_transformers import SentenceTransformer

# Instead of:
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Use quantized version:
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                           model_kwargs={'quantization_config': {'load_in_8bit': True}})
```

### 2. External API Integration (Eliminates local ML dependencies)
```python
# Replace local embeddings with OpenAI API
from openai import OpenAI

client = OpenAI(api_key=settings.openai_api_key)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
```

### 3. Redis Session Storage (Better than in-memory)
```python
# Current: In-memory session storage
# Optimized: Redis for production
import redis
from redis import Redis

redis_client = Redis(host='localhost', port=6379, db=0)

def save_session(session_id, data):
    redis_client.setex(f"session:{session_id}", 3600, json.dumps(data))

def get_session(session_id):
    data = redis_client.get(f"session:{session_id}")
    return json.loads(data) if data else None
```

### 4. Request Batching (Improves throughput)
```python
# Current: Process requests one at a time
# Optimized: Batch similar requests
from collections import defaultdict

class BatchProcessor:
    def __init__(self, batch_size=10, timeout=0.1):
        self.batch = []
        self.batch_size = batch_size
        self.timeout = timeout
    
    async def process(self, item):
        self.batch.append(item)
        if len(self.batch) >= self.batch_size:
            return await self.flush()
        return await self.wait_and_flush()
```

### 5. Caching Layer (Reduces redundant computations)
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_embedding(text: str) -> np.ndarray:
    return get_embedding_model().encode_one(text)

def get_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()
```

### 6. Connection Pooling (Reduces overhead)
```python
# For database/API connections
from httpx import AsyncClient

http_client = AsyncClient(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    timeout=30.0
)
```

### 7. Graceful Degradation (Handles memory pressure)
```python
import psutil
import gc

def check_memory_usage():
    process = psutil.Process()
    memory_percent = process.memory_info().rss / psutil.virtual_memory().total * 100
    
    if memory_percent > 80:
        logger.warning(f"High memory usage: {memory_percent:.1f}%")
        gc.collect()  # Force garbage collection
        
        if memory_percent > 90:
            # Unload least recently used models
            unload_idle_models()
```

## Performance Monitoring Additions

### 1. Request Timing
```python
import time
from fastapi import Request

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {duration:.3f}s")
    return response
```

### 2. Model Loading Metrics
```python
import time

def get_embedding_model() -> EmbeddingModel:
    global _embedding_model_instance, _load_time
    
    if _embedding_model_instance is not None:
        return _embedding_model_instance
    
    start_time = time.time()
    with _embedding_model_lock:
        if _embedding_model_instance is not None:
            return _embedding_model_instance
        
        settings = get_settings()
        _embedding_model_instance = EmbeddingModel(settings.embedding_model_name)
        _load_time = time.time() - start_time
        logger.info(f"Embedding model loaded in {_load_time:.2f}s")
        
    return _embedding_model_instance
```

## Configuration for Different Environments

### Development (Local)
```python
# .env.development
DEBUG_MODE=true
SKIP_MODEL_LOADING=false
SESSION_BACKEND=memory
# Use smaller models for faster iteration
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

### Production (Paid Hosting)
```python
# .env.production
DEBUG_MODE=false
SKIP_MODEL_LOADING=false
SESSION_BACKEND=redis
REDIS_URL=redis://your-redis-instance:6379/0
# Use larger models for better quality
EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
```

### Lightweight (Free Hosting)
```python
# .env.lightweight
DEBUG_MODE=false
SKIP_MODEL_LOADING=true  # Disable local models
SESSION_BACKEND=memory
# Use external APIs
USE_OPENAI_EMBEDDINGS=true
OPENAI_API_KEY=your-key
```

## Deployment-Specific Dockerfiles

### Lightweight Dockerfile (Free Hosting)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install only essential dependencies
COPY requirements-light.txt .
RUN pip install --no-cache-dir -r requirements-light.txt

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Full ML Dockerfile (Paid Hosting)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Non-root user for security
RUN useradd -r -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Migration Strategy

### Phase 1: Current Setup (No changes)
- Continue development locally
- Focus on features and functionality
- Test thoroughly

### Phase 2: Lightweight Version (Free deployment)
- Implement external API integrations
- Remove heavy ML dependencies
- Deploy to free platform
- Gather user feedback

### Phase 3: Optimized ML Version (Paid deployment)
- Add model quantization
- Implement Redis caching
- Optimize performance
- Upgrade to paid hosting if needed

## Cost Optimization Tips

1. **Start Free**: Use lightweight version on free hosting
2. **Scale Gradually**: Only upgrade when you hit limits
3. **Monitor Usage**: Track requests, memory, CPU usage
4. **Cache Aggressively**: Reduce API calls and computations
5. **Use Spot Instances**: For GPU workloads (80% cheaper)

## Monitoring Setup

```python
# Add to main.py
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    request_count.inc()
    request_duration.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

This guide gives you a clear path from current setup to production-ready deployment.