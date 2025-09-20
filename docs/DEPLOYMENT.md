# Ultimate Tennis Predictor - Deployment Guide

Complete production deployment guide for the Ultimate Tennis Predictor system.

## 🚀 Quick Start (Local Development)

```bash
# 1. Clone and setup
git clone https://github.com/Milkpainter/ultimate-tennis-predictor.git
cd ultimate-tennis-predictor

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Initialize database and load data
python scripts/train_models.py --load-atp --data-years 2020 2021 2022 2023 2024

# 4. Train models
python scripts/train_models.py --validation

# 5. Start API server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000

## 📦 Docker Deployment

### Development Environment
```bash
# Build development image
docker build --target development -t tennis-predictor:dev .

# Run development container
docker run -p 8000:8000 -v $(pwd):/app tennis-predictor:dev
```

### Production Environment  
```bash
# Build production image
docker build --target production -t tennis-predictor:prod .

# Run with environment variables
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/tennis \
  -e REDIS_URL=redis://redis:6379/0 \
  tennis-predictor:prod
```

### Docker Compose (Recommended)
```yaml
# docker-compose.yml
version: '3.8'

services:
  tennis-api:
    build:
      context: .
      target: production
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://tennis:password@postgres:5432/tennis_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=tennis
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=tennis_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

Run with: `docker-compose up -d`

## ⚙️ Kubernetes Production Deployment

### Prerequisites
- Kubernetes cluster (1.25+)
- kubectl configured
- Helm 3.x installed

### Configuration Files

#### Namespace and RBAC
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: tennis-predictor
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: tennis-predictor
  namespace: tennis-predictor
```

#### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tennis-predictor
  namespace: tennis-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tennis-predictor
  template:
    metadata:
      labels:
        app: tennis-predictor
    spec:
      serviceAccountName: tennis-predictor
      containers:
      - name: tennis-api
        image: ghcr.io/milkpainter/ultimate-tennis-predictor:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tennis-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

#### Service and Ingress
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: tennis-predictor-service
  namespace: tennis-predictor
spec:
  selector:
    app: tennis-predictor
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tennis-predictor-ingress
  namespace: tennis-predictor
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.tennis-predictor.com
    secretName: tennis-predictor-tls
  rules:
  - host: api.tennis-predictor.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: tennis-predictor-service
            port:
              number: 80
```

### Deploy to Kubernetes
```bash
# Apply all configurations
kubectl apply -f k8s/

# Wait for deployment
kubectl rollout status deployment/tennis-predictor -n tennis-predictor

# Check pods
kubectl get pods -n tennis-predictor

# View logs
kubectl logs -f deployment/tennis-predictor -n tennis-predictor
```

## 📊 Monitoring and Observability

### Prometheus Metrics
The API exposes metrics at `/metrics` endpoint:

- `tennis_predictions_total`: Total predictions made
- `tennis_prediction_duration_seconds`: Prediction latency
- `tennis_model_accuracy`: Current model accuracy
- `tennis_api_errors_total`: API error count

### Grafana Dashboards
Key metrics to monitor:

1. **API Performance**
   - Request rate (RPS)
   - Response latency (p50, p95, p99)
   - Error rate
   - Active users

2. **Model Performance**
   - Prediction accuracy (overall, by surface)
   - Model agreement levels
   - Feature importance drift
   - Calibration metrics

3. **Business Metrics**
   - Betting ROI performance
   - Sharpe ratio trends
   - Maximum drawdown
   - Prediction confidence distribution

### Logging
Structured JSON logs with fields:
```json
{
  "timestamp": "2024-09-20T05:30:00Z",
  "level": "INFO",
  "logger": "tennis_predictor.predictor",
  "message": "Match prediction completed",
  "player1": "Novak Djokovic",
  "player2": "Carlos Alcaraz", 
  "prediction": 0.68,
  "confidence": 0.82,
  "latency_ms": 45.2,
  "surface": "hard",
  "tournament_level": "Grand Slam"
}
```

## 🔧 Configuration

### Environment Variables

**Required:**
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

**Optional:**
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING)
- `MODEL_RETRAIN_SCHEDULE`: Cron expression for retraining
- `MAX_PREDICTION_BATCH_SIZE`: Maximum batch predictions (default: 50)
- `PREDICTION_CACHE_TTL`: Cache TTL in seconds (default: 300)
- `API_RATE_LIMIT`: Requests per minute per IP (default: 1000)

### Configuration File
Create `config/production.yaml`:
```yaml
models:
  ensemble_weights:
    xgboost: 0.25
    lightgbm: 0.20
    catboost: 0.15
    random_forest: 0.15
    neural_net: 0.15
    logistic: 0.10

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_batch_size: 50
  rate_limit: 1000

validation:
  accuracy_threshold: 0.65
  retrain_threshold: 0.60
  validation_frequency_days: 7

data:
  min_matches_per_player: 10
  feature_selection_threshold: 0.01
  cache_ttl_seconds: 300
```

## 🔍 Health Checks and Monitoring

### Health Check Endpoint
`GET /health` returns:
```json
{
  "status": "healthy",
  "timestamp": "2024-09-20T05:30:00Z",
  "models_loaded": true,
  "database_connected": true,
  "total_predictions": 12547,
  "average_latency_ms": 42.3,
  "model_accuracy": 0.723
}
```

### Alerting Rules (Prometheus)
```yaml
groups:
- name: tennis-predictor
  rules:
  - alert: HighErrorRate
    expr: rate(tennis_api_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: High error rate detected
  
  - alert: ModelAccuracyDrop
    expr: tennis_model_accuracy < 0.60
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: Model accuracy below threshold
  
  - alert: HighLatency
    expr: histogram_quantile(0.95, tennis_prediction_duration_seconds) > 0.2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: Prediction latency too high
```

## 📝 API Usage Examples

### Basic Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "player1": "Novak Djokovic",
       "player2": "Carlos Alcaraz",
       "surface": "hard",
       "tournament_level": "Grand Slam",
       "best_of": 5,
       "round_name": "Final",
       "temperature": 25.0,
       "humidity": 60.0,
       "match_importance": 2.0
     }'
```

### Batch Predictions
```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
     -H "Content-Type: application/json" \
     -d '[
       {
         "player1": "Novak Djokovic",
         "player2": "Carlos Alcaraz",
         "surface": "hard",
         "tournament_level": "Grand Slam"
       },
       {
         "player1": "Jannik Sinner", 
         "player2": "Daniil Medvedev",
         "surface": "clay",
         "tournament_level": "Masters"
       }
     ]'
```

### Player Information
```bash
curl "http://localhost:8000/api/v1/players/Novak%20Djokovic"
```

### Head-to-Head Analysis
```bash
curl "http://localhost:8000/api/v1/h2h/Novak%20Djokovic/Carlos%20Alcaraz"
```

## 🔄 Model Training and Updates

### Manual Retraining
```bash
# Retrain with latest data
curl -X POST "http://localhost:8000/api/v1/retrain" \
     -H "Content-Type: application/json" \
     -d '{
       "min_date": "2020-01-01",
       "surfaces": ["Clay", "Grass", "Hard"]
     }'
```

### Automated Retraining
Models automatically retrain weekly via GitHub Actions:
- Triggers: Every Sunday at 2 AM UTC
- Downloads latest ATP/WTA data
- Retrains all models with validation
- Deploys updated models if performance improved

### Training Script Options
```bash
# Full training pipeline
python scripts/train_models.py \
  --load-atp \
  --load-wta \
  --validation \
  --data-years 2018 2019 2020 2021 2022 2023 2024 \
  --min-matches 5000

# Surface-specific training
python scripts/train_models.py \
  --surfaces Clay Grass \
  --validation

# Quick training for development
python scripts/train_models.py \
  --min-matches 100 \
  --data-years 2023 2024
```

## 🔒 Security and Authentication

### API Security
- Rate limiting: 1000 requests/minute per IP
- CORS protection with configurable origins
- Request validation and sanitization
- Error handling without information leakage

### Production Security
```yaml
# Add to deployment.yaml
env:
- name: API_KEY_REQUIRED
  value: "true"
- name: ALLOWED_ORIGINS
  value: "https://tennis-predictor.com,https://app.tennis-predictor.com"
- name: MAX_BATCH_SIZE
  value: "10"
```

### Database Security
- Connection encryption (SSL/TLS)
- Separate read/write permissions
- Regular backups with encryption
- Access logging and monitoring

## 📊 Scaling Considerations

### Horizontal Scaling
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tennis-predictor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tennis-predictor
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Performance Optimization

**API Level:**
- Response caching with Redis
- Connection pooling
- Async request processing
- Request compression

**Model Level:**
- Model quantization for faster inference
- Feature preprocessing caching
- Batch prediction optimization
- GPU acceleration for neural networks

**Database Level:**
- Read replicas for queries
- Index optimization
- Connection pooling
- Query result caching

## 🚨 Troubleshooting

### Common Issues

**1. Models not loading:**
```bash
# Check model files exist
ls -la models/trained/

# Retrain if needed
python scripts/train_models.py --min-matches 100
```

**2. Database connection errors:**
```bash
# Test database connectivity
python -c "from tennis_predictor.data_loader import TennisDataLoader; import asyncio; asyncio.run(TennisDataLoader().get_database_stats())"
```

**3. High memory usage:**
- Reduce model ensemble size
- Implement model quantization
- Add memory limits in Kubernetes
- Use streaming for large datasets

**4. Slow predictions:**
- Check feature engineering performance
- Enable prediction caching
- Use async processing for batch requests
- Monitor database query performance

### Debug Commands
```bash
# Check API health
curl http://localhost:8000/health

# View model performance
curl http://localhost:8000/api/v1/performance

# Check database stats
curl http://localhost:8000/api/v1/stats/database

# View recent predictions
curl http://localhost:8000/api/v1/predictions/recent

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Log Analysis
```bash
# View application logs
kubectl logs -f deployment/tennis-predictor -n tennis-predictor

# Search for errors
kubectl logs deployment/tennis-predictor -n tennis-predictor | grep ERROR

# Monitor prediction latency
kubectl logs deployment/tennis-predictor -n tennis-predictor | grep "prediction_latency_ms"
```

## 🔄 Backup and Recovery

### Database Backups
```bash
# PostgreSQL backup
pg_dump tennis_db > backup_$(date +%Y%m%d).sql

# Automated daily backups
0 2 * * * pg_dump tennis_db | gzip > /backups/tennis_$(date +\%Y\%m\%d).sql.gz
```

### Model Backups
- Models are versioned in GitHub artifacts
- Automatic backup before each retrain
- Point-in-time recovery available

### Disaster Recovery
1. **Database Recovery**: Restore from latest backup
2. **Model Recovery**: Rollback to previous model version
3. **Infrastructure**: Recreate from Infrastructure as Code
4. **Data Recovery**: Re-download from original sources

## 🎯 Performance Targets

### API Performance
- **Response Time**: <100ms (p95)
- **Throughput**: 1000+ RPS
- **Availability**: 99.9% uptime
- **Error Rate**: <0.1%

### Model Performance  
- **Accuracy**: 70-75% overall
- **Calibration**: Brier score <0.25
- **Consistency**: <5% accuracy variance
- **Training Time**: <4 hours full retrain

### Resource Usage
- **Memory**: <4GB per replica
- **CPU**: <2 cores per replica
- **Storage**: <100GB for database
- **Network**: <1Gbps bandwidth

---

📞 **Support**: For deployment issues, create a GitHub issue or check the [troubleshooting guide](TROUBLESHOOTING.md).