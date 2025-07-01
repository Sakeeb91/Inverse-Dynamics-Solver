# 🚀 Deployment Guide

This guide covers deployment options for the Differentiable Trebuchet application.

## Local Development

### Prerequisites
- Python 3.8+ (recommended: 3.9-3.11)
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/Sakeeb91/Inverse-Dynamics-Solver.git
cd Inverse-Dynamics-Solver

# Install dependencies
pip install -r requirements.txt

# Run basic tests
python test_basic.py

# Start the Streamlit application
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Cloud Deployment Options

### 1. Streamlit Cloud (Recommended)

**Steps:**
1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and connect your GitHub account
4. Select this repository and `app.py` as the main file
5. Deploy!

**Advantages:**
- Free hosting for public repositories
- Automatic deployment on git push
- Built-in authentication and sharing
- No server management required

### 2. Heroku

Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### 3. Google Cloud Platform

Use Cloud Run for containerized deployment:

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
```

Deploy:
```bash
gcloud run deploy --source .
```

### 4. AWS EC2

1. Launch an EC2 instance (t2.micro for testing)
2. Install Python and dependencies
3. Clone the repository
4. Use nginx as reverse proxy (optional)
5. Run with systemd for automatic restart

Example systemd service (`/etc/systemd/system/trebuchet.service`):
```ini
[Unit]
Description=Differentiable Trebuchet App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/Inverse-Dynamics-Solver
ExecStart=/usr/bin/python3 -m streamlit run app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### 5. Docker

Build and run locally:
```bash
docker build -t trebuchet-app .
docker run -p 8501:8501 trebuchet-app
```

For production, use docker-compose with nginx:

`docker-compose.yml`:
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
    restart: unless-stopped
```

## Production Considerations

### Performance Optimization

1. **Model Caching**: Enable Streamlit caching for trained models
```python
@st.cache_resource
def load_trained_model():
    # Load pre-trained model
    pass
```

2. **Data Caching**: Cache expensive computations
```python
@st.cache_data
def generate_training_data(n_samples):
    # Cache generated datasets
    pass
```

3. **Resource Limits**: Set appropriate memory and CPU limits

### Security

1. **Environment Variables**: Use environment variables for sensitive configuration
2. **Input Validation**: Validate all user inputs
3. **Rate Limiting**: Implement rate limiting for API endpoints
4. **HTTPS**: Always use HTTPS in production

### Monitoring

1. **Logging**: Add comprehensive logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

2. **Health Checks**: Implement health check endpoints
3. **Error Tracking**: Use services like Sentry for error tracking
4. **Performance Monitoring**: Monitor response times and resource usage

### Scaling

1. **Horizontal Scaling**: Use load balancers for multiple instances
2. **Database**: Consider using external database for model storage
3. **CDN**: Use CDN for static assets
4. **Caching**: Implement Redis for shared caching

## Environment Variables

Create `.env` file for configuration:
```env
# App Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true

# Model Configuration  
MODEL_CACHE_SIZE=100
TRAINING_MAX_SAMPLES=10000

# Logging
LOG_LEVEL=INFO
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce training sample size or use smaller models
2. **Slow Training**: Use fewer iterations or simpler network architecture
3. **Import Errors**: Ensure all dependencies are installed correctly
4. **Port Conflicts**: Change the default port in Streamlit configuration

### Debug Mode

Run in debug mode for detailed error messages:
```bash
streamlit run app.py --logger.level=debug
```

### Performance Profiling

Use Python profilers to identify bottlenecks:
```python
import cProfile
cProfile.run('your_function()')
```

## Support

For deployment issues:
1. Check the [GitHub Issues](https://github.com/Sakeeb91/Inverse-Dynamics-Solver/issues)
2. Review Streamlit documentation
3. Check cloud provider documentation
4. Submit new issues with detailed error messages

---

*This deployment guide covers the most common scenarios. Adapt the instructions based on your specific requirements and infrastructure.*