# SeatRacer Railway Deployment Guide

## Quick Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

## Manual Deployment

1. **Connect Repository to Railway**
   - Go to [Railway](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select this repository

2. **Environment Configuration**
   - Railway will automatically detect the `railway.toml` configuration
   - The app will be deployed with the correct start command
   - No additional environment variables needed

3. **Deployment Files Included**
   - `railway.toml` - Railway configuration
   - `Procfile` - Process definition for Railway
   - `requirements.txt` - Python dependencies
   - `runtime.txt` - Python version specification
   - `.streamlit/config.toml` - Streamlit configuration

## Using the Deployed App

### CSV Data Upload
- The deployed app supports CSV file upload through the web interface
- Navigate to the "Data" tab
- Use "Upload Racing Data" section to upload your CSV files
- Maximum file size: 200MB

### Example Data
- If example datasets are included in the deployment, they will appear in the "Load Example Datasets" section
- If no example data is available, users can upload their own CSV files

## Local Development vs Production

### Local Development
- Includes debugpy for VS Code debugging (on port 5678)
- Debug setup is automatically disabled in Railway production environment

### Production (Railway)
- Debuggy is disabled for production
- CORS and XSRF protection are disabled for Streamlit functionality
- Headless mode enabled for server deployment

## Configuration Details

### Streamlit Configuration
```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[theme]
font = "B612"
```

### Railway Configuration
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "streamlit run seatracer/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"
```

## Troubleshooting

### Build Issues
- Ensure all dependencies are listed in `requirements.txt`
- Check Python version in `runtime.txt` is supported by Railway

### Runtime Issues
- Check Railway logs for error messages
- Verify CSV file format matches expected structure
- Ensure uploaded files are valid CSV format

### Memory Issues
- Railway provides 512MB RAM by default
- Large datasets may require upgrading Railway plan
- Consider data preprocessing for very large files

## CSV File Format Requirements

Your CSV file should include columns for:
- Athlete names/IDs
- Race results/times
- Boat configurations
- Race dates/sessions
- Other performance metrics as required by the analysis models

Refer to example datasets (if available) for the expected format.