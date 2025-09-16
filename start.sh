#!/bin/bash

# Railway deployment start script
echo "Starting SeatRacer application..."

# Install dependencies if needed
pip install -r requirements.txt

# Run the Streamlit app
exec streamlit run seatracer/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true