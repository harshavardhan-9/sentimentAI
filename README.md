# Sentiment Analysis API & Dashboard

## Overview
Full-stack application for analyzing sentiment in text data, featuring a FastAPI backend and React frontend with real-time visualization.

## Features
- 🔒 Secure JWT authentication
- 📊 Interactive data visualization
- 📈 Bulk CSV processing
- 🎨 Modern UI with Tailwind CSS
- 🐳 Docker deployment ready
- ✅ Comprehensive testing suite

## Reference Screenshots

## Login Screen 
![Screenshot (886)](https://github.com/user-attachments/assets/d4810cf6-9fb8-44ad-b58f-9b846815b47d)
## Main Dashboard
![Screenshot (887)](https://github.com/user-attachments/assets/0754de7e-12f1-431c-a86b-ae71c79ea1bf)
## API Documentation UI
![Screenshot (888)](https://github.com/user-attachments/assets/a58137c6-b6c7-4626-85b8-308fcbfe919c)



## Quick Start

### Using Docker
```bash
git clone https://github.com/harshavardhan-9/sentimentAI
cd sentiment-analysis
docker-compose up
```

### Manual Setup
1. Backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

2. Frontend:
```bash
cd frontend
npm install
npm start
```

## Usage

1. Access the dashboard at `http://localhost:3000`
2. Login with credentials:
   - Username: admin
   - Password: password
3. Upload CSV file with format:
```csv
id,text,timestamp
1,"Great product!",2024-01-01 10:00:00
```

## Development

### Project Structure
```
sentiment-analysis/
├── backend/          # FastAPI application
├── frontend/         # React application
└── docker-compose.yml
```

### Running Tests
```bash
# Backend tests
cd backend && pytest

# Frontend tests
cd frontend && npm test
```

## API Documentation
Swagger UI: `http://localhost:8000/docs`

## License
MIT License
