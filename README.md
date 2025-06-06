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

## Quick Start

### Using Docker
```bash
git clone https://github.com/mrigankraj/sentiment-analysis
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

## Contributing
1. Fork repository
2. Create feature branch
3. Commit changes
4. Open pull request

## License
MIT License

## Support
- Documentation: See `docs/` directory
- Issues: GitHub Issues
- Contact: mrigankraj2003@gmail.com
