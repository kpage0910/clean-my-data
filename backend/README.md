# Clean My Data - Backend API

FastAPI backend for CSV data cleaning and validation.

## Setup

1. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

| Method | Endpoint              | Description               |
| ------ | --------------------- | ------------------------- |
| POST   | `/upload`             | Upload a CSV file         |
| POST   | `/scan`               | Scan file for data issues |
| POST   | `/preview`            | Preview cleaning changes  |
| POST   | `/apply`              | Apply cleaning rules      |
| GET    | `/download/{file_id}` | Download cleaned file     |

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
