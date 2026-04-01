# Healthcare Symptom Checker

`Healthcare Symptom Checker` is a small FastAPI app with a simple frontend that lets someone describe symptoms in plain language and get an educational AI-generated summary.

It is a learning project, not a diagnosis tool. It should not replace a doctor, urgent care, or emergency services.

## What it does

- Takes symptom descriptions in plain language
- Runs a quick safety check before using the model
- Returns a structured response with a summary, possible conditions, next steps, and follow-up questions
- Stores recent requests in MongoDB so they can appear in the history panel

## How safety works

- `EMERGENCY`: certain high-risk phrases skip the LLM and return emergency guidance immediately
- `URGENT`: certain symptom combinations still use Gemini, but the response is prefixed with an urgent-care warning
- `NORMAL`: standard educational response

## Tech stack

- FastAPI and Uvicorn
- Gemini via `google-genai`
- MongoDB via PyMongo
- Static frontend in `frontend/index.html`
- Environment loading via `python-dotenv`

## Requirements

- Python 3.10 or newer
- A Gemini API key
- MongoDB

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-api-key
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=healthai

```

3. Start the app:

```bash
uvicorn backend.main:app --reload --port 8000
```
or
```bash
python backend/main.py 
```

4. Open `http://localhost:8000`

## Disclaimer

This project is for educational purposes only. Always talk to a qualified healthcare professional about medical symptoms. If someone may be having a medical emergency, call local emergency services right away.
