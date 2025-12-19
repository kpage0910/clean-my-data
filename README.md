# Clean My Data

A simple web tool that helps you clean messy CSV files—with clear, previewable fixes.

---

## What It Is

Clean My Data is a lightweight data cleaning tool built for people who work with spreadsheets but aren't data engineers.

Upload a CSV, see what's wrong, preview the fixes, and download a cleaned version. No scripting required.

---

## The Problem

Cleaning data is tedious. Most tools either:

- Do too much automatically (and break things)
- Require you to write code
- Hide what they're doing

This makes data cleaning feel risky and unpredictable—especially for users who aren't sure what "clean" even means for their dataset.

---

## How This Helps

Clean My Data takes a different approach:

1. **Upload** your CSV file
2. **See** detected issues (missing values, inconsistent formatting, duplicates, etc.)
3. **Preview** exactly what will change—before anything happens
4. **Download** the cleaned file when you're ready

Every transformation is visible. Nothing changes without your approval.

---

## Key Features

- **Issue Detection** — Scans for missing values, duplicates, formatting inconsistencies, and more
- **Side-by-Side Preview** — See original vs. cleaned values before committing
- **Deterministic Cleaning** — Same input always produces the same output
- **AI Summaries (Optional)** — Human-readable explanations of data quality issues (AI is never used to modify your data)

---

## Demo

| Upload                 | Scan                          | Preview                           |
| ---------------------- | ----------------------------- | --------------------------------- |
| Drag and drop your CSV | See detected issues by column | Compare original vs. cleaned data |

---

## How It Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Upload     │ ──▶ │    Scan      │ ──▶ │   Preview    │ ──▶ │   Download   │
│   CSV File   │     │   for Issues │     │   Changes    │     │   Cleaned    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

1. **Upload**: The backend stores your file temporarily and assigns it an ID
2. **Scan**: The scanner analyzes your data and detects issues (missing values, duplicates, type inconsistencies, etc.)
3. **Preview**: You see exactly what will change—row by row, column by column
4. **Apply & Download**: Cleaning rules are applied and you get your cleaned CSV

All transformations are deterministic. The same file will always produce the same result.

---

## Tech Stack

**Backend**

- Python 3.10+
- FastAPI
- Pandas
- OpenAI API (optional, for summaries only)

**Frontend**

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS

---

## Design Decisions

### Why deterministic transformations?

AI is powerful, but unpredictable. For data cleaning, predictability matters more than cleverness.

All cleaning operations in Clean My Data are rule-based:

- Trim whitespace
- Normalize capitalization (`JOHN DOE` → `John Doe`)
- Convert number words (`thirty` → `30`)
- Standardize date formats
- Remove duplicates

These rules are explicit, testable, and reversible.

### Why preview before apply?

Data cleaning tools that auto-fix everything can cause silent data loss. By showing changes first, users stay in control and can catch mistakes before they happen.

### Why AI is used sparingly

AI generates the **summary explanations** of data quality issues—not the transformations themselves. This keeps the core cleaning logic auditable while still providing helpful context for non-technical users.

---

## Tradeoffs & Limitations

| Decision              | Tradeoff                                          |
| --------------------- | ------------------------------------------------- |
| Deterministic only    | Can't handle complex imputation or fuzzy matching |
| Preview-first         | Adds an extra step before cleaning                |
| No in-place editing   | Must download the cleaned file separately         |
| AI for summaries only | Summaries require an OpenAI API key               |

This tool is best for:

- Small to medium CSV files
- Straightforward cleaning tasks
- Users who want visibility into what's changing

It's not designed for:

- Large-scale ETL pipelines
- Complex data transformations
- Real-time data processing

---

## Running Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) OpenAI API key for AI summaries

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:3000`

### Environment Variables

Create a `.env` file in the `backend/` directory:

```
OPENAI_API_KEY=your_api_key_here  # Optional, for AI summaries
```

---

## API Endpoints

| Method | Endpoint              | Description               |
| ------ | --------------------- | ------------------------- |
| POST   | `/upload`             | Upload a CSV file         |
| POST   | `/scan`               | Scan file for data issues |
| POST   | `/preview`            | Preview cleaning changes  |
| POST   | `/apply`              | Apply cleaning rules      |
| GET    | `/download/{file_id}` | Download cleaned file     |

Full API documentation: `http://localhost:8000/docs`

---

## Future Improvements

- Support for Excel files (.xlsx)
- Custom cleaning rules
- Batch processing for multiple files
- Export cleaning recipes for reuse
- More granular issue selection

---

## License

MIT

---

Built with care for anyone who's ever stared at a messy spreadsheet and wondered where to start.
