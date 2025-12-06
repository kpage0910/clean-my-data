# Clean My Data - Frontend

Minimal React/Next.js frontend for the Clean My Data application.

## Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── UploadForm.tsx      # CSV file upload with drag & drop
│   │   ├── IssuesList.tsx      # Displays detected issues by column
│   │   └── PreviewTable.tsx    # Side-by-side original vs cleaned preview
│   ├── lib/
│   │   └── api.ts              # API client for backend communication
│   ├── pages/
│   │   ├── _app.tsx            # App wrapper
│   │   ├── index.tsx           # Upload & scan page
│   │   └── preview.tsx         # Preview changes page
│   └── styles/
│       └── globals.css         # Tailwind CSS imports
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── next.config.js
```

## Setup

1. Install dependencies:

   ```bash
   cd frontend
   npm install
   ```

2. Configure the backend URL (optional):
   Create a `.env.local` file:

   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. Run the development server:

   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000)

## Features

- **Upload**: Drag & drop or click to upload CSV files
- **Scan**: Automatically scans uploaded files for data issues
- **Select Issues**: Choose which issues to fix (grouped by column)
- **Preview**: View original vs cleaned data side-by-side
- **Apply**: Download the cleaned file (requires backend /apply endpoint)

## API Endpoints Used

- `POST /upload` - Upload CSV file
- `POST /scan` - Scan file for issues
- `POST /preview` - Preview cleaning changes

## Tech Stack

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Fetch API for HTTP requests
