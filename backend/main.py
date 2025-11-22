# ...existing code...


from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Request
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
GOOGLE_CLIENT_ID = "867940367822-7lpadjpmjvktcd6jgq4o94ul58di7f3h.apps.googleusercontent.com"

def verify_google_token(token: str):
    try:
        idinfo = id_token.verify_oauth2_token(token, grequests.Request(), GOOGLE_CLIENT_ID)
        return idinfo
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token"
        )

def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = auth_header.split(" ")[1]
    return verify_google_token(token)

from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from ai_service import clean_dataset, generate_ai_summary
from db import database, users
import datetime


app = FastAPI(title="CSV Data Cleaner API")
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://clean-my-data.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Data Cleaner API is running. Upload CSV to /clean"}


@app.post("/clean")
async def clean_csv(file: UploadFile = File(...), user=Depends(get_current_user)):
    print("[DEBUG] Google user info received:", user)
    print("[DEBUG] Extracted email:", user.get("email"))
    # User management and usage tracking
    google_id = user["sub"]
    email = user["email"]
    now = datetime.datetime.utcnow()
    # Check if user exists
    query = users.select().where(users.c.google_id == google_id)
    user_record = await database.fetch_one(query)
    if not user_record:
        # New user, insert
        insert_query = users.insert().values(
            google_id=google_id,
            email=email,
            cleanings_this_month=0,
            last_reset=now
        )
        await database.execute(insert_query)
        cleanings_this_month = 0
        last_reset = now
    else:
        cleanings_this_month = user_record["cleanings_this_month"]
        last_reset = user_record["last_reset"]
        # Reset monthly usage if needed
        if not last_reset or (now.year != last_reset.year or now.month != last_reset.month):
            update_query = users.update().where(users.c.google_id == google_id).values(
                cleanings_this_month=0,
                last_reset=now
            )
            await database.execute(update_query)
            cleanings_this_month = 0
            last_reset = now
        # Enforce monthly limit (3 cleanings per month)
        if cleanings_this_month >= 3:
            return {
                "success": False,
                "error": "Monthly cleaning limit reached. Please try again next month."
            }

    """
    Upload CSV → Get cleaned CSV + AI summary.
    Does everything automatically.
    """
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Try to read CSV with error handling
        try:
            df_original = pd.read_csv(io.BytesIO(contents))
        except pd.errors.ParserError as e:
            # Try with error_bad_lines=False (for pandas < 1.3) or on_bad_lines='skip' (for pandas >= 1.3)
            try:
                df_original = pd.read_csv(io.BytesIO(contents), on_bad_lines='skip')
            except TypeError:
                # Fallback for older pandas versions
                df_original = pd.read_csv(io.BytesIO(contents), error_bad_lines=False, warn_bad_lines=True)
            
            return {
                "success": False,
                "error": f"CSV parsing error: {str(e)}. Some malformed rows were skipped. Please check your CSV format - make sure all rows have the same number of columns."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read CSV: {str(e)}. Please ensure your file is a valid CSV format."
            }

        # Validate we have data
        if df_original.empty:
            return {
                "success": False,
                "error": "The CSV file is empty or contains no valid data."
            }


        # Increment usage only if under the limit
        update_query = users.update().where(users.c.google_id == google_id).values(
            cleanings_this_month=cleanings_this_month + 1
        )
        await database.execute(update_query)

        # Apply all cleaning operations
        df_cleaned, operations, before_summary, after_summary, cleaning_instructions, warnings = clean_dataset(df_original)

        # Generate AI summary
        ai_summary = generate_ai_summary(before_summary, after_summary, operations)

        # Prepare cleaned CSV
        csv_buffer = io.StringIO()
        df_cleaned.to_csv(csv_buffer, index=False)

        # Preview (first 10 rows) for cleaned data
        preview_df = df_cleaned.head(10)
        # Ensure boolean columns are shown as True/False strings
        preview_df = preview_df.copy()
        for col in preview_df.columns:
            # Only apply .isin([True, False]) if the column is not datetime
            if preview_df[col].dtype == bool:
                preview_df[col] = preview_df[col].map(lambda x: 'True' if x is True else ('False' if x is False else None))
            elif str(preview_df[col].dtype) != 'datetime64[ns]' and preview_df[col].dropna().isin([True, False]).all():
                preview_df[col] = preview_df[col].map(lambda x: 'True' if x is True else ('False' if x is False else None))
        preview_data = preview_df.replace({float('nan'): None}).values.tolist()

        # Preview (first 10 rows) for original data
        before_preview_df = df_original.head(10)
        before_preview_data = before_preview_df.replace({float('nan'): None}).values.tolist()

        return {
            "success": True,
            "csv_string": csv_buffer.getvalue(),
            "summary": ai_summary,
            "operations": operations,
            "before": before_summary,
            "after": after_summary,
            "cleaning_instructions": cleaning_instructions,
            "warnings": warnings,
            "preview": {
                "columns": preview_df.columns.tolist(),
                "data": preview_data
            },
            "before_preview": {
                "columns": before_preview_df.columns.tolist(),
                "data": before_preview_data
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Cleaning failed: {str(e)}"
        }

@app.get("/usage")
async def get_usage(user=Depends(get_current_user)):
    google_id = user["sub"]
    now = datetime.datetime.utcnow()
    query = users.select().where(users.c.google_id == google_id)
    user_record = await database.fetch_one(query)
    if not user_record:
        # New user, insert
        insert_query = users.insert().values(
            google_id=google_id,
            email=user["email"],
            cleanings_this_month=0,
            last_reset=now
        )
        await database.execute(insert_query)
        cleanings_this_month = 0
        last_reset = now
    else:
        cleanings_this_month = user_record["cleanings_this_month"]
        last_reset = user_record["last_reset"]
        # Reset monthly usage if needed
        if not last_reset or (now.year != last_reset.year or now.month != last_reset.month):
            update_query = users.update().where(users.c.google_id == google_id).values(
                cleanings_this_month=0,
                last_reset=now
            )
            await database.execute(update_query)
            cleanings_this_month = 0
            last_reset = now
    usage_limit = 3
    next_reset = (last_reset.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
    return {
        "requestsUsed": cleanings_this_month,
        "requestsLimit": usage_limit,
        "requestsRemaining": usage_limit - cleanings_this_month,
        "nextResetDate": next_reset.strftime("%B %d, %Y")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
