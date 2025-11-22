import pandas as pd
import numpy as np
from typing import Optional
import re
from word2number import w2n

# Only keep a function to send data to OpenAI for cleaning
import pandas as pd
import os
from openai import OpenAI

def clean_with_openai(df: pd.DataFrame, model="gpt-5-nano") -> pd.DataFrame:
    """
    Clean the dataset using OpenAI GPT-5 Nano. All cleaning is handled by the model.
    """
    csv_data = df.to_csv(index=False)
    prompt = f"""
You are a data cleaning assistant. Clean the following CSV data. Remove duplicates, fix missing values, correct obvious errors, and standardize formats. Return the cleaned CSV data only, with the same columns.

CSV DATA:
{csv_data}
"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = "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"
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.2
        )
        cleaned_csv = response.choices[0].message.content.strip()
        from io import StringIO
        df_cleaned = pd.read_csv(StringIO(cleaned_csv))
        return df_cleaned
    except Exception as e:
        raise RuntimeError(f"OpenAI cleaning failed: {e}")