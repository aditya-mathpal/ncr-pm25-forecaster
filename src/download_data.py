import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")
SENSOR_ID = 23534

url = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/measurements"
headers = {"X-API-Key": API_KEY}

all_measurements = []
page = 1

while True:
    # Page size limit max is 1000
    params = {"limit": 1000, "page": page}
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        break
        
    data = response.json()
    results = data.get("results", [])
    
    if not results:
        break  # End of records
        
    all_measurements.extend(results)
    page += 1

print(f"Retrieved {len(all_measurements)} total measurement records.")

# 2. Flatten OpenAQ v3 nested dictionary structure
df = pd.json_normalize(all_measurements)

# Extract and convert UTC timestamp from the flattened 'period.datetimeFrom.utc' column
if "period.datetimeFrom.utc" in df.columns:
    df["datetime"] = pd.to_datetime(df["period.datetimeFrom.utc"])

# 3. Save to compressed Parquet format
df.to_parquet("sensor_data.parquet", engine="pyarrow", compression="snappy")

# Fast load verify
df_loaded = pd.read_parquet("sensor_data.parquet")
print(f"Saved and verified {len(df_loaded)} rows in Parquet format.")