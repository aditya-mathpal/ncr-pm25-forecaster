# Delhi PM2.5 Forecasting

## Setup

### Environment Setup
```
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Generate API key at https://airquality.xkdr.org/signup

Create a file `.env` in the root directory and add the following line:
```
AQI_API_KEY=<your API key from XKDR>
```

### Download data
```
cd src
python download_data.py
cd ..
python src/prepare_data.py
```

## Data Attribution

This project uses air quality data provided by the **Central Pollution Control Board (CPCB)** and the **United States Department of State**, accessed through the **XKDR Forum**.

Any publication resulting from this project will credit **CPCB, the US Department of State, and XKDR Forum** as the data sources/providers.