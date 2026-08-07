from dotenv import load_dotenv
import os
import requests

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

headers = {
    "X-API-KEY": API_KEY
}

base_url = "https://api.openaq.org/v3"

# coordinate limits for NCR
# north: 29.9940391
# south: 26.702677
# east: 78.4822761
# west: 75.4688561
# bbox=75.4688561,26.702677,78.4822761,29.9940391

params = {
    "bbox": "75.4688561,26.702677,78.4822761,29.9940391",
    "parameters_id": 2,
    #"limit": 1
}

response = requests.get(f"{base_url}/locations", headers = headers, params = params)

# print(response.status_code)
data = response.json()
# print(data['results'][0])

pm25_sensors = []

for location in data['results']:
    for sensor in location['sensors']:
        if sensor['parameter']['id'] == 2:
            pm25_sensors.append({
                "location_id": location['id'],
                "location_name": location['name'],
                "sensor_id": sensor['id']
            })

# print(pm25_sensors)
# for sensor in pm25_sensors:
#     print(sensor['location_name'])
# print(len(pm25_sensors))

# print(pm25_sensors[3:5])

pm25_sensors_with_count = []

for sensor in pm25_sensors:
    response = requests.get(f"{base_url}/sensors/{sensor['sensor_id']}", headers = headers)
    data = response.json()
    if data["results"][0]["coverage"] is None:
        continue
    sensor['observed_count'] = data['results'][0]['coverage']['observedCount']
    pm25_sensors_with_count.append(sensor)

# print(pm25_sensors)

pm25_sensors_sorted = sorted(pm25_sensors_with_count, key=lambda d: d['observed_count'], reverse=True)

# for sensor in pm25_sensors_sorted[:5]:
#     print(
#         sensor['sensor_id'],
#         sensor["location_name"],
#         sensor["observed_count"]
#     )

# sensor with the most pm2.5 observations:
# sensor ID: 23534, name: New Delhi, observations: 79112

