import requests
import json
import sys
# from haversine import haversine

# country = sys.argv[1]
country = "Ireland"

schema = {
    "city": {
        "type": "string",
        "description": "Name of the city"
    },

    "lat": {
        "type": "float",
        "description": "Decimal Latitude of the city"
    },

    "lon": {
        "type": "float",
        "description": "Decimal longitude of the city"
    }
}

payload = {
    "model": "llama3:8b-instruct-q8_0",
    "messages": [
        {"role": "system", "content": f"""You are a helpful AI assistant. The user will enter a
        country name and the assistant will return
        the decimal latitude and decimal longitude of
        the capital of the country. Output in JSON using the schema below {schema}."""},
        # {"role": "user", "content": "Ireland"},
        # {"role": "assistant", "content": "{'city': 'Dublin', 'lat': 53.3498, 'lon': -6.2603}"},
        {"role": "user", "content": country}
    ],
    "format": "json",
    "temperature": 0,
    "stream": False
}

response = requests.post("http://localhost:11434/api/chat", json=payload)

# for message in response.iter_lines():
#     jsonstr = json.loads(message)
#     print(jsonstr["message"]["content"], end="")

cityinfo = json.loads(response.json()["message"]["content"])

print(cityinfo)

# distance = haversine((mylat, mylon), (cityinfo['lat'], cityinfo['lon']), unit='mi')

# print(f"Bainbridge Island is about {int(round(distance, -1))} miles away from {cityinfo['city']}")