"""
You will need to visit https://api.nasa.gov/ and get a free API key
You will need to create the files:
    1) .env
    2) .gitignore
You will need to install:
    1) pip install python-dotenv
    2) pip install requests
    To use your environment variable:
    from dotenv import load_dotenv
    load_dotenv()
    os.getenv("SECRET_KEY")
"""

import requests
import main_functions
from dotenv import load_dotenv
import os
load_dotenv()

# NASA's API url
url = "https://api.nasa.gov/planetary/apod?api_key="

api_key = os.getenv("NASA_KEY")

final_url = url + api_key

response = requests.get(final_url).json()

print(type(response))
print(response.keys())
print(response["title"])
print(response["date"])
print(response["explanation"])

