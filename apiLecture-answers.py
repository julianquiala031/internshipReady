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

my_api_key = os.getenv("NASA_KEY")

#Build the final API url
final_url = url+my_api_key

#Make the API request
apod_retrieved = requests.get(final_url).json()

#Serialize the result to a JSON document
main_functions.save_to_file(apod_retrieved, "apod.json")

#Deserialize the recently create JSON document
apod= main_functions.read_from_file("apod.json")

#What is the type of the object deserialized?
print(type(apod))

#What are its keys?
print(apod.keys())

#Access some of their values passing their keys
print(apod["title"])

print(apod["explanation"])

print(apod["url"])