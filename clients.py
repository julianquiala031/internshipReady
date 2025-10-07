import requests

url = "https://127.0.0.1:5055/api/items"

response = requests.get(url).json()

print(response)