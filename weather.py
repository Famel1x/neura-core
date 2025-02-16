import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('WEATHER_API_KEY')

CITY = "Nizhny Novgorod"

url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
print(url)

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    weather_description = data['weather'][0]['description']
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    wind_speed = data['wind']['speed']
    
    print(f"Погода в Нижнем Новгороде: {weather_description}")
    print(f"Температура: {temperature}°C")
    print(f"Влажность: {humidity}%")
    print(f"Скорость ветра: {wind_speed} м/с")
else:
    print(f"Ошибка: {response.status_code}")