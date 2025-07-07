import requests

# Use your access token from step 2
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiJkZDFmODRkYTg2YWU0ODllOGFmZDcxNmIyYmJhOThhYiIsImp0aSI6IjIxNWEyMmRiMzNjMTAxM2U3YzM5NGE3ZDA3MmY3NTkzYjVlMjg0NGNkZmM1MmRhMDI0MGJhNTFlNTAwMmJkOTRjZDcxMDNjZDEwMTkwMDJkIiwiaWF0IjoxNzUxNDY2NzgwLjA0ODAzOSwibmJmIjoxNzUxNDY2NzgwLjA0ODA0LCJleHAiOjMzMzA4Mzc1NTgwLjA0MjA3LCJzdWIiOiI3ODY3MjkyNSIsImlzcyI6Imh0dHBzOi8vbWV0YS53aWtpbWVkaWEub3JnIiwicmF0ZWxpbWl0Ijp7InJlcXVlc3RzX3Blcl91bml0Ijo1MDAwLCJ1bml0IjoiSE9VUiJ9LCJzY29wZXMiOlsiYmFzaWMiXX0.DKRG73FtprwTj8iLWil96iHaSn2Lue_Qmhmt9K1C_9_7oRUJEbvCs_jRqCTaD5BcF6bs70EBV5zgdDIfK0NYkBRYeanlhq1vFyGdrh3JoQIvrFaP4qN0OljxDmMXMH8fksHn0Ab2mtOEsTATJpHrwYEr-MUqj9kpi4B9AFnVk1utQWNCkdMUw0M1cePKW_4tgUkGS8LLH4kDyByPkB8nIiXTW1mggZK7b_H76uBolHkU8GlmWXcwvwLvrSZ-e4H2LV--TL9YItsxcLAra11CQtl__QX9OzJv52QE7MrH22sgt5j2gq-VEeiOy-ZjGS2GgPbT_6G0Y5-U_agIAxBSJXAzXksb7vvhpjQA65gp5YboziuAelwUH-v2xOJGJJfmvYWZMH18IhJUDae2nWPX_jAoy7yHdmuyvdwMx-Vj4BFlF6wKSTfWD613x3WLhKeQaARA4C0TMNEgPlduqK74bmN4_Lp5RZ7Qm6zJUBiyOowx7AonqtKk8wQfuwcEjT7LEdpO8Ajx1DyL7F-ax_i45wqcXyMHfux4U_E4KFgFtvEugXpKUo3E_gYUB93yyyN4C5mtSeYVvbQyF-Ofqc1zlbey6dzgNr1cRZd-no7HqTyxvecTH9C4SYuL0d-kg27pdfcAA1jqNK_HFt2BU-HmtqCnblon2p7DoE4cNtAMaqc"  # Truncated here for safety

# Example revision ID to test
# Get from : curl "https://en.wikipedia.org/w/api.php?action=query&list=recentchanges&rcnamespace=118&rclimit=1&format=json"


REV_ID = 1298442750 # Replace with a real rev_id from Wikipedia

# API endpoint
url = "https://api.wikimedia.org/service/lw/inference/v1/models/enwiki-drafttopic:predict"

# HTTP headers
headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

# Request body
data = {
    "rev_id": REV_ID
}

# Send the POST request
response = requests.post(url, headers=headers, json=data)

# Print the result
if response.status_code == 200:
    print("Prediction:")
    print(response.json())
else:
    print(f"Error {response.status_code}: {response.text}")
