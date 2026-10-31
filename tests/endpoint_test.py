import requests
import json

url = "https://doctordekho-dev.nexeagle.com/api/search/parse"

payload = "{\"query\":\"sar dard\",\"variables\":{}}"
headers = {
  'accept': '*/*',
  'accept-language': 'en-US,en;q=0.6',
  'content-type': 'application/json',
  'origin': 'https://doctordekho-dev.nexeagle.com',
  'priority': 'u=1, i',
  'referer': 'https://doctordekho-dev.nexeagle.com/',
  'sec-ch-ua': '"Not:A-Brand";v="8", "Chromium";v="156", "Brave";v="156"',
  'sec-ch-ua-mobile': '?0',
  'sec-ch-ua-platform': '"Windows"',
  'sec-fetch-dest': 'empty',
  'sec-fetch-mode': 'cors',
  'sec-fetch-site': 'same-origin',
  'sec-gpc': '1',
  'user-agent': 'Mozilla/5.0 ^(Windows NT 10.0; Win64; x64^) AppleWebKit/537.36 ^(KHTML, like Gecko^) Chrome/156.0.0.0 Safari/537.36'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
