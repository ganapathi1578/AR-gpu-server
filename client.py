import requests

url = 'http://10.23.66.242:12345/api/'
video_path = r"c:\Users\GANAPATHI\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\TempState\3BBA69A182957D20ED02F89A5AD436BA\WhatsApp Video 2025-05-25 at 23.24.58_2022b2e2.mp4"
files = {'video': open(video_path, 'rb')}

response = requests.post(url, files=files)

print(response.json())
