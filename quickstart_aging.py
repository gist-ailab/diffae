import cv2
import time
import requests
import replicate

data = 'mid1'
target_age = "25"

image = open(f"imgs_align/{data}.png", "rb")

startTime = time.time()
output = replicate.run(
    "yuval-alaluf/sam:9222a21c181b707209ef12b5e0d7e94c994b58f01c7b2fec075d2e892362f13c",
    input={
        "image": image,
        "target_age": target_age
    }
)

endTime = time.time()

print(output)

img_data = requests.get(output).content
with open(f'outputs/aging/{data}_{target_age}.png', 'wb') as handler:
    handler.write(img_data)

print('time:', endTime-startTime)