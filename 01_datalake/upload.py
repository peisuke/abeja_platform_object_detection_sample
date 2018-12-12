import io
import time
from datetime import datetime
import cv2
from PIL import Image
import numpy as np
from abeja.datalake import APIClient

ORGANIZATION_ID = 'XXXXXXXXXXX'
CHANNEL_ID = 'XXXXXXXXXXXXX'
DATASOURCE_ID = 'datasource-XXXXXXXXXXXXXXX'
DATASOURCE_TOKEN = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

credential = {
    'user_id': DATASOURCE_ID,
    'personal_access_token': DATASOURCE_TOKEN
}

client = APIClient(credential=credential)

def upload(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = Image.fromarray(img)
    img_byte = io.BytesIO()
    img.save(img_byte, format='jpeg')
    data = img_byte.getvalue()
    content_type = 'image/jpeg'
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + '.jpg'
    metadata = {
        'x-abeja-meta-filename': filename
    }
    client.post_channel_file_upload(CHANNEL_ID, data, content_type, metadata=metadata)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
    
    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame, (width, height))
        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 13:
            upload(img)
