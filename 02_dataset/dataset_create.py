import json
import argparse
import os
import abeja

from abeja.datasets import APIClient
from abejacli.config import (
    ABEJA_PLATFORM_USER_ID, ABEJA_PLATFORM_TOKEN
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotation Data Importer: Detection')
    parser.add_argument('--input', '-i', required=True, help='Annotated data from Annotation Tool')
    parser.add_argument('--organization', '-o', required=True, help='Organization ID')
    parser.add_argument('--attributes', '-a', default='attributes/attributes_detection.json', help='Attributes file')
    parser.add_argument('--dataset', '-d', required=True, help='Output Dataset Name')
    args = parser.parse_args()

    credential = {
        'user_id': ABEJA_PLATFORM_USER_ID,
        'personal_access_token': ABEJA_PLATFORM_TOKEN
    }
    
    organization_id = args.organization
    datasetname = args.dataset

    with open(args.attributes, 'r') as f:
        attributes = json.load(f)

    category_name = attributes['attributes'][0]['name']
    categories = attributes['attributes'][0]['categories']

    labels =[]
    label_to_id = {}
    for count, cat in enumerate(categories):
        label_to_id[cat['name']] = count
        labels.append({
            'label': cat['name'],
            'label_id': count
        })

    with open(args.input, 'r') as f:
        data = json.load(f)

    category = {
        'labels': labels,
        'category_id': 0,
        'name': datasetname
    }
    props = {'categories': [category]}

    api_client = APIClient(credential)
    dataset = api_client.create_dataset(organization_id, datasetname, 'custom', props)

    dataset_id = dataset['dataset_id']

    for d in data:
        channel_id = d['task']['metadata'][0]['channel_id']
  
        '''
        "information": [
          {
            "rect": [
              359.1836734693877,
              193.9591836734694,
              469.55102040816325,
              325.8775510204082
            ],
            "color": "rgb(249, 123, 106)",
            "classes": [
              {
                "id": 1,
                "name": "Cup",
                "category_id": 0
              }
            ]
        }]
        '''
        info = []
        for rect in d['information']:    
            label = rect['classes'][0]['name']
            label_id = label_to_id[label]
            bbox = rect['rect']
            info.append({
                'category_id': 0,
                'label': label,
                'label_id': int(label_id),
                'rect': {
                    'xmin': bbox[0],
                    'ymin': bbox[1],
                    'xmax': bbox[2],
                    'ymax': bbox[3],
                }
            })
        filename = d['task']['metadata'][0]['information']['filename']
        file_id = d['task']['metadata'][0]['source']
    
        if os.path.splitext(filename)[1].lower() == '.jpg' or        os.path.splitext(filename)[1].lower() == '.jpeg':
            content_type = 'image/jpeg'
        elif os.path.splitext(filename)[1].lower() == '.png':
            content_type = 'image/png'
        else:
            print('{} is invalid file type.'.format(filename))
            continue
 
        data_uri = 'datalake://{}/{}'.format(channel_id, file_id)
        source_data = [{'data_uri': data_uri, 'data_type': content_type}]
        attributes = {'detection': info}
        api_client.create_dataset_item(organization_id, dataset_id, source_data, attributes)