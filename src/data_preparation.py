from dotenv import load_dotenv
import os
import labelbox as lb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
from typing import List
from shutil import copyfile

load_dotenv()

api_key = os.getenv("LB_API_KEY")
project_id = os.getenv("PROJECT_ID")

# Create a labelbox client
client = lb.Client(api_key=api_key)
project = client.get_project(project_id=project_id)

export_params = {
    "attachments": False,
    "metadata_fields": False,
    "data_row_details": True,
    "projects_details": True,
    "label_details": True,
    "performance_details": False
}

filters = {
    "workflow_status": "Done"
}

export_task = project.export_v2(
    params=export_params,
    filters=filters
)

export_task.wait_till_done()

if export_task.errors:
    print(export_task.errors)

export_json = export_task.result

res_df = pd.DataFrame(
    columns=[
        'data_row_id',
        'external_id',
        'img_height',
        'img_width',
        'feature_id',
        'class',
        'xmin',
        'ymin',
        'xmax',
        'ymax'
    ]
)

df_dict = {}

for i in range(len(export_json)):
    data = export_json[i]
    data_row_id = data['data_row']['id']
    external_id = data['data_row']['external_id']
    height = data['media_attributes']['height']
    width = data['media_attributes']['width']

    label_list = data['projects'][project_id]['labels'][0]['annotations']['objects']

    for row in label_list:
        feature_id = row['feature_id']
        feature_class = row['name']
        bbox = row['bounding_box']
        xmin = bbox['left']
        ymin = bbox['top']
        xmax = xmin + bbox['width']
        ymax = ymin + bbox['height']
        df_dict[feature_id] = [
            data_row_id,
            external_id, 
            feature_id,
            height,
            width,
            feature_class,
            xmin,
            ymin,
            xmax,
            ymax
        ]
res_df = pd.DataFrame.from_dict(
    data=df_dict, 
    orient='index',
    columns=[
        'data_row_id',
        'external_id',
        'feature_id',
        'img_height',
        'img_width',
        'class',
        'xmin',
        'ymin',
        'xmax',
        'ymax'
    ]
).reset_index(drop=True)

data_rows = res_df.external_id.unique()

def clear_dirs(dirs: list[str]):
    for dir in dirs:
        file_glob = glob('*', root_dir=dir)
        if len(file_glob)>0:
            print(f"Removing existing files in {dir}")
            os.system(f'rm -rf {dir}*')
    
def split_mv_imgs(src_dir: str, train_dir: str, val_dir: str, id_list: list[str], train_size: float=0.8, remove: bool=False):
    if remove:
        clear_dirs([train_dir, val_dir])

    train_ids, val_ids = train_test_split(pd.DataFrame({"data_rows": data_rows}), train_size=train_size)
    train_list = train_ids['data_rows'].tolist()
    val_list = val_ids['data_rows'].tolist()
    
    # move training images
    for f in train_list:
        print(f)
        copyfile(src=os.path.join(src_dir, f), dst=os.path.join(train_dir, f))

    # move validation images
    for f in val_list:
        print(f)
        copyfile(src=os.path.join(src_dir, f), dst=os.path.join(val_dir, f))

    return train_list, val_list


train_list, val_list = split_mv_imgs(
    src_dir='./data/images/all_images',
    train_dir='./data/images/train/', 
    val_dir='./data/images/val/',
    id_list=data_rows,
    remove=True
)

train_df = res_df.loc[res_df['external_id'].isin(train_list)]
val_df = res_df.loc[res_df['external_id'].isin(val_list)]

train_df.to_csv('./data/annotations/train_annotations.csv')
val_df.to_csv('./data/annotations/val_annotations.csv')

print(train_df, val_df)