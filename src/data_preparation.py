"""
data_preparation.py

Export annotations from LabelBox
Perform train/test split
Move images into proper directories

BoMeyering 2025
"""

from dotenv import load_dotenv
import os
import labelbox as lb
import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
from typing import List
from shutil import copyfile

# Load .env file
load_dotenv()

# extract API key and project ID
api_key = os.getenv("LB_API_KEY")
project_id = os.getenv("PROJECT_ID")

# Create a labelbox client
client = lb.Client(api_key=api_key)
project = client.get_project(project_id=project_id)

# Set the data export parameters
export_params = {
    "attachments": False,
    "metadata_fields": False,
    "data_row_details": True,
    "projects_details": True,
    "label_details": True,
    "performance_details": False
}

# Filter for images that are 'done'
filters = {
    "workflow_status": "Done"
}

# Export and wait till done
export_task = project.export(
    params=export_params,
    filters=filters
)
print("Exporting Data from Labelbox")
export_task.wait_till_done()

# Stream the export using a callback function
def json_stream_handler(output: lb.BufferedJsonConverterOutput):
  return output.json

export_task.get_buffered_stream(stream_type=lb.StreamType.RESULT).start(stream_handler=json_stream_handler)

# Collect all exported data into a list
export_json = [data_row.json for data_row in export_task.get_buffered_stream()]

print("file size: ", export_task.get_total_file_size(stream_type=lb.StreamType.RESULT))
print("line count: ", export_task.get_total_lines(stream_type=lb.StreamType.RESULT))

df_list = []

# Main loop through JSON results and append data to df_list
print("Looping through results to create labels")
for i in range(len(export_json)):
    data = export_json[i]
    data_row_id = data['data_row']['id']
    external_id = data['data_row']['external_id']
    height = data['media_attributes']['height']
    width = data['media_attributes']['width']

    label_list = data['projects'][project_id]['labels'][0]['annotations']['objects']

    for row in label_list:
        feature_id = str(row['feature_id'])
        feature_class = str(row['name'])
        bbox = row['bounding_box']
        xmin = int(bbox['left'])
        ymin = int(bbox['top'])
        xmax = int(xmin + bbox['width'])
        ymax = int(ymin + bbox['height'])

        label_row = {
            'data_row_id': data_row_id,
            'external_id': external_id,
            'feature_id': feature_id,
            'height': height,
            'width': width,
            'feature_class': feature_class,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        }

        df_list.append(label_row)

# Create a dataframe out of the dict-list
res_df = pl.DataFrame(df_list)

# Value counts for each labeled object in all images for train-test-split stratification
print("Aggregating data")
data_agg = res_df.group_by(pl.col('external_id'))\
    .agg(pl.col('feature_class').value_counts())\
    .explode('feature_class')\
    .unnest('feature_class')\
    .pivot(values="count", index="external_id", on="feature_class", aggregate_function="first")\
    .fill_null(0)

# Create quantiles for data stratification
pod_quantiles = data_agg['pod'].qcut(quantiles=4, labels=['a', 'b', 'c', 'd'], allow_duplicates=True)
seed_quantiles = data_agg['seed'].qcut(quantiles=4, labels=['a', 'b', 'c', 'd'], allow_duplicates=True)
split_quantiles = data_agg['split'].qcut(quantiles=4, labels=['a', 'b', 'c', 'd'], allow_duplicates=True)

quantile_df = pl.DataFrame({
    "pod_q": pod_quantiles,
    "seed_q": seed_quantiles,
    "split_q": split_quantiles
})

print(data_agg.shape, quantile_df.shape)

# Add quantiles to dataframe
data_agg = pl.concat([data_agg, quantile_df], how='horizontal')

data_agg.write_csv("data_agg.csv")

# Clear directory function
def clear_dirs(dirs: list[str]):
    for dir in dirs:
        file_glob = glob('*', root_dir=dir)
        if len(file_glob)>0:
            print(f"Removing existing files in {dir}")
            os.system(f'rm -rf {dir}*')

# Split data and move images
def split_mv_imgs(src_dir: str, train_dir: str, val_dir: str, id_df: pl.DataFrame, train_size: float=0.8, remove: bool=False):
    if remove:
        clear_dirs([train_dir, val_dir])

    train_ids, val_ids = train_test_split(id_df, train_size=train_size, stratify=id_df['seed_q', 'split_q'])
    train_list = train_ids['external_id'].to_list()
    val_list = val_ids['external_id'].to_list()
    
    # move training images
    for f in train_list:
        print(f)
        copyfile(src=os.path.join(src_dir, f), dst=os.path.join(train_dir, f))

    # move validation images
    for f in val_list:
        print(f)
        copyfile(src=os.path.join(src_dir, f), dst=os.path.join(val_dir, f))

    return train_list, val_list

# Clear directories and move images
train_list, val_list = split_mv_imgs(
    src_dir='./data/images/all_images',
    train_dir='./data/images/train/', 
    val_dir='./data/images/val/',
    id_df=data_agg,
    remove=True
)

# Separate train and val dataframes and write to csv
print(train_list, val_list)
train_df = res_df.filter(pl.col('external_id').is_in(train_list))
val_df = res_df.filter(pl.col('external_id').is_in(val_list))

train_df.write_csv('./data/annotations/train_annotations.csv')
val_df.write_csv('./data/annotations/val_annotations.csv')

data_agg.write_csv('./data/annotations/data_label_counts.csv')

print(train_df, val_df)