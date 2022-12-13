from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import pickle
import numpy as np
import pandas as pd
import io
import csv
import codecs


app = FastAPI()

def to_df(file):
    data = file.file
    data = csv.reader(codecs.iterdecode(data,'utf-8'), delimiter=',')
    header = data.__next__()
    df = pd.DataFrame(data, columns=header)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df

def get_float_from_file(filename):
    res = 0
    with open(filename, 'r') as r:
        res = float(r.read())
    print(res)
    return res

def prepare_data(df, transformer):
    mileage_med = get_float_from_file('meleage_med')
    engine_med = get_float_from_file('engine_med')
    max_power_med = get_float_from_file('max_power_med')
    seats_med = get_float_from_file('seats_med')

    df['mileage'] = df['mileage'].str.extract('([-+]?\d*\.?\d+)').astype(float)
    df['engine'] = df['engine'].str.extract('([-+]?\d*\.?\d+)').astype(float)
    df['max_power'] = df['max_power'].str.extract('([-+]?\d*\.?\d+)').astype(float)
    df = df.drop('torque', axis=1)
    df['mileage'] = df.mileage.fillna(mileage_med)
    df['engine'] = df['engine'].fillna(engine_med)
    df['max_power'] = df['max_power'].fillna(max_power_med)
    df['seats'] = df['seats'].fillna(seats_med)
    df = transformer.transform(df)
    df = pd.DataFrame(df, columns=transformer.get_feature_names())
    assert (df.isnull().sum().sum() == 0) and (df.isnull().sum().sum() == 0)
    df['mileage'] = df['mileage'].astype(float)
    df['seats'] = df['seats'].astype(float)
    df['engine'] = df['engine'].astype(float)
    df['km_driven'] = df['km_driven'].astype(float)
    df['year'] = df['year'].astype(float)
    df['my_new_feature_1'] = np.log(1 + df['mileage'])
    df['my_new_feature_2'] = np.log(df['km_driven'] + 1)
    df['my_new_feature_3'] = np.log(df['engine'] + 1)
    df['my_new_feature_4'] = df['year'] - 2000
    df = df.drop(['onehotencoder__x0_CNG', 'onehotencoder__x1_Dealer', 'onehotencoder__x2_Automatic', 'onehotencoder__x3_First Owner'], axis=1)
    return df

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
async def predict_item(item_file: UploadFile) -> float:
    sc = pickle.load(open('./scaler.pkl','rb'))
    transformer = pickle.load(open('./transformer.pkl', 'rb'))
    ridge = pickle.load(open('./ridge.pkl', 'rb'))
    data = to_df(item_file) 
    data = prepare_data(data, transformer)
    return ridge.predict(data)[0]

@app.post("/predict_items")
def predict_items(item_file: UploadFile) -> List[float]:
    sc = pickle.load(open('./scaler.pkl','rb'))
    transformer = pickle.load(open('./transformer.pkl', 'rb'))
    ridge = pickle.load(open('./ridge.pkl', 'rb'))
    data = to_df(item_file) 
    print('before get data')
    data = prepare_data(data, transformer)
    print('after get data')
    return list(ridge.predict(data))


@app.post("/")
async def root():
    return {"message": "helo"}
