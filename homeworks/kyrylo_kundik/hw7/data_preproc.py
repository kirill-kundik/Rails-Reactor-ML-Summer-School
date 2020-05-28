import re

import numpy as np
import pandas as pd

STREET_TYPES = ['проспект', 'улица', 'шоссе', 'переулок', 'бульвар', 'дорога', 'плато', 'майдан', 'проезд']

cols = ['id',
        'title',
        'created_at',
        'price_usd',
        'description',
        'street_name',
        'state_name',
        'city_name',
        'total_square_meters',
        'living_square_meters',
        'kitchen_square_meters',
        'rooms_count',
        'floor',
        'wall_type',
        'inspected',
        'verified_price',
        'latitude',
        'longitude',
        'construction_year',
        'heating',
        'seller',
        'water',
        'building_condition',
        'dist_to_center',
        'dist_to_railway_station',
        'dist_to_airport',
        'floors_count',
        'price_uah']


def process():
    categorical = [cols[5], cols[7], cols[13], cols[19], cols[20], cols[21], cols[22]]
    numeric = [cols[8], cols[9], cols[10], cols[11], cols[12], cols[18], cols[26]]
    print(numeric)
    print(categorical)
    print(len(categorical))
    print(len(numeric))
    print(len(cols))
    exit(1)
    df = pd.read_csv('data/apartments_raw.csv')
    df = df.drop(
        [cols[0], cols[1], cols[2], cols[3], cols[4], cols[6], cols[15], cols[16], cols[17], cols[23], cols[24],
         cols[25]], axis=1).astype(object)

    print(df.columns)

    for i, x in enumerate(df['floor']):
        try:
            float(x)
        except ValueError:
            df['floor'][i] = np.nan

    df[cols[5]] = df[cols[5]].fillna('other')
    for i, x in enumerate(df[cols[5]]):
        if x == 'other':
            continue
        for j, y in enumerate(STREET_TYPES):
            try:
                if y in x.lower():
                    df[cols[5]][i] = STREET_TYPES[j]
                    break
                else:
                    if j == len(STREET_TYPES) - 1:
                        df[cols[5]][i] = 'other'
            except TypeError:
                df[cols[5]][i] = 'other'
            except AttributeError:
                df[cols[5]][i] = 'other'

    df[cols[7]] = df[cols[7]].fillna('other')
    df[cols[13]] = df[cols[13]].fillna('other')

    df[cols[14]] = df[cols[14]].fillna(0)
    df[cols[14]] = df[cols[14]].astype(bool)

    for i, x in enumerate(df[cols[18]]):
        if str(x) == 'nan':
            continue
        y = re.findall(r'\d+', str(x))
        if len(y) == 0:
            continue
        df[cols[18]][i] = y[0]

    df[cols[19]] = df[cols[19]].fillna('other')
    df[cols[20]] = df[cols[20]].fillna('other')
    df[cols[21]] = df[cols[21]].fillna('other')

    df[cols[22]] = df[cols[22]].fillna('other')
    for i, x in enumerate(df[cols[22]]):
        if x == 'other':
            continue
        y = re.findall(r'состояние дома и подъезда: ([^\s]+)', str(x))
        if len(y) == 0:
            df[cols[22]][i] = 'other'
        else:
            df[cols[22]][i] = y[0]

    for n in numeric:
        median = df[n].median()
        df[n] = df[n].fillna(median)

    # df = pd.get_dummies(df, columns=categorical)

    df.to_csv('data/apartments.csv')


if __name__ == '__main__':
    process()
