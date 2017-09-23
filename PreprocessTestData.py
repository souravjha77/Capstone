import  pandas as pd
import ast
from utils import *

def preprocessT(path):
    df = pd.read_csv(path)
    # removing the data doesn,t
    df.drop(df[(df.MISSING_DATA == True) | (df['POLYLINE'].map(len) == 0)].index, inplace=True)
    df['POLYLINE'] = [np.array(ast.literal_eval(x)) for x in df['POLYLINE']]
    #create the regression value


    df['startPLong'] = df['POLYLINE'].apply(lambda x: x[0][0])
    df['startPLat'] = df['POLYLINE'].apply(lambda x: x[0][1])
    df['endPLat'] = df['POLYLINE'].apply(lambda x: x[len(x) - 1][1])
    df['endPLong'] = df['POLYLINE'].apply(lambda x: x[len(x) - 1][0])
    df['dist'] = df.apply(dista, axis=1)


    df.drop(['POLYLINE',  'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'DAY_TYPE', 'MISSING_DATA'], axis=1,
            inplace=True)
    df.drop(['startPLong', 'startPLat', 'endPLong', 'endPLat'], axis=1, inplace=True)


    return df