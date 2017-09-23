import  pandas as pd
import ast
from utils import *



def preprocess(path,savePath) :

      df = pd.read_csv(path)
      # removing the data doesn,t
      df.drop(df[(df.MISSING_DATA == True)].index, inplace=True)
      # #create the regression value
      print("loading done ...........")
      df.drop(['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'DAY_TYPE', 'ORIGIN_STAND', 'MISSING_DATA'], axis=1,
              inplace=True)
      df['POLYLINE'] = [np.array(ast.literal_eval(x)) for x in df['POLYLINE']]
      df.drop(df[(df['POLYLINE'].map(len) == 0)].index, inplace=True)
      triptime = (df['POLYLINE'].map(len) - 1) * 15.0

      df['startPLong'] = df['POLYLINE'].apply(lambda x: x[0][0])
      df['startPLat'] = df['POLYLINE'].apply(lambda x: x[0][1])
      df['endPLat'] = df['POLYLINE'].apply(lambda x: x[len(x) - 1][1])
      df['endPLong'] = df['POLYLINE'].apply(lambda x: x[len(x) - 1][0])

      # #removing the features that are not relevant
      df['triptime'] = triptime
      df['dist'] = df.apply(dista, axis=1)
      df.drop(['POLYLINE','startPLong','startPLat','endPLong','endPLat'], axis=1, inplace=True)

      df.to_csv(savePath, index=False)
      # print( df['startPLong'])
      return df, triptime