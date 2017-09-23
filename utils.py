import numpy as np
#online code from kaggle
def haversine(lonlat1, lonlat2):
  lon_diff = np.abs(lonlat1[0]-lonlat2[0])*np.pi/360.0
  lat_diff = np.abs(lonlat1[1]-lonlat2[1])*np.pi/360.0
  a = np.sin(lat_diff)**2 + np.cos(lonlat1[1]) * np.cos(lonlat2[1]) * np.sin(lon_diff)**2
  d = 2*6371*np.arctan2(np.sqrt(a), np.sqrt(1-a))
  return(d)

def dista(row):
    return haversine([row['startPLong'],row['endPLong']],[row['endPLat'],row['startPLat']])

def rmsle(predicted,real):
    sum=0.0
   #print(real)
    for x in range(len(predicted)):
        p = np.log(predicted[x]+1)

        a = np.log(real[x]+1)
        sum = sum + (p - a)**2
    return (sum/len(predicted))**0.5