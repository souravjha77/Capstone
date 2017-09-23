import  pandas as pd
from SupportVectorR import *
from PreprocessTestData import *

df = pd.read_csv('/home/sourav/PycharmProjects/capstone/train_1.csv')
y = df['triptime']
X = df

X.drop(['triptime'],axis= 1,inplace=True)

print("training started ..........")

model = svr(X,y,5)
print("training done .......")

x_test = preprocessT('/home/sourav/PycharmProjects/capstone/test.csv')

final_df = pd.DataFrame()
final_df["TRIP_ID"] = x_test['TRIP_ID']
x_test.drop(['TRIP_ID'],axis = 1,inplace=True)
output = model.predict((x_test.values))
final_df["TRAVEL_TIME"] = output
final_df.to_csv("Output_SVR.csv",index=False)