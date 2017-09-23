from  XGBoost import *
from PreprocessTestData import *


df = pd.read_csv('/home/sourav/PycharmProjects/capstone/train_1.csv')
y = df['triptime']
X = df
X.drop(['triptime'],axis= 1,inplace=True)

print("training started ..........")

model = xgboost(X,y)
#model.save("/home/sourav/PycharmProjects/capstone/Xgmodel")
print("training done .......")
x_test = preprocessT('/home/sourav/PycharmProjects/capstone/test.csv')
#prediction values  for public leaderboard(test case availaible)
final_df = pd.DataFrame()

final_df["TRIP_ID"] = x_test['TRIP_ID']
x_test.drop(['TRIP_ID'],axis = 1,inplace=True)
print('prediction ............')
output = model.best_estimator_.predict(x_test)

final_df["TRAVEL_TIME"] = output
#save the file
print('saving file .............')
final_df.to_csv("Output_XGB.csv",index=False)