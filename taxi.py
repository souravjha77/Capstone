from MLPRegressorTrain import *
from PreprocessTestData import *
from runPkl import *
import xgboost as xgb
from matplotlib import pyplot

#X,y = preprocess('/home/sourav/PycharmProjects/capstone/train.csv')
df = pd.read_csv('/home/sourav/PycharmProjects/capstone/train_1.csv')
X = df
#X['dist'] = df.apply(dista,axis=1)
y = X['triptime']
#
X.drop(['startPLong','startPLat','endPLong','endPLat'], axis=1, inplace=True)
X.drop(['triptime'],axis=1,inplace=True)

#runPkl('/home/sourav/PycharmProjects/capstone/test.csv',modelPath='xgboostModel.pkl',submissionPath="out.csv")
#X.drop(['TAXI_ID','triptime','startPLong','startPLat','endPLong','endPLat'],axis= 1,inplace=True)
model = xgb.XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
       learning_rate=0.4, max_delta_step=0, max_depth=4,
       min_child_weight=3, n_estimators=100, nthread=4,
       objective='reg:linear', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)
print("training")
bst =model.fit(X,y)


xgb.plot_importance(bst)
pyplot.show()



# #print(X.describe())
# print("training started ..........")
# #model = joblib.load('/home/sourav/PycharmProjects/capstone/xgboostModel.pkl')
# model = mlp(X,y)
#
# #model.save("/home/sourav/PycharmProjects/capstone/Xgmodel")
# #model = svr(X,y,5)
print("training done .......")
x_test = preprocessT('/home/sourav/PycharmProjects/capstone/test.csv')
final_df = pd.DataFrame()
final_df["TRIP_ID"] = x_test['TRIP_ID']
x_test.drop(['TRIP_ID'],axis = 1,inplace=True)
output = model. predict(x_test)
final_df["TRAVEL_TIME"] = output
final_df.to_csv("Output_XGB2.csv",index=False)

