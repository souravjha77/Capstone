from sklearn.externals import joblib
from PreprocessTestData import*

def runPkl(testPath,modelPath,submissionPath):
  print('loading pkl file ...........')
  model = joblib.load(modelPath)
  print('best estimators of the model ........')
  print(model.best_estimator_)
  x_test = preprocessT(testPath)
  final_df = pd.DataFrame()
  final_df["TRIP_ID"] = x_test['TRIP_ID']
  x_test.drop(['TRIP_ID'], axis=1, inplace=True)
  output = model.predict(x_test)
  final_df["TRAVEL_TIME"] = output
  final_df.to_csv(submissionPath, index=False)
  print('saved the predicted values.....')

