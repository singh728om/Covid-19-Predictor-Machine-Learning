import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data,ratio):
  np.random.seed(42)
  shuffled = np.random.permutation(len(data))
  test_set_size = int(len(data)*ratio)
  test_indices = shuffled[:test_set_size]
  train_indices = shuffled[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]



if __name__ == "__main__":
    df = pd.read_csv('corona.csv')
    train,test = data_split(df,0.2)

    #Read_Data
    x_train = train[['fever', 'bodyPain', 'age', 'runnyNose', 'Moderate_severe_cough','Dry_Cough','Gender','Sore_throat','Weakness','Change_in_Appetite','Feeling_breathless','close_contact','Diabetes','heart_dis','progressin48hr','kidneydis']].to_numpy()
    x_test = test[['fever', 'bodyPain', 'age', 'runnyNose', 'Moderate_severe_cough','Dry_Cough','Gender','Sore_throat','Weakness','Change_in_Appetite','Feeling_breathless','close_contact','Diabetes','heart_dis','progressin48hr','kidneydis']].to_numpy()

    y_train = train[['infectionProb']].to_numpy().reshape(2035,)
    y_test = test[['infectionProb']].to_numpy().reshape(508 ,)

    clf = LogisticRegression()
    clf.fit(x_train,y_train)

    file = open('model.pkl', 'wb')
    # dump information to that file
    pickle.dump(clf, file)
    
file.close()