import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_breast_cancer
from warnings import filterwarnings

filterwarnings("ignore")

cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc")
model.fit(X, y)

from yanapy.interpretators.baseinterpretator import BaseInterpretator

bi = BaseInterpretator(model)

bi.fit_shap()
bi.fit_skater(X)


bi.shap(X)