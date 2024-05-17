import pandas as pd
df = pd.read_csv('/Users/DCL/Downloads/heart_failure_clinical_records_dataset.csv')

x=df.drop(['DEATH_EVENT'], axis = 'columns')
y=df['DEATH_EVENT']

#!pip install -U imbalanced-learn
from imblearn.combine import SMOTETomek
sm = SMOTETomek()
x_sm, y_sm = sm.fit_resample(x, y)

features_to_drop = ['anaemia', 'high_blood_pressure', 'sex']  
x_sm.drop(columns=features_to_drop, inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
X_train, X_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.2, random_state=42)

estimators = [
    ('rf', RandomForestClassifier())
]
sehm = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),  
)
sehm.fit(X_train, y_train)

#!pip install lime
from lime.lime_tabular import LimeTabularExplainer 
explainer = LimeTabularExplainer(X_train.values, 
                      feature_names=X_train.columns, 
                      class_names=['Survive', 'Death'], 
                      discretize_continuous=True, 
                      verbose=True)
lime = explainer.explain_instance(X_train.iloc[1], sehm.predict_proba)
lime.show_in_notebook(show_table=True)
