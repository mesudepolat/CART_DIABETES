################################
# DIABETES PREDICTION with CART
################################
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.tree import export_graphviz, export_text
from skompiler import skompile
from helpers.eda import *

pd.set_option('display.max_columns', None)

df = pd.read_csv("datasets/diabetes.csv")
df.info()
check_df(df)
df.describe().T
################################
# DATA PREPROCESSING
################################
nan_cols = ["Glucose", "SkinThickness", "Insulin", "BloodPressure", "BMI"]

for col in nan_cols:
    df[col].replace(0, np.NaN, inplace=True)

def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

median_target(var="Glucose")

for col in nan_cols:
    df.loc[(df['Outcome'] == 0) & (df[col].isnull()), col] = median_target(col)[col][0]
    df.loc[(df['Outcome'] == 1) & (df[col].isnull()), col] = median_target(col)[col][1]

df.describe().T

##########################################################
# FEATURE EXTRACTION
##########################################################
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# 25-30 kg/m2: Fazla kilolu
# BMI = 30 – 39, 9 kg/m. arasında olanlar: Obez
# 40 kg/m2 ve üzerinde olanlar : Morbid obez (İleri derecede obez)
df.loc[(df['BMI'] < 18), 'WEIGHT'] = 'slim'
df.loc[(df['BMI'] > 18) & (df['BMI'] < 25), 'WEIGHT'] = 'normal'
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'WEIGHT'] = 'overweight'
df.loc[(df['BMI'] >= 30) & (df['BMI'] < 40), 'WEIGHT'] = 'obese'
df.loc[df['BMI'] >= 40, 'WEIGHT'] = 'morbidobesity'

df["BMI_THICK"] = df['BMI'] * df['SkinThickness']

df['KID'] = (df['Pregnancies'] >= 4).astype('int')

# 30 yaşın üzerinde, fazla kilolu hamileler gestasyonel diyabet açısından risk taşırlar.
# hamilelik sonrası %50 düzelir
df['gestation'] = ((df['Age'] > 30) & (df['BMI'] > 28) & (df['SkinThickness'] > 30) & (df['Pregnancies'] != 0)).astype('int')

df.loc[df["Glucose"] <= 70, 'NEW_GLUCOSE'] = 'low'
df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), 'NEW_GLUCOSE'] = 'normal'
df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), 'NEW_GLUCOSE'] = 'risk'
df.loc[df["Glucose"] > 126, 'NEW_GLUCOSE'] = 'high_risk'

df['sugar'] = df.Glucose * df.Insulin

# %75 insülin değeri 169.50000
df.loc[df['Insulin'] < 170, 'NEW_INSULIN'] = 'normal'
df.loc[df['Insulin'] >= 170, 'NEW_INSULIN'] = 'abnormal'


df.columns = [col.upper() for col in df.columns]
check_df(df)


#############################################
# ONE-HOT ENCODING
###########################################
ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
from helpers.data_prep import one_hot_encoder
df = one_hot_encoder(df, ohe_cols, drop_first=True)
df.columns = [col.upper() for col in df.columns]
df.info()

#############################################
# LABEL ENCODING
###########################################
binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']
from helpers.data_prep import label_encoder

for col in binary_cols:
    df = label_encoder(df, col)

################################
# CART
################################
y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,  random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

################################
# HOLDOUT YÖNTEMİ İLE MODEL DOĞRULAMA
################################
# train hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# test hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)


################################
# HİPERPARAMETRE OPTİMİZASYONU
################################
cart_model.get_params()

cart_model = DecisionTreeClassifier(random_state=17)

# arama yapılacak hiperparametre setleri
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": [2, 3, 4],
               "min_samples_leaf": [2, 3, 4]}

cart_cv = GridSearchCV(cart_model, cart_params, cv=10, n_jobs=-1, verbose=True)
cart_cv.fit(X_train, y_train)

cart_cv.best_params_
cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_).fit(X_train, y_train)


################################
# HOLDOUT YÖNTEMİ İLE MODEL DOĞRULAMA
################################
# train hatası
y_pred = cart_tuned.predict(X_train)
y_prob = cart_tuned.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# test hatası
y_pred = cart_tuned.predict(X_test)
y_prob = cart_tuned.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

################################
# DEĞİŞKEN ÖNEM DÜZEYLERİNİ İNCELEMEK
################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cart_tuned, X_train)


################################
# FİNAL MODELİN YENİDEN TÜM VERİYE FİT EDİLMESİ
################################

cart_tuned_final = DecisionTreeClassifier(**cart_cv.best_params_).fit(X, y)

################################
# MODELİN DAHA SONRA KULLANILMAK ÜZERE KAYDEDİLMESİ
################################

import joblib
joblib.dump(cart_tuned_final, "cart_tuned_final.pkl")
cart_model_from_disk = joblib.load("cart_tuned_final.pkl")
cart_model_from_disk.predict(X_test)


