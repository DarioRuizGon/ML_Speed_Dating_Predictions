import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import imblearn.pipeline as imbpipe
import sklearn.pipeline as skpipe

import sys
sys.path.insert(0, "../utils")
import bootcampviztools as viz
import ToolBox as tb
import functions as fn

np.random.seed(42)

from collections import Counter

from scipy.io import arff # Necesario para leer los datos, ya que vienen en este formato
from scipy.stats import f_oneway, pearsonr, spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def feature_selection_from_model(X, y, n_folds, n_features_to_select, scoring, threshold = "median", estimator = None):
    if not estimator:
        estimator = RandomForestClassifier(max_depth= 8, random_state=42, class_weight= "balanced")

    cv = KFold(n_splits= n_folds, shuffle = True, random_state= 42)
    # Selectores basados en estimador
    model_selector = SelectFromModel(estimator= estimator, threshold= threshold, max_features= n_features_to_select)
    rfe = RFE(estimator= estimator, n_features_to_select= n_features_to_select, step=1)
    rfecv = RFECV(estimator= estimator, cv= cv, scoring = scoring)

    for selector in [model_selector, rfe, rfecv]:
        selector.fit(X, y)

    model_selection = model_selector.get_feature_names_out().tolist()
    rfe_selection = rfe.get_feature_names_out().tolist()
    rfecv_selection = rfecv.get_feature_names_out().tolist()

    return (model_selection, rfe_selection, rfecv_selection)



def hard_voting(list):
    voting = Counter(list) # Cuento cuántas veces ha salido cada una
    df_voting = pd.DataFrame(voting.values(), columns = ["votes"], index = voting.keys()).sort_values("votes", ascending=False)
    return df_voting



def kbest_selector(X, y, k_anova = 15, k_mutual_info = 15, num_vars = None, cat_vars = None):
    if num_vars:
        anova_selector = SelectKBest(f_classif, k = k_anova)
        anova_selector.fit(X.dropna()[num_vars], y.dropna())
    
    if cat_vars:
        mutual_info_selector = SelectKBest(mutual_info_classif, k = k_mutual_info)
        mutual_info_selector.fit(X.dropna()[cat_vars], y.dropna())

    kbest_selection = anova_selector.get_feature_names_out().tolist() + mutual_info_selector.get_feature_names_out().tolist()

    return kbest_selection



def feature_set_selection(X, y, n_folds, scoring, set_names, sets, model_names = None, models = None):
    if not model_names:
        model_names = ["Decision Tree", "Random Forest", "XGBoost", "LightGBM"]

    if not models:
        dtc = DecisionTreeClassifier(max_depth= 8, random_state=42, class_weight= "balanced")
        rfc = RandomForestClassifier(max_depth=8, random_state=42, class_weight= "balanced")
        xgbc = XGBClassifier(max_depth=8, random_state=42, scale_pos_weight = len(y[y == 0]) / len(y[y == 1]))
        lgbmc = LGBMClassifier(max_depth=8, random_state=42, verbose = -1, class_weight= "balanced")
        models = [dtc, rfc, xgbc, lgbmc]
    
    results = []

    cv = KFold(n_splits= n_folds, shuffle = True, random_state= 43) # Semilla distinta de la utilizada para el selector para intentar evitar el posible "sobreajuste" que podría producirse al seleccionar las features con cv

    for model, model_name in zip(models, model_names):
        for feature_set, name in zip(sets, set_names):
            score = cross_val_score(model, X[feature_set], y, cv= cv, scoring= scoring).mean()
            results.append({"model": model_name, "features_set": name, f"{scoring}": score})

    df_scores = pd.DataFrame(results).sort_values(f"{scoring}", ascending=False)

    return df_scores



def binned_plots(df, bin_num, list_num, cat_col):
    col_num = bin_num
    row_num = len(list_num)
    fig, axs = plt.subplots(row_num, col_num, figsize = (6 * col_num, 6 * row_num))
    for i, col in enumerate(list_num):
        for bin in range(bin_num):
            if row_num == 1:
                ax = axs[bin]
            else:
                ax = axs[i, bin]
            df_binning = df.loc[pd.cut(df[col], bin_num, labels= range(bin_num)) == bin, [col, cat_col]]
            for cat in df_binning[cat_col].unique():
                sns.histplot(df_binning.loc[df_binning[cat_col] == cat][col], kde = True, ax = ax, label = str(cat))
            ax.set_title(f"Histogramas de {col} y {cat_col}, bin {bin}")
            ax.set_xlabel(col)
            ax.set_ylabel("frequency")
            ax.legend()
    plt.tight_layout()
    plt.show();



def binned_value_counts(df, bin_num, list_num, cat_col):
    for col in list_num:
        for bin in range(bin_num):
            serie = df.loc[pd.cut(df[col], bin_num, labels= range(bin_num)) == bin, cat_col]
            print(f"% match para {col}, bin {bin}, nº de instancias: {serie.count()}", serie.value_counts(True), sep= "\n")

def ready_for_pipeline(df):
    df_ready = df.drop_duplicates(keep = False, inplace = True)
