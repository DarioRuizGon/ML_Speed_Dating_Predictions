import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
            print(f"% match para {col}, bin {bin}", serie.value_counts(True), sep= "\n")

def ready_for_pipeline(df):
    df_ready = df.drop_duplicates(keep = False, inplace = True)