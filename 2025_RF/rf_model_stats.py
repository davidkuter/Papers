import pandas as pd

from scipy import stats

shuffled_data_path = "input_data/RF_Shuffled.csv"
unshuffled_data_path = "input_data/RF_Unshuffled.csv"


def reorientate_shuffled_data(df):

    # Strip trailing spaces from Metric names
    df['Metric'] = df['Metric'].str.strip()

    active_inactive_cols = ['Recall', 'Precision', 'Sensitivity', 'Specificity', 'F-measure']
    overall_cols = ['Accuracy', "Cohen's kappa", 'Area Under Curve']

    data = []
    for metric in active_inactive_cols:
        new_df = df[df['Metric'] == metric].copy()
        new_df = new_df[["Partition", "Y-Shuffle", "Actives", "Inactives"]]
        new_df = new_df.rename(columns={"Actives": f"{metric}_Actives", "Inactives": f"{metric}_Inactives"})
        new_df = pd.melt(new_df, id_vars=["Partition", "Y-Shuffle"], value_vars=[f"{metric}_Actives", f"{metric}_Inactives"],
                         var_name="Metric", value_name="Value")
        data.append(new_df)

    for metric in overall_cols:
        new_df = df[df['Metric'] == metric].copy()
        new_df = new_df[["Partition", "Y-Shuffle", "Overall"]]
        new_df = new_df.rename(columns={"Overall": f"{metric}"})
        new_df = pd.melt(new_df, id_vars=["Partition", "Y-Shuffle"], value_vars=[f"{metric}"],
                         var_name="Metric", value_name="Value")
        data.append(new_df)

    reoriented_df = pd.concat(data, ignore_index=True)

    return reoriented_df


def get_shuffled_stats(shuffled_data_path):

    # Load shuffled data
    df_shuff_raw = pd.read_csv(shuffled_data_path)
    df_shuff = reorientate_shuffled_data(df_shuff_raw)

    # Get average and std Metric
    data = []
    for metric, df_subgroup in df_shuff.groupby("Metric"):
        df_stats = df_subgroup["Value"].describe()
        data.append([metric, df_stats["mean"], df_stats["std"]])

    df_shuff_stats = pd.DataFrame(data, columns=["Metric", "Mean", "Std"])
    return df_shuff_stats


# Get shuffled results
df_shuff_stats = get_shuffled_stats(shuffled_data_path)
print("Shuffled Results:")
print(df_shuff_stats)

# Get unshuffled results
df_unshuff_raw = pd.read_csv(unshuffled_data_path)

data = []
for metric in df_shuff_stats["Metric"]:
    df_stats = df_unshuff_raw[metric].describe()
    data.append([metric, df_stats["mean"], df_stats["std"]])

df_unshuff_stats = pd.DataFrame(data, columns=["Metric", "Mean", "Std"])
print("Unshuffled Results:")
print(df_unshuff_stats)

# Compute one-tailed t-test between shuffled and unshuffled results
p_stats = []
for metric in df_shuff_stats["Metric"]:
    shuff_mean = df_shuff_stats[df_shuff_stats["Metric"] == metric]["Mean"].values[0]
    shuff_stdev = df_shuff_stats[df_shuff_stats["Metric"] == metric]["Std"].values[0]

    unshuff_mean = df_unshuff_stats[df_unshuff_stats["Metric"] == metric]["Mean"].values[0]
    unshuff_stdev = df_unshuff_stats[df_unshuff_stats["Metric"] == metric]["Std"].values[0]

    t_stat, p_value = stats.ttest_ind_from_stats(mean1=unshuff_mean, std1=unshuff_stdev, nobs1=50,
                                      mean2=shuff_mean, std2=shuff_stdev, nobs2=500,
                                      alternative='greater')
    p_stats.append([metric, t_stat, p_value])

df_p_stats = pd.DataFrame(p_stats, columns=["Metric", "T-statistic", "P-value"])
print("T-test Results (Unshuffled > Shuffled):")
print(df_p_stats)
