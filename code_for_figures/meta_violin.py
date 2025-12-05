import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_volume = pd.read_csv("../meta_results_volume/meta_analysis_without_covars.csv")
df_area = pd.read_csv("../meta_results_area/meta_analysis_without_covars.csv")
df_thickness = pd.read_csv("../meta_results_thickness/meta_analysis_without_covars.csv")

structural_dfs = [df_volume, df_area, df_thickness]

# load functional data

schaefer_partial_path = "../meta_analysis_without_covars_with_yeo.csv"
schaefer_partial_df = pd.read_csv(schaefer_partial_path)
yeo_mapping = {
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal Attention",
    4: "Ventral Attention",
    5: "Limbic",
    6: "Frontoparietal",
    7: "Default",
    8: "Subcortical"
}
schaefer_partial_df['Yeo'] = schaefer_partial_df['Yeo1']
schaefer_partial_df['Yeo_name'] = schaefer_partial_df['Yeo1'].map(yeo_mapping)

functional_dfs = [schaefer_partial_df]

df_volume["estimate"] = df_volume["estimate"].abs()
df_volume["imaging_metric"] = "Gray Matter Volume"
df_area["estimate"] = df_area["estimate"].abs()
df_area["imaging_metric"] = "Cortical Surface Area"
df_thickness["estimate"] = df_thickness["estimate"].abs()
df_thickness["imaging_metric"] = "Cortical Thickness"
schaefer_partial_df["estimate"] = schaefer_partial_df["estimate"].abs()
schaefer_partial_df["imaging_metric"] = "Functional Connectivity"

df_total = pd.concat([df_volume, df_area, df_thickness, schaefer_partial_df], ignore_index=True)

print(df_total.head())

print(schaefer_partial_df.columns)
print(schaefer_partial_df["phen_group"].unique())
print(max(schaefer_partial_df["estimate"]))

all_dfs = structural_dfs + functional_dfs


df_total.loc[df_total['phen_group'] == 'Neuroticism', 'phen_group'] = 'Predisposition'
df_total.loc[df_total['phen_group'] == 'Depression', 'phen_group'] = 'Severity'
df_total['Depression phenotype']=df_total['phen_group']
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_total, x='imaging_metric', y='estimate', hue='Depression phenotype', dodge=True,inner='box', palette=['lightgray','skyblue'])
plt.xlabel('Imaging Metric Type')
plt.ylabel('Absolute Meta-analysis Estimate')
plt.ylim(-0.02,0.09)
plt.tight_layout()
#plt.show()
plt.savefig("../FIGURE3.png", dpi=600)

df_volume['estimate_abs'] = df_volume['estimate'].abs()
df_area['estimate_abs'] = df_area['estimate'].abs()
df_volume.loc[df_volume['phen_group'] == 'Neuroticism', 'phen_group'] = 'Predisposition'
df_volume.loc[df_volume['phen_group'] == 'Depression', 'phen_group'] = 'Severity'
df_volume['Depression phenotype']=df_volume['phen_group']
df_area['Depression phenotype']=df_area['phen_group']
plt.figure(figsize=(12, 6))
sns.violinplot(data=df_volume, x='Yeo_name', y='estimate_abs', hue='Depression phenotype', dodge=True,palette=['lightgray','skyblue'])
# plt.title('ARI Score Distribution by k')
plt.xlabel('Spatial Network')
plt.ylabel('Absolute Meta-analysis Estimate')
plt.ylim(-0.02,0.1)
plt.tight_layout()
#plt.show()
plt.savefig("../FIGURE4_A.png", dpi=600)

df_area.loc[df_area['phen_group'] == 'Neuroticism', 'phen_group'] = 'Predisposition'
df_area.loc[df_area['phen_group'] == 'Depression', 'phen_group'] = 'Severity'
plt.figure(figsize=(12, 6))
sns.violinplot(data=df_area, x='Yeo_name', y='estimate_abs', hue='Depression phenotype', dodge=True,palette=['lightgray','skyblue'])
# plt.title('ARI Score Distribution by k')
plt.xlabel('Spatial Network')
plt.ylim(-0.02,0.1)
plt.ylabel('Absolute Meta-analysis Estimate')
plt.legend().remove()
plt.tight_layout()
#plt.show()
plt.savefig("../FIGURE4-B.png", dpi=600)

