import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pymare import meta_regression
from pymare import Dataset
from pymare.estimators import VarianceBasedLikelihoodEstimator
from statsmodels.stats.multitest import fdrcorrection

def set_confounds_type(confounds):
    if pd.isna(confounds):
        return 'None'
    else:
        return 'all'

def set_confounds_type_detail(confounds):
    if pd.isna(confounds):
        return 'None'
    else:
        return confounds

def get_all_val(indir):
    vals = pd.DataFrame()
    file_list = [file for file in os.listdir(indir) if file.endswith('.csv')]
    for i in file_list:
        valset = pd.read_csv(os.path.join(indir, i), index_col=0, header=0)
        vals = pd.concat([vals, valset], axis=0, ignore_index=True)
    vals['confounds_type'] = vals['confounds'].apply(set_confounds_type)
    vals['confounds_type_details'] = vals['confounds'].apply(set_confounds_type_detail)
    vals = vals[vals['regressor'].str.contains('IDP')]

    return vals

def map_yeo(results, dictionary, func_dictionary):
    dataset_list = ['UKB', 'HCP_YA', 'HCP_Aging', 'HCP_Development', 'ABCD', 'ANXPE']
    mappings = {dataset: pd.Series(dictionary['Yeo'].values, index=dictionary[dataset]).to_dict() for dataset in dataset_list}
    for dataset in dataset_list:
        results.loc[results['Dataset'] == dataset, 'Yeo'] = results['IDP_name'].map(mappings[dataset])

    # map func_idps
    func_map = dict(zip(func_dictionary['node'].values, func_dictionary['Yeo'].values))
    results.loc[results['Yeo'].isna(), 'Yeo'] = results['IDP_name'].map(func_map)

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
    results['Yeo_name'] = results['Yeo'].map(yeo_mapping)
    results = results.sort_values(by='Yeo')
    idp_name_mapping = dict(zip(dictionary['UKB'].values, dictionary['HCP_YA'].values))
    results.loc[results['Dataset'] == 'UKB', 'IDP_name'] = results['IDP_name'].map(idp_name_mapping)
    results.sort_values(['Dataset', 'Yeo'], inplace=True)
    return results

curr_path = os.getcwd()

dictionary = pd.read_csv(f"{curr_path}/Source_Code/maps/structural_dictionary_new.csv")
func_dictionary = pd.read_csv(f"{curr_path}/Source_Code/maps/functional_dictionary.csv")
output_path = f"{curr_path}/Results"

# map yeo network
df = get_all_val(f'{curr_path}/Data/TGIF')
df = map_yeo(df, dictionary, func_dictionary)
df.loc[df['phenotype'] == 'phq', 'Dataset'] = 'UKB_phq'

#drop rows with phq, cbclscrsynanxdepr, or cbclanxdep
df = df[(df['phenotype']!= 'phq') & (df['phenotype']!= 'cbclscrsynanxdepr') & (df['phenotype']!= 'cbclanxdep')]

dataset_list = ['ABCD', 'ANXPE', 'HCP_Aging', 'HCP_Development', 'HCP_YA', 'UKB']
# covars = {'ABCD': 9.93, 'ANXPE': 28.45, 'HCP_Aging': 62.48, 'HCP_Development': 12.66, 'HCP_YA': 28.75, 'UKB': 62.92} #mean age covariate
covars = {'ABCD': 1, 'ANXPE': .512, 'HCP_Aging': .512, 'HCP_Development': .512, 'HCP_YA': .343, 'UKB': 1} #voxel size covariate


# Initialize results dataframes
columns = ['IDP_name', 'phenotype', 'estimate', 'se', 'z_score', 'p_val']

# Phenotype group dictionaries
phen_neuroticism = {'ABCD': 'uppsyssnegativeurgency', 'UKB': 'N', 'HCP_Development': 'uppsneg', 'HCP_Aging': 'neon', 'HCP_YA': 'Neuroticism', 'ANXPE': 'neov'}
phen_depress = {'ABCD': 'cbclscrdsmdepressr', 'UKB': 'rds', 'HCP_Development': 'dsmscale', 'HCP_Aging': 'sadRawScore', 'HCP_YA': 'Sadness', 'ANXPE': 'hamdtotal'}

df_volume = df[(df["IDP_name"].str.contains("volume")) | (df["Yeo_name"] == "Subcortical")]
df_area = df[(df["IDP_name"].str.contains("area")) & (df["Yeo_name"] != "Subcortical")]
df_thickness = df[(df["IDP_name"].str.contains("thickness")) & (df["Yeo_name"] != "Subcortical")]

structural_dfs = [df_volume, df_area, df_thickness]
metrics = ['volume', 'area', 'thickness']

# List of phenotype groups for looping
phen_groups = {
    "Neuroticism": phen_neuroticism,
    "Depression": phen_depress
}

df.to_csv(f"{output_path}/concat_reg_results.csv", index=False)

# Perform meta-analysis separately for each structural DataFrame and phenotype group
for metric, structural_df in zip(metrics, structural_dfs):
    results_with_covars_all = []
    results_without_covars_all = []

    for phen_group_name, phen_group in phen_groups.items():
        results_with_covars = []
        results_without_covars = []
        all_p_values_with_covars = []
        all_p_values_without_covars = []

        for idp_name in structural_df['IDP_name'].unique():
            y, v, n, covars_array = [], [], [], []

            for dataset in dataset_list:
                phen = phen_group.get(dataset)
                if phen is None:
                    print(f"Phenotype missing for dataset {dataset} in group {phen_group_name}")
                    continue
                # Filter for the corresponding row
                row = structural_df[(structural_df['IDP_name'] == idp_name) & (structural_df['Dataset'] == dataset) & (structural_df['phenotype'] == phen)]
                if row.empty:
                    print(f"Missing data for IDP {idp_name}, dataset {dataset}, phenotype {phen}")
                    continue
                yeo_value = row['Yeo'].values[0]
                yeo_name_value = row['Yeo_name'].values[0]
                # Append y (coefficient), v (variance), N, and covariates
                y.append(row['coef_'].values[0])
                v.append(row['var_coef_'].values[0])
                n.append(row['N'].values[0])
                covars_array.append(covars[dataset])

            if len(y) < 2:
                print(f"Insufficient data for IDP {idp_name} in phenotype group {phen_group_name}")
                continue

            # Convert to numpy arrays
            y = np.array(y)
            v = np.array(v)
            n = np.array(n)
            covars_array = np.array(covars_array).reshape(-1, 1)

            # Meta-analysis with covariates
            result_with_covars = meta_regression(y, v, covars_array, n, X_names=['voxel_size'], add_intercept=True, method='REML')
            df_with_covars = result_with_covars.to_df()
            df_with_covars.drop(df_with_covars[df_with_covars.name == 'voxel_size'].index, inplace=True)

            df_with_covars['IDP_name'] = idp_name
            df_with_covars['phen_group'] = phen_group_name
            df_with_covars['Yeo'] = yeo_value
            df_with_covars['Yeo_name'] = yeo_name_value

            # Collect p-values for FDR correction
            all_p_values_with_covars.extend(df_with_covars['p-value'].tolist())

            results_with_covars.append(df_with_covars)

            # Meta-analysis without covariates
            result_no_covars = meta_regression(y, v, n=n, add_intercept=True, method='REML')
            df_no_covars = result_no_covars.to_df()
            df_no_covars['IDP_name'] = idp_name
            df_no_covars['phen_group'] = phen_group_name
            df_no_covars['Yeo'] = yeo_value
            df_no_covars['Yeo_name'] = yeo_name_value

            # Collect p-values for FDR correction
            all_p_values_without_covars.extend(df_no_covars['p-value'].tolist())

            results_without_covars.append(df_no_covars)

        # Perform FDR correction across all comparisons within the phenotype group
        _, p_fdr_with_covars = fdrcorrection(all_p_values_with_covars)
        _, p_fdr_without_covars = fdrcorrection(all_p_values_without_covars)

        # Debug prints for sanity check
        print(f"Original p-values with covars ({phen_group_name}): {all_p_values_with_covars}")
        print(f"FDR corrected p-values with covars ({phen_group_name}): {p_fdr_with_covars}")
        print(f"Original p-values without covars ({phen_group_name}): {all_p_values_without_covars}")
        print(f"FDR corrected p-values without covars ({phen_group_name}): {p_fdr_without_covars}")

        # Append FDR corrected p-values to the DataFrames
        for df_with_covars in results_with_covars:
            df_with_covars['p_fdr'] = p_fdr_with_covars[:len(df_with_covars)]
            df_with_covars['Significance'] = df_with_covars['p_fdr'] < 0.05  # Significance flag
            p_fdr_with_covars = p_fdr_with_covars[len(df_with_covars):]  # Remove used p-values

        for df_no_covars in results_without_covars:
            df_no_covars['p_fdr'] = p_fdr_without_covars[:len(df_no_covars)]
            df_no_covars['Significance'] = df_no_covars['p_fdr'] < 0.05  # Significance flag
            p_fdr_without_covars = p_fdr_without_covars[len(df_no_covars):]  # Remove used p-values

        # Concatenate results for each phenotype group
        results_with_covars_df = pd.concat(results_with_covars)
        results_without_covars_df = pd.concat(results_without_covars)

        # Append to the overall results
        results_with_covars_all.append(results_with_covars_df)
        results_without_covars_all.append(results_without_covars_df)

    # Concatenate all results for the structural metric
    results_with_covars_all_df = pd.concat(results_with_covars_all)
    results_without_covars_all_df = pd.concat(results_without_covars_all)

    # Save results
    output_dir = f"{curr_path}/Results/meta_results_{metric}_voxel"
    os.makedirs(output_dir, exist_ok=True)
    results_with_covars_all_df.to_csv(f"{output_dir}/meta_analysis_with_covars.csv", index=False)
    results_without_covars_all_df.to_csv(f"{output_dir}/meta_analysis_without_covars.csv", index=False)

print("Meta-analyses completed and results saved!")

