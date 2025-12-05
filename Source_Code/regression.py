import re
import os
import sys
import argparse
import numpy as np
import pandas as pd
import pprint
from pprint import pprint
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sys.path.append('/scratch/l.lexi/WAPIAW2024/Source_Code/utils_py')
import utils_py

## GET INPUT DATA -- move all helper functions (starting with "_") to utils
##################################################################################################################################
# given a filepath to a list of subject IDs and a general datapath string, returns train-test split data and labels
def get_input_data(
        X_fpath,
        Y_fpath,
        cnfvars_fpath=None,
        sublist_fpath=None,
        idplist_fpath=None,
        ptylist_fpath=None,
        cnflist_fpath=None,
        verbose=True,
        debug=False
        ):

    if 'func_idp' in X_fpath and not X_fpath.endswith('.csv'):    
        # load one-file-per-subject data given data directory and a path to a list of subject IDs
        X_df = utils_py.data._load_func_idp(X_fpath, sublist_fpath)
    else:
        # assumes that eids/session-ids/subject identifiers are always in first column of .csv file
        X_df = pd.read_csv(X_fpath, index_col=0)

    # assumes that eids/session-ids/etc are always in first column of .csv file
    Y_df = pd.read_csv(Y_fpath, index_col=0)

    if debug:
        ### debugging code ###
        print("Full data array has shape:", X_df.shape)
        print("Input data array has shape:", X_fit.shape)
        print("Full phenotype array has shape:", Y_df.shape)
        ### debugging code ###

    if cnfvars_fpath:
        # assumes that eids/session-ids/etc are always in first column of .csv file
        cnfvars_all = utils_py.data._normalize_df(pd.read_csv(cnfvars_fpath, index_col=0))
        # selects subset of input data based on subject ID and confound variable lists
        conf_fit = utils_py.data._subset_data(cnfvars_all, sublist_fpath=sublist_fpath, colvars_fpath=cnflist_fpath)
    else:
        conf_fit = None
        
    # selects subset of input data based on subject ID and IDP variable lists
    X_fit = utils_py.data._subset_data(X_df, sublist_fpath=sublist_fpath, colvars_fpath=idplist_fpath)

    # selects subset of input data based on subject ID and MH phenotype variable lists
    Y_fit = utils_py.data._subset_data(Y_df, sublist_fpath=sublist_fpath, colvars_fpath=ptylist_fpath)

    if not sublist_fpath:
        X_fit, Y_fit, conf_fit = utils_py.data._enforce_same_subjs(X_fit, Y_fit, conf=conf_fit)

    if verbose:
        # give detailed summary of input variables
        utils_py.data._show_input_params(X_fit, Y_fit, cnfvars=conf_fit)

    X_fit = utils_py.data._normalize_df(X_fit) # applies normalization (i.e., 0-mean, unit variance)
    Y_fit = utils_py.data._normalize_df(Y_fit) # applies normalization (i.e., 0-mean, unit variance)
    if conf_fit is not None:
        conf_fit = utils_py.data._normalize_df(conf_fit) # applies normalization (i.e., 0-mean, unit variance)

    metadata = utils_py.data._get_metadata(X_fpath, Y_fpath, cnfvars_fpath=cnfvars_fpath, 
            idplist_fpath=idplist_fpath, ptylist_fpath=ptylist_fpath, cnflist_fpath=cnflist_fpath)

    return X_fit, Y_fit, conf_fit, metadata
##################################################################################################################################



### send help functions (i.e., switcher) to utils or two model_specification?
## DO REGRESSION
##################################################################################################################################
def all_regressions(
        X_fit, 
        Y_fit, 
        conf, 
        fit_type='lstsq', 
        debug=False
        ):
    regress_list = []

    # we fit one feature (i.e, IDP) at a time
    for col in X_fit.columns:
        # we need to regress each phenotype separately!
        for pheno in Y_fit.columns:
            regress_list.append(
                    do_regression(
                    X_fit[col].to_frame(), 
                    Y_fit[pheno].to_frame(), 
                    conf, 
                    fit_type=fit_type
                    )
                    )

    if debug:
        ### debugging code ###
        print(f"Columns of X data are: {X_fit.columns}")
        print(f"Regression list length is {len(regress_list)}")
        print("First and last entry of regression list:", regress_list[0], regress_list[-1])
        ### debugging code ###

    return regress_list

def do_regression(
        X, 
        Y, 
        conf, 
        fit_type='lstsq'
        ):
    # Extract column names for IDP and depression score
    #print(f'X columns: {X.columns}')
    X = X.rename(columns={X.columns[0]: re.sub(r'[^a-zA-Z]', '', X.columns[0])})
    Y = Y.rename(columns={Y.columns[0]: re.sub(r'[^a-zA-Z]', '', Y.columns[0])})
    #X = X.rename(columns={X.columns[0]: 'idp'})
    #Y = Y.rename(columns={Y.columns[0]: 'dep'})

    idp_col = X.columns[0]  # Assuming there's only one IDP column per run
    dep_col = Y.columns[0]  # Assuming there's only one dependent variable column per run


    X_aug = pd.concat([X, conf], axis=1)

    # List of covariates
    covariates = ['age', 'sex', 'hm', 'FS_IntraCranial_Vol']

    # Identify random effects
    random_effects = [col for col in X_aug.columns if col not in covariates and col != idp_col]

    # Construct formula
    formula = fr'{dep_col} ~ {idp_col} + {" + ".join(covariates)} + ' + ' + '.join([f'(1|{re})' for re in random_effects])

    #regmodel = utils_py.analysis._get_regmodel(fit_type)  <<old code pre-lmer
    regmodel_callable = utils_py.analysis._get_regmodel(fit_type)
    if fit_type == 'lmer':
        regmodel = regmodel_callable(formula=formula, X_cols=X_aug.columns) #was X_cols=pd.concat([X_aug, Y])
        print("formula: ", formula)
    else:
        regmodel = regmodel_callable # linear regression is not callable, fix this so that linreg works! old version in local 

    # if custom regresser is written, should follow scikit-learn API as closely as possible.
    if fit_type == 'lmer':
        #X_aug['site'], _ = pd.factorize(X_aug['site']) #might have to change if re is not only site (i.e. familial)
        regmodel.fit(data=pd.concat([X_aug,Y], axis=1))
    else:
        regmodel.fit(X_aug, Y)

    setattr(regmodel, 'N_fit', len(Y.values))
    setattr(regmodel, 'target_name_', list(Y.columns))
    if not hasattr(regmodel, "feature_names_in_"):
        setattr(regmodel, "feature_names_in_", X_aug.columns)

    if fit_type == 'lmer':
        preds = regmodel.predict(X_aug.values, verify_predictions=False)

        # Manually compute the R2 score
        r2 = r2_score(Y.values, preds)

        # Set the R2_score attribute
        setattr(regmodel, 'R2_score', r2)
    else:
        setattr(regmodel, 'R2_score', regmodel.score(X_aug.values, Y.values))

    utils_py.analysis._do_hetBP_test(regmodel, X_aug.values, Y.values)   # tests heteroskedasticity assumptions

    if hasattr(regmodel, 'statistics_'):
        setattr(regmodel, 'p_raw', regmodel.statistics_["p_values"])    # for GAM, should ensure that we *ALWAYS* choose functional forms with one term per column!
        setattr(regmodel, 'model_degrees_of_freedom', regmodel.statistics_["edof"])
        print(regmodel.statistics_["se"])
        setattr(regmodel, 'var_coef_', regmodel.statistics_["se"]**2)
    else:
        # If X_aug includes the intercept, remove it
        X_aug = X_aug.drop(columns=random_effects)
        X_aug.to_csv("/ceph/chpc/shared/janine_bijsterbosch_group/l.lexi_scratch/WAPIAW2024/Experiments/HCP_team/hcp_d/temp_X_aug_new.csv", index=False)
        utils_py.analysis._get_raw_pvals(regmodel, X_aug.values, Y.values)   # assumes linear regression assumptions hold!!
        # add code here to get coefficient standard error from sklearn model
        print(f"regmodel in regression.py after get_raw_pvals: {vars(regmodel)}")
        print(f"regmodel.coef_ = {regmodel.coef_}")
        print("var_coef_:")
        print(getattr(regmodel, 'var_coef_'))


    return regmodel
##################################################################################################################################


## SAVE OUTPUT
##################################################################################################################################
def save_output(
        outdir,
        regress_list, 
        metadata,
        colsout_fpath=None, 
        outname=None, 
        verbose=False,
        debug=False
        ):

    if not outname:
        outname = "regressions"


    if debug:
        ### debugging code ###
        print("Experimental metadata is:", metadata)
        ### debugging code ###

    outpath = os.path.join(outdir, f"{outname}.csv")

    if verbose:
        raw_outpath = outpath.replace(".csv","_raw.npy")
        np.save(raw_outpath, regress_list)
        print(f"raw list of fit models saved to {raw_outpath}")
        txt_outpath = outpath.replace(".csv","_textsumm.txt")
        with open(txt_outpath, 'w') as fout:
            for regmodel in regress_list:
                print(f"\nFIT FOR FEATURE: {regmodel.feature_names_in_[0]}", file=fout)
                for par in regmodel.__dict__:
                    print(f"Regression model attribute \'{par}\' has value:", regmodel.__dict__[par], file=fout)
                print('', file=fout)
                
            print(f"human-readable text summary saved to {txt_outpath}")

    df_out = get_full_regression_df(regress_list, metadata, colsout_fpath=colsout_fpath)
    print("Debug step 1:")
    print(df_out[['coef_', 'var_coef_']].head())
    df_out.to_csv(outpath)
    print(f"Dataframe of regression results saved to {outpath}")

def get_full_regression_df(
        regress_list, 
        metadata, 
        colsout_fpath=None, 
        param_type="coef_", 
        debug=False
        ):

    if colsout_fpath:
        with open(colsout_fpath,'r') as fin:
            df_cols = fin.read().split()
    else:
        df_cols = [
                "IDP_name", "regressor", "edof", "N",
                param_type, f"var_{param_type}", "p_fdr", "p_raw", "score (R2)", "p_PBhet", 
                "phenotype", "confounds", "Dataset", "data type"
                ]

        df_data = []
        for regmodel in regress_list:
            regresults_dictlist = utils_py.analysis._get_regression_dicts(regmodel, df_cols, metadata, param_type)
            df_data = df_data + regresults_dictlist

        full_regression_df = pd.DataFrame(data = df_data)
        full_regression_df = utils_py.analysis._correct_pvals(full_regression_df)

    return full_regression_df
##################################################################################################################################




## PARSER
##################################################################################################################################
# parses inputs, generates derivative intermediates, loads and splits data, fits and predicts with RF classifiers, and saves results
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Binary classification: patient vs. control from resting-state data features")
    parser.add_argument(
            '-x',
            '--IDP_filepath',
            type=str,
            default='/scratch/l.lexi/WAPIAW2024/Data/UKB/func_idp/partial_NMs',
            help='filepath to full set of IDPs (ie, indpendent variables) -- is directory in functional IDP case'
            )
    parser.add_argument(
            '-y',
            '--phenotype_filepath',
            type=str,
            default="/scratch/l.lexi/WAPIAW2024/Data/UKB/phenotypes.csv",
            help='filepath to full set of phenotypes'
            )
    parser.add_argument(
            '-c',
            '--confound_filepath',
            type=str,
            default=None,
            help='filepath to full set of confounds'
            )
    parser.add_argument(
            '-s',
            '--sublist_fpath',
            type=str,
            default=None,
            help='filepath to (sub)list of subject EIDs to include in regression'
            )
    parser.add_argument(
            '-d',
            '--idplist_fpath',
            type=str,
            default=None,
            help='filepath to (sub)list of column names of idp variables to include in regression'
            )
    parser.add_argument(
            '-p',
            '--ptylist_fpath',
            type=str,
            default=None,
            help='filepath to (sub)list of column names of phenotype variables to include in regression'    # maybe change this to direct input instead of readin?
            )
    parser.add_argument(
            '-C',
            '--cnflist_fpath',
            type=str,
            default=None,
            help='filepath to (sub)list of column names of confounds to include in regression'    # maybe change this to direct input instead of readin?
            )
    parser.add_argument(
            '-R',
            '--fit_type',
            type=str,
            default='lstsq',
            help='name of regression model to fit to data -- the default \'lstsq\' means \"least squares\" and is an ordinary linear regression'
            )
    parser.add_argument(
            '-o',
            '--outdir',
            type=str,
            default='/scratch/l.lexi/WAPIAW2024/Experiments/testing/Results',
            help='output directory for linear regression results'
            )
    parser.add_argument(
            '-N',
            '--outname',
            type=str,
            default=None,
            help='filename (NOT path) for results files'
            )
    parser.add_argument(
            '-v',
            '--verbose',
            default=False,
            action='store_true',
            help="verbose flag: send problem parameters to output stream?"
            )
    args = parser.parse_args()

    # send selection of options to standard output
    print('')
    print('Path to IDP (indepedent variable):', args.IDP_filepath)
    print('Path to mental health phenotypes:', args.phenotype_filepath)
    print('Path to confounds:', args.confound_filepath)
    print('Fit type used:', args.fit_type)
    print('Path to list of included subject EIDs:', args.sublist_fpath)
    print('Path to (sub)list of IDP column names:', args.idplist_fpath)
    print('Path to (sub)list of phenotype column names:', args.ptylist_fpath)
    print('Path to (sub)list of confound column names:', args.cnflist_fpath)
    print('Path to output directory:', args.outdir)
    print('')

    X_fit, Y_fit, confounds, metadata = get_input_data(
            args.IDP_filepath,
            args.phenotype_filepath,
            cnfvars_fpath=args.confound_filepath,
            sublist_fpath=args.sublist_fpath,
            idplist_fpath=args.idplist_fpath,
            ptylist_fpath=args.ptylist_fpath,
            cnflist_fpath=args.cnflist_fpath,
            verbose=args.verbose
            )

    ### debugging code ###
    X_fit.to_csv(os.path.join(args.outdir,'X_fit.csv'))
    Y_fit.to_csv(os.path.join(args.outdir,'Y_fit.csv'))
    if args.confound_filepath is not None:
        confounds.to_csv(os.path.join(args.outdir,'confounds.csv'))
    ### debugging code ###
       

    regress_list = all_regressions(
            X_fit, 
            Y_fit, 
            confounds, 
            fit_type=args.fit_type, 
            debug=True
            )

    save_output(
            args.outdir, 
            regress_list, 
            metadata,
            outname=args.outname,
            verbose=args.verbose
            )
