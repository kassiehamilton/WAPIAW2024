# Overview

This repository contains the code for the analyses and visualization performed in the manuscript "The neuroimaging correlates of depression established across six large-scale datasets", [preprint available on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.02.660888v1). In this project, we used a linear mixed effects model to relate depression to imaging derived phenotypes from structural and resting state functional MRI data within each dataset. We then performed a meta analysis to establish the neuroimaging correlates of depression across 23,417 participants from six large datasets. It is important to note that data is not shared in this repository, as data sharing is in conflict with data use agreements of dataset providers. However, how data may be obtained directly from dataset providers is detailed in the "data availability" statement below. 

# Source Code

## `regression.py`
Perform within-dataset linear mixed effects regressions and output regression statistics (including coefficient, variance, and raw and FDR-corrected p-value) 

## `utils_py/analysis.py`
Includes helper functions used by `regression.py` which calculate statistics and run regression based on selected regression model

## `maps`
`functional_dictionary.csv` and `structural_dictionary_new.csv` 
CSVs containing the mapping of each node/region to Yeo networks.

## `meta.py`
Perform meta-analysis on outputs from individual dataset linear mixed effects regressions results (`regression.py`)

## `ANOVA_notebook.ipynb`
Perform all ANOVAs on meta-analysis outputs

# Visualization code (`code_for_figures`)

## `brain_visual.Rmd`
Generate individual brain plots to display meta-analytical estimates and corresponding significance for figure two using `ggseg` and `ggplot`

## `combine_figures.py`
Generate scatterplots and combine with individual brain plots to create figure two

## `meta_violin.py`
Create violin plots of meta-analysis estimates for figures three and four

# Code dependencies
This repository leverages several Python and R packages. Dependencies are listed below by script.

## General Use
* [numpy] (https://numpy.org/)
* [pandas] (https://pandas.pydata.org/)

## Linear Mixed-Effects Regression (`regression.py` and `analysis.py`)
* [sklearn-lmer] (https://pypi.org/project/sklearn-lmer/)
* [scipy] (https://scipy.org/)
* [scikit-learn] (https://scikit-learn.org/stable/)
* [statsmodels] (https://www.statsmodels.org/stable/index.html)

## Meta-analysis (`meta.py`)
* [pymare] (https://pymare.readthedocs.io/en/latest/)
* [statsmodels] (https://www.statsmodels.org/stable/index.html)

## ANOVAs (`ANOVA_notebook.ipynb`)
* [scipy] (https://scipy.org/)
* [statsmodels] (https://www.statsmodels.org/stable/index.html)

Visualizations (`code_for_figures/`)
* [ggseg] (https://ggseg.r-universe.dev)
* [ggplot] (https://ggplot2.tidyverse.org/)
* [matplotlib] (https://matplotlib.org/)
* [seaborn] (https://seaborn.pydata.org/)

# Data availability
* UKB data are available following an access application process: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access. This research was performed under UK Biobank application number 47267.
* ABCD data was available from the NIMH Data Archive when data were obtained for this study.
* HCP-A, HCP-D, and ANXPE (referred to as Dimensional Connectomics of Anxious Misery [DCAM]) data are available from the NIMH Data Archive collectively under the header of ‘CCF Data from the Human Connectome Projects’.
* HCP-YA data are available from the connectomeDB: https://db.humanconnectome.org/.


