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

## `heterogeneity_test.py`
Perform three statistical tests, namely the Cochran's Q statistic (weighted sum of squared differences between individual study effects and the pooled effect across studies), I2 index (percentage of total variation across studies that is due to heterogeneity rather than chance), and Tau2 (estimates the true variability between the effect sizes of included studies).

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
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)

## Linear Mixed-Effects Regression (`regression.py` and `analysis.py`)
* [sklearn-lmer](https://pypi.org/project/sklearn-lmer/)
* [scipy](https://scipy.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [statsmodels](https://www.statsmodels.org/stable/index.html)

## Meta-analysis (`meta.py`)
* [pymare](https://pymare.readthedocs.io/en/latest/)
* [statsmodels](https://www.statsmodels.org/stable/index.html)

## ANOVAs (`ANOVA_notebook.ipynb`)
* [scipy](https://scipy.org/)
* [statsmodels](https://www.statsmodels.org/stable/index.html)

## Visualizations (`code_for_figures/`)
* [ggseg](https://ggseg.r-universe.dev)
* [ggplot](https://ggplot2.tidyverse.org/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)

# Data availability
* UKB data are available following an access application process: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access. This research was performed under UK Biobank application number 47267.
* ABCD data was available from the NIMH Data Archive when data were obtained for this study.
* HCP-A, HCP-D, and ANXPE (referred to as Dimensional Connectomics of Anxious Misery [DCAM]) data are available from the NIMH Data Archive collectively under the header of ‘CCF Data from the Human Connectome Projects’.
* HCP-YA data are available from the connectomeDB: https://db.humanconnectome.org/.

# Data preprocessing
Data preprocessing was performed with the following external code repositories:
* UK Biobank:
  * Paper:
    * Alfaro-Almagro, F., Jenkinson, M., Bangerter, N. K., Andersson, J. L. R., Griffanti, L., Douaud, G., Sotiropoulos, S. N., Jbabdi, S., Hernandez-Fernandez, M., Vallee, E., Vidaurre, D., Webster, M., McCarthy, P., Rorden, C., Daducci, A., Alexander, D. C., Zhang, H., Dragonu, I., Matthews, P. M., … Smith, S. M. (2018). Image processing and Quality Control for the first 10,000 brain imaging datasets from UK Biobank. NeuroImage, 166, 400–424. https://doi.org/10.1016/j.neuroimage.2017.10.034 
  * Code:
    * https://git.fmrib.ox.ac.uk/falmagro/UK_biobank_pipeline_v_1 
* ABCD
  * Paper:
    * Feczko, E., Conan, G., Marek, S., Tervo-Clemmens, B., Cordova, M., Doyle, O., Earl, E., Perrone, A., Sturgeon, D., Klein, R., Harman, G., Kilamovich, D., Hermosillo, R., Miranda-Dominguez, O., Adebimpe, A., Bertolero, M., Cieslak, M., Covitz, S., Hendrickson, T., … Fair, D. A. (2021). Adolescent brain cognitive development (ABCD) community MRI collection and utilities. In bioRxiv (p. 2021.07.09.451638). bioRxiv. https://doi.org/10.1101/2021.07.09.451638 
  * Code: https://docs.abcdstudy.org/latest/documentation/imaging/abcc_pipeline.html 
* HCP (young adult, aging, developing, ANXPE)
  * Paper:
Glasser, M. F., Sotiropoulos, S. N., Wilson, J. A., Coalson, T. S., Fischl, B., Andersson, J. L., Xu, J., Jbabdi, S., Webster, M., Polimeni, J. R., Van Essen, D. C., Jenkinson, M., & WU-Minn HCP Consortium. (2013). The minimal preprocessing pipelines for the Human Connectome Project. NeuroImage, 80, 105–124. https://doi.org/10.1016/j.neuroimage.2013.04.127
  * Code: https://github.com/Washington-University/HCPpipelines


