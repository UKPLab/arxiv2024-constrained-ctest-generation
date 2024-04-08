# User Study Analysis

We provide simple preprocessing scripts for R as well as the respective R scripts for the significance analysis conducted in the paper. First setup a virtual envirionment (e.g., conda) and install the required packages:

    conda create --name=analysis python=3.10
    conda activate analysis
    pip install -r requirements.txt

Next, install [R](https://www.r-project.org/) and [RStudio](https://posit.co/download/rstudio-desktop/) (RStudio is not required, but convenient). The R scripts use following packages:

* [mgcv](https://cran.r-project.org/web/packages/mgcv/index.html): Mixed GAM Computation Vehicle with Automatic Smoothness Estimation
* [itsadug](https://cran.r-project.org/web/packages/itsadug/index.html): Interpreting Time Series and Autocorrelated Data Using GAMMs
* [report](https://cran.r-project.org/web/packages/report/index.html): Automated Reporting of Results and Statistical Models
* [xtable](https://cran.r-project.org/web/packages/xtable/index.html): Export Tables to LaTeX or HTML
* [hash](https://cran.r-project.org/web/packages/hash/index.html): Full Featured Implementation of Hash Tables/Associative Arrays/Dictionaries 

The raw data found in `study_data_raw` can be converted via `format_data_for_r.py` and `format_data_for_r_feedback.py`, which will be written into the `r_data` folder. The statistical models are provided in `r_models`.