Operational Guide for Reproducibility – Topic Modelling on Summarized News Articles  
By Giovanni Noè and Francesco Volpi Ghirardini  
University of Milan Bicocca – Text Mining and Search Course


This project investigates the effect of text summarization on topic modeling using the CNN/DailyMail news dataset. The workflow is implemented in Google Colab notebooks and is designed to run with GPU acceleration due to the computational cost of embedding generation.

The original dataset is available at: https://cs.nyu.edu/~kcho/DMQA/

The project folder is organized into the following content:
-  1_Cleaning_and_Exploration.ipynb - This notebook handles all data preprocessing and 
    exploratory data analysis.
-   2_Topic_Modelling.ipynb - This notebook performs topic modeling on the original (full)
    documents. 
-   3_Text_Summarization.ipynb - This notebook is dedicated to text summarization.
-   4_Topic_modeling_on_Summaries.ipynb -  This notebook applies the same topic 
    modeling pipeline used on full documents to the summarized texts. 
-   Function_Setup.ipynb - This file writes all utility functions used across the notebooks.
-   text_mining_project_utils.py - Python file containing the scripts for the functions created in 
    the function setup notebook

Before executing code it is necessary to download the data and correctly set the directories in the variuos notebooks.

The notebooks should be executed in the following order:
-   Function_Setup.ipynb
-   1_Cleaning_and_Exploration.ipynb
-   2_Topic_Modelling.ipynb
-   3_Summarization
-   4_Topic_modeling_on_Summaries

