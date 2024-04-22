import PySimpleGUI as sg
from Function import preprocessing_features_extraction, Analysis, complete_plot, kfanalysis

choose = sg.PopupYesNo("Do you want complete execution ?")
if choose == "Yes":
    ## Preprocessing and Feature Extraction
    path = "Skincancer Dataset"
    preprocessing_features_extraction(path)
    ## Training Percentage Varying Analysis
    Analysis()
    ## Cross Validation Analysis
    kfanalysis()
    ## Graph Plotting
    complete_plot(1)
else:
    ## Graph Plotting
    complete_plot(1)
