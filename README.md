# GTD
Global Terrorism Database

### The goal of this project was to perform an exploratory data analysis (EDA) on the Golabl Terrorism Database, an OPEN SOURCE data set maintained by the National Consortium for the Study of Terrorism and Responses to Terrorism (START) at the University of Maryland. For more information on the data or to download the data please visit https://www.start.umd.edu/gtd/

### Once the initial EDA was performed I wanted to look at the events attributed to "Unknown" organizations and see how a machine learning model would perform when tasked with reclassifying the events to known organizations.

### For viewing the analysis on GitHub- open up "eda.ipynb" file to view the exploratory data analysis and characterization of major organizations. Open up "yearly_gname_preds.ipynb" to look at reclassifying events classified "Unknown" into new labels of other known organizations. The file "gtd_funcs.py" contains custom built functions to process the data and perform various tasks.

### To recreate the code: clone the repo and download the data from https://www.start.umd.edu/gtd/access/. The distribution used in this analysis was the '0617dist' edition (there is a newer distribution available which includes the year 2018).