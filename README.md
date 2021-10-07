# Bike Sharing Utilization Prediction
### * Conda environment:
The file conda_environment.yml contains a working conda environment for this code.
- install conda or anaconda
- run in terminal: 
  - conda env create -f environment.yml
  - conda activate mlearn2

### * Code structure:
The code can be run in two ways:
- "terminal" mode using run.py: It can be used as a test run. The input for prediction has to be introduced by hand in the run.py script
- "api-server" mode using run_api.py: deployment mode which can be linked to any front end application like Java, Tableau, etc.

The file settings.py contains some flexible configurations for the code like the path for training data, model to be used, forcing the re-training, etc. Check the file for more options.