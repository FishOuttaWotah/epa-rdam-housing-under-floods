# Thesis model code: "Underwater" Real Estate; exploring housing market dynamics under severe flooding in Rotterdam.

_I'll first preface this section by saying I am not very familiar with git/Github, and have had some terrible accidents while trying to publish this thesis code here. Therefore, please forgive any sloppy git decisions in this repository._

This code is the agent-based model (ABM) developed for the author's MSc. thesis in Engineering and Policy Analysis, TU Delft. It is a rudimentary ABM that simulates simplified housebuyer agents deciding whether to buy houses that were flood-damaged/-discounted. A caveat to be mentioned upfront is that this model is extensive (and extensive models = great respect++) but not sufficiently validated, so I advise against using this for any actual policy. 

# Code structure

The code consists of three core phases:

1. the data prep phase before simulation
2. the simulation phase of multiple scenarios (termed "experiments")
3. the post-simulation output data prep phase

Here is an overview of the file structure:

1. (root) The core .py dependencies are here, with the different aspects of the code sorted, albeit not very neatly. There are 4 classes of files: 
   1. the @experiments.py file which initiates the experiments setup and runs the model under various scenarios
   2. "agent" prefix: these mostly contain functions or classes that involve agents and/or the creation of agents. (agent_household is a misnomer, households were agents in this model. Additionally, env_housing_market is technically an agent, but the current version of Pycharm is very unreliable with renaming.)
   3. "env" prefix, which contains functions/classes that form the model environment (not the code environment), i.e. the flooding mechanisms, flood damage and housing  market.
   4. "model" prefix, which contains functions/classes that are necessary for the simulation part, i.e. the ABM scheduler and data updating schemes. model_init is the script containing the MESA Model definition, model_scheduler for the MESA scheduler, and model_ledger for functions updating the state of the model per step.
2. The ```data``` directory, which contains the raw data from Dutch open data. Not all data is included due to file size restrictionsThese are specifically the terrain data (DSM maps), the wijk and buurt (district and neighbourhood) vector data, and the flood data themselves. **I may look to include some instructions how to download them yourself, for now I'll assume the data is there.**
3. The ```data_exploration``` directory, which contains Jupyter notebooks that handle the data prep and the post-simulation data analysis. 
4. The ```data_model_inputs``` directory, which hold intermediate data (cleaned from the Data Prep phase) and are used for the model simulation itself.
5. The ```data_model_outputs``` directory, which hold output data generated from the Simulation phase, but still need to be processed for further statistical work or graphical plotting. The raw output data are also not included (~300MB) due to file size limitations.

# Data prep

This study uses various raw data from Dutch open data sources, and these data sources need to be modified somewhat to be used in the model. _The raw data is stored in the ```data``` directory, the Jupyter notebooks used to process these data is in the ```data_exploration``` directory, and the output of these raw data is in the ```data_model_inputs``` directory.

To generate the model input data, several Jupyter notebooks need to run:

1. map_dtm_merge to merge the terrain DSM maps (they come in tiles)
2. pre-flooding-specific to generate the necessary flood submaps per district 
3. pre-rdam-city-only to filter out districts that were not used

# ABM simulation

Next is the actual simulation run itself. The main file is @experiments.py. The model runs in 2 parts: first the control scenarios and single flood scenarios (with 40 replications) and the multiple flood scenarios (with 1 replication each). You'd need to change the variable ```run_pt``` from 1 to 2 to ensure the entire experiment is run. Total time takes ~30 mins on my 2017 Thinkpad with 24GB RAM and i7-7700hq, note the simulation is mostly RAM-limited. By default, the script runs on 7 logical cores, so be careful.

# Post-simulation

Most of the post simulation data output is generated in the plotting_aggregation.ipynb Jupyter notebook in the data_exploration directory. This takes the raw output data (which is not provided here in Github due to its large filesize of ~300MB)and converts them into nice graphs and statistics. This also does a data prep operation for the plotting of price indices per district, which is done in the pre-flooding-specific-plotting.ipynb notebook in the data_exploration directory.

# Contact

Questions can be sent to fishermanlee97@gmail.com. I don't intend to expand on this project in the near future, but I am happy to elaborate on the work.
