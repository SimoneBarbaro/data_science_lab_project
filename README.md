#Dependency

to install all the library dependecies for this project, run the following command inside the project directory:

```
pip install -r requirements.txt
```

#Preparation

In order to use this repository, first the data files needs to be added in the data/ folder. Read the data/README.md file for further instructions on how to do this. 

#Usage

In order to produce the results we collected during our project, run on src/ folder the main.py file.
The command has many arguments, run

```
python main.py  --help
```

to learn the full usage.

A shorter usage is the follow: fix the dataset with  --dataset between spider or tiger. chooose the clustering method with --clustering then choose the appropriate --clustering_config from the config/ directory, then choose an appropriate --run_name where the results will be saved.
Leave the rest of the results as default to obtain the best runs on our report, add the parameter --no_normalize to remove the data normalization.

Examples of runs we used are:

```
python main.py --dataset spider --clustering aggl --clustering_config ../config/aggl_5.json --run_name spider_aggl5norm

python main.py --dataset tiger --clustering aggl --clustering_config ../config/aggl_5.json --run_name tiger_aggl5norm 
```

Alternatively to running main, which does the full analysis as well as clustering, you can run clustering_main.py for only the clustering and statistical_analysis_main.py for only the statistical analysis. The usage is also similar, run with --help to learn the arguments. However this is not recommended.

For instructions on the search of the clustering parameters, use 

```
python search_clustering_main.py  --help
```

All the configs in config/ that contain the word search are the ones used to run in this program.

#Notebook

Finally, the jupiter notebook src/DSL-Notebook.ipyb can be used to explore the results after running the main file to create them. For instructions for the notebook read src/DSL-Notebook-instructions.md
