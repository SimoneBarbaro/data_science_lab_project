## Dependencies

To install all the library dependecies for this project, run the following command inside the project directory:

```
pip install -r requirements.txt
```

## Preparation

In order to use this repository, first the data files needs to be added in the `data/` folder. Read `data/README.md` for further instructions on how to do this. 

## Usage

In order to produce the results we collected during our project, use the `src/` folder as the working directory to run the `main.py` script.
The script allows many arguments to be specified: run

```
python main.py  --help
```

for complete usage instructions.

**Brief usage instructions** are as follows: fix the dataset with `--dataset` to `spider` or `tiger`. Choose the clustering method with `--clustering`, then choose the appropriate `--clustering_config` from the `config/` directory, and finally choose an appropriate `--run_name` specifying the subfolder under `results/` in which to save the results.
Leave the other arguments to their defaults to obtain the best runs in our report. Add the parameter `--no_normalize` to remove feature normalization prior to clustering.

Examples of runs we used are:

```
python main.py --dataset spider --clustering aggl --clustering_config ../config/aggl_5.json --run_name spider_aggl5norm

python main.py --dataset tiger --clustering aggl --clustering_config ../config/aggl_5.json --run_name tiger_aggl5norm 
```

As an alternative to running `main.py`, which does the full analysis as well as clustering, you can run `clustering_main.py` for only the clustering and `statistical_analysis_main.py` for only the statistical analysis. The usage is similar: run with `--help` for descriptions of the arguments. This is, however, only recommended for debugging purposes.

For instructions on the search of the clustering parameters, use 

```
python search_clustering_main.py  --help
```

All the configs in `config/` that contain the word search are the ones used to run in this program.

## Notebook

Finally, the Jupyter notebook `src/DSL-Notebook.ipynb` can be used to explore the results created by running the `main.py` script. Instructions for running the notebook are provided in `src/DSL-Notebook-Instructions.md` while usage instructions are within the notebook itself.
