#### Usage of `load_sample_with_names`

```python
data_sample, sample_names = load_sample_with_names(frac=0.15, random_state=1, save=False)
```
**Input**
- `frac`: fraction of data from the original Excel file
- `random_state`: seed
- `save`: whether to save (and subsequently load) the output dataframes as compressed pickles (specified by `frac` and `random_state`) in the data folder

**Output**
- `data_sample`: dataframe matrix of interactions (from `create_matrix`)
- `sample_names`: two-column dataframe with the names of drug pairs corresponding to the rows of the matrix

The indices of the the output dataframes match each other.

#### Usage of `load_full_matrix_with_names`

```python
data_full, names_full = load_full_matrix_with_names()
```
**Output**
- `data_full`: dataframe matrix of interactions (from `create_matrix`)
- `sample_full`: two-column dataframe with the names of drug pairs corresponding to the rows of the matrix

The indices of the the output dataframes match each other.

The output is saved as compressed pickles in the data folder.
