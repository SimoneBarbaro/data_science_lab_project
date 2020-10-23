### Usage of `load_sample_with_names`

```python
data_sample, sample_names = load_sample_with_names(frac=0.15, random_state=1, save=True)
```
**Input**
- `frac`: fraction of data from the original Excel file
- `random_state`: seed
- `save`: whether to save (and subsequently load) the output dataframes as compressed pickles (specified by `frac` and `random_state`) in the data folder

**Output**
- `data_sample`: dataframe matrix of interactions (from `create_matrix`)
- `sample_names`: two-column dataframe with the names of drug pairs corresponding to the rows of the matrix

The indices of the the output dataframes match each other.

<details>
  <summary>Sample Output</summary>
  
  `data_sample` (target names replaced with T{n})
  |   | T1 | T2 | T3 | ... | T249 | T250 | T251 |
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
  | 0 | 0.498 | 0.000 | 0.0 | ... | 0.0 | 0.432 | 0.000 |
  | 1 | 0.498 | 0.000 | 0.0 | ... | 0.0 | 0.432 | 0.000 |
  | 2 | 0.937 | 0.955 | 0.0 | ... | 0.0 | 0.832 | 0.000 |
  | ... | ... | ... | ... | ... | ... | ... | ... |
  | 10582 | 0.000 | 0.496 | 0.0 | ... | 0.0 | 0.000 | 0.000 |
  | 10583 | 0.000 | 0.496 | 0.0 | ... | 0.0 | 0.000 | 0.495 |
  | 10584 | 0.975 | 0.496 | 0.0 | ... | 0.0 | 0.492 | 0.000 |

  `sample_names`
  |       | name1         | name2         |
  |:-----:|---------------|---------------|
  | 0     | amprenavir    | amikacin      |
  | 1     | amprenavir    | thioridazine  |
  | 2     | amprenavir    | acyclovir     |
  | ...   | ...           | ...           |
  | 10582 | atracurium    | diphenoxylate |
  | 10583 | atracurium    | cortisone     |
  | 10584 | diphenoxylate | cortisone     |
</details>

### Usage of `load_full_matrix_with_names`

```python
data_full, names_full = load_full_matrix_with_names()
```
**Output**
- `data_full`: dataframe matrix of interactions (from `create_matrix`)
- `sample_full`: two-column dataframe with the names of drug pairs corresponding to the rows of the matrix

The indices of the the output dataframes match each other.

The output is saved as compressed pickles in the data folder.
