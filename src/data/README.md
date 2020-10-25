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

### Usage of `get_twosides_meddra`

```python
twosides = get_twosides_meddra(pickle=True)
```
**Input**
- `pickle`: whether to read `data/TWOSIDES_medDRA.pkl.gz` (smaller, faster) instead of `data/TWOSIDES_medDRA.csv.gz`

**Output**
- `twosides`: the TWOSIDES database with SPiDER drug pairs and side effect classifications according to medDRA

<details>
  <summary>Sample Output</summary>
  
  |  | drug_1_name | drug_2_name | soc_code | hlgt_code | hlt_code | pt_code | soc_term | hlgt_term | hlt_term | pt_term |
  |-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-|
  | 0 | tamoxifen | prednisone | 10017947 | 10017977 | 10012736 | 10012735 | Gastrointestinal disorders | Gastrointestinal motility and defaecation cond... | Diarrhoea (excl infective) | Diarrhoea |
  | 1 | tamoxifen | prednisone | 10017947 | 10018012 | 10028817 | 10047700 | Gastrointestinal disorders | Gastrointestinal signs and symptoms | Nausea and vomiting symptoms | Vomiting |
  | 2 | tamoxifen | prednisone | 10017947 | 10018012 | 10013949 | 10013946 | Gastrointestinal disorders | Gastrointestinal signs and symptoms | Dyspeptic signs and symptoms | Dyspepsia |
  | 3 | tamoxifen | prednisone | 10018065 | 10018073 | 10003550 | 10025482 | General disorders and administration site cond... | General system disorders NEC | Asthenic conditions | Malaise |
</details>

### Usage of `filter_twosides`:

```python
data_matrix_ts, data_names_ts = filter_twosides(data_matrix, data_names, twosides)
```
**Input**
- `data_matrix`: dataframe matrix of interactions (from `create_matrix`)
- `data_names`: two-column dataframe with the names of drug pairs corresponding to the rows of the matrix
- `twosides`: the TWOSIDES database with columns `drug_1_name` and `drug_2_name` representing the drug pairs (as read by `get_twosides_meddra`)

**Output**
- `data_matrix_ts`: dataframe matrix of only the interactions where the pair is present in TWOSIDES
- `data_names_ts`: two-column dataframe of names of drug pairs present in TWOSIDES

<details>
  <summary>Sample Output</summary>
  
  `data_matrix_ts` (target names replaced with T{n})
  |   | T1 | T2 | T3 | ... | T249 | T250 | T251 |
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
  | 2 | 0.937 | 0.955 | 0.0 | ... | 0.0 | 0.832 | 0.0 |
  | 43 | 0.757 | 0.000 | 0.0 | ... | 0.0 | 0.432 | 0.0 |
  | 89 | 0.498 | 0.429 | 0.0 | ... | 0.0 | 0.432 | 0.0 |
  | ... | ... | ... | ... | ... | ... | ... | ... |
  | 10571 | 0.000 | 1.110 | 0.0 | ... | 0.0 | 0.460 | 0.0 |
  | 10573 | 0.460 | 0.993 | 0.0 | ... | 0.0 | 0.000 | 0.0 |
  | 10574 | 0.000 | 0.496 | 0.0 | ... | 0.0 | 0.000 | 0.4 |
  
  `data_names_ts`
  |       |      name1 |          name2 |
  |:-----:|-----------:|----------------|
  |     2 | amprenavir |      acyclovir |
  |    43 | amprenavir |     nevirapine |
  |    89 | amprenavir |     saquinavir |
  |   ... |        ... |            ... |
  | 10571 | paroxetine | amphotericin b |
  | 10573 | paroxetine |  diphenoxylate |
  | 10574 | paroxetine |      cortisone |
</details>

### Usage of `match_meddra`:

```python
meddra = match_meddra(data_names_ts, twosides)
```

**Input**
- `data_names_ts`: two-column dataframe of names of drug pairs present in TWOSIDES
- `twosides`: the TWOSIDES database with columns `drug_1_name` and `drug_2_name` representing the drug pairs (as read by `get_twosides_meddra`)

**Output**
- `meddra`: dataframe with the medDRA classifications of side effects for the drug pairs given in `data_names_ts`

The columns with the names of drug pairs are named `name1` and `name2` and the pairs are ordered as in `data_names_ts` (to facilitate merging later).

<details>
  <summary>Sample Output</summary>
  
  `meddra`
  |   | name1 | name2 | soc_code | hlgt_code | hlt_code | pt_code | soc_term | hlgt_term | hlt_term | pt_term |
  |-|-|-|-|-|-|-|-|-|-|-|
  | 3578 | pyridostigmine | ciprofloxacin | 10005329 | 10002086 | 10002067 | 10002034 | Blood and lymphatic system disorders | Anaemias nonhaemolytic and marrow depression | Anaemias NEC | Anaemia |
  | 3579 | pyridostigmine | ciprofloxacin | 10007541 | 10007521 | 10037908 | 10043071 | Cardiac disorders | Cardiac arrhythmias | Rate and rhythm disorders NEC | Tachycardia |
  | 3580 | pyridostigmine | ciprofloxacin | 10007541 | 10007521 | 10042600 | 10003658 | Cardiac disorders | Cardiac arrhythmias | Supraventricular arrhythmias | Atrial fibrillation |
  | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
  | 22480586 | leucovorin | acebutolol | 10047065 | 10014523 | 10034572 | 10051055 | Vascular disorders | Embolism and thrombosis | Peripheral embolism and thrombosis | Deep vein thrombosis |
  | 22480587 | leucovorin | acebutolol | 10047065 | 10047075 | 10018987 | 10018852 | Vascular disorders | Vascular haemorrhagic disorders | Haemorrhages NEC | Haematoma |
  | 22480588 | leucovorin | acebutolol | 10047065 | 10057166 | 10020774 | 10020772 | Vascular disorders | Vascular hypertensive disorders | Vascular hypertensive disorders NEC | Hypertension |
</details>
