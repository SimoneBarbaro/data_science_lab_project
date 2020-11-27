# Simple Method
If the data and results can be uploaded on Google servers for the duration of using the notebook, launch the exploration notebook by clicking on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SimoneBarbaro/data_science_lab_project/blob/main/src/DSL-Notebook.ipynb).
You may download the results ([`results_2020_11_27.zip`](https://polybox.ethz.ch/index.php/f/2160436597)) and the paired SPiDER data ([`matrix_spider_full.pkl.gz`](https://polybox.ethz.ch/index.php/f/2160073828) and [`matrix_spider_names_full.pkl.gz`](https://polybox.ethz.ch/index.php/f/2160073720)) beforehand.

---
# Local Installation
:warning: The following instructions are currently not up to date.

### Preparation
- Install Anaconda:
https://docs.anaconda.com/anaconda/install/windows/
- Download and extract this repository with results and the required data: https://polybox.ethz.ch/index.php/f/2157321373 (protected access)
  - <details>
    <summary>Manually update for the latest code and results</summary>

    - Download this code repository: https://github.com/SimoneBarbaro/data_science_lab_project/archive/main.zip
      - Extract the `.zip` archive into some folder.
    - Download the results archive `results_2020_11_26.zip`: https://polybox.ethz.ch/index.php/f/2157320838 (protected access)
      - Extract the `results` folder to the root of the previous extraction (next to the `src` and `data` folders)

    The following files need to be downloaded once only and can be re-used when updating:
    - Download the processed TWOSIDES database (`TWOSIDES_medDRA.csv.gz`): https://polybox.ethz.ch/index.php/s/Uemf21AIiZ7ooNi/download
      - Place it into the `data` folder.
    - Download the processed SPiDER dataset (`alldrugs_twosides_merged.csv`): https://polybox.ethz.ch/index.php/f/2152429962 (protected access)
      - Place it into the `data` folder.
  </details>

### Opening the notebook
1. Run Jupyter Notebook:
    - launch the [Anaconda Navigator](https://docs.anaconda.com/anaconda/user-guide/getting-started/) from the Start Menu
    - follow steps 1-2 of https://docs.anaconda.com/anaconda/user-guide/getting-started/#run-python-in-a-jupyter-notebook
2. In the embedded file browser, navigate to the repository set up in Preparation. Open the notebook called `DSL-Notebook.ipynb` in the `src` folder.
3. Follow the instructions in the notebook.
    - You can execute a cell by clicking on the black triangle icon at the top-left of a cell.
  Alternatively, you can press `Shift-Enter` to execute the selected cell (indicated by a blue bar on the left; move the selection with the up and down arrow keys or by clicking on another cell with the mouse).

### Ending the session
1. Close the notebook browser tab.
2. On the browser tab with file navigation, click on Quit in the top-right corner.
