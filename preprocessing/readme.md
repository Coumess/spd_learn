# DATA PREPROCESSING

## Folder Setup
```text
Folder
├── configs/
│   ├── config.yaml
├── data_scripts/
│   └── get_eeg_data.py
│   └── preprocess_data.py
├── utils/
│   └── data_helpers.py
│   └── precond.jl
├── tests/
│   ├── test_stat.py
│   ├── test_stat_ju.jl
│   ├── test_expP.py
│   ├── test_coshP.py
│   └── test_expT.py
└── extractCovMat.jl*
```

## Useful Julia Commands

| Purpose | Julia Command |
|-----------|---------------|
| Open a Julia terminal | `Alt + J + O` |
| Install a Julia package | `Alt + ]` and write `add PackageName` |
| Check current directory | `pwd()` |
| Change directory | `cd("chemin")` |
| Run a Julia script | `include("nom.jl")` |

---

## Download the FII_BCI_Corpus

1. Open a Julia terminal.
2. Install **Eegle.jl**.
3. Execute the following commands :
```julia
using Eegle
downloadDB()
```

### Some informations about the database :
![Details of the dataset of motor imagery](preprocessing/FII_BCI_MI_database_info.png)

---

## Extract Covariance Matrices from the Desired Databases

1. Open the Julia script `extractCovMat.jl`.
2. Modify the paths :
   - `MIDir` : path to the corpus databases.
   - `outDir` : output path to write the covariance matrices and associated labels.
3. Modify the parameters :
   - Classes to use.
   - Minimum number of trials.
   - Frequency band `bandPass`.
   - Artifact rejection threshold `upperLimit`.
4. Run the script.

### Output

The following files are generated :

- `covmat.npy` : covariance matrices.
- `labels.npy` : associated labels.

---

## Preprocess the Data

1. Open the `config.yaml` file.
2. Modify the variables in the `data/eeg` section :

| Variable | Description |
|-----------|------------|
| `data_dir` | Path to the folder containing the covariance matrices and labels |
| `processed_root` | Path to the folder where the preprocessed databases will be saved |
| `db_prefix` | Prefix of the database to preprocess |
| `precond_explVar` | PDimensionality reduction parameter : <br>- `0` : automatic reduction for databases with more than 64 electrodes <br>- `1` : no dimensionality reduction <br>- a value between `0` and `1` : variance ratio to keep |

3. Run the preprocessing script: :

⚠️ Make sure you are in the correct directory, if not : 
```bash
cd("D:/BCI/SPD_LEARN/preprocessing")
```

Run the script

```bash
python -m data_scripts.preprocess_data --config configs/config.yaml
```

## Results

The different folds of the preprocessed data are generated and saved in the folder defined by processed_root.