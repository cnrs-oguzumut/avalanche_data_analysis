# Amorphous Plasticity Avalanche Analyzer

This Python script performs statistical analysis on avalanche data—energy drops $E$ and stress drops $S$—from amorphous plasticity simulations.

It reads one or more simulation log files, filters the data based on user-defined criteria, and generates plots, binned data, and fit parameters.

## 🚀 Features

* **Avalanche Distributions:** Generates log-log histograms for energy $E$ and stress $S$ avalanches.
* **Scaling Exponents:** Fits distributions to a truncated power law $P(x) \sim x^{-\epsilon} e^{-\lambda x}$ to find the exponent $\epsilon$.
* **Energy-Stress Scaling:** Calculates the scaling exponent $\gamma$ from the relationship $E \sim S^\gamma$ by fitting binned log-log data.
* **Inter-Event Distributions:** Analyzes the distribution of $\Delta\alpha$ (e.g., change in strain) between consecutive avalanche events.

## 📋 Prerequisites

The script requires the following Python libraries:

* `numpy`
* `matplotlib`
* `scipy`

You can install them using `pip`:

```bash
pip install numpy matplotlib scipy
````

## 📁 File Structure

The script is designed to work with the following directory structure:

```
project_folder/
│
├── analyze_avalanches.py     # <-- The analysis script
│
├── build1/                   # <-- Simulation 1 output
│   └── energy_stress_log.csv
├── build2/                   # <-- Simulation 2 output
│   └── energy_stress_log.csv
├── ...
│
└── statistics/               # <-- Output directory (created by the script)
    ├── all_distributions.png
    ├── fit_parameters.txt
    ├── data_histogram_energy.dat
    └── ...
```

## ⌨️ Input Data Format

The script reads `energy_stress_log.csv` files with a specific format:

  * **Comma-separated** (CSV).
  * The first row (header) is **skipped**.
  * The script expects the following columns:
      * `alpha` (Column index 1)
      * `energy_change` (Column index 6)
      * `stress_change` (Column index 7)
      * `plasticity_flag` (Column index 8)

-----

## ⚙️ How to Use

### 1\. Place Your Data

Arrange your simulation output files as shown in the **File Structure** section.

### 2\. Configure the Script

Open `analyze_avalanches.py` and edit the `CONFIG` dictionary located inside the `main()` function.

```python
def main():
    # ===== CONFIGURATION =====
    CONFIG = {
        'USE_MULTIPLE_FILES': True,
        'FILE_PATTERN': "./build*/energy_stress_log.csv",
        'SINGLE_FILE': "energy_stress_log.csv",
        'OUTPUT_DIR': './statistics',
        'SHOW_PLOTS': True, 
        
        'FILTERS': {
            'BY_PLASTICITY': False,
            'BY_ALPHA': True,
            'ALPHA_MIN': 0.1401,
            'ALPHA_MAX': 0.4,
            # ... etc ...
        },
        
        'ANALYSIS': {
            'FIT_DATA': True,
            'FIT_METHOD': 'logspace',
            # ... etc ...
        }
    }
    # =========================
    
    # ... (rest of the script) ...
```

See the **Configuration Details** section below for a full explanation of each parameter.

### 3\. Run the Script

Execute the script from your terminal:

```bash
python analyze_avalanches.py
```

The script will print its progress, including file processing, filtering results, and fitted parameters.

-----

## 🔧 Configuration Details

All settings are controlled by the `CONFIG` dictionary.

### Main Settings

  * `USE_MULTIPLE_FILES` (bool):
      * `True`: Processes and combines all files matching `FILE_PATTERN`.
      * `False`: Analyzes only the single file specified in `SINGLE_FILE`.
  * `FILE_PATTERN` (str): The [glob pattern](https://en.wikipedia.org/wiki/Glob_\(programming\)) to find files when `USE_MULTIPLE_FILES = True`.
  * `SINGLE_FILE` (str): The path to a single file to use when `USE_MULTIPLE_FILES = False`.
  * `OUTPUT_DIR` (str): Directory to save all plots and data. It will be created if it doesn't exist.
  * `SHOW_PLOTS` (bool): If `True`, `plt.show()` will be called at the end to display the combined plot.

### `FILTERS` Dictionary

This nested dictionary controls all data filtering.

  * `BY_PLASTICITY` (bool): If `True`, only includes data rows where `plasticity_flag == 1`.
  * `BY_ALPHA` (bool): If `True`, only includes data from the specified `[ALPHA_MIN, ALPHA_MAX]` range.
  * `ALPHA_MIN`, `ALPHA_MAX` (float): The min/max `alpha` values to include.
  * `BY_XMIN`, `BY_XMAX` (bool): These apply to the *positive avalanche data*. They set a minimum/maximum size for avalanches to be included in the histograms and fits.
  * `ENERGY_XMIN`, `STRESS_XMIN` (float): Minimum $E$ or $S$ to be included.
  * `ENERGY_XMAX`, `STRESS_XMAX` (float): Maximum $E$ or $S$ to be included.

### `ANALYSIS` Dictionary

This nested dictionary controls the analysis and fitting parameters.

  * `FIT_DATA` (bool): If `True`, fits the Energy and Stress distributions.
  * `FIT_METHOD` (str): Method for fitting. Currently `logspace` (default) or `weighted`.
  * `ANALYZE_ALPHA_DIFF` (bool): If `True`, calculates and plots the `delta_alpha` (inter-event) distribution.
  * `FIT_ALPHA_DIFF` (bool): If `True`, attempts to fit the `delta_alpha` distribution.
  * `NBIN` (int): Number of bins for logarithmic histograms ($E$, $S$, $\Delta\alpha$-log).
  * `NBIN_LINEAR` (int): Number of bins for the linear $\Delta\alpha$ histogram.
  * `NBIN_SCALING` (int): Number of bins to use for the $E \sim S^\gamma$ scaling analysis.

-----

## 📈 Output

All results are saved in the directory specified by `OUTPUT_DIR` (e.g., `./statistics/`). File names are appended with suffixes based on the filters used.

### 📊 Plots (`.png`)

  * `energy_vs_stress_scaling*.png`: Log-log plot showing $E$ vs. $S$. Includes raw data, binned averages, and the linear fit for $\gamma$.
  * `energy_distribution*.png`: Log-log plot of the energy avalanche distribution, $P(E)$, with its fit.
  * `stress_distribution*.png`: Log-log plot of the stress avalanche distribution, $P(S)$, with its fit.
  * `dalpha_distribution_log*.png`: Log-log plot of the inter-event ($\Delta\alpha$) distribution.
  * `dalpha_distribution_linear*.png`: Linear plot of the $\Delta\alpha$ distribution.
  * `all_distributions*.png`: A single summary image combining all distribution plots.

### 💾 Data Files (`.dat`)

These files contain the processed data used to generate the plots.

  * `data_energy_vs_stress_raw*.dat`: The raw, paired $\log_{10}(S)$, $\log_{10}(E)$ data.
  * `data_energy_vs_stress_binned*.dat`: The binned $\log_{10}(S)$, $\log_{10}(E)_{\text{mean}}$, $\log_{10}(E)_{\text{std}}$, `count` data.
  * `data_histogram_energy*.dat`: The binned $\log_{10}(E)$, $\log_{10}(P(E))$ data.
  * `data_histogram_stress*.dat`: The binned $\log_{10}(S)$, $\log_{10}(P(S))$ data.
  * `data_histogram_dalpha_log*.dat`: The log-binned $\log_{10}(\Delta\alpha)$, $\log_{10}(P(\Delta\alpha))$ data.
  * `data_histogram_dalpha_linear*.dat`: The linearly-binned $\Delta\alpha$, $P(\Delta\alpha)$ data.

### 📝 Fit Results (`.txt`)

  * `fit_parameters*.txt`: A text file summarizing the fitted parameters ($\epsilon$, $\lambda$, $A$) for the Energy, Stress, and $\Delta\alpha$ distributions.

-----

## 🔬 Analysis Details

### Truncated Power Law

The script fits distributions to the following function:
$$P(x) = A \cdot x^{-\epsilon} \cdot \exp(-\lambda x)$$
where:

  * $\epsilon$ (`epsilon`) is the power-law exponent.
  * $\lambda$ (`lambda_`) is the exponential cutoff parameter.
  * $A$ (`A`) is the normalization constant.

### Energy-Stress Scaling

The script determines the exponent $\gamma$ by fitting a line to the binned data in log-log space:
$$\log_{10}(E) = \gamma \cdot \log_{10}(S) + C$$

```
```