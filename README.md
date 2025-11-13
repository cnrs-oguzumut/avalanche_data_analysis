# Amorphous Plasticity Avalanche Analyzer

This Python script performs statistical analysis on avalanche data (energy drops, ``$`E`$``, and stress drops, ``$`S`$``) from amorphous plasticity simulations.

It reads one or more simulation log files, filters the data based on user-defined criteria, and then generates:
1.  **Avalanche Distributions:** Histograms for energy (``$`E`$``) and stress (``$`S`$``) avalanches.
2.  **Scaling Exponents:** Fits distributions to a truncated power law (``$`P(x) \sim x^{-\epsilon} e^{-\lambda x}`$``) to find the exponent ``$`\epsilon`$``.
3.  **Energy-Stress Scaling:** Calculates the scaling exponent ``$`\gamma`$`` from the relationship ``$`E \sim S^\gamma`$`` by fitting the binned data.
4.  **Waiting Time Distributions:** Analyzes the distribution of "waiting times" (``$`\Delta\alpha`$``) between consecutive avalanche events.

All plots, processed data, and fit parameters are saved to an output directory.

## Prerequisites

The script requires the following Python libraries:

* `numpy`
* `matplotlib`
* `scipy`

You can install them using pip:
```bash
pip install numpy matplotlib scipy
```

## Input Data Format

The script is designed to read `energy_stress_log.csv` files with a specific format.

* The file must be comma-separated (CSV).
* The script skips the first row (header).
* The script expects the following columns:
    * `alpha` (Column 1)
    * `energy_change` (Column 6)
    * `stress_change` (Column 7)
    * `plasticity_flag` (Column 8)

## How to Use

### 1. Place Your Data

Place your `energy_stress_log.csv` file(s) in the appropriate location. The script is designed to find them using a file pattern, for example, in multiple `build*/` directories.

### 2. Configure the Script

Open the script (e.g., `analyze_avalanches.py`) and edit the `===== CONFIGURATION =====` block at the top of the `if __name__ == "__main__":` section.

```python
    # ===== CONFIGURATION =====
    USE_MULTIPLE_FILES = True     # *** NEW: Set to True for multiple files ***
    FILE_PATTERN = "./build*/energy_stress_log.csv"  # *** NEW: Pattern to match files ***
    SINGLE_FILE = "energy_stress_log.csv"    # Used only if USE_MULTIPLE_FILES = False
    
    FILTER_BY_PLASTICITY = False  # True: Only use events with plasticity_flag == 1
    FILTER_BY_ALPHA = True      # True: Filter by the alpha range below
    ALPHA_MIN = 0.16            # Minimum alpha to include
    ALPHA_MAX = .4              # Maximum alpha to include
    
    FILTER_BY_XMIN = True       # True: Apply a minimum threshold for E and S
    ENERGY_XMIN = 1e-3
    STRESS_XMIN = 20
    
    FILTER_BY_XMAX = False      # True: Apply a maximum threshold for E and S
    ENERGY_XMAX = 1e-2
    STRESS_XMAX = 1e-2
    
    FIT_DATA = True             # True: Fit distributions to a power law
    FIT_METHOD = 'logspace'     # 'logspace', 'weighted', or 'both'
    
    ANALYZE_ALPHA_DIFF = True   # True: Analyze the delta_alpha "waiting time" distribution
    FIT_ALPHA_DIFF = False      # True: Fit the delta_alpha distribution
    
    OUTPUT_DIR = './statistics' # Directory to save all plots and data
    
    nbin = 13                   # Number of bins for logarithmic histograms
    nbin_linear = 30            # Number of bins for linear histograms (delta_alpha)
    # =========================
```

**Key Parameters:**

* `USE_MULTIPLE_FILES`: Set to `True` to process and combine many files. Set to `False` to analyze only the file specified in `SINGLE_FILE`.
* `FILE_PATTERN`: The [glob pattern](https://en.wikipedia.org/wiki/Glob_(programming)) to find files when `USE_MULTIPLE_FILES = True`.
* `FILTER_BY_PLASTICITY`: If `True`, only includes data rows where `plasticity_flag == 1`.
* `FILTER_BY_ALPHA`: If `True`, only includes data from the specified `[ALPHA_MIN, ALPHA_MAX]` range.
* `FILTER_BY_XMIN` / `FILTER_BY_XMAX`: These apply to the *positive avalanche data* (events where `energy_change > 0`). They set a minimum/maximum size for the avalanches to be included in the histograms and fits.
* `FIT_DATA`: Set to `True` to fit the Energy and Stress distributions.
* `ANALYZE_ALPHA_DIFF`: Set to `True` to calculate and plot the distribution of `alpha` differences between consecutive filtered energy avalanches.
* `OUTPUT_DIR`: The name of the folder where all results will be saved. It will be created if it doesn't exist.

### 3. Run the Script

Execute the script from your terminal:

```bash
python analyze_avalanches.py
```

The script will print its progress to the console, including which files it's processing, the results of filtering, and the final fitted parameters.

## Output

All results are saved in the directory specified by `OUTPUT_DIR` (e.g., `./statistics/`).

### Plots (`.png`)

* `energy_vs_stress_scaling*.png`: A log-log plot showing the scaling relationship between Energy and Stress. It includes raw data points, binned averages, and a linear fit to find the exponent ``$`\gamma`$``.
* `energy_distribution*.png`: Log-log plot of the energy avalanche distribution, ``$`P(E)`$``, with the truncated power-law fit.
* `stress_distribution*.png`: Log-log plot of the stress avalanche distribution, ``$`P(S)`$``, with the truncated power-law fit.
* `dalpha_distribution_log*.png`: Log-log plot of the "waiting time" (``$`\Delta\alpha`$``) distribution.
* `dalpha_distribution_linear*.png`: Linear plot of the ``$`\Delta\alpha`$`` distribution.
* `all_distributions*.png`: A single image combining all distribution plots.

### Data Files (`.dat`)

* `data_energy_vs_stress_raw*.dat`: The raw, paired (``$`\log_{10}(S)`$``, ``$`\log_{10}(E)`$``) data used for the scaling plot.
* `data_energy_vs_stress_binned*.dat`: The binned (``$`\log_{10}(S)`$``, ``$`\log_{10}(E)_{\text{mean}}`$``, ``$`\log_{10}(E)_{\text{std}}`$``, `count`) data used for the ``$`\gamma`$`` fit.
* `data_histogram_energy*.dat`: The binned (``$`\log_{10}(E)`$``, ``$`\log_{10}(P(E))`$``) data for the energy histogram.
* `data_histogram_stress*.dat`: The binned (``$`\log_{10}(S)`$``, ``$`\log_{10}(P(S))`$``) data for the stress histogram.
* `data_histogram_dalpha_log*.dat`: The binned (``$`\log_{10}(\Delta\alpha)`$``, ``$`\log_{10}(P(\Delta\alpha))`$``) data.
* `data_histogram_dalpha_linear*.dat`: The binned (``$`\Delta\alpha`$``, ``$`P(\Delta\alpha)`$``) data.

### Fit Results (`.txt`)

* `fit_parameters*.txt`: A summary of the fitted parameters (``$`\epsilon`$``, ``$`\lambda`$``, ``$`A`$``) for the Energy, Stress, and ``$`\Delta\alpha`$`` distributions.

## Analysis Details

### Truncated Power Law

The script fits distributions to the following function:
```math
P(x) = A \cdot x^{-\epsilon} \cdot \exp(-\lambda x)
```
where:
* ``$`\epsilon`$`` (`epsilon`) is the power-law exponent.
* ``$`\lambda`$`` (`lambda_`) is the exponential cutoff parameter.
* ``$`A`$`` (`A`) is the normalization constant.

### Energy-Stress Scaling

The script determines the exponent ``$`\gamma`$`` by fitting a line to the binned data in log-log space:
```math
\log_{10}(E) = \gamma \cdot \log_{10}(S) + C
```
