# Avalanche Statistics Analysis for Materials

A comprehensive Python tool for analyzing avalanche statistics in materials undergoing plastic deformation, with support for critical point detection and before/after transition analysis.

## Overview

This tool analyzes energy and stress avalanches from mechanical simulations of crystalline and architected materials. It provides:

- **Power-law distribution analysis** with truncated exponential cutoffs
- **Critical point detection** based on maximum energy
- **Before/after transition analysis** to study regime changes during deformation
- **Energy-stress scaling relationships** (E ~ S^γ)
- **Time-between-events statistics** (Δα distributions)
- **Comprehensive visualization** with comparison plots

Perfect for studying:
- Plasticity in crystals and amorphous materials
- Shear band formation and localization
- Avalanche behavior and critical phenomena
- Transitions from elastic to plastic regimes

## Features

### Core Analysis Capabilities

1. **Avalanche Size Distributions**
   - Logarithmic binning for power-law analysis
   - Truncated power-law fitting: P(x) = A·x^(-ε)·exp(-λx)
   - Automatic parameter estimation with uncertainty quantification

2. **Critical Point Analysis**
   - Automatic detection of critical α (maximum energy point)
   - Split analysis for before/after transition regimes
   - Separate statistics for pre-critical and post-critical behavior

3. **Energy-Stress Scaling**
   - Power-law scaling analysis: E ~ S^γ
   - Binned analysis with error bars
   - Comparison of scaling exponents across regimes

4. **Inter-event Statistics**
   - Δα (time between events) distributions
   - Both logarithmic and linear binning options
   - Power-law analysis of event clustering

5. **Comprehensive Visualization**
   - Time series plots with running averages
   - Distribution comparison plots (before/after/full)
   - Scaling relationship visualizations
   - Combined multi-panel figures

## Installation

### Requirements

python >= 3.7
numpy >= 1.19
matplotlib >= 3.3
scipy >= 1.5

### Quick Install

git clone https://github.com/cnrs-oguzumut/avalanche_data_analysis.git
cd avalanche_data_analysis
pip install -r requirements.txt

Or install dependencies manually:
pip install numpy matplotlib scipy

## Usage

### Basic Usage

python avalanche_analysis.py

The script will:
1. Search for energy_stress_log.csv files in ./build* directories
2. Process and combine all datasets
3. Perform statistical analysis
4. Generate plots and save results to ./statistics/

### Input Data Format

Your CSV file should have the following structure:

step,alpha,energy_total,stress_total,energy_cumul,stress_cumul,energy_change,stress_change,plasticity_flag
0,0.000000,1.234e-05,2.345e+01,1.234e-05,2.345e+01,0.000e+00,0.000e+00,0
1,0.000100,1.235e-05,2.346e+01,1.235e-05,2.346e+01,1.000e-07,1.000e-01,1

**Columns used:**
- alpha (column 2): Loading parameter
- energy_cumul (column 5): Cumulative energy (for time series)
- stress_cumul (column 6): Cumulative stress (for time series)
- energy_change (column 7): Energy avalanche size
- stress_change (column 8): Stress drop size
- plasticity_flag (column 9): 1 for plastic events, 0 for elastic

## Configuration

Edit the CONFIG dictionary in main() to customize analysis:

### Key Configuration Options

**Split Mode (SPLIT_BY_MAX_ENERGY)**
- True: Analyze separately before and after maximum energy point
- False: Analyze full dataset only

**Filters**
- BY_PLASTICITY: Only analyze plastic events (flag = 1)
- BY_ALPHA: Restrict analysis to specific loading range
- BY_XMIN/BY_XMAX: Apply threshold filters to avalanche sizes

**Analysis Methods**
- FIT_METHOD = 'logspace': Standard log-space fitting (recommended)
- FIT_METHOD = 'weighted': Weighted fitting for emphasis on specific regions

## Outputs

### Directory Structure

statistics/
├── critical_alphas_*.txt                    # Critical α values for each dataset
├── fit_parameters_*.txt                     # All fit parameters (ε, λ, γ)
├── data_histogram_energy_*.dat              # Binned energy distributions
├── data_histogram_stress_*.dat              # Binned stress distributions
├── data_histogram_dalpha_log_*.dat          # Δα distributions
├── data_energy_vs_stress_binned_*.dat       # E-S scaling (binned)
├── data_energy_vs_stress_raw_*.dat          # E-S scaling (raw)
├── *_before_*.dat                           # Before critical α data
├── *_after_*.dat                            # After critical α data
├── alpha_vs_energy_timeseries_*.png         # Time series plots
├── alpha_vs_stress_timeseries_*.png
├── energy_comparison_*.png                  # Before/after/full comparisons
├── stress_comparison_*.png
├── energy_vs_stress_scaling_comparison_*.png
└── all_distributions_*.png                  # Combined overview

### Fit Parameters File

Example output from fit_parameters_*.txt:

============================================================
TRUNCATED POWER LAW FIT PARAMETERS
============================================================
Truncated Power Law Fit: P(x) = A * x^(-epsilon) * exp(-lambda * x)
Fitting method: logspace

============================================================
FULL DATASET
============================================================
ENERGY:
  A       = 1.234567e+02 ± 5.678901e+00
  epsilon = 1.45 ± 0.03
  lambda  = 2.345678e+02 ± 1.234567e+01

STRESS:
  A       = 9.876543e+01 ± 4.321098e+00
  epsilon = 1.52 ± 0.04
  lambda  = 1.234567e+02 ± 6.789012e+00

============================================================
ENERGY-STRESS SCALING PARAMETERS
============================================================
Power Law Scaling: E ~ S^γ (log10(E) = γ * log10(S) + intercept)

E-S SCALING (FULL):
  gamma (γ)  = 1.234567
  intercept  = -2.345678
  R²         = 0.987654

## Scientific Background

### Avalanche Statistics in Materials

When materials undergo plastic deformation, they often exhibit avalanche-like behavior where stress and energy are released in discrete bursts. These avalanches follow power-law statistics with exponential cutoffs.

### Critical Transitions

The critical α (maximum energy point) often corresponds to:
- Transition from elastic to plastic regime
- Onset of shear band localization
- System-spanning avalanches

Analyzing before/after this transition reveals:
- Changes in avalanche statistics (ε, λ parameters)
- Evolution of energy-stress coupling (γ exponent)
- Transformation of inter-event timing (Δα distributions)

### Energy-Stress Scaling

The relationship between energy and stress drops:

E ~ S^γ

- γ < 1: Energy-efficient avalanches
- γ = 1: Linear coupling
- γ > 1: Energy-intensive avalanches

Changes in γ at the critical point indicate fundamental shifts in deformation mechanisms.

## Examples

### Example 1: Basic Analysis

# Analyze single file
CONFIG['USE_MULTIPLE_FILES'] = False
CONFIG['SINGLE_FILE'] = 'my_simulation.csv'
CONFIG['SPLIT_BY_MAX_ENERGY'] = False

### Example 2: Multiple Simulations with Split Analysis

# Analyze all simulations in build directories
CONFIG['USE_MULTIPLE_FILES'] = True
CONFIG['FILE_PATTERN'] = './build*/energy_stress_log.csv'
CONFIG['SPLIT_BY_MAX_ENERGY'] = True

### Example 3: Focused Analysis on Plastic Events

CONFIG['FILTERS']['BY_PLASTICITY'] = True
CONFIG['FILTERS']['BY_ALPHA'] = True
CONFIG['FILTERS']['ALPHA_MIN'] = 0.1
CONFIG['FILTERS']['ALPHA_MAX'] = 0.5

## Troubleshooting

**Problem: No data after filtering**
Solution: Check filter thresholds (ALPHA_MIN, ALPHA_MAX, XMIN values)
         Verify plasticity flags are correctly set in input data

**Problem: Fitting fails**
Solution: Try switching FIT_METHOD between 'logspace' and 'weighted'
         Increase NBIN for better statistics
         Check if data has sufficient range for power-law fit

**Problem: Memory issues with large datasets**
Solution: Process files individually (USE_MULTIPLE_FILES = False)
         Reduce NBIN_LINEAR for Δα analysis
         Use stricter filters to reduce data volume

## Citation

If you use this tool in your research, please cite:

@software{avalanche_analysis,
  author = {Umut, Oguz},
  title = {Avalanche Statistics Analysis for Materials},
  year = {2024},
  url = {https://github.com/cnrs-oguzumut/avalanche_data_analysis}
}

## Related Publications

This tool was developed as part of research on:
- "Mechanics of Displacive Instabilities in Solids" - HDR Thesis (in preparation)
- "Quasi-amorphous Crystals: Bridging Crystal Plasticity and Amorphous Materials" - Physical Review Letters (in preparation)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For bug reports or feature requests, open an issue on GitHub.

## License

MIT License - see LICENSE file for details

## Author

**Oguz Umut**  
Chargé de Recherche CNRS  
LSPM Laboratory (Université Sorbonne Paris Nord)

## Acknowledgments

This work was supported by CNRS and developed as part of research at LSPM laboratory.

---

**Version:** 2.0 (with critical point analysis)  
**Last Updated:** November 2024
