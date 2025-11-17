#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import os
import glob
import shutil


# =============================================================================
# == DATA IO FUNCTIONS
# =============================================================================

def read_energy_stress_log(filename="energy_stress_log.csv"):
    """
    Read the CSV file created by ConfigurationSaver::logEnergyAndStress
    
    Returns:
        alpha, energy (col 5), stress (col 6), energy_change (col 6), stress_change (col 7), plasticity_flag
    """
    try:
        # Skip header row and read all data
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        
        # Handle empty or single-line files
        if data.ndim == 0 or data.size == 0:
            print(f"  WARNING: File '{filename}' is empty or has no data. Skipping.")
            return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        
        # Handle files with only one data row
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Extract columns
        alpha = data[:, 1]                    # Column B (index 1)
        energy = data[:, 4]                   # Column 5 (index 4) - for time series
        stress = data[:, 5]                   # Column 6 (index 5) - for time series
        energy_change = data[:, 6]            # Column 7 (index 6) - for avalanche statistics
        stress_change = data[:, 7]            # Column 8 (index 7) - for avalanche statistics
        plasticity_flag = data[:, 8].astype(int)
        
        return alpha, energy, stress, energy_change, stress_change, plasticity_flag
        
    except Exception as e:
        print(f"  ERROR: Could not read file '{filename}'. Reason: {e}. Skipping.")
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))


def find_max_energy_alpha(alpha, energy):
    """
    Find the alpha value corresponding to the maximum energy.
    
    Returns:
        alpha_max_energy: the alpha value at maximum energy
        max_energy: the maximum energy value
        max_idx: the index of the maximum
    """
    if len(energy) == 0:
        return None, None, None
    
    max_idx = np.argmax(energy)
    alpha_max_energy = alpha[max_idx]
    max_energy = energy[max_idx]
    
    return alpha_max_energy, max_energy, max_idx


def load_and_process_data(filenames, config):
    """
    Process one or more simulation files, apply filters, and
    return all necessary combined raw data streams for analysis.
    
    If SPLIT_BY_MAX_ENERGY is enabled, also returns data split into
    "before" and "after" the maximum energy alpha for each dataset.
    """
    all_delta_alpha = []
    all_energy_data = []
    all_stress_data = []
    all_paired_energy = []
    all_paired_stress = []
    
    # For split analysis
    all_delta_alpha_before = []
    all_energy_data_before = []
    all_stress_data_before = []
    all_paired_energy_before = []
    all_paired_stress_before = []
    
    all_delta_alpha_after = []
    all_energy_data_after = []
    all_stress_data_after = []
    all_paired_energy_after = []
    all_paired_stress_after = []
    
    # Store UNFILTERED time series data per file for plotting
    time_series_data = []
    
    # Store critical alpha values for each file
    critical_alphas = []
    
    cfg_filters = config['FILTERS']
    split_mode = config.get('SPLIT_BY_MAX_ENERGY', False)
    
    for filename in filenames:
        print(f"  Processing: {filename}")
        
        # Read data - now getting both time series (col 5,6) and change data (col 6,7)
        alpha, energy, stress, energy_change, stress_change, plasticity_flag = read_energy_stress_log(filename)
        
        if len(alpha) == 0:
            continue
        
        # Find critical alpha (maximum energy point) for this dataset
        alpha_critical, max_energy, max_idx = find_max_energy_alpha(alpha, energy)
        
        if split_mode and alpha_critical is not None:
            print(f"    Critical α (max energy): {alpha_critical:.6f} (E_max = {max_energy:.6e})")
            critical_alphas.append({
                'filename': filename,
                'alpha_critical': alpha_critical,
                'max_energy': max_energy,
                'max_idx': max_idx
            })
        
        # Apply common filters (for statistical analysis only, using columns 6 and 7)
        mask = np.ones(len(alpha), dtype=bool)
        
        if cfg_filters['BY_PLASTICITY']:
            plasticity_mask = (plasticity_flag == 1)
            mask = mask & plasticity_mask
        
        if cfg_filters['BY_ALPHA']:
            alpha_mask = (alpha >= cfg_filters['ALPHA_MIN']) & (alpha <= cfg_filters['ALPHA_MAX'])
            mask = mask & alpha_mask
        
        energy_change_filtered = energy_change[mask]
        stress_change_filtered = stress_change[mask]
        alpha_filtered = alpha[mask]
        
        # Track indices of accepted avalanches for visualization
        original_indices = np.arange(len(alpha))
        filtered_indices = original_indices[mask]
        
        # --- Stream 1 & 2: Energy Distribution & Delta Alpha ---
        positive_energy_mask = energy_change_filtered > 0
        alpha_for_energy = alpha_filtered[positive_energy_mask]
        energy_data = energy_change_filtered[positive_energy_mask]
        accepted_energy_indices = filtered_indices[positive_energy_mask]
        
        # Apply xmin/xmax filters
        if cfg_filters['BY_XMIN']:
            energy_xmin_mask = energy_data >= cfg_filters['ENERGY_XMIN']
            alpha_for_energy = alpha_for_energy[energy_xmin_mask]
            energy_data = energy_data[energy_xmin_mask]
            accepted_energy_indices = accepted_energy_indices[energy_xmin_mask]
        
        if cfg_filters['BY_XMAX']:
            energy_xmax_mask = energy_data <= cfg_filters['ENERGY_XMAX']
            alpha_for_energy = alpha_for_energy[energy_xmax_mask]
            energy_data = energy_data[energy_xmax_mask]
            accepted_energy_indices = accepted_energy_indices[energy_xmax_mask]
        
        # Compute delta_alpha for THIS simulation
        if len(alpha_for_energy) > 1:
            all_delta_alpha.append(np.diff(alpha_for_energy))
        
        all_energy_data.append(energy_data)

        # --- Split analysis: before and after critical alpha ---
        if split_mode and alpha_critical is not None:
            # Before critical alpha
            mask_before = alpha_for_energy < alpha_critical
            energy_before = energy_data[mask_before]
            alpha_before = alpha_for_energy[mask_before]
            
            if len(alpha_before) > 1:
                all_delta_alpha_before.append(np.diff(alpha_before))
            all_energy_data_before.append(energy_before)
            
            # After critical alpha
            mask_after = alpha_for_energy >= alpha_critical
            energy_after = energy_data[mask_after]
            alpha_after = alpha_for_energy[mask_after]
            
            if len(alpha_after) > 1:
                all_delta_alpha_after.append(np.diff(alpha_after))
            all_energy_data_after.append(energy_after)
            
            print(f"    Energy avalanches before α_c: {len(energy_before)}, after α_c: {len(energy_after)}")

        # --- Stream 3: Stress Distribution ---
        positive_stress_mask = stress_change_filtered > 0
        stress_data = stress_change_filtered[positive_stress_mask]
        alpha_for_stress = alpha_filtered[positive_stress_mask]
        accepted_stress_indices = filtered_indices[positive_stress_mask]
        
        if cfg_filters['BY_XMIN']:
            stress_xmin_mask = stress_data >= cfg_filters['STRESS_XMIN']
            stress_data = stress_data[stress_xmin_mask]
            alpha_for_stress = alpha_for_stress[stress_xmin_mask]
            accepted_stress_indices = accepted_stress_indices[stress_xmin_mask]
        
        if cfg_filters['BY_XMAX']:
            stress_xmax_mask = stress_data <= cfg_filters['STRESS_XMAX']
            stress_data = stress_data[stress_xmax_mask]
            alpha_for_stress = alpha_for_stress[stress_xmax_mask]
            accepted_stress_indices = accepted_stress_indices[stress_xmax_mask]
            
        all_stress_data.append(stress_data)
        
        # Split stress data
        if split_mode and alpha_critical is not None:
            mask_before = alpha_for_stress < alpha_critical
            all_stress_data_before.append(stress_data[mask_before])
            
            mask_after = alpha_for_stress >= alpha_critical
            all_stress_data_after.append(stress_data[mask_after])
            
            print(f"    Stress avalanches before α_c: {np.sum(mask_before)}, after α_c: {np.sum(mask_after)}")
        
        # --- Stream 4: Paired E-S Data for Scaling ---
        both_positive_mask = (energy_change_filtered > 0) & (stress_change_filtered > 0)
        energy_paired = energy_change_filtered[both_positive_mask]
        stress_paired = stress_change_filtered[both_positive_mask]
        alpha_paired = alpha_filtered[both_positive_mask]
        
        # Apply xmin/xmax filters to the pairs
        pair_mask = np.ones(len(energy_paired), dtype=bool)
        if cfg_filters['BY_XMIN']:
            pair_mask = pair_mask & (energy_paired >= cfg_filters['ENERGY_XMIN'])
            pair_mask = pair_mask & (stress_paired >= cfg_filters['STRESS_XMIN'])
        
        if cfg_filters['BY_XMAX']:
            pair_mask = pair_mask & (energy_paired <= cfg_filters['ENERGY_XMAX'])
            pair_mask = pair_mask & (stress_paired <= cfg_filters['STRESS_XMAX'])
        
        all_paired_energy.append(energy_paired[pair_mask])
        all_paired_stress.append(stress_paired[pair_mask])
        
        # Split paired data
        if split_mode and alpha_critical is not None:
            alpha_paired_filtered = alpha_paired[pair_mask]
            energy_paired_filtered = energy_paired[pair_mask]
            stress_paired_filtered = stress_paired[pair_mask]
            
            mask_before = alpha_paired_filtered < alpha_critical
            all_paired_energy_before.append(energy_paired_filtered[mask_before])
            all_paired_stress_before.append(stress_paired_filtered[mask_before])
            
            mask_after = alpha_paired_filtered >= alpha_critical
            all_paired_energy_after.append(energy_paired_filtered[mask_after])
            all_paired_stress_after.append(stress_paired_filtered[mask_after])
        
        # Store UNFILTERED raw time series data WITH accepted avalanche indices
        time_series_data.append({
            'filename': filename,
            'alpha': alpha.copy(),
            'energy': energy.copy(),
            'stress': stress.copy(),
            'accepted_energy_indices': accepted_energy_indices,
            'accepted_stress_indices': accepted_stress_indices,
            'alpha_critical': alpha_critical,
            'max_idx': max_idx
        })
    
    # Combine all arrays from all files
    raw_data = {
        'delta_alpha': np.concatenate(all_delta_alpha) if all_delta_alpha else np.array([]),
        'energy': np.concatenate(all_energy_data) if all_energy_data else np.array([]),
        'stress': np.concatenate(all_stress_data) if all_stress_data else np.array([]),
        'paired_energy': np.concatenate(all_paired_energy) if all_paired_energy else np.array([]),
        'paired_stress': np.concatenate(all_paired_stress) if all_paired_stress else np.array([]),
        'time_series': time_series_data,
        'critical_alphas': critical_alphas
    }
    
    # Add split data if enabled
    if split_mode:
        raw_data['before'] = {
            'delta_alpha': np.concatenate(all_delta_alpha_before) if all_delta_alpha_before else np.array([]),
            'energy': np.concatenate(all_energy_data_before) if all_energy_data_before else np.array([]),
            'stress': np.concatenate(all_stress_data_before) if all_stress_data_before else np.array([]),
            'paired_energy': np.concatenate(all_paired_energy_before) if all_paired_energy_before else np.array([]),
            'paired_stress': np.concatenate(all_paired_stress_before) if all_paired_stress_before else np.array([]),
        }
        raw_data['after'] = {
            'delta_alpha': np.concatenate(all_delta_alpha_after) if all_delta_alpha_after else np.array([]),
            'energy': np.concatenate(all_energy_data_after) if all_energy_data_after else np.array([]),
            'stress': np.concatenate(all_stress_data_after) if all_stress_data_after else np.array([]),
            'paired_energy': np.concatenate(all_paired_energy_after) if all_paired_energy_after else np.array([]),
            'paired_stress': np.concatenate(all_paired_stress_after) if all_paired_stress_after else np.array([]),
        }

    print(f"\n{'='*60}")
    print(f"COMBINED RESULTS:")
    print(f"  Total delta_alpha: {len(raw_data['delta_alpha'])}")
    print(f"  Total energy avalanches: {len(raw_data['energy'])}")
    print(f"  Total stress avalanches: {len(raw_data['stress'])}")
    print(f"  Total paired E-S events: {len(raw_data['paired_energy'])}")
    print(f"  Number of time series files: {len(time_series_data)}")
    
    if split_mode:
        print(f"\n  BEFORE CRITICAL α:")
        print(f"    Energy avalanches: {len(raw_data['before']['energy'])}")
        print(f"    Stress avalanches: {len(raw_data['before']['stress'])}")
        print(f"    Paired E-S events: {len(raw_data['before']['paired_energy'])}")
        print(f"\n  AFTER CRITICAL α:")
        print(f"    Energy avalanches: {len(raw_data['after']['energy'])}")
        print(f"    Stress avalanches: {len(raw_data['after']['stress'])}")
        print(f"    Paired E-S events: {len(raw_data['after']['paired_energy'])}")
    
    print(f"{'='*60}\n")
    
    return raw_data



# =============================================================================
# == ANALYSIS FUNCTIONS (BINNING, FITTING)
# =============================================================================

def compute_running_average(x, y, window_size):
    """
    Compute running average of y as a function of x using a sliding window.
    
    Args:
        x: array of x values (e.g., alpha)
        y: array of y values (e.g., energy or stress)
        window_size: number of points in the sliding window
    
    Returns:
        x_avg, y_avg: arrays of averaged values
    """
    if len(x) < window_size:
        return x, y
    
    # Sort by x to ensure proper ordering
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    x_avg = []
    y_avg = []
    
    for i in range(len(x_sorted) - window_size + 1):
        x_avg.append(np.mean(x_sorted[i:i+window_size]))
        y_avg.append(np.mean(y_sorted[i:i+window_size]))
    
    return np.array(x_avg), np.array(y_avg)


def logarithmic_binning(data, nbin=14, xmin=None, xmax=None, density=True, padding=0.01):
    """
    Perform logarithmic binning
    
    Parameters:
    -----------
    data : array-like
        Input data to bin
    nbin : int
        Number of bins
    xmin : float, optional
        Minimum value for binning range
    xmax : float, optional
        Maximum value for binning range
    density : bool
        If True, returns probability density. If False, returns counts/bin_width
    padding : float
        Fraction of log-range to add as buffer on each side (default 0.05 = 5%)
    """
    # Filter: keep only positive values
    data = data[data > 0]
    
    if len(data) == 0:
        return None, None, None, None, None
    
    # Determine range
    if xmin is None:
        xmin = np.min(data)
    if xmax is None:
        xmax = np.max(data)

    print(f"    Log-binning range: xmin={xmin:.6e}, xmax={xmax:.6e}")

    
    # Convert to log space for padding
    log_xmin = np.log10(xmin)
    log_xmax = np.log10(xmax)
    
    # Add buffer zone (padding) to log-range
    log_range = log_xmax - log_xmin
    log_xmin -= padding * log_range
    log_xmax += padding * log_range
    
    # Create logarithmic bins with padded range
    bins = np.logspace(log_xmin, log_xmax, nbin+1)
    
    # Compute histogram
    hist, bin_edges = np.histogram(data, bins=bins, density=density)
    
    # Compute bin centers (arithmetic mean)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute bin widths
    dx = bin_edges[1:] - bin_edges[:-1]
    
    # Compute log10 values
    log_bin_centers = np.log10(bin_centers)
    log_hist = np.log10(hist)
    
    return bin_centers, hist, log_bin_centers, log_hist, dx

def linear_binning(data, nbin=20, xmin=None, xmax=None, density=True, padding=0.05):
    """
    Perform linear binning for data
    
    Parameters:
    -----------
    data : array-like
        Input data to bin
    nbin : int
        Number of bins
    xmin : float, optional
        Minimum value for binning range
    xmax : float, optional
        Maximum value for binning range
    density : bool
        If True, returns probability density. If False, returns counts/bin_width
    padding : float
        Fraction of range to add as buffer on each side (default 0.05 = 5%)
    """
    # Filter: keep only positive values
    data = data[data > 0]
    
    if len(data) == 0:
        return None, None, None
    
    # Determine range
    if xmin is None:
        xmin = np.min(data)
    if xmax is None:
        xmax = np.max(data)
    
    print(f"    Linear-binning range: xmin={xmin:.6e}, xmax={xmax:.6e}")
    # Add buffer zone (padding) to range
    data_range = xmax - xmin
    xmin -= padding * data_range
    xmax += padding * data_range
    
    # Create linear bins
    bins = np.linspace(xmin, xmax, nbin+1)
    
    # Compute histogram
    hist, bin_edges = np.histogram(data, bins=bins, density=density)
    
    # Compute bin centers (arithmetic mean)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute bin widths
    dx = bin_edges[1:] - bin_edges[:-1]
    
    return bin_centers, hist, dx

def truncated_powerlaw(x, A, epsilon, lambda_):
    """
    Truncated power law: P(x) = A * x^(-epsilon) * exp(-lambda * x)
    """
    return A * x**(-epsilon) * np.exp(-lambda_ * x)


def log_truncated_powerlaw(log_x, log_A, epsilon, lambda_):
    """
    Log of truncated power law for fitting in log space
    log(P(x)) = log(A) - epsilon * log(x) - lambda * x * log(e)
    """
    x = 10**log_x
    return log_A - epsilon * log_x - lambda_ * x * np.log10(np.e)


def fit_truncated_powerlaw_logspace(bin_centers, hist, method='curve_fit'):
    """
    Fit data to truncated power law in log-log space
    """
    # Remove zero or negative values
    mask = (hist > 0) & (bin_centers > 0) & ~np.isinf(hist) & ~np.isnan(hist)
    x_data = bin_centers[mask]
    y_data = hist[mask]
    
    if len(x_data) < 3:
        print("  Not enough data points for fitting")
        return None, None
    
    # Convert to log space
    log_x_data = np.log10(x_data)
    log_y_data = np.log10(y_data)
    
    # Estimate initial parameters from the data
    n_fit = max(3, len(log_x_data) // 2)
    coeffs = np.polyfit(log_x_data[:n_fit], log_y_data[:n_fit], 1)
    epsilon_init = -coeffs[0]
    log_A_init = coeffs[1]
    
    if len(x_data) > 3:
        characteristic_scale = np.median(x_data)
        lambda_init = 1.0 / characteristic_scale
    else:
        lambda_init = 1e-6
    
    print(f"  Initial guess: log_A={log_A_init:.3f}, epsilon={epsilon_init:.3f}, lambda={lambda_init:.3e}")
    
    if method == 'curve_fit':
        try:
            p0 = [log_A_init, epsilon_init, lambda_init]
            bounds = ([-np.inf, 0.1, 0], [np.inf, 10, np.inf])
            
            popt_log, pcov = curve_fit(log_truncated_powerlaw, log_x_data, log_y_data, 
                                       p0=p0, bounds=bounds, maxfev=50000)
            
            popt = np.array([10**popt_log[0], popt_log[1], popt_log[2]])
            
            perr_log = np.sqrt(np.diag(pcov))
            perr = np.array([
                popt[0] * np.log(10) * perr_log[0],
                perr_log[1],
                perr_log[2]
            ])
            
            return popt, perr
            
        except Exception as e:
            print(f"  curve_fit failed: {e}")
            print("  Trying minimize method...")
            method = 'minimize'
    
    if method == 'minimize':
        try:
            def cost_function(params):
                log_A, epsilon, lambda_ = params
                if epsilon <= 0 or lambda_ < 0:
                    return 1e10
                y_pred = log_truncated_powerlaw(log_x_data, log_A, epsilon, lambda_)
                residuals = log_y_data - y_pred
                return np.sum(residuals**2)
            
            x0 = [log_A_init, max(0.5, epsilon_init), max(1e-8, lambda_init)]
            bounds = [(-np.inf, np.inf), (0.1, 10), (0, np.inf)]
            
            result = minimize(cost_function, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 10000, 'ftol': 1e-9})
            
            if result.success:
                popt_log = result.x
                popt = np.array([10**popt_log[0], popt_log[1], popt_log[2]])
                return popt, None
            else:
                print(f"  Optimization failed: {result.message}")
                return None, None
                
        except Exception as e:
            print(f"  minimize failed: {e}")
            return None, None


def fit_truncated_powerlaw_with_weights(bin_centers, hist):
    """
    Fit with weights to emphasize different parts of the distribution
    """
    mask = (hist > 0) & (bin_centers > 0) & ~np.isinf(hist) & ~np.isnan(hist)
    x_data = bin_centers[mask]
    y_data = hist[mask]
    
    if len(x_data) < 3:
        print("  Not enough data points for fitting")
        return None, None
    
    log_x_data = np.log10(x_data)
    log_y_data = np.log10(y_data)
    
    weights = np.ones_like(log_y_data)
    
    n_fit = max(3, len(log_x_data) // 2)
    coeffs = np.polyfit(log_x_data[:n_fit], log_y_data[:n_fit], 1)
    epsilon_init = -coeffs[0]
    log_A_init = coeffs[1]
    
    if len(x_data) > 3:
        characteristic_scale = np.median(x_data)
        lambda_init = 1.0 / characteristic_scale
    else:
        lambda_init = 1e-6
    
    print(f"  Weighted fit initial guess: log_A={log_A_init:.3f}, epsilon={epsilon_init:.3f}, lambda={lambda_init:.3e}")
    
    try:
        p0 = [log_A_init, epsilon_init, lambda_init]
        bounds = ([-np.inf, 0.1, 0], [np.inf, 10, np.inf])
        
        popt_log, pcov = curve_fit(log_truncated_powerlaw, log_x_data, log_y_data, 
                                   p0=p0, sigma=1/weights, bounds=bounds, 
                                   maxfev=50000, absolute_sigma=False)
        
        popt = np.array([10**popt_log[0], popt_log[1], popt_log[2]])
        perr_log = np.sqrt(np.diag(pcov))
        perr = np.array([
            popt[0] * np.log(10) * perr_log[0],
            perr_log[1],
            perr_log[2]
        ])
        
        return popt, perr
        
    except Exception as e:
        print(f"  Weighted fit failed: {e}")
        return None, None


def analyze_data_subset(data_subset, config, subset_name=""):
    """
    Run analysis on a subset of data (e.g., before or after critical alpha).
    Returns analysis results dictionary.
    """
    results = {}
    cfg_analysis = config['ANALYSIS']
    prefix = f"  [{subset_name}] " if subset_name else "  "
    
    # --- Energy Analysis ---
    print(f"{prefix}Analyzing Energy...")
    results['energy'] = {}
    if len(data_subset['energy']) == 0:
        print(f"{prefix}No energy data.")
        results['energy']['bins'] = (None, None, None, None, None)
    else:
        print(f"{prefix}Energy range: [{np.min(data_subset['energy']):.6e}, {np.max(data_subset['energy']):.6e}]")
        bin_results = logarithmic_binning(data_subset['energy'], nbin=cfg_analysis['NBIN'])
        results['energy']['bins'] = bin_results
        
        if bin_results[0] is not None and cfg_analysis['FIT_DATA']:
            fit_func = (fit_truncated_powerlaw_with_weights if cfg_analysis['FIT_METHOD'] == 'weighted' 
                        else fit_truncated_powerlaw_logspace)
            popt, perr = fit_func(bin_results[0], bin_results[1])
            results['energy']['fit'] = (popt, perr)
            if popt is not None:
                print(f"{prefix}Energy fit: ε={popt[1]:.4f}, λ={popt[2]:.4e}")
    
    # --- Stress Analysis ---
    print(f"{prefix}Analyzing Stress...")
    results['stress'] = {}
    if len(data_subset['stress']) == 0:
        print(f"{prefix}No stress data.")
        results['stress']['bins'] = (None, None, None, None, None)
    else:
        print(f"{prefix}Stress range: [{np.min(data_subset['stress']):.6e}, {np.max(data_subset['stress']):.6e}]")
        bin_results = logarithmic_binning(data_subset['stress'], nbin=cfg_analysis['NBIN'])
        results['stress']['bins'] = bin_results
        
        if bin_results[0] is not None and cfg_analysis['FIT_DATA']:
            fit_func = (fit_truncated_powerlaw_with_weights if cfg_analysis['FIT_METHOD'] == 'weighted' 
                        else fit_truncated_powerlaw_logspace)
            popt, perr = fit_func(bin_results[0], bin_results[1])
            results['stress']['fit'] = (popt, perr)
            if popt is not None:
                print(f"{prefix}Stress fit: ε={popt[1]:.4f}, λ={popt[2]:.4e}")
    
    # --- Delta Alpha Analysis ---
    results['delta_alpha'] = {}
    if cfg_analysis['ANALYZE_ALPHA_DIFF'] and len(data_subset['delta_alpha']) > 0:
        print(f"{prefix}Analyzing Delta Alpha...")
        delta_alpha = data_subset['delta_alpha']
        positive_delta_alpha = delta_alpha[delta_alpha > 0]
        
        if len(positive_delta_alpha) > 0:
            filtered_delta_alpha = positive_delta_alpha
            log_bin_results = logarithmic_binning(filtered_delta_alpha, nbin=cfg_analysis['NBIN'])
            results['delta_alpha']['log_bins'] = log_bin_results
            
            lin_bin_results = linear_binning(positive_delta_alpha, nbin=cfg_analysis['NBIN_LINEAR'])
            results['delta_alpha']['lin_bins'] = lin_bin_results
            
            if cfg_analysis['FIT_ALPHA_DIFF'] and log_bin_results[0] is not None:
                popt, perr = fit_truncated_powerlaw_logspace(log_bin_results[0], log_bin_results[1])
                results['delta_alpha']['fit'] = (popt, perr)
    
    # --- E-S Scaling ---
    print(f"{prefix}Analyzing E-S Scaling...")
    results['scaling'] = {}
    if len(data_subset['paired_energy']) >= 2:
        energy_paired = data_subset['paired_energy']
        stress_paired = data_subset['paired_stress']
        
        log_stress = np.log10(stress_paired)
        log_energy = np.log10(energy_paired)
        
        stress_min, stress_max = np.min(log_stress), np.max(log_stress)
        bins_ES = np.linspace(stress_min, stress_max, cfg_analysis['NBIN_SCALING'] + 1)
        
        binned_stress, binned_energy_mean, binned_energy_std, binned_counts = [], [], [], []
        
        for i in range(len(bins_ES) - 1):
            mask = (log_stress >= bins_ES[i]) & (log_stress < bins_ES[i+1])
            if np.sum(mask) > 0:
                bin_center = (bins_ES[i] + bins_ES[i+1]) / 2
                binned_stress.append(bin_center)
                binned_energy_mean.append(np.mean(log_energy[mask]))
                binned_energy_std.append(np.std(log_energy[mask]))
                binned_counts.append(np.sum(mask))
        
        if len(binned_stress) >= 2:
            results['scaling']['binned_data'] = (
                np.array(binned_stress), 
                np.array(binned_energy_mean), 
                np.array(binned_energy_std), 
                np.array(binned_counts)
            )
            
            coeffs = np.polyfit(binned_stress, binned_energy_mean, 1)
            gamma, intercept = coeffs[0], coeffs[1]
            residuals = binned_energy_mean - (gamma * np.array(binned_stress) + intercept)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((binned_energy_mean - np.mean(binned_energy_mean))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            results['scaling']['fit'] = (gamma, intercept, r_squared)
            print(f"{prefix}E-S scaling: γ = {gamma:.4f}, R² = {r_squared:.4f}")
    
    return results


def run_analyses(raw_data, config):
    """
    Run all statistical analyses (binning, fitting) on the raw data.
    If SPLIT_BY_MAX_ENERGY is enabled, also analyzes before/after subsets.
    """
    print("\n" + "="*50)
    print("ANALYZING FULL DATASET")
    print("="*50)
    
    # Analyze full dataset
    results = analyze_data_subset(raw_data, config)
    
    # Analyze split datasets if enabled
    if config.get('SPLIT_BY_MAX_ENERGY', False):
        print("\n" + "="*50)
        print("ANALYZING BEFORE CRITICAL α")
        print("="*50)
        results['before'] = analyze_data_subset(raw_data['before'], config, "BEFORE")
        
        print("\n" + "="*50)
        print("ANALYZING AFTER CRITICAL α")
        print("="*50)
        results['after'] = analyze_data_subset(raw_data['after'], config, "AFTER")
    
    return results

# =============================================================================
# == PLOTTING & SAVING FUNCTIONS
# =============================================================================

def _get_filename_suffix(config):
    """Helper to generate a consistent filename suffix from config."""
    cfg_filters = config['FILTERS']
    filename_suffix = ""
    
    if config.get('SPLIT_BY_MAX_ENERGY', False):
        filename_suffix += "_split"
    
    if cfg_filters['BY_PLASTICITY']:
        filename_suffix += "_plasticity"
    if cfg_filters['BY_ALPHA']:
        filename_suffix += f"_alpha_{cfg_filters['ALPHA_MIN']}_{cfg_filters['ALPHA_MAX']}"
    
    xmin_str = f"{cfg_filters['ENERGY_XMIN']:.0e}_{cfg_filters['STRESS_XMIN']:.0e}"
    xmax_str = f"{cfg_filters['ENERGY_XMAX']:.0e}_{cfg_filters['STRESS_XMAX']:.0e}"
    
    if cfg_filters['BY_XMIN'] and cfg_filters['BY_XMAX']:
        filename_suffix += f"_xrange_{xmin_str}_{xmax_str}"
    elif cfg_filters['BY_XMIN']:
        filename_suffix += f"_xmin_{xmin_str}"
    elif cfg_filters['BY_XMAX']:
        filename_suffix += f"_xmax_{xmax_str}"
    
    return filename_suffix

def _get_title_suffix(config):
    """Helper to generate a consistent plot title suffix from config."""
    cfg_filters = config['FILTERS']
    title_suffix = ""
    
    if cfg_filters['BY_PLASTICITY']:
        title_suffix += " (Plasticity)"
    if cfg_filters['BY_ALPHA']:
        title_suffix += f" (α∈[{cfg_filters['ALPHA_MIN']},{cfg_filters['ALPHA_MAX']}])"
    if cfg_filters['BY_XMIN'] or cfg_filters['BY_XMAX']:
        title_suffix += " (x filtered)"
    
    return title_suffix

def _plot_distribution(ax, bin_results, fit_results, title, xlabel, color='b', label_prefix=''):
    """Helper to plot a single distribution (energy, stress, or d-alpha)."""
    
    if bin_results[0] is None:
        ax.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=13)
        return

    centers, hist, log_centers, log_hist, dx = bin_results
    
    mask = (hist > 0) & ~np.isinf(log_hist) & ~np.isnan(log_hist)
    data_label = f'{label_prefix}Data' if label_prefix else 'Data'
    ax.plot(log_centers[mask], log_hist[mask], 
            linestyle='None', marker='.', markersize=20, color=color, label=data_label)
    
    if fit_results is not None:
        popt, perr = fit_results
        if popt is not None:
            x_fit = np.logspace(np.log10(centers[mask].min()), np.log10(centers[mask].max()), 100)
            y_fit = truncated_powerlaw(x_fit, *popt)
            fit_label = f'{label_prefix}Fit: ε={popt[1]:.2f}, λ={popt[2]:.2e}' if label_prefix else f'Fit: ε={popt[1]:.2f}, λ={popt[2]:.2e}'
            ax.plot(np.log10(x_fit), np.log10(y_fit), '-', linewidth=2, 
                    label=fit_label, color=color, linestyle='--')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('log₁₀(Density)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_scaling_comparison(raw_data, analysis_results, config):
    """Create comparison plots for E-S scaling before/after critical alpha."""
    if not config.get('SPLIT_BY_MAX_ENERGY', False):
        return
    
    output_dir = config['OUTPUT_DIR']
    filename_suffix = _get_filename_suffix(config)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    colors = ['blue', 'green', 'red']
    titles = ['Full Dataset', 'Before α_c', 'After α_c']
    data_keys = [None, 'before', 'after']
    
    for idx, (ax, color, title, data_key) in enumerate(zip(axes, colors, titles, data_keys)):
        if data_key is None:
            results = analysis_results['scaling']
            paired_stress = raw_data['paired_stress']
            paired_energy = raw_data['paired_energy']
        else:
            results = analysis_results.get(data_key, {}).get('scaling', {})
            paired_stress = raw_data.get(data_key, {}).get('paired_stress', np.array([]))
            paired_energy = raw_data.get(data_key, {}).get('paired_energy', np.array([]))
        
        if 'binned_data' not in results or len(paired_stress) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue
        
        ax.plot(np.log10(paired_stress), np.log10(paired_energy), '.', markersize=8, 
               color=color, alpha=0.2, label='Data', zorder=1)
        
        binned_stress, binned_energy_mean, binned_energy_std, _ = results['binned_data']
        ax.errorbar(binned_stress, binned_energy_mean, yerr=binned_energy_std,
                   fmt='o', markersize=8, capsize=5, color=color, linewidth=2, 
                   alpha=0.8, label='Binned', zorder=3)
        
        if 'fit' in results:
            gamma, intercept, r_squared = results['fit']
            stress_fit = np.linspace(binned_stress.min(), binned_stress.max(), 100)
            energy_fit = gamma * stress_fit + intercept
            ax.plot(stress_fit, energy_fit, '--', linewidth=2.5, 
                   label=f'γ={gamma:.3f}, R²={r_squared:.3f}', color='black', zorder=2)
            title += f'\nγ = {gamma:.3f}'
        
        ax.set_xlabel('log₁₀(Stress)', fontsize=12, fontweight='bold')
        ax.set_ylabel('log₁₀(Energy)', fontsize=12, fontweight='bold')
        ax.set_title(f'E-S Scaling - {title}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'energy_vs_stress_scaling_comparison{filename_suffix}.png'), dpi=200)
    print(f"  Saved E-S scaling comparison")
    plt.close(fig)

def plot_time_series(raw_data, config):
    """
    Plot alpha vs energy and alpha vs stress time series with running averages.
    If SPLIT_BY_MAX_ENERGY is enabled, marks the critical alpha values.
    """
    print("\n" + "="*50)
    print("GENERATING TIME SERIES PLOTS")
    print("="*50)
    
    output_dir = config['OUTPUT_DIR']
    filename_suffix = _get_filename_suffix(config)
    
    time_series_data = raw_data['time_series']
    split_mode = config.get('SPLIT_BY_MAX_ENERGY', False)
    
    if len(time_series_data) == 0:
        print("  No time series data available.")
        return
    
    window_size = config['ANALYSIS'].get('RUNNING_AVG_WINDOW', 100)
    
    # Generate random colors for different files
    np.random.seed(12)
    colors = [np.random.rand(3,) for _ in range(len(time_series_data))]
    
    # --- Plot 1: Alpha vs Energy ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    all_alpha_energy = []
    all_energy_values = []
    
    for idx, data in enumerate(time_series_data):
        alpha = data['alpha']
        energy = data['energy']
        accepted_indices = data['accepted_energy_indices']
        
        ax1.plot(alpha, energy, 'o-', markersize=3, alpha=0.3, 
                color=colors[idx], zorder=1, rasterized=True)
        
        ax1.scatter(alpha[accepted_indices], energy[accepted_indices], 
                   s=50, alpha=0.2, color="black", 
                   label=f'#{idx+1} (accepted)', zorder=2, 
                   edgecolors=colors[idx], linewidths=.5)
        
        # Mark critical alpha if in split mode
        if split_mode and data['alpha_critical'] is not None:
            ax1.axvline(data['alpha_critical'], color=colors[idx], linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'#{idx+1} α_c={data["alpha_critical"]:.4f}')
            # Mark the maximum energy point
            ax1.scatter([data['alpha_critical']], [energy[data['max_idx']]], 
                       s=200, marker='*', color='red', edgecolors='black', 
                       linewidths=2, zorder=5)
        
        all_alpha_energy.extend(alpha)
        all_energy_values.extend(energy)
    
    all_alpha_energy = np.array(all_alpha_energy)
    all_energy_values = np.array(all_energy_values)
    
    if len(all_alpha_energy) >= window_size:
        alpha_avg_all, energy_avg_all = compute_running_average(all_alpha_energy, all_energy_values, window_size)
        ax1.plot(alpha_avg_all, energy_avg_all, '-', linewidth=2, color='black', 
                label='Overall average', zorder=10)

    ax1.set_xlabel('α (Loading Parameter)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Energy', fontsize=13, fontweight='bold')
    title = 'Energy vs Alpha - Accepted Avalanches'
    if split_mode:
        title += ' (Critical α Marked)'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'alpha_vs_energy_timeseries{filename_suffix}.png')
    plt.savefig(plot_path, dpi=200)
    print(f"  Saved: {plot_path}")
    plt.close(fig1)
    
    # --- Plot 2: Alpha vs Stress ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    all_alpha_stress = []
    all_stress_values = []
    
    for idx, data in enumerate(time_series_data):
        alpha = data['alpha']
        stress = data['stress']
        accepted_indices = data['accepted_stress_indices']
        
        ax2.plot(alpha, stress, 'o-', markersize=3, alpha=0.3, 
                color=colors[idx], zorder=1, rasterized=True)
        
        ax2.scatter(alpha[accepted_indices], stress[accepted_indices], 
                   s=50, alpha=0.3, color=colors[idx], 
                   label=f'#{idx+1} (accepted)', zorder=2,
                   edgecolors='black', linewidths=0.5)
        
        # Mark critical alpha if in split mode
        if split_mode and data['alpha_critical'] is not None:
            ax2.axvline(data['alpha_critical'], color=colors[idx], linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'#{idx+1} α_c={data["alpha_critical"]:.4f}')
        
        all_alpha_stress.extend(alpha)
        all_stress_values.extend(stress)
    
    all_alpha_stress = np.array(all_alpha_stress)
    all_stress_values = np.array(all_stress_values)
    
    if len(all_alpha_stress) >= window_size:
        alpha_avg_all, stress_avg_all = compute_running_average(all_alpha_stress, all_stress_values, window_size)
        ax2.plot(alpha_avg_all, stress_avg_all, '-', linewidth=4, color='black', 
                label='Overall average', zorder=10)
    
    ax2.set_xlabel('α (Loading Parameter)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Stress', fontsize=13, fontweight='bold')
    title = 'Stress vs Alpha - Accepted Avalanches'
    if split_mode:
        title += ' (Critical α Marked)'
    ax2.set_title(title, fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'alpha_vs_stress_timeseries{filename_suffix}.png')
    plt.savefig(plot_path, dpi=200)
    print(f"  Saved: {plot_path}")
    plt.close(fig2)


def plot_comparison(analysis_results, config, quantity='energy'):
    """
    Create comparison plots for before/after critical alpha.
    """
    if not config.get('SPLIT_BY_MAX_ENERGY', False):
        return
    
    output_dir = config['OUTPUT_DIR']
    filename_suffix = _get_filename_suffix(config)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Determine which key to use for bins (delta_alpha uses 'log_bins', others use 'bins')
    bins_key = 'log_bins' if quantity == 'delta_alpha' else 'bins'
    
    # Set appropriate labels
    if quantity == 'delta_alpha':
        quantity_label = 'Δα'
        xlabel = 'log₁₀(Δα)'
    else:
        quantity_label = quantity.capitalize()
        xlabel = f'log₁₀({quantity.capitalize()})'
    
    # Full dataset
    if bins_key in analysis_results.get(quantity, {}):
        _plot_distribution(axes[0], 
                          analysis_results[quantity][bins_key],
                          analysis_results[quantity].get('fit'),
                          f'{quantity_label} - Full Dataset',
                          xlabel,
                          color='blue', label_prefix='')
    
    # Before critical alpha
    if 'before' in analysis_results and bins_key in analysis_results['before'].get(quantity, {}):
        _plot_distribution(axes[1],
                          analysis_results['before'][quantity][bins_key],
                          analysis_results['before'][quantity].get('fit'),
                          f'{quantity_label} - Before α_c',
                          xlabel,
                          color='green', label_prefix='Before ')
    
    # After critical alpha
    if 'after' in analysis_results and bins_key in analysis_results['after'].get(quantity, {}):
        _plot_distribution(axes[2],
                          analysis_results['after'][quantity][bins_key],
                          analysis_results['after'][quantity].get('fit'),
                          f'{quantity_label} - After α_c',
                          xlabel,
                          color='red', label_prefix='After ')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{quantity}_comparison{filename_suffix}.png')
    plt.savefig(plot_path, dpi=200)
    print(f"  Saved: {plot_path}")
    plt.close(fig)


def generate_plots(raw_data, analysis_results, config):
    """
    Generate and save all plots based on the analysis results.
    """
    print("\n" + "="*50)
    print("GENERATING PLOTS")
    print("="*50)
    
    output_dir = config['OUTPUT_DIR']
    filename_suffix = _get_filename_suffix(config)
    title_suffix = _get_title_suffix(config)
    
    cfg_analysis = config['ANALYSIS']
    split_mode = config.get('SPLIT_BY_MAX_ENERGY', False)
    
    # Time Series Plots
    plot_time_series(raw_data, config)
    
    # If split mode, create comparison plots
    if split_mode:
        print("\nGenerating comparison plots (before/after α_c)...")
        plot_comparison(analysis_results, config, 'energy')
        plot_comparison(analysis_results, config, 'stress')
        if cfg_analysis['ANALYZE_ALPHA_DIFF']:
            plot_comparison(analysis_results, config, 'delta_alpha')
        plot_scaling_comparison(raw_data, analysis_results, config)  # <-- ADD THIS

    
    # Original plots (keeping your existing plotting code)
    # --- Plot 0: Energy vs Stress Scaling ---
    if 'binned_data' in analysis_results['scaling']:
        fig0, ax0 = plt.subplots(figsize=(8, 6))
        
        ax0.plot(np.log10(raw_data['paired_stress']), np.log10(raw_data['paired_energy']), 
                linestyle='None', marker='.', markersize=12, color='blue', alpha=0.3, 
                label='Data points', zorder=1)
        
        binned_stress, binned_energy_mean, binned_energy_std, _ = analysis_results['scaling']['binned_data']
        ax0.errorbar(binned_stress, binned_energy_mean, yerr=binned_energy_std,
                    fmt='o', markersize=8, capsize=5, capthick=2, 
                    color='red', ecolor='darkred', linewidth=2,
                    label='Binned mean ± std', zorder=3)
        
        if 'fit' in analysis_results['scaling']:
            gamma, intercept, r_squared = analysis_results['scaling']['fit']
            stress_fit = np.linspace(np.min(binned_stress), np.max(binned_stress), 100)
            energy_fit = gamma * stress_fit + intercept
            ax0.plot(stress_fit, energy_fit, 'b--', linewidth=2.5, 
                    label=f'Power law fit: γ = {gamma:.3f} (R²={r_squared:.3f})', zorder=2)
        
        ax0.set_xlabel('log₁₀(Stress Change)', fontsize=13, fontweight='bold')
        ax0.set_ylabel('log₁₀(Energy Change)', fontsize=13, fontweight='bold')
        ax0.set_title('Energy-Stress Scaling: E ~ S^γ' + title_suffix, fontsize=14, fontweight='bold')
        ax0.grid(True, alpha=0.3, linestyle='--')
        ax0.legend(fontsize=11, loc='best')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'energy_vs_stress_scaling{filename_suffix}.png')
        plt.savefig(plot_path, dpi=200)
        print(f"  Saved: {plot_path}")
        plt.close(fig0)
    
    # Individual distribution plots
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    _plot_distribution(ax1, analysis_results['energy']['bins'], 
                       analysis_results['energy'].get('fit'),
                       'Energy Change Distribution' + title_suffix,
                       'log₁₀(Energy Change)')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'energy_distribution{filename_suffix}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    _plot_distribution(ax2, analysis_results['stress']['bins'], 
                       analysis_results['stress'].get('fit'),
                       'Stress Change Distribution' + title_suffix,
                       'log₁₀(Stress Change)')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'stress_distribution{filename_suffix}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")
    plt.close(fig2)

    # Delta Alpha plots
    if cfg_analysis['ANALYZE_ALPHA_DIFF'] and 'log_bins' in analysis_results['delta_alpha']:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        _plot_distribution(ax3, analysis_results['delta_alpha']['log_bins'], 
                           analysis_results['delta_alpha'].get('fit'),
                           'Δα Distribution (Log-Log)' + title_suffix,
                           'log₁₀(Δα)')
        
        if 'lin_bins' in analysis_results['delta_alpha']:
            lin_bin_results = analysis_results['delta_alpha']['lin_bins']
            if lin_bin_results[0] is not None:
                centers, hist, dx = lin_bin_results
                mask_linear = (hist > 0)
                ax3.plot(np.log10(centers[mask_linear]), np.log10(hist[mask_linear]), 
                        linestyle='None', marker='x', markersize=12, color='orange', label='Linear binning')
        
        ax3.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'dalpha_distribution_log{filename_suffix}.png')
        plt.savefig(plot_path, dpi=150)
        print(f"  Saved: {plot_path}")
        plt.close(fig3)

        if 'lin_bins' in analysis_results['delta_alpha']:
            lin_bin_results = analysis_results['delta_alpha']['lin_bins']
            if lin_bin_results[0] is not None:
                fig4, ax4 = plt.subplots(figsize=(7, 5))
                centers, hist, dx = lin_bin_results
                ax4.plot(centers, hist, linestyle='None', marker='.', markersize=20, color='g', label='Data')
                ax4.set_xlabel('Δα', fontsize=12)
                ax4.set_ylabel('Density', fontsize=12)
                ax4.set_title('Δα Distribution (Linear)' + title_suffix, fontsize=13)
                ax4.grid(True, alpha=0.3, axis='y')
                ax4.legend()
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'dalpha_distribution_linear{filename_suffix}.png')
                plt.savefig(plot_path, dpi=150)
                print(f"  Saved: {plot_path}")
                plt.close(fig4)

    # Combined plot
    num_plots = 2
    if cfg_analysis['ANALYZE_ALPHA_DIFF'] and 'log_bins' in analysis_results['delta_alpha']:
        num_plots += 1
    if cfg_analysis['ANALYZE_ALPHA_DIFF'] and 'lin_bins' in analysis_results['delta_alpha']:
        num_plots += 1
        
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    
    _plot_distribution(axes[0], analysis_results['energy']['bins'], 
                       analysis_results['energy'].get('fit'),
                       'Energy Change Distribution' + title_suffix,
                       'log₁₀(Energy Change)')
    
    _plot_distribution(axes[1], analysis_results['stress']['bins'], 
                       analysis_results['stress'].get('fit'),
                       'Stress Change Distribution' + title_suffix,
                       'log₁₀(Stress Change)')

    if num_plots >= 3:
        _plot_distribution(axes[2], analysis_results['delta_alpha']['log_bins'], 
                           analysis_results['delta_alpha'].get('fit'),
                           'Δα Distribution (Log-Log)' + title_suffix,
                           'log₁₀(Δα)')
    
    if num_plots == 4:
        lin_bin_results = analysis_results['delta_alpha']['lin_bins']
        if lin_bin_results[0] is not None:
            centers, hist, dx = lin_bin_results
            axes[3].plot(centers, hist, linestyle='None', marker='.', markersize=20, color='g', label='Data')
            axes[3].set_xlabel('Δα', fontsize=12)
            axes[3].set_ylabel('Density', fontsize=12)
            axes[3].set_title('Δα Distribution (Linear)' + title_suffix, fontsize=13)
            axes[3].grid(True, alpha=0.3, axis='y')
            axes[3].legend()
    
    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, f'all_distributions{filename_suffix}.png')
    plt.savefig(combined_plot_path, dpi=150)
    print(f"  Saved combined plot: {combined_plot_path}")
    
    if config['SHOW_PLOTS']:
        plt.show()
    
    plt.close('all')


def _save_histogram_data(filepath, log_centers, log_hist):
    """Helper to save log-binned histogram data."""
    if log_centers is None:
        return
    with open(filepath, 'w') as f:
        f.write("# log10(bin_center), log10(density)\n")
        for x, y in zip(log_centers, log_hist):
            if not np.isinf(y) and not np.isnan(y):
                f.write(f"{x:.8e}, {y:.8e}\n")
    print(f"  Saved: {filepath}")

def _write_fit_params(f, name, fit_results):
    """Helper to write fit parameters to a file."""
    if fit_results is None:
        f.write(f"{name}:\n  Fitting failed or was not performed.\n\n")
        return
        
    popt, perr = fit_results
    if popt is None:
        f.write(f"{name}:\n  Fitting failed.\n\n")
        return

    f.write(f"{name}:\n")
    f.write(f"  A       = {popt[0]:.6e}")
    if perr is not None: f.write(f" ± {perr[0]:.6e}\n")
    else: f.write("\n")
    
    f.write(f"  epsilon = {popt[1]:.6f}")
    if perr is not None: f.write(f" ± {perr[1]:.6f}\n")
    else: f.write("\n")
    
    f.write(f"  lambda  = {popt[2]:.6e}")
    if perr is not None: f.write(f" ± {perr[2]:.6e}\n\n")
    else: f.write("\n\n")


def _write_scaling_params(f, name, fit_results):
    """Helper to write E-S scaling parameters to a file."""
    if fit_results is None or len(fit_results) != 3:
        f.write(f"{name}:\n  Fitting failed or was not performed.\n\n")
        return
    
    gamma, intercept, r_squared = fit_results
    f.write(f"{name}:\n")
    f.write(f"  gamma (γ)  = {gamma:.6f}\n")
    f.write(f"  intercept  = {intercept:.6f}\n")
    f.write(f"  R²         = {r_squared:.6f}\n\n")


def save_results(raw_data, analysis_results, config):
    """
    Save all binned data and fit parameters to text files.
    """
    print("\n" + "="*50)
    print("SAVING DATA FILES")
    print("="*50)

    output_dir = config['OUTPUT_DIR']
    filename_suffix = _get_filename_suffix(config)
    cfg_analysis = config['ANALYSIS']
    split_mode = config.get('SPLIT_BY_MAX_ENERGY', False)

    # Save critical alpha values if in split mode
    if split_mode and 'critical_alphas' in raw_data:
        filepath = os.path.join(output_dir, f'critical_alphas{filename_suffix}.txt')
        with open(filepath, 'w') as f:
            f.write("# Critical alpha values (maximum energy) for each dataset\n")
            f.write("# Filename, Alpha_critical, Max_Energy, Index\n")
            for info in raw_data['critical_alphas']:
                f.write(f"{info['filename']}, {info['alpha_critical']:.8f}, {info['max_energy']:.8e}, {info['max_idx']}\n")
        print(f"  Saved: {filepath}")

    # Save binned E-S Scaling Data (Full dataset)
    if 'binned_data' in analysis_results['scaling']:
        binned_stress, binned_energy_mean, binned_energy_std, binned_counts = analysis_results['scaling']['binned_data']
        
        filepath = os.path.join(output_dir, f'data_energy_vs_stress_binned{filename_suffix}.dat')
        with open(filepath, 'w') as f:
            f.write("# Binned Energy-Stress relationship\n")
            f.write("# log10(Stress_center), log10(Energy_mean), log10(Energy_std), count\n")
            for s, e, std, cnt in zip(binned_stress, binned_energy_mean, binned_energy_std, binned_counts):
                f.write(f"{s:.8e}, {e:.8e}, {std:.8e}, {int(cnt)}\n")
        print(f"  Saved: {filepath}")
    
    # NEW: Save binned E-S Scaling Data (Before/After) if in split mode
    if split_mode:
        # Before critical alpha
        if 'before' in analysis_results and 'binned_data' in analysis_results['before'].get('scaling', {}):
            binned_stress, binned_energy_mean, binned_energy_std, binned_counts = analysis_results['before']['scaling']['binned_data']
            filepath = os.path.join(output_dir, f'data_energy_vs_stress_binned_before{filename_suffix}.dat')
            with open(filepath, 'w') as f:
                f.write("# Binned Energy-Stress relationship (BEFORE critical α)\n")
                f.write("# log10(Stress_center), log10(Energy_mean), log10(Energy_std), count\n")
                for s, e, std, cnt in zip(binned_stress, binned_energy_mean, binned_energy_std, binned_counts):
                    f.write(f"{s:.8e}, {e:.8e}, {std:.8e}, {int(cnt)}\n")
            print(f"  Saved: {filepath}")
        
        # After critical alpha
        if 'after' in analysis_results and 'binned_data' in analysis_results['after'].get('scaling', {}):
            binned_stress, binned_energy_mean, binned_energy_std, binned_counts = analysis_results['after']['scaling']['binned_data']
            filepath = os.path.join(output_dir, f'data_energy_vs_stress_binned_after{filename_suffix}.dat')
            with open(filepath, 'w') as f:
                f.write("# Binned Energy-Stress relationship (AFTER critical α)\n")
                f.write("# log10(Stress_center), log10(Energy_mean), log10(Energy_std), count\n")
                for s, e, std, cnt in zip(binned_stress, binned_energy_mean, binned_energy_std, binned_counts):
                    f.write(f"{s:.8e}, {e:.8e}, {std:.8e}, {int(cnt)}\n")
            print(f"  Saved: {filepath}")
    
    # Save Raw E-S Paired Data
    if len(raw_data['paired_energy']) > 0:
        filepath = os.path.join(output_dir, f'data_energy_vs_stress_raw{filename_suffix}.dat')
        with open(filepath, 'w') as f:
            f.write("# Raw paired data\n")
            f.write("# log10(Stress), log10(Energy)\n")
            for s, e in zip(raw_data['paired_stress'], raw_data['paired_energy']):
                f.write(f"{np.log10(s):.8e}, {np.log10(e):.8e}\n")
        print(f"  Saved: {filepath}")

    # Save Histograms
    _, _, log_centers, log_hist, _ = analysis_results['energy']['bins']
    filepath = os.path.join(output_dir, f'data_histogram_energy{filename_suffix}.dat')
    _save_histogram_data(filepath, log_centers, log_hist)

    _, _, log_centers, log_hist, _ = analysis_results['stress']['bins']
    filepath = os.path.join(output_dir, f'data_histogram_stress{filename_suffix}.dat')
    _save_histogram_data(filepath, log_centers, log_hist)

    if cfg_analysis['ANALYZE_ALPHA_DIFF'] and 'log_bins' in analysis_results['delta_alpha']:
        _, _, log_centers, log_hist, _ = analysis_results['delta_alpha']['log_bins']
        filepath = os.path.join(output_dir, f'data_histogram_dalpha_log{filename_suffix}.dat')
        _save_histogram_data(filepath, log_centers, log_hist)
        
        if 'lin_bins' in analysis_results['delta_alpha']:
            centers, hist, _ = analysis_results['delta_alpha']['lin_bins']
            if centers is not None:
                filepath = os.path.join(output_dir, f'data_histogram_dalpha_linear{filename_suffix}.dat')
                with open(filepath, 'w') as f:
                    f.write("# bin_center, density\n")
                    for x, y in zip(centers, hist):
                        f.write(f"{x:.8e}, {y:.8e}\n")
                print(f"  Saved: {filepath}")
    
    # Save before/after histograms if in split mode
    if split_mode:
        print("\n  Saving before/after histogram data...")
        
        # Energy - before
        if 'before' in analysis_results:
            _, _, log_centers, log_hist, _ = analysis_results['before']['energy']['bins']
            filepath = os.path.join(output_dir, f'data_histogram_energy_before{filename_suffix}.dat')
            _save_histogram_data(filepath, log_centers, log_hist)
            
            # Stress - before
            _, _, log_centers, log_hist, _ = analysis_results['before']['stress']['bins']
            filepath = os.path.join(output_dir, f'data_histogram_stress_before{filename_suffix}.dat')
            _save_histogram_data(filepath, log_centers, log_hist)
            
            # Delta alpha - before
            if cfg_analysis['ANALYZE_ALPHA_DIFF'] and 'log_bins' in analysis_results['before']['delta_alpha']:
                _, _, log_centers, log_hist, _ = analysis_results['before']['delta_alpha']['log_bins']
                filepath = os.path.join(output_dir, f'data_histogram_dalpha_log_before{filename_suffix}.dat')
                _save_histogram_data(filepath, log_centers, log_hist)
        
        # Energy - after
        if 'after' in analysis_results:
            _, _, log_centers, log_hist, _ = analysis_results['after']['energy']['bins']
            filepath = os.path.join(output_dir, f'data_histogram_energy_after{filename_suffix}.dat')
            _save_histogram_data(filepath, log_centers, log_hist)
            
            # Stress - after
            _, _, log_centers, log_hist, _ = analysis_results['after']['stress']['bins']
            filepath = os.path.join(output_dir, f'data_histogram_stress_after{filename_suffix}.dat')
            _save_histogram_data(filepath, log_centers, log_hist)
            
            # Delta alpha - after
            if cfg_analysis['ANALYZE_ALPHA_DIFF'] and 'log_bins' in analysis_results['after']['delta_alpha']:
                _, _, log_centers, log_hist, _ = analysis_results['after']['delta_alpha']['log_bins']
                filepath = os.path.join(output_dir, f'data_histogram_dalpha_log_after{filename_suffix}.dat')
                _save_histogram_data(filepath, log_centers, log_hist)

    # Save Fit Parameters
    if cfg_analysis['FIT_DATA'] or (cfg_analysis['ANALYZE_ALPHA_DIFF'] and cfg_analysis['FIT_ALPHA_DIFF']):
        filepath = os.path.join(output_dir, f'fit_parameters{filename_suffix}.txt')
        with open(filepath, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TRUNCATED POWER LAW FIT PARAMETERS\n")
            f.write("="*60 + "\n")
            f.write("Truncated Power Law Fit: P(x) = A * x^(-epsilon) * exp(-lambda * x)\n")
            f.write(f"Fitting method: {cfg_analysis['FIT_METHOD']}\n\n")
            
            f.write("="*60 + "\n")
            f.write("FULL DATASET\n")
            f.write("="*60 + "\n")
            _write_fit_params(f, "ENERGY", analysis_results['energy'].get('fit'))
            _write_fit_params(f, "STRESS", analysis_results['stress'].get('fit'))
            if cfg_analysis['ANALYZE_ALPHA_DIFF']:
                _write_fit_params(f, "ALPHA DIFFERENCES (Δα)", analysis_results['delta_alpha'].get('fit'))
            
            # NEW: Add E-S scaling parameters
            f.write("\n" + "="*60 + "\n")
            f.write("ENERGY-STRESS SCALING PARAMETERS\n")
            f.write("="*60 + "\n")
            f.write("Power Law Scaling: E ~ S^γ (log10(E) = γ * log10(S) + intercept)\n\n")
            _write_scaling_params(f, "E-S SCALING (FULL)", analysis_results['scaling'].get('fit'))
            
            if split_mode and 'before' in analysis_results:
                f.write("\n" + "="*60 + "\n")
                f.write("BEFORE CRITICAL α\n")
                f.write("="*60 + "\n")
                _write_fit_params(f, "ENERGY (BEFORE)", analysis_results['before']['energy'].get('fit'))
                _write_fit_params(f, "STRESS (BEFORE)", analysis_results['before']['stress'].get('fit'))
                if cfg_analysis['ANALYZE_ALPHA_DIFF']:
                    _write_fit_params(f, "ALPHA DIFFERENCES (Δα, BEFORE)", 
                                    analysis_results['before']['delta_alpha'].get('fit'))
                # NEW: Add E-S scaling parameters for before
                _write_scaling_params(f, "E-S SCALING (BEFORE)", analysis_results['before']['scaling'].get('fit'))
            
            if split_mode and 'after' in analysis_results:
                f.write("\n" + "="*60 + "\n")
                f.write("AFTER CRITICAL α\n")
                f.write("="*60 + "\n")
                _write_fit_params(f, "ENERGY (AFTER)", analysis_results['after']['energy'].get('fit'))
                _write_fit_params(f, "STRESS (AFTER)", analysis_results['after']['stress'].get('fit'))
                if cfg_analysis['ANALYZE_ALPHA_DIFF']:
                    _write_fit_params(f, "ALPHA DIFFERENCES (Δα, AFTER)", 
                                    analysis_results['after']['delta_alpha'].get('fit'))
                # NEW: Add E-S scaling parameters for after
                _write_scaling_params(f, "E-S SCALING (AFTER)", analysis_results['after']['scaling'].get('fit'))
        
        print(f"  Saved: {filepath}")

# =============================================================================
# == MAIN SCRIPT
# =============================================================================

def main():
    """
    Main execution workflow for the avalanche analysis.
    """
    
    # ===== CONFIGURATION =====
    CONFIG = {
        'USE_MULTIPLE_FILES': True,
        'FILE_PATTERN': "./build*/energy_stress_log.csv",
        'SINGLE_FILE': "energy_stress_log.csv",
        'OUTPUT_DIR': './statistics',
        'SHOW_PLOTS': False,
        
        # NEW: Enable split analysis at maximum energy alpha
        'SPLIT_BY_MAX_ENERGY': True,  # Set to False to use original behavior
        
        'FILTERS': {
            'BY_PLASTICITY': False,
            'BY_ALPHA': True,
            'ALPHA_MIN': 0.140001,
            'ALPHA_MAX': .7,
            
            'BY_XMIN': True,
            'ENERGY_XMIN': 1e-5,
            'STRESS_XMIN': 2,
            
            'BY_XMAX': False,
            'ENERGY_XMAX': 1e-2,
            'STRESS_XMAX': 1e-2,
        },
        
        'ANALYSIS': {
            'FIT_DATA': True,
            'FIT_METHOD': 'logspace',
            
            'ANALYZE_ALPHA_DIFF': True,
            'FIT_ALPHA_DIFF': False,
            
            'NBIN': 12,
            'NBIN_LINEAR': 120,
            'NBIN_SCALING': 20,
            
            'RUNNING_AVG_WINDOW': 250,
        }
    }
    # =========================
    
    # Setup Environment
    if os.path.exists(CONFIG['OUTPUT_DIR']):
        shutil.rmtree(CONFIG['OUTPUT_DIR'])
        print(f"Cleaned existing output directory: {CONFIG['OUTPUT_DIR']}")

    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    print(f"Output directory: {CONFIG['OUTPUT_DIR']}")
    
    # Find Files
    if CONFIG['USE_MULTIPLE_FILES']:
        print(f"\nSearching for files matching: {CONFIG['FILE_PATTERN']}")
        filenames = sorted(glob.glob(CONFIG['FILE_PATTERN']))
    else:
        print(f"\nUsing single file: {CONFIG['SINGLE_FILE']}")
        filenames = [CONFIG['SINGLE_FILE']]
    
    if len(filenames) == 0:
        print(f"ERROR: No files found matching pattern '{CONFIG['FILE_PATTERN']}'")
        exit(1)
        
    print(f"\nFound {len(filenames)} files to process:")
    for f in filenames:
        print(f"  - {f}")
    
    if CONFIG['SPLIT_BY_MAX_ENERGY']:
        print("\n*** SPLIT MODE ENABLED: Will analyze before/after maximum energy α ***\n")

    # Load and Process Data
    print("\n" + "="*60)
    print("STEP 1: LOADING AND PROCESSING DATA")
    print("="*60)
    raw_data = load_and_process_data(filenames, CONFIG)
    
    if all(len(v) == 0 for k, v in raw_data.items() if k not in ['time_series', 'critical_alphas', 'before', 'after']):
        print("ERROR: No data was loaded after processing. Check files and filters.")
        exit(1)

    # Run Analyses
    print("\n" + "="*60)
    print("STEP 2: RUNNING ANALYSES")
    print("="*60)
    analysis_results = run_analyses(raw_data, CONFIG)

    # Generate Plots
    print("\n" + "="*60)
    print("STEP 3: GENERATING PLOTS")
    print("="*60)
    generate_plots(raw_data, analysis_results, CONFIG)

    # Save Data Files
    print("\n" + "="*60)
    print("STEP 4: SAVING DATA FILES")
    print("="*60)
    save_results(raw_data, analysis_results, CONFIG)
    
    print(f"\nDone! All output files saved in {CONFIG['OUTPUT_DIR']}/")


if __name__ == "__main__":
    main()