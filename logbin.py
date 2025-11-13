#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import os
import glob

# =============================================================================
# == DATA IO FUNCTIONS
# =============================================================================

def read_energy_stress_log(filename="energy_stress_log.csv"):
    """
    Read the CSV file created by ConfigurationSaver::logEnergyAndStress
    
    Returns:
        alpha, energy_change, stress_change, plasticity_flag as numpy arrays
    """
    try:
        # Skip header row and read all data
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        
        # Handle empty or single-line files
        if data.ndim == 0 or data.size == 0:
            print(f"  WARNING: File '{filename}' is empty or has no data. Skipping.")
            return (np.array([]), np.array([]), np.array([]), np.array([]))
        
        # Handle files with only one data row
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Extract columns
        alpha = data[:, 1]
        energy_change = data[:, 6]
        stress_change = data[:, 7]
        plasticity_flag = data[:, 8].astype(int)
        
        return alpha, energy_change, stress_change, plasticity_flag
        
    except Exception as e:
        print(f"  ERROR: Could not read file '{filename}'. Reason: {e}. Skipping.")
        return (np.array([]), np.array([]), np.array([]), np.array([]))


def load_and_process_data(filenames, config):
    """
    Process one or more simulation files, apply filters, and
    return all necessary combined raw data streams for analysis.
    """
    all_delta_alpha = []
    all_energy_data = []
    all_stress_data = []
    all_paired_energy = []
    all_paired_stress = []
    
    cfg_filters = config['FILTERS']
    
    for filename in filenames:
        print(f"  Processing: {filename}")
        
        # Read data
        alpha, energy_change, stress_change, plasticity_flag = read_energy_stress_log(filename)
        
        if len(alpha) == 0:
            continue
            
        # Apply common filters
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
        
        # --- Stream 1 & 2: Energy Distribution & Delta Alpha ---
        # These are linked, as delta_alpha is based on accepted energy avalanches
        
        positive_energy_mask = energy_change_filtered > 0
        alpha_for_energy = alpha_filtered[positive_energy_mask]
        energy_data = energy_change_filtered[positive_energy_mask]
        
        # Apply xmin/xmax filters
        if cfg_filters['BY_XMIN']:
            energy_xmin_mask = energy_data >= cfg_filters['ENERGY_XMIN']
            alpha_for_energy = alpha_for_energy[energy_xmin_mask]
            energy_data = energy_data[energy_xmin_mask]
        
        if cfg_filters['BY_XMAX']:
            energy_xmax_mask = energy_data <= cfg_filters['ENERGY_XMAX']
            alpha_for_energy = alpha_for_energy[energy_xmax_mask]
            energy_data = energy_data[energy_xmax_mask]
        
        # Compute delta_alpha for THIS simulation only
        if len(alpha_for_energy) > 1:
            all_delta_alpha.append(np.diff(alpha_for_energy))
        
        all_energy_data.append(energy_data)

        # --- Stream 3: Stress Distribution ---
        # Processed independently for its own distribution
        
        positive_stress_mask = stress_change_filtered > 0
        stress_data = stress_change_filtered[positive_stress_mask]
        
        if cfg_filters['BY_XMIN']:
            stress_data = stress_data[stress_data >= cfg_filters['STRESS_XMIN']]
        
        if cfg_filters['BY_XMAX']:
            stress_data = stress_data[stress_data <= cfg_filters['STRESS_XMAX']]
            
        all_stress_data.append(stress_data)
        
        # --- Stream 4: Paired E-S Data for Scaling ---
        # We need events where *both* are positive and pass filters
        
        both_positive_mask = (energy_change_filtered > 0) & (stress_change_filtered > 0)
        energy_paired = energy_change_filtered[both_positive_mask]
        stress_paired = stress_change_filtered[both_positive_mask]
        
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
    
    # Combine all arrays from all files
    raw_data = {
        'delta_alpha': np.concatenate(all_delta_alpha) if all_delta_alpha else np.array([]),
        'energy': np.concatenate(all_energy_data) if all_energy_data else np.array([]),
        'stress': np.concatenate(all_stress_data) if all_stress_data else np.array([]),
        'paired_energy': np.concatenate(all_paired_energy) if all_paired_energy else np.array([]),
        'paired_stress': np.concatenate(all_paired_stress) if all_paired_stress else np.array([]),
    }

    print(f"\n{'='*60}")
    print(f"COMBINED RESULTS:")
    print(f"  Total delta_alpha: {len(raw_data['delta_alpha'])}")
    print(f"  Total energy avalanches: {len(raw_data['energy'])}")
    print(f"  Total stress avalanches: {len(raw_data['stress'])}")
    print(f"  Total paired E-S events: {len(raw_data['paired_energy'])}")
    print(f"{'='*60}\n")
    
    return raw_data


# =============================================================================
# == ANALYSIS FUNCTIONS (BINNING, FITTING)
# =============================================================================

def logarithmic_binning(data, nbin=14, xmin=None, xmax=None):
    """
    Perform logarithmic binning exactly as shown by user
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
    
    # Create logarithmic bins - exactly as user showed
    bins = np.logspace(np.log10(xmin), np.log10(xmax), nbin+1)
    
    # Compute histogram
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    
    # Compute bin centers (arithmetic mean)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute bin widths
    dx = bin_edges[1:] - bin_edges[:-1]
    
    # Compute log10 values
    log_bin_centers = np.log10(bin_centers)
    log_hist = np.log10(hist)
    
    return bin_centers, hist, log_bin_centers, log_hist, dx


def linear_binning(data, nbin=20, xmin=None, xmax=None):
    """
    Perform linear binning for data
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
    
    # Create linear bins
    bins = np.linspace(xmin, xmax, nbin+1)
    
    # Compute histogram
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    
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


def run_analyses(raw_data, config):
    """
    Run all statistical analyses (binning, fitting) on the raw data.
    Returns a dictionary with all analysis results.
    """
    results = {}
    cfg_analysis = config['ANALYSIS']
    
    # --- Energy Analysis ---
    print("\n" + "="*50)
    print("ANALYZING ENERGY")
    print("="*50)
    results['energy'] = {}
    if len(raw_data['energy']) == 0:
        print("  No energy data to analyze.")
        results['energy']['bins'] = (None, None, None, None, None)
    else:
        print(f"  Final energy range: [{np.min(raw_data['energy']):.6e}, {np.max(raw_data['energy']):.6e}]")
        bin_results = logarithmic_binning(raw_data['energy'], nbin=cfg_analysis['NBIN'])
        results['energy']['bins'] = bin_results
        
        if bin_results[0] is None:
            print("  ERROR: Energy binning failed!")
        else:
            print(f"  Binned energy data into {len(bin_results[0])} bins.")
            if cfg_analysis['FIT_DATA']:
                print("\n  Fitting Energy Data...")
                fit_func = (fit_truncated_powerlaw_with_weights if cfg_analysis['FIT_METHOD'] == 'weighted' 
                            else fit_truncated_powerlaw_logspace)
                popt, perr = fit_func(bin_results[0], bin_results[1])
                results['energy']['fit'] = (popt, perr)
                if popt is not None:
                    print(f"  Fit successful: ε={popt[1]:.4f}, λ={popt[2]:.4e}")
                else:
                    print("  Energy fitting failed!")
    
    # --- Stress Analysis ---
    print("\n" + "="*50)
    print("ANALYZING STRESS")
    print("="*50)
    results['stress'] = {}
    if len(raw_data['stress']) == 0:
        print("  No stress data to analyze.")
        results['stress']['bins'] = (None, None, None, None, None)
    else:
        print(f"  Final stress range: [{np.min(raw_data['stress']):.6e}, {np.max(raw_data['stress']):.6e}]")
        bin_results = logarithmic_binning(raw_data['stress'], nbin=cfg_analysis['NBIN'])
        results['stress']['bins'] = bin_results
        
        if bin_results[0] is None:
            print("  ERROR: Stress binning failed!")
        else:
            print(f"  Binned stress data into {len(bin_results[0])} bins.")
            if cfg_analysis['FIT_DATA']:
                print("\n  Fitting Stress Data...")
                fit_func = (fit_truncated_powerlaw_with_weights if cfg_analysis['FIT_METHOD'] == 'weighted' 
                            else fit_truncated_powerlaw_logspace)
                popt, perr = fit_func(bin_results[0], bin_results[1])
                results['stress']['fit'] = (popt, perr)
                if popt is not None:
                    print(f"  Fit successful: ε={popt[1]:.4f}, λ={popt[2]:.4e}")
                else:
                    print("  Stress fitting failed!")

    # --- Delta Alpha Analysis ---
    results['delta_alpha'] = {}
    if cfg_analysis['ANALYZE_ALPHA_DIFF']:
        print("\n" + "="*50)
        print("ANALYZING ALPHA DIFFERENCES")
        print("="*50)
        delta_alpha = raw_data['delta_alpha']
        
        if len(delta_alpha) == 0:
            print("  No delta_alpha data to analyze.")
        else:
            print(f"  Number of alpha differences: {len(delta_alpha)}")
            print(f"  Alpha difference range: [{np.min(delta_alpha):.6e}, {np.max(delta_alpha):.6e}]")
            positive_delta_alpha = delta_alpha[delta_alpha > 0]
            print(f"  Positive alpha differences: {len(positive_delta_alpha)} / {len(delta_alpha)}")
            
            if len(positive_delta_alpha) > 0:
                # Logarithmic binning
                print("\n  Logarithmic Binning (Δα)...")
                log_bin_results = logarithmic_binning(positive_delta_alpha, nbin=cfg_analysis['NBIN'])
                results['delta_alpha']['log_bins'] = log_bin_results
                if log_bin_results[0] is None:
                    print("  ERROR: Delta alpha log-binning failed!")
                
                # Linear binning
                print("  Linear Binning (Δα)...")
                lin_bin_results = linear_binning(positive_delta_alpha, nbin=cfg_analysis['NBIN_LINEAR'])
                results['delta_alpha']['lin_bins'] = lin_bin_results
                if lin_bin_results[0] is None:
                    print("  ERROR: Delta alpha linear-binning failed!")
                
                # Fitting (based on log-binned data)
                if cfg_analysis['FIT_ALPHA_DIFF'] and log_bin_results[0] is not None:
                    print("\n  Fitting Delta Alpha Data...")
                    popt, perr = fit_truncated_powerlaw_logspace(log_bin_results[0], log_bin_results[1])
                    results['delta_alpha']['fit'] = (popt, perr)
                    if popt is not None:
                        print(f"  Fit successful: ε={popt[1]:.4f}, λ={popt[2]:.4e}")
                    else:
                        print("  Delta alpha fitting failed!")
            else:
                print("  WARNING: No positive alpha differences found!")
    
    # --- E-S Scaling Analysis ---
    print("\n" + "="*50)
    print("ANALYZING E-S SCALING")
    print("="*50)
    results['scaling'] = {}
    energy_paired = raw_data['paired_energy']
    stress_paired = raw_data['paired_stress']
    
    if len(energy_paired) < 2:
        print("  Not enough paired data to analyze scaling.")
    else:
        print(f"  Number of paired energy-stress events: {len(energy_paired)}")
        log_stress = np.log10(stress_paired)
        log_energy = np.log10(energy_paired)
        
        stress_min, stress_max = np.min(log_stress), np.max(log_stress)
        
        # Create bins for stress
        bins_ES = np.linspace(stress_min, stress_max, cfg_analysis['NBIN_SCALING'] + 1)
        
        # Compute mean energy for each stress bin
        binned_stress, binned_energy_mean, binned_energy_std, binned_counts = [], [], [], []
        
        for i in range(len(bins_ES) - 1):
            mask = (log_stress >= bins_ES[i]) & (log_stress < bins_ES[i+1])
            if np.sum(mask) > 0:
                bin_center = (bins_ES[i] + bins_ES[i+1]) / 2
                binned_stress.append(bin_center)
                binned_energy_mean.append(np.mean(log_energy[mask]))
                binned_energy_std.append(np.std(log_energy[mask]))
                binned_counts.append(np.sum(mask))
        
        binned_stress = np.array(binned_stress)
        binned_energy_mean = np.array(binned_energy_mean)
        binned_energy_std = np.array(binned_energy_std)
        binned_counts = np.array(binned_counts)
        
        results['scaling']['binned_data'] = (binned_stress, binned_energy_mean, binned_energy_std, binned_counts)
        
        # Fit power law: log(E) = γ * log(S) + const
        if len(binned_stress) >= 2:
            coeffs = np.polyfit(binned_stress, binned_energy_mean, 1)
            gamma, intercept = coeffs[0], coeffs[1]
            
            # Compute R²
            residuals = binned_energy_mean - (gamma * binned_stress + intercept)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((binned_energy_mean - np.mean(binned_energy_mean))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            results['scaling']['fit'] = (gamma, intercept, r_squared)
            print(f"  E ~ S^γ power law fit: γ = {gamma:.4f}, R² = {r_squared:.4f}")
        else:
            print("  Not enough binned data points to fit scaling relationship.")
            
    return results

# =============================================================================
# == PLOTTING & SAVING FUNCTIONS
# =============================================================================

def _get_filename_suffix(config):
    """Helper to generate a consistent filename suffix from config."""
    cfg_filters = config['FILTERS']
    filename_suffix = ""
    
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

def _plot_distribution(ax, bin_results, fit_results, title, xlabel):
    """Helper to plot a single distribution (energy, stress, or d-alpha)."""
    
    if bin_results[0] is None:
        ax.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=13)
        return

    centers, hist, log_centers, log_hist, dx = bin_results
    
    mask = (hist > 0) & ~np.isinf(log_hist) & ~np.isnan(log_hist)
    ax.plot(log_centers[mask], log_hist[mask], 
            linestyle='None', marker='.', markersize=20, color='b', label='Data')
    
    if fit_results is not None:
        popt, perr = fit_results
        if popt is not None:
            x_fit = np.logspace(np.log10(centers[mask].min()), np.log10(centers[mask].max()), 100)
            y_fit = truncated_powerlaw(x_fit, *popt)
            ax.plot(np.log10(x_fit), np.log10(y_fit), 'r-', linewidth=2, 
                    label=f'Fit: ε={popt[1]:.2f}, λ={popt[2]:.2e}')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('log₁₀(Density)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()


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
    
    # --- Plot 0: Energy vs Stress Scaling ---
    if 'binned_data' in analysis_results['scaling']:
        fig0, ax0 = plt.subplots(figsize=(8, 6))
        
        # Plot raw data
        ax0.plot(np.log10(raw_data['paired_stress']), np.log10(raw_data['paired_energy']), 
                linestyle='None', marker='.', markersize=12, color='blue', alpha=0.3, 
                label='Data points', zorder=1)
        
        # Plot binned data
        binned_stress, binned_energy_mean, binned_energy_std, _ = analysis_results['scaling']['binned_data']
        ax0.errorbar(binned_stress, binned_energy_mean, yerr=binned_energy_std,
                    fmt='o', markersize=8, capsize=5, capthick=2, 
                    color='red', ecolor='darkred', linewidth=2,
                    label='Binned mean ± std', zorder=3)
        
        # Plot fit
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
    
    # --- Plot 1 & 2: Individual Energy and Stress Distributions ---
    # (These are also part of the combined plot, but saved individually)
    
    # Energy
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

    # Stress
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

    # --- Plot 3 & 4: Delta Alpha Distributions ---
    if cfg_analysis['ANALYZE_ALPHA_DIFF'] and 'log_bins' in analysis_results['delta_alpha']:
        
        # Plot 3: Log-Log plot
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        
        # Plot log-binned data
        _plot_distribution(ax3, analysis_results['delta_alpha']['log_bins'], 
                           analysis_results['delta_alpha'].get('fit'),
                           'Δα Distribution (Log-Log)' + title_suffix,
                           'log₁₀(Δα)')
        
        # Overlay linear-binned data
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

        # Plot 4: Linear-binned plot
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

    # --- Plot 5: Combined Plot ---
    num_plots = 2
    if cfg_analysis['ANALYZE_ALPHA_DIFF'] and 'log_bins' in analysis_results['delta_alpha']:
        num_plots += 1
    if cfg_analysis['ANALYZE_ALPHA_DIFF'] and 'lin_bins' in analysis_results['delta_alpha']:
        num_plots += 1
        
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    
    # Energy
    _plot_distribution(axes[0], analysis_results['energy']['bins'], 
                       analysis_results['energy'].get('fit'),
                       'Energy Change Distribution' + title_suffix,
                       'log₁₀(Energy Change)')
    
    # Stress
    _plot_distribution(axes[1], analysis_results['stress']['bins'], 
                       analysis_results['stress'].get('fit'),
                       'Stress Change Distribution' + title_suffix,
                       'log₁₀(Stress Change)')

    # Delta Alpha (Log)
    if num_plots >= 3:
        _plot_distribution(axes[2], analysis_results['delta_alpha']['log_bins'], 
                           analysis_results['delta_alpha'].get('fit'),
                           'Δα Distribution (Log-Log)' + title_suffix,
                           'log₁₀(Δα)')
    
    # Delta Alpha (Linear)
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
    
    # Optionally show the final combined plot
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

    # --- Save Binned E-S Scaling Data ---
    if 'binned_data' in analysis_results['scaling']:
        binned_stress, binned_energy_mean, binned_energy_std, binned_counts = analysis_results['scaling']['binned_data']
        
        # Save binned data
        filepath = os.path.join(output_dir, f'data_energy_vs_stress_binned{filename_suffix}.dat')
        with open(filepath, 'w') as f:
            f.write("# Binned Energy-Stress relationship\n")
            f.write("# log10(Stress_center), log10(Energy_mean), log10(Energy_std), count\n")
            for s, e, std, cnt in zip(binned_stress, binned_energy_mean, binned_energy_std, binned_counts):
                f.write(f"{s:.8e}, {e:.8e}, {std:.8e}, {int(cnt)}\n")
        print(f"  Saved: {filepath}")
    
    # --- Save Raw E-S Paired Data ---
    if len(raw_data['paired_energy']) > 0:
        filepath = os.path.join(output_dir, f'data_energy_vs_stress_raw{filename_suffix}.dat')
        with open(filepath, 'w') as f:
            f.write("# Raw paired data\n")
            f.write("# log10(Stress), log10(Energy)\n")
            for s, e in zip(raw_data['paired_stress'], raw_data['paired_energy']):
                f.write(f"{np.log10(s):.8e}, {np.log10(e):.8e}\n")
        print(f"  Saved: {filepath}")

    # --- Save Histograms ---
    # Energy
    _, _, log_centers, log_hist, _ = analysis_results['energy']['bins']
    filepath = os.path.join(output_dir, f'data_histogram_energy{filename_suffix}.dat')
    _save_histogram_data(filepath, log_centers, log_hist)

    # Stress
    _, _, log_centers, log_hist, _ = analysis_results['stress']['bins']
    filepath = os.path.join(output_dir, f'data_histogram_stress{filename_suffix}.dat')
    _save_histogram_data(filepath, log_centers, log_hist)

    # Delta Alpha (Log)
    if cfg_analysis['ANALYZE_ALPHA_DIFF'] and 'log_bins' in analysis_results['delta_alpha']:
        _, _, log_centers, log_hist, _ = analysis_results['delta_alpha']['log_bins']
        filepath = os.path.join(output_dir, f'data_histogram_dalpha_log{filename_suffix}.dat')
        _save_histogram_data(filepath, log_centers, log_hist)
        
        # Delta Alpha (Linear)
        if 'lin_bins' in analysis_results['delta_alpha']:
            centers, hist, _ = analysis_results['delta_alpha']['lin_bins']
            if centers is not None:
                filepath = os.path.join(output_dir, f'data_histogram_dalpha_linear{filename_suffix}.dat')
                with open(filepath, 'w') as f:
                    f.write("# bin_center, density\n")
                    for x, y in zip(centers, hist):
                        f.write(f"{x:.8e}, {y:.8e}\n")
                print(f"  Saved: {filepath}")

    # --- Save Fit Parameters ---
    if cfg_analysis['FIT_DATA'] or (cfg_analysis['ANALYZE_ALPHA_DIFF'] and cfg_analysis['FIT_ALPHA_DIFF']):
        filepath = os.path.join(output_dir, f'fit_parameters{filename_suffix}.txt')
        with open(filepath, 'w') as f:
            f.write("Truncated Power Law Fit: P(x) = A * x^(-epsilon) * exp(-lambda * x)\n")
            f.write(f"Fitting method: {cfg_analysis['FIT_METHOD']}\n\n")
            
            _write_fit_params(f, "ENERGY", analysis_results['energy'].get('fit'))
            _write_fit_params(f, "STRESS", analysis_results['stress'].get('fit'))
            if cfg_analysis['ANALYZE_ALPHA_DIFF']:
                _write_fit_params(f, "ALPHA DIFFERENCES (Δα)", analysis_results['delta_alpha'].get('fit'))
        
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
        'SHOW_PLOTS': True, # Whether to call plt.show() at the end
        
        'FILTERS': {
            'BY_PLASTICITY': False,
            'BY_ALPHA': True,
            'ALPHA_MIN': 0.1401,
            'ALPHA_MAX': 0.4,
            
            'BY_XMIN': True,
            'ENERGY_XMIN': 1e-3,
            'STRESS_XMIN': 1,
            
            'BY_XMAX': False,
            'ENERGY_XMAX': 1e-2,
            'STRESS_XMAX': 1e-2,
        },
        
        'ANALYSIS': {
            'FIT_DATA': True,
            'FIT_METHOD': 'logspace', # 'logspace', 'weighted', or 'both' (not implemented, defaults to logspace)
            
            'ANALYZE_ALPHA_DIFF': True,
            'FIT_ALPHA_DIFF': False,
            
            'NBIN': 13,
            'NBIN_LINEAR': 30,
            'NBIN_SCALING': 15,
        }
    }
    # =========================
    
    # 1. Setup Environment
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    print(f"Output directory: {CONFIG['OUTPUT_DIR']}")
    
    # 2. Find Files
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

    # 3. Load and Process Data
    print("\n" + "="*60)
    print("STEP 1: LOADING AND PROCESSING DATA")
    print("="*60)
    raw_data = load_and_process_data(filenames, CONFIG)
    
    if all(len(v) == 0 for v in raw_data.values()):
        print("ERROR: No data was loaded after processing. Check files and filters.")
        exit(1)

    # 4. Run Analyses
    print("\n" + "="*60)
    print("STEP 2: RUNNING ANALYSES")
    print("="*60)
    analysis_results = run_analyses(raw_data, CONFIG)

    # 5. Generate Plots
    print("\n" + "="*60)
    print("STEP 3: GENERATING PLOTS")
    print("="*60)
    generate_plots(raw_data, analysis_results, CONFIG)

    # 6. Save Data Files
    print("\n" + "="*60)
    print("STEP 4: SAVING DATA FILES")
    print("="*60)
    save_results(raw_data, analysis_results, CONFIG)
    
    print(f"\nDone! All output files saved in {CONFIG['OUTPUT_DIR']}/")


if __name__ == "__main__":
    main()