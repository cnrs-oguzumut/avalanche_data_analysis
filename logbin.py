import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import os
import glob  

def process_multiple_files(filenames, **kwargs):
    """
    Process multiple simulation files and combine delta_alpha arrays
    """
    all_delta_alpha = []
    all_energy_data = []
    all_stress_data = []
    
    for filename in filenames:
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        # Read data
        alpha, energy_change, stress_change, plasticity_flag = read_energy_stress_log(filename)
        
        # Apply filters
        mask = np.ones(len(alpha), dtype=bool)
        
        if kwargs.get('FILTER_BY_PLASTICITY', False):
            plasticity_mask = (plasticity_flag == 1)
            mask = mask & plasticity_mask
        
        if kwargs.get('FILTER_BY_ALPHA', False):
            alpha_mask = (alpha >= kwargs['ALPHA_MIN']) & (alpha <= kwargs['ALPHA_MAX'])
            mask = mask & alpha_mask
        
        energy_change_filtered = energy_change[mask]
        stress_change_filtered = stress_change[mask]
        alpha_filtered = alpha[mask]
        
        # Get positive energy
        positive_energy_mask = energy_change_filtered > 0
        positive_stress_mask = stress_change_filtered > 0
        
        alpha_for_energy = alpha_filtered[positive_energy_mask]
        energy_data = energy_change_filtered[positive_energy_mask]
        stress_data = stress_change_filtered[positive_stress_mask]
        
        # Apply xmin/xmax filters
        if kwargs.get('FILTER_BY_XMIN', False):
            energy_xmin_mask = energy_data >= kwargs['ENERGY_XMIN']
            alpha_for_energy = alpha_for_energy[energy_xmin_mask]
            energy_data = energy_data[energy_xmin_mask]
            
            stress_data = stress_data[stress_data >= kwargs['STRESS_XMIN']]
        
        if kwargs.get('FILTER_BY_XMAX', False):
            energy_xmax_mask = energy_data <= kwargs['ENERGY_XMAX']
            alpha_for_energy = alpha_for_energy[energy_xmax_mask]
            energy_data = energy_data[energy_xmax_mask]
            
            stress_data = stress_data[stress_data <= kwargs['STRESS_XMAX']]
        
        # Compute delta_alpha for THIS simulation only
        if len(alpha_for_energy) > 1:
            delta_alpha = np.diff(alpha_for_energy)
            all_delta_alpha.append(delta_alpha)
            print(f"  Avalanches: {len(alpha_for_energy)}")
            print(f"  Delta_alpha computed: {len(delta_alpha)}")
        
        # Collect energy and stress data
        all_energy_data.append(energy_data)
        all_stress_data.append(stress_data)
    
    # Combine all arrays
    combined_delta_alpha = np.concatenate(all_delta_alpha) if all_delta_alpha else np.array([])
    combined_energy = np.concatenate(all_energy_data) if all_energy_data else np.array([])
    combined_stress = np.concatenate(all_stress_data) if all_stress_data else np.array([])
    
    print(f"\n{'='*60}")
    print(f"COMBINED RESULTS:")
    print(f"  Total delta_alpha: {len(combined_delta_alpha)}")
    print(f"  Total energy avalanches: {len(combined_energy)}")
    print(f"  Total stress avalanches: {len(combined_stress)}")
    print(f"{'='*60}\n")
    
    return combined_delta_alpha, combined_energy, combined_stress


def read_energy_stress_log(filename="energy_stress_log.csv"):
    """
    Read the CSV file created by ConfigurationSaver::logEnergyAndStress
    
    Returns:
        alpha, energy_change, stress_change, plasticity_flag as numpy arrays
    """
    # Skip header row and read all data
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    
    # Extract columns
    alpha = data[:, 1]
    energy_change = data[:, 6]
    stress_change = data[:, 7]
    plasticity_flag = data[:, 8].astype(int)
    
    return alpha, energy_change, stress_change, plasticity_flag


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
    div = (np.log10(xmax) - np.log10(xmin)) / nbin
    
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
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    
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
    mask = (hist > 0) & (bin_centers > 0) & ~np.isinf(hist)
    x_data = bin_centers[mask]
    y_data = hist[mask]
    
    if len(x_data) < 3:
        print("Not enough data points for fitting")
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
    
    print(f"Initial guess: log_A={log_A_init:.3f}, epsilon={epsilon_init:.3f}, lambda={lambda_init:.3e}")
    
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
            print(f"curve_fit failed: {e}")
            print("Trying minimize method...")
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
                print(f"Optimization failed: {result.message}")
                return None, None
                
        except Exception as e:
            print(f"minimize failed: {e}")
            return None, None


def fit_truncated_powerlaw_with_weights(bin_centers, hist):
    """
    Fit with weights to emphasize different parts of the distribution
    """
    mask = (hist > 0) & (bin_centers > 0) & ~np.isinf(hist)
    x_data = bin_centers[mask]
    y_data = hist[mask]
    
    if len(x_data) < 3:
        print("Not enough data points for fitting")
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
    
    print(f"Weighted fit initial guess: log_A={log_A_init:.3f}, epsilon={epsilon_init:.3f}, lambda={lambda_init:.3e}")
    
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
        print(f"Weighted fit failed: {e}")
        return None, None


# Main script
if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    USE_MULTIPLE_FILES = True     # *** NEW: Set to True for multiple files ***
    FILE_PATTERN = "./build*/energy_stress_log.csv"  # *** NEW: Pattern to match files ***
    SINGLE_FILE = "energy_stress_log.csv"    # Used only if USE_MULTIPLE_FILES = False
    
    FILTER_BY_PLASTICITY = False
    FILTER_BY_ALPHA = True
    ALPHA_MIN = 0.16
    ALPHA_MAX = .4
    
    FILTER_BY_XMIN = True
    ENERGY_XMIN = 1e-3
    STRESS_XMIN = 20
    
    FILTER_BY_XMAX = False
    ENERGY_XMAX = 1e-2
    STRESS_XMAX = 1e-2
    
    FIT_DATA = True
    FIT_METHOD = 'logspace'
    
    ANALYZE_ALPHA_DIFF = True
    FIT_ALPHA_DIFF = False
    
    OUTPUT_DIR = './statistics'
    
    nbin = 13
    nbin_linear = 30
    # =========================
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # ===== DATA LOADING =====
    if USE_MULTIPLE_FILES:
        # Get all files matching pattern
        print(f"\nSearching for files matching: {FILE_PATTERN}")
        filenames = sorted(glob.glob(FILE_PATTERN))
        
        if len(filenames) == 0:
            print(f"ERROR: No files found matching pattern '{FILE_PATTERN}'")
            print(f"\nAvailable directories:")
            for d in sorted(glob.glob("./build*/")):
                print(f"  {d}")
            exit(1)
        
        print(f"\nFound {len(filenames)} files:")
        for f in filenames:
            print(f"  - {f}")
        
        # Process all files and combine results
        delta_alpha_all, energy_data, stress_data = process_multiple_files(
            filenames,
            FILTER_BY_PLASTICITY=FILTER_BY_PLASTICITY,
            FILTER_BY_ALPHA=FILTER_BY_ALPHA,
            ALPHA_MIN=ALPHA_MIN,
            ALPHA_MAX=ALPHA_MAX,
            FILTER_BY_XMIN=FILTER_BY_XMIN,
            ENERGY_XMIN=ENERGY_XMIN,
            STRESS_XMIN=STRESS_XMIN,
            FILTER_BY_XMAX=FILTER_BY_XMAX,
            ENERGY_XMAX=ENERGY_XMAX,
            STRESS_XMAX=STRESS_XMAX
        )
        
        # Store delta_alpha for later use
        alpha_for_energy_exists = len(delta_alpha_all) > 0
        
    else:
        # Single file processing (original code)
        print(f"\nProcessing single file: {SINGLE_FILE}")
        
        # Read the data
        alpha, energy_change, stress_change, plasticity_flag = read_energy_stress_log(SINGLE_FILE)
        
        print(f"Loaded {len(alpha)} data points")
        print(f"Alpha range in data: [{np.min(alpha):.6e}, {np.max(alpha):.6e}]")
        print(f"Plasticity events: {np.sum(plasticity_flag)} / {len(plasticity_flag)}")
        
        # Store original indices to track alpha values
        original_indices = np.arange(len(alpha))
        
        # Apply filters
        mask = np.ones(len(alpha), dtype=bool)
        
        if FILTER_BY_PLASTICITY:
            print("\n*** FILTERING: Only using plasticity events (flag=1) ***")
            plasticity_mask = (plasticity_flag == 1)
            mask = mask & plasticity_mask
            print(f"Data points after plasticity filter: {np.sum(mask)}")
        
        if FILTER_BY_ALPHA:
            print(f"\n*** FILTERING: Only using alpha in range [{ALPHA_MIN}, {ALPHA_MAX}] ***")
            alpha_mask = (alpha >= ALPHA_MIN) & (alpha <= ALPHA_MAX)
            mask = mask & alpha_mask
            print(f"Data points after alpha filter: {np.sum(mask)}")
        
        # Apply mask to all arrays
        energy_change_filtered = energy_change[mask]
        stress_change_filtered = stress_change[mask]
        alpha_filtered = alpha[mask]
        original_indices_filtered = original_indices[mask]
        
        print(f"\nTotal data points after all filters: {len(alpha_filtered)}")
        
        if len(alpha_filtered) == 0:
            print("ERROR: No data points remaining after filtering!")
            exit(1)
        
        print(f"Alpha range after filtering: [{np.min(alpha_filtered):.6e}, {np.max(alpha_filtered):.6e}]")
        
        # Filter for positive values
        positive_energy_mask = energy_change_filtered > 0
        positive_stress_mask = stress_change_filtered > 0
        
        print(f"Positive energy changes: {np.sum(positive_energy_mask)} / {len(energy_change_filtered)}")
        print(f"Positive stress changes: {np.sum(positive_stress_mask)} / {len(stress_change_filtered)}")
        
        # Get positive data and track corresponding alpha values for accepted energy avalanches
        alpha_for_energy = alpha_filtered[positive_energy_mask]
        energy_data = energy_change_filtered[positive_energy_mask]
        stress_data = stress_change_filtered[positive_stress_mask]

        # Apply xmin and xmax filters to energy and stress, tracking alpha for energy
        if FILTER_BY_XMIN:
            print(f"\n*** FILTERING XMIN: Energy >= {ENERGY_XMIN:.6e}, Stress >= {STRESS_XMIN:.6e} ***")
            print(f"Energy data points before xmin filter: {len(energy_data)}")
            print(f"Stress data points before xmin filter: {len(stress_data)}")
            
            energy_xmin_mask = energy_data >= ENERGY_XMIN
            alpha_for_energy = alpha_for_energy[energy_xmin_mask]
            energy_data = energy_data[energy_xmin_mask]
            
            stress_data = stress_data[stress_data >= STRESS_XMIN]
            
            print(f"Energy data points after xmin filter: {len(energy_data)}")
            print(f"Stress data points after xmin filter: {len(stress_data)}")

        if FILTER_BY_XMAX:
            print(f"\n*** FILTERING XMAX: Energy <= {ENERGY_XMAX:.6e}, Stress <= {STRESS_XMAX:.6e} ***")
            print(f"Energy data points before xmax filter: {len(energy_data)}")
            print(f"Stress data points before xmax filter: {len(stress_data)}")
            
            energy_xmax_mask = energy_data <= ENERGY_XMAX
            alpha_for_energy = alpha_for_energy[energy_xmax_mask]
            energy_data = energy_data[energy_xmax_mask]
            
            stress_data = stress_data[stress_data <= STRESS_XMAX]
            
            print(f"Energy data points after xmax filter: {len(energy_data)}")
            print(f"Stress data points after xmax filter: {len(stress_data)}")

        if len(energy_data) == 0 or len(stress_data) == 0:
            print("ERROR: No positive data points to analyze!")
            exit(1)
        
        print(f"Number of accepted energy avalanches: {len(alpha_for_energy)}")
        
        # Compute delta_alpha for single file
        if len(alpha_for_energy) > 1:
            delta_alpha_all = np.diff(alpha_for_energy)
            alpha_for_energy_exists = True
        else:
            delta_alpha_all = np.array([])
            alpha_for_energy_exists = False
    
    # Common statistics
    energy_xmin = np.min(energy_data)
    energy_xmax = np.max(energy_data)
    stress_xmin = np.min(stress_data)
    stress_xmax = np.max(stress_data)

    print(f"\nFinal energy range: [{energy_xmin:.6e}, {energy_xmax:.6e}]")
    print(f"Final stress range: [{stress_xmin:.6e}, {stress_xmax:.6e}]")

    # Compute alpha differences for accepted energy avalanches
    dalpha_linear_centers = None
    dalpha_linear_hist = None

    if ANALYZE_ALPHA_DIFF and alpha_for_energy_exists and len(delta_alpha_all) > 0:
        print("\n" + "="*50)
        print("ANALYZING ALPHA DIFFERENCES (Accepted Energy Avalanches)")
        print("="*50)
        
        # Use the already computed delta_alpha
        delta_alpha = delta_alpha_all
        print(f"Number of alpha differences: {len(delta_alpha)}")
        print(f"Alpha difference range: [{np.min(delta_alpha):.6e}, {np.max(delta_alpha):.6e}]")
        
        # Diagnostics
        print(f"\n=== DIAGNOSTICS ===")
        print(f"Negative delta_alpha count: {np.sum(delta_alpha < 0)}")
        print(f"Zero delta_alpha count: {np.sum(delta_alpha == 0)}")
        print(f"Mean delta_alpha: {np.mean(delta_alpha):.6e}")
        print(f"Median delta_alpha: {np.median(delta_alpha):.6e}")
        
        # Check for positive differences
        positive_delta_alpha = delta_alpha[delta_alpha > 0]
        print(f"\nPositive alpha differences: {len(positive_delta_alpha)} / {len(delta_alpha)}")
        
        if len(positive_delta_alpha) > 0:
            print(f"Positive delta alpha range: [{np.min(positive_delta_alpha):.6e}, {np.max(positive_delta_alpha):.6e}]")
            
            # Logarithmic binning for alpha differences
            print("\n=== Alpha Difference Logarithmic Binning ===")
            result = logarithmic_binning(positive_delta_alpha, nbin=nbin)
            if result[0] is not None:
                dalpha_centers, dalpha_hist, dalpha_log_centers, dalpha_log_hist, dalpha_dx = result
                print(f"Number of bins: {len(dalpha_centers)}")
                
                # Fit alpha difference data if requested
                dalpha_popt = None
                dalpha_perr = None
                dalpha_fit_params = None
                
                if FIT_ALPHA_DIFF:
                    print("\n=== Fitting Alpha Difference Distribution (Log) ===")
                    print("Model: P(Δα) = A * (Δα)^(-ε) * exp(-λ * Δα)")
                    
                    dalpha_popt, dalpha_perr = fit_truncated_powerlaw_logspace(dalpha_centers, dalpha_hist, method='curve_fit')
                    if dalpha_popt is not None:
                        dalpha_fit_params = dalpha_popt
                        print(f"\nAlpha difference fit parameters:")
                        print(f"  A       = {dalpha_popt[0]:.6e}", end="")
                        if dalpha_perr is not None:
                            print(f" ± {dalpha_perr[0]:.6e}")
                        else:
                            print()
                        print(f"  epsilon = {dalpha_popt[1]:.4f}", end="")
                        if dalpha_perr is not None:
                            print(f" ± {dalpha_perr[1]:.4f}")
                        else:
                            print()
                        print(f"  lambda  = {dalpha_popt[2]:.6e}", end="")
                        if dalpha_perr is not None:
                            print(f" ± {dalpha_perr[2]:.6e}")
                        else:
                            print()
                    else:
                        print("Alpha difference fitting failed!")
            else:
                print("ERROR: Alpha difference logarithmic binning failed!")
            
            # Linear binning for alpha differences
            print("\n=== Alpha Difference Linear Binning ===")
            result_linear = linear_binning(positive_delta_alpha, nbin=nbin_linear)
            if result_linear[0] is not None:
                dalpha_linear_centers, dalpha_linear_hist, dalpha_linear_dx = result_linear
                print(f"Number of bins: {len(dalpha_linear_centers)}")
            else:
                print("ERROR: Alpha difference linear binning failed!")
                ANALYZE_ALPHA_DIFF = False
        else:
            print("WARNING: No positive alpha differences found!")
            ANALYZE_ALPHA_DIFF = False
    elif ANALYZE_ALPHA_DIFF:
        print("\nWARNING: Not enough accepted energy avalanches to compute alpha differences!")
        ANALYZE_ALPHA_DIFF = False
    
    # Binning for energy and stress
    print("\n=== Energy Change Binning ===")
    result = logarithmic_binning(energy_data, nbin=nbin)
    if result[0] is None:
        print("ERROR: Energy binning failed!")
        exit(1)
    energy_centers, energy_hist, energy_log_centers, energy_log_hist, energy_dx = result
    print(f"Number of bins: {len(energy_centers)}")
    
    print("\n=== Stress Change Binning ===")
    result = logarithmic_binning(stress_data, nbin=nbin)
    if result[0] is None:
        print("ERROR: Stress binning failed!")
        exit(1)
    stress_centers, stress_hist, stress_log_centers, stress_log_hist, stress_dx = result
    print(f"Number of bins: {len(stress_centers)}")
    
    # ===== ENERGY vs STRESS RELATIONSHIP =====
    print("\n=== Energy vs Stress Relationship ===")
    
    # For multiple files, we need to reprocess to get paired energy-stress data
    if USE_MULTIPLE_FILES:
        print("Extracting paired energy-stress data from multiple files...")
        all_energy_stress_pairs = []
        
        for filename in filenames:
            alpha, energy_change, stress_change, plasticity_flag = read_energy_stress_log(filename)
            
            # Apply same filters as before
            mask = np.ones(len(alpha), dtype=bool)
            
            if FILTER_BY_PLASTICITY:
                mask = mask & (plasticity_flag == 1)
            
            if FILTER_BY_ALPHA:
                mask = mask & (alpha >= ALPHA_MIN) & (alpha <= ALPHA_MAX)
            
            energy_change_filtered = energy_change[mask]
            stress_change_filtered = stress_change[mask]
            
            # Keep only events where BOTH energy and stress are positive
            both_positive_mask = (energy_change_filtered > 0) & (stress_change_filtered > 0)
            energy_paired = energy_change_filtered[both_positive_mask]
            stress_paired = stress_change_filtered[both_positive_mask]
            
            # Apply xmin/xmax filters if needed
            if FILTER_BY_XMIN or FILTER_BY_XMAX:
                pair_mask = np.ones(len(energy_paired), dtype=bool)
                
                if FILTER_BY_XMIN:
                    pair_mask = pair_mask & (energy_paired >= ENERGY_XMIN) & (stress_paired >= STRESS_XMIN)
                
                if FILTER_BY_XMAX:
                    pair_mask = pair_mask & (energy_paired <= ENERGY_XMAX) & (stress_paired <= STRESS_XMAX)
                
                energy_paired = energy_paired[pair_mask]
                stress_paired = stress_paired[pair_mask]
            
            all_energy_stress_pairs.append((energy_paired, stress_paired))
        
        # Combine all pairs
        energy_for_scaling = np.concatenate([e for e, s in all_energy_stress_pairs])
        stress_for_scaling = np.concatenate([s for e, s in all_energy_stress_pairs])
        
    else:
        # Single file: extract paired data
        print("Extracting paired energy-stress data from single file...")
        
        # Keep only events where BOTH energy and stress are positive
        both_positive_mask = (energy_change_filtered > 0) & (stress_change_filtered > 0)
        energy_paired = energy_change_filtered[both_positive_mask]
        stress_paired = stress_change_filtered[both_positive_mask]
        
        # Apply xmin/xmax filters if needed
        if FILTER_BY_XMIN or FILTER_BY_XMAX:
            pair_mask = np.ones(len(energy_paired), dtype=bool)
            
            if FILTER_BY_XMIN:
                pair_mask = pair_mask & (energy_paired >= ENERGY_XMIN) & (stress_paired >= STRESS_XMIN)
            
            if FILTER_BY_XMAX:
                pair_mask = pair_mask & (energy_paired <= ENERGY_XMAX) & (stress_paired <= STRESS_XMAX)
            
            energy_paired = energy_paired[pair_mask]
            stress_paired = stress_paired[pair_mask]
        
        energy_for_scaling = energy_paired
        stress_for_scaling = stress_paired
    
    print(f"Number of paired energy-stress events: {len(energy_for_scaling)}")
    print(f"Energy range: [{np.min(energy_for_scaling):.6e}, {np.max(energy_for_scaling):.6e}]")
    print(f"Stress range: [{np.min(stress_for_scaling):.6e}, {np.max(stress_for_scaling):.6e}]")
    
    # Fit data
    energy_fit_params = None
    stress_fit_params = None
    energy_popt = None
    energy_perr = None
    stress_popt = None
    stress_perr = None
    
    if FIT_DATA:
        print("\n=== Fitting Truncated Power Law ===")
        print("Model: P(x) = A * x^(-epsilon) * exp(-lambda * x)")
        print(f"Fitting method: {FIT_METHOD}")
        
        # Fit energy data
        print("\n" + "="*50)
        print("FITTING ENERGY DATA")
        print("="*50)
        
        if FIT_METHOD in ['logspace', 'both']:
            print("\nMethod: Log-space fitting")
            energy_popt, energy_perr = fit_truncated_powerlaw_logspace(energy_centers, energy_hist, method='curve_fit')
            if energy_popt is not None:
                energy_fit_params = energy_popt
                print(f"\nEnergy fit parameters (log-space):")
                print(f"  A       = {energy_popt[0]:.6e}", end="")
                if energy_perr is not None:
                    print(f" ± {energy_perr[0]:.6e}")
                else:
                    print()
                print(f"  epsilon = {energy_popt[1]:.4f}", end="")
                if energy_perr is not None:
                    print(f" ± {energy_perr[1]:.4f}")
                else:
                    print()
                print(f"  lambda  = {energy_popt[2]:.6e}", end="")
                if energy_perr is not None:
                    print(f" ± {energy_perr[2]:.6e}")
                else:
                    print()
            else:
                print("Energy fitting failed!")
        
        if FIT_METHOD in ['weighted', 'both']:
            print("\nMethod: Weighted fitting")
            energy_popt2, energy_perr2 = fit_truncated_powerlaw_with_weights(energy_centers, energy_hist)
            if energy_popt2 is not None:
                print(f"\nEnergy fit parameters (weighted):")
                print(f"  A       = {energy_popt2[0]:.6e}", end="")
                if energy_perr2 is not None:
                    print(f" ± {energy_perr2[0]:.6e}")
                else:
                    print()
                print(f"  epsilon = {energy_popt2[1]:.4f}", end="")
                if energy_perr2 is not None:
                    print(f" ± {energy_perr2[1]:.4f}")
                else:
                    print()
                print(f"  lambda  = {energy_popt2[2]:.6e}", end="")
                if energy_perr2 is not None:
                    print(f" ± {energy_perr2[2]:.6e}")
                else:
                    print()
                if FIT_METHOD == 'both' and energy_popt is None:
                    energy_fit_params = energy_popt2
                    energy_popt = energy_popt2
                    energy_perr = energy_perr2
        
        # Fit stress data
        print("\n" + "="*50)
        print("FITTING STRESS DATA")
        print("="*50)
        
        if FIT_METHOD in ['logspace', 'both']:
            print("\nMethod: Log-space fitting")
            stress_popt, stress_perr = fit_truncated_powerlaw_logspace(stress_centers, stress_hist, method='curve_fit')
            if stress_popt is not None:
                stress_fit_params = stress_popt
                print(f"\nStress fit parameters (log-space):")
                print(f"  A       = {stress_popt[0]:.6e}", end="")
                if stress_perr is not None:
                    print(f" ± {stress_perr[0]:.6e}")
                else:
                    print()
                print(f"  epsilon = {stress_popt[1]:.4f}", end="")
                if stress_perr is not None:
                    print(f" ± {stress_perr[1]:.4f}")
                else:
                    print()
                print(f"  lambda  = {stress_popt[2]:.6e}", end="")
                if stress_perr is not None:
                    print(f" ± {stress_perr[2]:.6e}")
                else:
                    print()
            else:
                print("Stress fitting failed!")
        
        if FIT_METHOD in ['weighted', 'both']:
            print("\nMethod: Weighted fitting")
            stress_popt2, stress_perr2 = fit_truncated_powerlaw_with_weights(stress_centers, stress_hist)
            if stress_popt2 is not None:
                print(f"\nStress fit parameters (weighted):")
                print(f"  A       = {stress_popt2[0]:.6e}", end="")
                if stress_perr2 is not None:
                    print(f" ± {stress_perr2[0]:.6e}")
                else:
                    print()
                print(f"  epsilon = {stress_popt2[1]:.4f}", end="")
                if stress_perr2 is not None:
                    print(f" ± {stress_perr2[1]:.4f}")
                else:
                    print()
                print(f"  lambda  = {stress_popt2[2]:.6e}", end="")
                if stress_perr2 is not None:
                    print(f" ± {stress_perr2[2]:.6e}")
                else:
                    print()
                if FIT_METHOD == 'both' and stress_popt is None:
                    stress_fit_params = stress_popt2
                    stress_popt = stress_popt2
                    stress_perr = stress_perr2
    
    # Build filename suffix
    filename_suffix = ""
    if FILTER_BY_PLASTICITY:
        filename_suffix += "_plasticity"
    if FILTER_BY_ALPHA:
        filename_suffix += f"_alpha_{ALPHA_MIN}_{ALPHA_MAX}"
    if FILTER_BY_XMIN and FILTER_BY_XMAX:
        filename_suffix += f"_xrange_{ENERGY_XMIN:.0e}_{ENERGY_XMAX:.0e}_{STRESS_XMIN:.0e}_{STRESS_XMAX:.0e}"
    elif FILTER_BY_XMIN:
        filename_suffix += f"_xmin_{ENERGY_XMIN:.0e}_{STRESS_XMIN:.0e}"
    elif FILTER_BY_XMAX:
        filename_suffix += f"_xmax_{ENERGY_XMAX:.0e}_{STRESS_XMAX:.0e}"
    
    title_suffix = ""
    if FILTER_BY_PLASTICITY:
        title_suffix += " (Plasticity)"
    if FILTER_BY_ALPHA:
        title_suffix += f" (α∈[{ALPHA_MIN},{ALPHA_MAX}])"
    if FILTER_BY_XMIN or FILTER_BY_XMAX:
        title_suffix += " (x filtered)"
    
    # Create and save individual plots
    print("\n=== Creating and saving plots ===")
    
# Plot 0: Energy vs Stress scaling relationship with binning
if len(energy_for_scaling) > 0 and len(stress_for_scaling) > 0:
    fig0, ax0 = plt.subplots(figsize=(8, 6))
    
    # Plot raw data points
    ax0.plot(np.log10(stress_for_scaling), np.log10(energy_for_scaling), 
            linestyle='None', marker='.', markersize=12, color='blue', alpha=0.3, 
            label='Data points', zorder=1)
    
    # Compute binned averages
    print("\n=== Computing binned E~S relationship ===")
    n_bins_ES = 15  # Number of bins for E-S relationship
    
    # Create logarithmic bins for stress
    log_stress = np.log10(stress_for_scaling)
    log_energy = np.log10(energy_for_scaling)
    
    stress_min = np.min(log_stress)
    stress_max = np.max(log_stress)
    
    bins_ES = np.linspace(stress_min, stress_max, n_bins_ES + 1)
    
    # Compute mean energy for each stress bin
    binned_stress = []
    binned_energy_mean = []
    binned_energy_std = []
    binned_counts = []
    
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
    
    # Plot binned averages with error bars
    ax0.errorbar(binned_stress, binned_energy_mean, yerr=binned_energy_std,
                fmt='o', markersize=8, capsize=5, capthick=2, 
                color='red', ecolor='darkred', linewidth=2,
                label='Binned mean ± std', zorder=3)
    
    # Fit power law to binned data: log(E) = γ * log(S) + const
    if len(binned_stress) >= 2:
        # Linear fit in log-log space
        coeffs = np.polyfit(binned_stress, binned_energy_mean, 1)
        gamma = coeffs[0]
        intercept = coeffs[1]
        
        # Create fit line
        stress_fit = np.linspace(stress_min, stress_max, 100)
        energy_fit = gamma * stress_fit + intercept
        
        ax0.plot(stress_fit, energy_fit, 'b--', linewidth=2.5, 
                label=f'Power law fit: γ = {gamma:.3f}', zorder=2)
        
        print(f"E ~ S^γ power law fit:")
        print(f"  γ (exponent) = {gamma:.4f}")
        print(f"  Intercept = {intercept:.4f}")
        print(f"  10^intercept = {10**intercept:.4e}")
        
        # Compute R² for the fit
        residuals = binned_energy_mean - (gamma * binned_stress + intercept)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((binned_energy_mean - np.mean(binned_energy_mean))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"  R² = {r_squared:.4f}")
    
    ax0.set_xlabel('log₁₀(Stress Change)', fontsize=13, fontweight='bold')
    ax0.set_ylabel('log₁₀(Energy Change)', fontsize=13, fontweight='bold')
    ax0.set_title('Energy-Stress Scaling: E ~ S^γ' + title_suffix, fontsize=14, fontweight='bold')
    ax0.grid(True, alpha=0.3, linestyle='--')
    ax0.legend(fontsize=11, loc='best')
    plt.tight_layout()
    
    energy_stress_plot_path = os.path.join(OUTPUT_DIR, f'energy_vs_stress_scaling{filename_suffix}.png')
    plt.savefig(energy_stress_plot_path, dpi=200)
    print(f"Saved: {energy_stress_plot_path}")
    plt.close(fig0)
    
    # Save the binned data
    binned_data_path = os.path.join(OUTPUT_DIR, f'data_energy_vs_stress_binned{filename_suffix}.dat')
    with open(binned_data_path, 'w') as f:
        f.write("# Binned Energy-Stress relationship\n")
        f.write("# log10(Stress_center), log10(Energy_mean), log10(Energy_std), count\n")
        for s, e, std, cnt in zip(binned_stress, binned_energy_mean, binned_energy_std, binned_counts):
            f.write(f"{s:.6e}, {e:.6e}, {std:.6e}, {int(cnt)}\n")
    print(f"Saved: {binned_data_path}")
    
    # Also save the raw paired data
    energy_stress_data_path = os.path.join(OUTPUT_DIR, f'data_energy_vs_stress_raw{filename_suffix}.dat')
    with open(energy_stress_data_path, 'w') as f:
        f.write("# Raw paired data\n")
        f.write("# log10(Stress), log10(Energy)\n")
        for s, e in zip(stress_for_scaling, energy_for_scaling):
            f.write(f"{np.log10(s):.6e}, {np.log10(e):.6e}\n")
    print(f"Saved: {energy_stress_data_path}")

    # Plot 1: Energy change distribution
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    mask = ~np.isinf(energy_log_hist)
    ax1.plot(energy_log_centers[mask], energy_log_hist[mask], 
            linestyle='None', marker='.', markersize=20, color='b', label='Data')
    
    if FIT_DATA and energy_fit_params is not None:
        x_fit = np.logspace(np.log10(energy_centers[0]), np.log10(energy_centers[-1]), 100)
        y_fit = truncated_powerlaw(x_fit, *energy_fit_params)
        ax1.plot(np.log10(x_fit), np.log10(y_fit), 'r-', linewidth=2, 
                label=f'Fit: ε={energy_fit_params[1]:.2f}, λ={energy_fit_params[2]:.2e}')
    
    ax1.set_xlabel('log₁₀(Energy Change)', fontsize=12)
    ax1.set_ylabel('log₁₀(Density)', fontsize=12)
    ax1.set_title('Energy Change Distribution' + title_suffix, fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    energy_plot_path = os.path.join(OUTPUT_DIR, f'energy_distribution{filename_suffix}.png')
    plt.savefig(energy_plot_path, dpi=150)
    print(f"Saved: {energy_plot_path}")
    plt.close(fig1)
    
    # Plot 2: Stress change distribution
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    mask = ~np.isinf(stress_log_hist)
    ax2.plot(stress_log_centers[mask], stress_log_hist[mask], 
            linestyle='None', marker='.', markersize=20, color='b', label='Data')
    
    if FIT_DATA and stress_fit_params is not None:
        x_fit = np.logspace(np.log10(stress_centers[0]), np.log10(stress_centers[-1]), 100)
        y_fit = truncated_powerlaw(x_fit, *stress_fit_params)
        ax2.plot(np.log10(x_fit), np.log10(y_fit), 'r-', linewidth=2,
                label=f'Fit: ε={stress_fit_params[1]:.2f}, λ={stress_fit_params[2]:.2e}')
    
    ax2.set_xlabel('log₁₀(Stress Change)', fontsize=12)
    ax2.set_ylabel('log₁₀(Density)', fontsize=12)
    ax2.set_title('Stress Change Distribution' + title_suffix, fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    stress_plot_path = os.path.join(OUTPUT_DIR, f'stress_distribution{filename_suffix}.png')
    plt.savefig(stress_plot_path, dpi=150)
    print(f"Saved: {stress_plot_path}")
    plt.close(fig2)
    
    # Plot 3 & 4: Alpha difference distributions (if available)
    if ANALYZE_ALPHA_DIFF and 'dalpha_centers' in locals():
        # Plot 3: Log-log plot with both binning methods
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        mask = ~np.isinf(dalpha_log_hist)
        ax3.plot(dalpha_log_centers[mask], dalpha_log_hist[mask], 
                linestyle='None', marker='.', markersize=20, color='g', label='Log binning')

        # Add linear binning to log-log plot
        if dalpha_linear_centers is not None:
            mask_linear = (dalpha_linear_hist > 0)
            ax3.plot(np.log10(dalpha_linear_centers[mask_linear]), np.log10(dalpha_linear_hist[mask_linear]), 
                    linestyle='None', marker='x', markersize=12, color='orange', label='Linear binning')

        if FIT_ALPHA_DIFF and 'dalpha_fit_params' in locals() and dalpha_fit_params is not None:
            x_fit = np.logspace(np.log10(dalpha_centers[0]), np.log10(dalpha_centers[-1]), 100)
            y_fit = truncated_powerlaw(x_fit, *dalpha_fit_params)
            ax3.plot(np.log10(x_fit), np.log10(y_fit), 'r-', linewidth=2,
                    label=f'Fit: ε={dalpha_fit_params[1]:.2f}, λ={dalpha_fit_params[2]:.2e}')

        ax3.set_xlabel('log₁₀(Δα)', fontsize=12)
        ax3.set_ylabel('log₁₀(Density)', fontsize=12)
        ax3.set_title('Δα Distribution (Log-Log)' + title_suffix, fontsize=13)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        plt.tight_layout()
        dalpha_log_plot_path = os.path.join(OUTPUT_DIR, f'dalpha_distribution_log{filename_suffix}.png')
        plt.savefig(dalpha_log_plot_path, dpi=150)
        print(f"Saved: {dalpha_log_plot_path}")
        plt.close(fig3)
        
        # Plot 4: Linear binning
        if dalpha_linear_centers is not None:
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            ax4.plot(dalpha_linear_centers, dalpha_linear_hist, 
                    linestyle='None', marker='.', markersize=20, color='g', label='Data')
            ax4.set_xlabel('Δα', fontsize=12)
            ax4.set_ylabel('Density', fontsize=12)
            ax4.set_title('Δα Distribution (Linear)' + title_suffix, fontsize=13)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.legend()
            plt.tight_layout()
            dalpha_linear_plot_path = os.path.join(OUTPUT_DIR, f'dalpha_distribution_linear{filename_suffix}.png')
            plt.savefig(dalpha_linear_plot_path, dpi=150)
            print(f"Saved: {dalpha_linear_plot_path}")
            plt.close(fig4)
    
    # Create combined plot with all subplots
    num_plots = 4 if (ANALYZE_ALPHA_DIFF and dalpha_linear_centers is not None) else 2
    if ANALYZE_ALPHA_DIFF and 'dalpha_centers' in locals() and dalpha_linear_centers is None:
        num_plots = 3
    
    fig, axes = plt.subplots(1, num_plots, figsize=(7*num_plots, 5))
    
    if num_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Energy change distribution
    ax = axes[0]
    mask = ~np.isinf(energy_log_hist)
    ax.plot(energy_log_centers[mask], energy_log_hist[mask], 
            linestyle='None', marker='.', markersize=20, color='b', label='Data')
    
    if FIT_DATA and energy_fit_params is not None:
        x_fit = np.logspace(np.log10(energy_centers[0]), np.log10(energy_centers[-1]), 100)
        y_fit = truncated_powerlaw(x_fit, *energy_fit_params)
        ax.plot(np.log10(x_fit), np.log10(y_fit), 'r-', linewidth=2, 
                label=f'Fit: ε={energy_fit_params[1]:.2f}, λ={energy_fit_params[2]:.2e}')
    
    ax.set_xlabel('log₁₀(Energy Change)', fontsize=12)
    ax.set_ylabel('log₁₀(Density)', fontsize=12)
    ax.set_title('Energy Change Distribution' + title_suffix, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Stress change distribution
    ax = axes[1]
    mask = ~np.isinf(stress_log_hist)
    ax.plot(stress_log_centers[mask], stress_log_hist[mask], 
            linestyle='None', marker='.', markersize=20, color='b', label='Data')
    
    if FIT_DATA and stress_fit_params is not None:
        x_fit = np.logspace(np.log10(stress_centers[0]), np.log10(stress_centers[-1]), 100)
        y_fit = truncated_powerlaw(x_fit, *stress_fit_params)
        ax.plot(np.log10(x_fit), np.log10(y_fit), 'r-', linewidth=2,
                label=f'Fit: ε={stress_fit_params[1]:.2f}, λ={stress_fit_params[2]:.2e}')
    
    ax.set_xlabel('log₁₀(Stress Change)', fontsize=12)
    ax.set_ylabel('log₁₀(Density)', fontsize=12)
    ax.set_title('Stress Change Distribution' + title_suffix, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Alpha difference distributions
    if ANALYZE_ALPHA_DIFF and 'dalpha_centers' in locals():
        # Log-log plot
        ax = axes[2]
        mask = ~np.isinf(dalpha_log_hist)
        ax.plot(dalpha_log_centers[mask], dalpha_log_hist[mask], 
                linestyle='None', marker='.', markersize=20, color='g', label='Log binning')
        
        # Add linear binning to log-log plot
        if dalpha_linear_centers is not None:
            mask_linear = (dalpha_linear_hist > 0)
            ax.plot(np.log10(dalpha_linear_centers[mask_linear]), np.log10(dalpha_linear_hist[mask_linear]), 
                    linestyle='None', marker='x', markersize=12, color='orange', label='Linear binning')
        
        if FIT_ALPHA_DIFF and 'dalpha_fit_params' in locals() and dalpha_fit_params is not None:
            x_fit = np.logspace(np.log10(dalpha_centers[0]), np.log10(dalpha_centers[-1]), 100)
            y_fit = truncated_powerlaw(x_fit, *dalpha_fit_params)
            ax.plot(np.log10(x_fit), np.log10(y_fit), 'r-', linewidth=2,
                    label=f'Fit: ε={dalpha_fit_params[1]:.2f}, λ={dalpha_fit_params[2]:.2e}')
        
        ax.set_xlabel('log₁₀(Δα)', fontsize=12)
        ax.set_ylabel('log₁₀(Density)', fontsize=12)
        ax.set_title('Δα Distribution (Log-Log)' + title_suffix, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Linear plot
        if dalpha_linear_centers is not None:
            ax = axes[3]
            ax.plot(dalpha_linear_centers, dalpha_linear_hist, 
                    linestyle='None', marker='.', markersize=20, color='g', label='Data')
            ax.set_xlabel('Δα', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title('Δα Distribution (Linear)' + title_suffix, fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
    
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = os.path.join(OUTPUT_DIR, f'all_distributions{filename_suffix}.png')
    plt.savefig(combined_plot_path, dpi=150)
    print(f"Saved combined plot: {combined_plot_path}")
    plt.show()
    
    # Save binned data to files
    print("\n=== Saving binned data ===")
    
    # Save energy
    energy_data_path = os.path.join(OUTPUT_DIR, f'data_histogram_energy{filename_suffix}.dat')
    with open(energy_data_path, 'w') as f:
        for x, y in zip(energy_log_centers, energy_log_hist):
            if not np.isinf(y):
                f.write(f"{x}, {y}\n")
    print(f"Saved: {energy_data_path}")
    
    # Save stress
    stress_data_path = os.path.join(OUTPUT_DIR, f'data_histogram_stress{filename_suffix}.dat')
    with open(stress_data_path, 'w') as f:
        for x, y in zip(stress_log_centers, stress_log_hist):
            if not np.isinf(y):
                f.write(f"{x}, {y}\n")
    print(f"Saved: {stress_data_path}")
    
    # Save alpha differences if analyzed
    if ANALYZE_ALPHA_DIFF and 'dalpha_log_centers' in locals():
        # Save logarithmic binning
        dalpha_data_path = os.path.join(OUTPUT_DIR, f'data_histogram_dalpha_log{filename_suffix}.dat')
        with open(dalpha_data_path, 'w') as f:
            for x, y in zip(dalpha_log_centers, dalpha_log_hist):
                if not np.isinf(y):
                    f.write(f"{x}, {y}\n")
        print(f"Saved: {dalpha_data_path}")
        
        # Save linear binning
        if dalpha_linear_centers is not None:
            dalpha_linear_data_path = os.path.join(OUTPUT_DIR, f'data_histogram_dalpha_linear{filename_suffix}.dat')
            with open(dalpha_linear_data_path, 'w') as f:
                for x, y in zip(dalpha_linear_centers, dalpha_linear_hist):
                    f.write(f"{x}, {y}\n")
            print(f"Saved: {dalpha_linear_data_path}")
    
    # Save fit parameters if available
    if FIT_DATA or (FIT_ALPHA_DIFF and ANALYZE_ALPHA_DIFF):
        if energy_popt is not None or stress_popt is not None or (ANALYZE_ALPHA_DIFF and 'dalpha_popt' in locals() and dalpha_popt is not None):
            fit_params_path = os.path.join(OUTPUT_DIR, f'fit_parameters{filename_suffix}.txt')
            with open(fit_params_path, 'w') as f:
                f.write("Truncated Power Law Fit: P(x) = A * x^(-epsilon) * exp(-lambda * x)\n")
                f.write(f"Fitting method: {FIT_METHOD}\n\n")
                
                if energy_popt is not None:
                    f.write("ENERGY:\n")
                    f.write(f"  A       = {energy_popt[0]:.6e}")
                    if energy_perr is not None:
                        f.write(f" ± {energy_perr[0]:.6e}\n")
                    else:
                        f.write("\n")
                    f.write(f"  epsilon = {energy_popt[1]:.6f}")
                    if energy_perr is not None:
                        f.write(f" ± {energy_perr[1]:.6f}\n")
                    else:
                        f.write("\n")
                    f.write(f"  lambda  = {energy_popt[2]:.6e}")
                    if energy_perr is not None:
                        f.write(f" ± {energy_perr[2]:.6e}\n\n")
                    else:
                        f.write("\n\n")
                
                if stress_popt is not None:
                    f.write("STRESS:\n")
                    f.write(f"  A       = {stress_popt[0]:.6e}")
                    if stress_perr is not None:
                        f.write(f" ± {stress_perr[0]:.6e}\n")
                    else:
                        f.write("\n")
                    f.write(f"  epsilon = {stress_popt[1]:.6f}")
                    if stress_perr is not None:
                        f.write(f" ± {stress_perr[1]:.6f}\n")
                    else:
                        f.write("\n")
                    f.write(f"  lambda  = {stress_popt[2]:.6e}")
                    if stress_perr is not None:
                        f.write(f" ± {stress_perr[2]:.6e}\n\n")
                    else:
                        f.write("\n\n")
                
                if ANALYZE_ALPHA_DIFF and 'dalpha_popt' in locals() and dalpha_popt is not None:
                    f.write("ALPHA DIFFERENCES (Δα) - Logarithmic:\n")
                    f.write(f"  A       = {dalpha_popt[0]:.6e}")
                    if dalpha_perr is not None:
                        f.write(f" ± {dalpha_perr[0]:.6e}\n")
                    else:
                        f.write("\n")
                    f.write(f"  epsilon = {dalpha_popt[1]:.6f}")
                    if dalpha_perr is not None:
                        f.write(f" ± {dalpha_perr[1]:.6f}\n")
                    else:
                        f.write("\n")
                    f.write(f"  lambda  = {dalpha_popt[2]:.6e}")
                    if dalpha_perr is not None:
                        f.write(f" ± {dalpha_perr[2]:.6e}\n")
                    else:
                        f.write("\n")
            print(f"Saved: {fit_params_path}")
    
    print(f"\nDone! All output files saved in {OUTPUT_DIR}/")