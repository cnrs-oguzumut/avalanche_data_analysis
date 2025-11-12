import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

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
    
    Parameters:
    -----------
    bin_centers : array
        Bin centers (x values)
    hist : array
        Histogram values (y values, density)
    method : str
        'curve_fit' or 'minimize'
    
    Returns:
    --------
    popt : array
        Optimal parameters [A, epsilon, lambda]
    perr : array
        Standard errors of parameters (or None for minimize method)
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
    # For power law: log(P) ≈ log(A) - epsilon * log(x)
    # Use linear fit on the first half of data (before cutoff dominates)
    n_fit = max(3, len(log_x_data) // 2)
    coeffs = np.polyfit(log_x_data[:n_fit], log_y_data[:n_fit], 1)
    epsilon_init = -coeffs[0]  # slope
    log_A_init = coeffs[1]      # intercept
    
    # Estimate lambda from the tail behavior
    # Look at how much the data deviates from pure power law
    if len(x_data) > 3:
        # Rough estimate: lambda ~ 1 / characteristic_scale
        characteristic_scale = np.median(x_data)
        lambda_init = 1.0 / characteristic_scale
    else:
        lambda_init = 1e-6
    
    print(f"Initial guess: log_A={log_A_init:.3f}, epsilon={epsilon_init:.3f}, lambda={lambda_init:.3e}")
    
    if method == 'curve_fit':
        try:
            # Fit in log-log space
            p0 = [log_A_init, epsilon_init, lambda_init]
            # Bounds: log_A in (-inf, inf), epsilon > 0, lambda >= 0
            bounds = ([-np.inf, 0.1, 0], [np.inf, 10, np.inf])
            
            popt_log, pcov = curve_fit(log_truncated_powerlaw, log_x_data, log_y_data, 
                                       p0=p0, bounds=bounds, maxfev=50000)
            
            # Convert back: A = 10^(log_A)
            popt = np.array([10**popt_log[0], popt_log[1], popt_log[2]])
            
            # Calculate uncertainties
            # For A, use error propagation: σ_A = A * ln(10) * σ_log_A
            perr_log = np.sqrt(np.diag(pcov))
            perr = np.array([
                popt[0] * np.log(10) * perr_log[0],  # A error
                perr_log[1],                          # epsilon error
                perr_log[2]                           # lambda error
            ])
            
            return popt, perr
            
        except Exception as e:
            print(f"curve_fit failed: {e}")
            print("Trying minimize method...")
            method = 'minimize'
    
    if method == 'minimize':
        try:
            # Define cost function (sum of squared residuals in log space)
            def cost_function(params):
                log_A, epsilon, lambda_ = params
                if epsilon <= 0 or lambda_ < 0:
                    return 1e10
                y_pred = log_truncated_powerlaw(log_x_data, log_A, epsilon, lambda_)
                residuals = log_y_data - y_pred
                return np.sum(residuals**2)
            
            # Initial guess
            x0 = [log_A_init, max(0.5, epsilon_init), max(1e-8, lambda_init)]
            
            # Bounds
            bounds = [(-np.inf, np.inf), (0.1, 10), (0, np.inf)]
            
            # Optimize
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
    
    # Use weights based on the counts (more weight to well-sampled bins)
    # In log space, equal weights often work well
    weights = np.ones_like(log_y_data)
    
    # Estimate initial parameters
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
    FILTER_BY_PLASTICITY = False  # Set to True to only use plasticity events (flag=1)
    FILTER_BY_ALPHA = True        # Set to True to filter by alpha range
    ALPHA_MIN = 0.0               # Minimum alpha value (inclusive)
    ALPHA_MAX = 2.               # Maximum alpha value (inclusive)
    
    FILTER_BY_XMIN = True         # Set to True to filter by minimum energy/stress values
    ENERGY_XMIN = 1e-3            # Minimum energy change value (only data >= this value)
    STRESS_XMIN = 20            # Minimum stress change value (only data >= this value)
    
    FILTER_BY_XMAX = True         # Set to True to filter by maximum energy/stress values
    ENERGY_XMAX = 1000            # Maximum energy change value (only data <= this value)
    STRESS_XMAX = 10000           # Maximum stress change value (only data <= this value)
    
    FIT_DATA = True               # Set to True to fit truncated power law
    FIT_METHOD = 'logspace'       # 'logspace', 'weighted', or 'both'
    
    nbin = 12
    # =========================
    
    # Read the data
    alpha, energy_change, stress_change, plasticity_flag = read_energy_stress_log()
    
    print(f"Loaded {len(alpha)} data points")
    print(f"Alpha range in data: [{np.min(alpha):.6e}, {np.max(alpha):.6e}]")
    print(f"Plasticity events: {np.sum(plasticity_flag)} / {len(plasticity_flag)}")
    
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
    
    energy_change = energy_change[mask]
    stress_change = stress_change[mask]
    alpha = alpha[mask]
    
    print(f"\nTotal data points after all filters: {len(alpha)}")
    
    if len(alpha) == 0:
        print("ERROR: No data points remaining after filtering!")
        exit(1)
    
    print(f"Alpha range after filtering: [{np.min(alpha):.6e}, {np.max(alpha):.6e}]")
    
    # Filter for positive values
    positive_energy_mask = energy_change > 0
    positive_stress_mask = stress_change > 0
    
    print(f"Positive energy changes: {np.sum(positive_energy_mask)} / {len(energy_change)}")
    print(f"Positive stress changes: {np.sum(positive_stress_mask)} / {len(stress_change)}")
    
    energy_data = energy_change[positive_energy_mask]
    stress_data = stress_change[positive_stress_mask]
    
    if FILTER_BY_XMIN:
        print(f"\n*** FILTERING XMIN: Energy >= {ENERGY_XMIN:.6e}, Stress >= {STRESS_XMIN:.6e} ***")
        print(f"Energy data points before xmin filter: {len(energy_data)}")
        print(f"Stress data points before xmin filter: {len(stress_data)}")
        
        energy_data = energy_data[energy_data >= ENERGY_XMIN]
        stress_data = stress_data[stress_data >= STRESS_XMIN]
        
        print(f"Energy data points after xmin filter: {len(energy_data)}")
        print(f"Stress data points after xmin filter: {len(stress_data)}")
    
    if FILTER_BY_XMAX:
        print(f"\n*** FILTERING XMAX: Energy <= {ENERGY_XMAX:.6e}, Stress <= {STRESS_XMAX:.6e} ***")
        print(f"Energy data points before xmax filter: {len(energy_data)}")
        print(f"Stress data points before xmax filter: {len(stress_data)}")
        
        energy_data = energy_data[energy_data <= ENERGY_XMAX]
        stress_data = stress_data[stress_data <= STRESS_XMAX]
        
        print(f"Energy data points after xmax filter: {len(energy_data)}")
        print(f"Stress data points after xmax filter: {len(stress_data)}")
    
    if len(energy_data) == 0 or len(stress_data) == 0:
        print("ERROR: No positive data points to analyze!")
        exit(1)
    
    energy_xmin = np.min(energy_data)
    energy_xmax = np.max(energy_data)
    stress_xmin = np.min(stress_data)
    stress_xmax = np.max(stress_data)
    
    print(f"\nFinal energy range: [{energy_xmin:.6e}, {energy_xmax:.6e}]")
    print(f"Final stress range: [{stress_xmin:.6e}, {stress_xmax:.6e}]")
    
    # Binning
    print("\n=== Energy Change Binning ===")
    energy_centers, energy_hist, energy_log_centers, energy_log_hist, energy_dx = logarithmic_binning(energy_data, nbin=nbin)
    print(f"Number of bins: {len(energy_centers)}")
    
    print("\n=== Stress Change Binning ===")
    stress_centers, stress_hist, stress_log_centers, stress_log_hist, stress_dx = logarithmic_binning(stress_data, nbin=nbin)
    print(f"Number of bins: {len(stress_centers)}")
    
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
                print(f"  A       = {energy_popt[0]:.6e} ± {energy_perr[0]:.6e}")
                print(f"  epsilon = {energy_popt[1]:.4f} ± {energy_perr[1]:.4f}")
                print(f"  lambda  = {energy_popt[2]:.6e} ± {energy_perr[2]:.6e}")
        
        if FIT_METHOD in ['weighted', 'both']:
            print("\nMethod: Weighted fitting")
            energy_popt2, energy_perr2 = fit_truncated_powerlaw_with_weights(energy_centers, energy_hist)
            if energy_popt2 is not None:
                print(f"\nEnergy fit parameters (weighted):")
                print(f"  A       = {energy_popt2[0]:.6e} ± {energy_perr2[0]:.6e}")
                print(f"  epsilon = {energy_popt2[1]:.4f} ± {energy_perr2[1]:.4f}")
                print(f"  lambda  = {energy_popt2[2]:.6e} ± {energy_perr2[2]:.6e}")
                # Use weighted fit as default if both methods used
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
                print(f"  A       = {stress_popt[0]:.6e} ± {stress_perr[0]:.6e}")
                print(f"  epsilon = {stress_popt[1]:.4f} ± {stress_perr[1]:.4f}")
                print(f"  lambda  = {stress_popt[2]:.6e} ± {stress_perr[2]:.6e}")
        
        if FIT_METHOD in ['weighted', 'both']:
            print("\nMethod: Weighted fitting")
            stress_popt2, stress_perr2 = fit_truncated_powerlaw_with_weights(stress_centers, stress_hist)
            if stress_popt2 is not None:
                print(f"\nStress fit parameters (weighted):")
                print(f"  A       = {stress_popt2[0]:.6e} ± {stress_perr2[0]:.6e}")
                print(f"  epsilon = {stress_popt2[1]:.4f} ± {stress_perr2[1]:.4f}")
                print(f"  lambda  = {stress_popt2[2]:.6e} ± {stress_perr2[2]:.6e}")
                if FIT_METHOD == 'both' and stress_popt is None:
                    stress_fit_params = stress_popt2
                    stress_popt = stress_popt2
                    stress_perr = stress_perr2
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    title_suffix = ""
    if FILTER_BY_PLASTICITY:
        title_suffix += " (Plasticity)"
    if FILTER_BY_ALPHA:
        title_suffix += f" (α∈[{ALPHA_MIN},{ALPHA_MAX}])"
    if FILTER_BY_XMIN or FILTER_BY_XMAX:
        title_suffix += " (x filtered)"
    
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
    
    plt.tight_layout()
    
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
    
    plt.savefig(f'energy_stress_distributions_log10{filename_suffix}.png', dpi=150)
    plt.show()
    
    # Save data and fit parameters
    print("\n=== Saving binned data ===")
    
    with open(f'data_histogram_energy{filename_suffix}.dat', 'w') as f:
        for x, y in zip(energy_log_centers, energy_log_hist):
            if not np.isinf(y):
                f.write(f"{x}, {y}\n")
    
    with open(f'data_histogram_stress{filename_suffix}.dat', 'w') as f:
        for x, y in zip(stress_log_centers, stress_log_hist):
            if not np.isinf(y):
                f.write(f"{x}, {y}\n")
    
    if FIT_DATA:
        with open(f'fit_parameters{filename_suffix}.txt', 'w') as f:
            f.write("Truncated Power Law Fit: P(x) = A * x^(-epsilon) * exp(-lambda * x)\n")
            f.write(f"Fitting method: {FIT_METHOD}\n\n")
            if energy_popt is not None:
                f.write("ENERGY:\n")
                f.write(f"  A       = {energy_popt[0]:.6e} ± {energy_perr[0]:.6e}\n")
                f.write(f"  epsilon = {energy_popt[1]:.6f} ± {energy_perr[1]:.6f}\n")
                f.write(f"  lambda  = {energy_popt[2]:.6e} ± {energy_perr[2]:.6e}\n\n")
            if stress_popt is not None:
                f.write("STRESS:\n")
                f.write(f"  A       = {stress_popt[0]:.6e} ± {stress_perr[0]:.6e}\n")
                f.write(f"  epsilon = {stress_popt[1]:.6f} ± {stress_perr[1]:.6f}\n")
                f.write(f"  lambda  = {stress_popt[2]:.6e} ± {stress_perr[2]:.6e}\n")
    
    print(f"Done! Saved files with suffix: {filename_suffix}")