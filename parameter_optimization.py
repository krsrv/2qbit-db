import numpy as np
from scipy.optimize import minimize, least_squares, differential_evolution
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def calculate_rmse(predictions, targets):
    """
    Calculate Root Mean Square Error between predictions and targets
    """
    return np.sqrt(np.mean((predictions - targets)**2))

def objective_function(params, func, x_data, y_data):
    """
    Objective function to minimize: RMSE between function predictions and data
    """
    predictions = func(x_data, *params)
    return calculate_rmse(predictions, y_data)

def fit_parameters_scipy_minimize(func, x_data, y_data, initial_guess, bounds=None):
    """
    Fit parameters using scipy.optimize.minimize
    
    Parameters:
    func: function that takes (x, *params) and returns predictions
    x_data: input data
    y_data: target data
    initial_guess: initial parameter values
    bounds: optional bounds for parameters [(min1, max1), (min2, max2), ...]
    """
    
    # Define objective function
    obj_func = lambda params: objective_function(params, func, x_data, y_data)
    
    # Perform optimization
    if bounds is not None:
        result = minimize(obj_func, initial_guess, bounds=bounds, method='L-BFGS-B')
    else:
        result = minimize(obj_func, initial_guess, method='Nelder-Mead')
    
    return result.x, result.fun, result

def fit_parameters_least_squares(func, x_data, y_data, initial_guess, bounds=None):
    """
    Fit parameters using scipy.optimize.least_squares
    
    Parameters:
    func: function that takes (x, *params) and returns predictions
    x_data: input data
    y_data: target data
    initial_guess: initial parameter values
    bounds: optional bounds for parameters [(min1, max1), (min2, max2), ...]
    """
    
    # Define residual function
    def residuals(params):
        predictions = func(x_data, *params)
        return predictions - y_data
    
    # Perform optimization
    if bounds is not None:
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
        result = least_squares(residuals, initial_guess, bounds=(lb, ub))
    else:
        result = least_squares(residuals, initial_guess)
    
    # Calculate RMSE
    rmse = calculate_rmse(func(x_data, *result.x), y_data)
    
    return result.x, rmse, result

def fit_parameters_curve_fit(func, x_data, y_data, initial_guess, bounds=None, y_err=None):
    """
    Fit parameters using scipy.optimize.curve_fit
    
    Parameters:
    func: function that takes (x, *params) and returns predictions
    x_data: input data
    y_data: target data
    initial_guess: initial parameter values (p0)
    bounds: optional bounds for parameters [(min1, max1), (min2, max2), ...]
    """
    
    try:
        if bounds is not None:
            lb = [b[0] for b in bounds]
            ub = [b[1] for b in bounds]
            popt, pcov = curve_fit(func, x_data, y_data, p0=initial_guess, bounds=(lb, ub), sigma=y_err)
        else:
            popt, pcov = curve_fit(func, x_data, y_data, p0=initial_guess, sigma=y_err)
        
        # Calculate RMSE
        predictions = func(x_data, *popt)
        rmse = calculate_rmse(predictions, y_data)
        
        return popt, rmse, pcov
    
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        return None, None, None

def fit_parameters_global_optimization(func, x_data, y_data, bounds = None):
    """
    Fit parameters using global optimization (differential evolution)
    
    Parameters:
    func: function that takes (x, *params) and returns predictions
    x_data: input data
    y_data: target data
    bounds: bounds for parameters [(min1, max1), (min2, max2), ...]
    """
    
    obj_func = lambda params: objective_function(params, func, x_data, y_data)
    
    result = differential_evolution(obj_func, bounds, maxiter=1000, popsize=15)
    
    return result.x, result.fun, result

# Example usage and testing
if __name__ == "__main__":
    # Example function: exponential decay with noise
    def example_func(x, a, b, c):
        """Example function: a * exp(-b * x) + c"""
        return a * np.exp(-b * x) + c
    
    # Generate synthetic data
    np.random.seed(42)
    x_data = np.linspace(0, 10, 50)
    true_params = [2.0, 0.5, 0.1]  # a=2, b=0.5, c=0.1
    y_data = example_func(x_data, *true_params) + 0.1 * np.random.normal(size=len(x_data))
    
    # Initial guess and bounds
    initial_guess = [1.0, 1.0, 0.0]
    bounds = [(0, 5), (0, 2), (-1, 1)]
    
    print("True parameters:", true_params)
    print("\n" + "="*50)
    
    # Test different optimization methods
    methods = [
        ("scipy.minimize", fit_parameters_scipy_minimize),
        ("least_squares", fit_parameters_least_squares),
        ("curve_fit", fit_parameters_curve_fit),
        ("global_optimization", fit_parameters_global_optimization)
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        print(f"\n{method_name.upper()}:")
        try:
            if method_name == "global_optimization":
                params, rmse, _ = method_func(example_func, x_data, y_data, bounds)
            else:
                params, rmse, _ = method_func(example_func, x_data, y_data, initial_guess, bounds)
            
            print(f"  Optimal parameters: {params}")
            print(f"  RMSE: {rmse:.6f}")
            results[method_name] = (params, rmse)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot data
    plt.subplot(2, 2, 1)
    plt.scatter(x_data, y_data, alpha=0.6, label='Data', color='black')
    plt.plot(x_data, example_func(x_data, *true_params), 'r-', label='True function', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data and True Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot fits from different methods
    colors = ['blue', 'green', 'orange', 'purple']
    for i, (method_name, (params, rmse)) in enumerate(results.items()):
        plt.subplot(2, 2, i+2)
        plt.scatter(x_data, y_data, alpha=0.6, label='Data', color='black')
        plt.plot(x_data, example_func(x_data, *params), color=colors[i], 
                label=f'{method_name} (RMSE: {rmse:.4f})', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{method_name.upper()} Fit')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    for method_name, (params, rmse) in results.items():
        print(f"{method_name}: RMSE = {rmse:.6f}, Params = {params}") 