import numpy as np
from components import DeviceCharacteristics
from parameter_optimization import fit_parameters_scipy_minimize, fit_parameters_curve_fit, fit_parameters_global_optimization, fit_parameters_least_squares
from typing import Callable, Literal

ExpType = Literal["00", "01", "10", "11", "0+", "+0", "++"]

def _update_in_position(kwargs: dict[str, float | None], params: tuple[float, ...]) -> dict[str, float]:
    """Update the kwargs in the order of the params.
    
    This works because python stores dicts in insertion order.
    """
    counter = 0
    for k,v in kwargs.items():
        if v is None:
            try:
                kwargs[k] = params[counter]
                counter = counter + 1
            except IndexError as e:
                raise ValueError(f"Not enough parameters provided") from e

    # check if all parameters were used
    if len(params) > counter:
        raise ValueError(f"Too many parameters: {params}")

    # check if all parameters are set in the kwargs
    for k,v in kwargs.items():
        if v is None:
            raise ValueError(f"Missing parameter: {k}")

    return kwargs

def _get_func(func_type: ExpType) -> Callable[..., float]:
    # do some special casing if required
    match func_type:
        case "11":
            return spam_curve_for_11
        case "0+":
            return spam_curve_for_0P
        case "+0":
            return spam_curve_for_P0
        case "++":
            return spam_curve_for_PP
    raise ValueError(f"Unsupported function type: {func_type}")

def ideal_curve_for_11(n, evolution_time, device_characteristics: DeviceCharacteristics):
    """Ideal fidelity value if the only coherent errors in the CZ gate are ZI, IZ, ZZ and the
    initial state is 11.
    """
    decay_factor = np.exp(-evolution_time/device_characteristics.T1_1);
    decay_factor = decay_factor * np.exp(-evolution_time/device_characteristics.T1_2);
    decay_factor = decay_factor ** (4 * n)
    return decay_factor;

def ideal_curve_for_0P(n, evolution_time, device_characteristics: DeviceCharacteristics):
    """Ideal fidelity value if the only coherent errors in the CZ gate are ZI, IZ, ZZ and the
    initial state is 0+.
    Under an ideal XI DD-sequence, the ZZ, ZI components in the CZ Hamiltonian should vanish and to do
    that, we should set ZZ, ZI values to -np.pi/4.
    """
    decay_factor = (np.exp(-evolution_time/device_characteristics.T2_2)) ** (4 * n);
    return 0.5 * (1 + np.cos(8 * (device_characteristics.ZZ + device_characteristics.IZ) * n) * decay_factor);

def ideal_curve_for_P0(n, evolution_time, device_characteristics: DeviceCharacteristics):
    """Ideal fidelity value if the only coherent errors in the CZ gate are ZI, IZ, ZZ and the
    initial state is +0.
    Under an ideal IX DD-sequence, the ZZ, IZ components in the CZ Hamiltonian should vanish and to do
    that, we should set ZZ, IZ values to -np.pi/4.
    """
    decay_factor = (np.exp(-evolution_time/device_characteristics.T2_1)) ** (4 * n);
    return 0.5 * (1 + np.cos(8 * (device_characteristics.ZZ + device_characteristics.ZI) * n) * decay_factor);

def ideal_curve_for_PP(n, evolution_time, device_characteristics: DeviceCharacteristics):
    """Ideal fidelity value if the only coherent errors in the CZ gate are ZI, IZ, ZZ and the
    initial state is ++.
    Under an ideal XX DD-sequence, the ZI and IZ components in the CZ Hamiltonian should vanish and to do
    that, we should set ZI, IZ values to -np.pi/4.
    """
    d1 = (np.exp(-evolution_time/device_characteristics.T2_1)) ** (4);
    d2 = (np.exp(-evolution_time/device_characteristics.T2_2)) ** (4);
    r1 = (np.exp(-evolution_time/device_characteristics.T1_1)) ** (4);
    r2 = (np.exp(-evolution_time/device_characteristics.T1_2)) ** (4);
    
    eps = device_characteristics.ZI;
    kap = device_characteristics.IZ;
    eta = device_characteristics.ZZ;
    
    return 1/16 * (4 + 
        2 * (d1 ** n) * np.cos(8 * (eps + eta) * n) + 
        2 * (d2 ** n) * np.cos(8 * (kap + eta) * n) + 
        2 * (d1 ** n) * (d2 ** n) * np.cos(8 * (eps - kap) * n) + 
        2 * (d1 ** n) * (d2 ** n) * np.cos(8 * (eps + kap) * n) +
        2 * (d2 ** n) * (r1 ** n) * np.cos(8 * (eta - kap) * n) +
        2 * (d1 ** n) * (r2 ** n) * np.cos(8 * (eta - eps) * n) + 
        2 * np.real((-1 + r1) * (d2 ** n) * (np.exp(8j * (eta + kap) * n) *  - np.exp(-8j * (eta - kap) * n) * (r1 ** n)) / (np.exp(4j * eta) + r1)) +
        2 * np.real((-1 + r2) * (d1 ** n) * (np.exp(8j * (eta + eps) * n) *  - np.exp(-8j * (eta - eps) * n) * (r2 ** n)) / (np.exp(4j * eta) + r2))
        )

def spam_curve_for_11(
    n : np.ndarray,
    device_characteristics : DeviceCharacteristics):
    # spam variables lie between 0 and 1
    # (A + B * e^(-4 * n * t / T_D)
    ideal_fidelity = ideal_curve_for_11(n, 1, device_characteristics);
    return device_characteristics.infinite_time_spam + device_characteristics.zero_time_spam * (ideal_fidelity - 0.5) / 0.5;

def spam_curve_for_0P(
    n : np.ndarray,
    device_characteristics : DeviceCharacteristics):
    # spam variables lie between 0 and 1
    # (A + B * e^(-4 * n * t / T_D) * Cos(8 * n * theta))/2
    ideal_fidelity = ideal_curve_for_0P(n, 1, device_characteristics);
    return device_characteristics.infinite_time_spam + device_characteristics.zero_time_spam * (ideal_fidelity - 0.5) / 0.5;

def spam_curve_for_P0(
    n : np.ndarray,
    device_characteristics : DeviceCharacteristics):
    # spam variables lie between 0 and 1
    # (A + B * e^(-4 * n * t / T_D) * Cos(8 * n * theta))/2
    ideal_fidelity = ideal_curve_for_P0(n, 1, device_characteristics);
    return device_characteristics.infinite_time_spam + device_characteristics.zero_time_spam * (ideal_fidelity - 0.5) / 0.5;

def spam_curve_for_PP(
    n : np.ndarray,
    device_characteristics : DeviceCharacteristics):
    # spam variables lie between 0 and 1
    # ideal zero_time_spam = 1/16, infinite_time_spam = 4/16
    ideal_fidelity = ideal_curve_for_PP(n, 1, device_characteristics);
    return device_characteristics.infinite_time_spam + device_characteristics.zero_time_spam * (ideal_fidelity - 4 / 16) * 16;

class OptimizableFunction:
    """Generic class for holding an optimizable function.

    Could've been implemented just using methods too instead of a class.
    But holding known values across calls seems like a useful thing in my mind + can 
    manage state if required.

    Entrypoint is __call__
    """
    def __init__(self, func_type: ExpType, device_characteristics: DeviceCharacteristics):
        self._func_type = func_type
        self._func = _get_func(func_type)
        self._known_param_dict = device_characteristics.get_fixed_value_sub_dict()
        self._dict_keys = device_characteristics.get_not_none_keys()

    def _initialize_dictionary(self) -> dict[str, float | None]:
        return {param: None for param in self._dict_keys}

    def _prepare_dictionary(self, params: tuple[float, ...]) -> dict[str, float]:
        """Prepare kwargs for the function call.
        
        A bit conflicted on the implementation because this substitutes visible 
        complexity for invisble (but slightly less verbose/complex) complexity. Can
        discuss more on getting feedback.
        """
        kwargs = self._initialize_dictionary()
        # update with known values first
        kwargs.update(self._known_param_dict)
        # then update the remainder with scipy provided positional arguments
        return _update_in_position(kwargs, params)

    def __call__(self, x, *params: float) -> float:
        """Caller.
        
        `params` are the guesses for the unknown params."""
        kwargs = self._prepare_dictionary(params)
        new_device_characteristics = DeviceCharacteristics(kwargs)
        # print(new_device_characteristics.get_dict())
        return self._func(x, new_device_characteristics)

def find_best_fit(
    func_type: ExpType, 
    x: np.ndarray, 
    y: np.ndarray, 
    device_characteristics: DeviceCharacteristics, 
    initial_guess: np.ndarray,
    opt: Literal["global", "minimize", "curve", "least_squares"] = "global"
    ):
    """Find the best fit for the given function type.

    func_type takes in the experiment type, i.e., the initial state.

    device_characteristics: DeviceCharacteristics instance where:
        * float values represent fixed variables
        * tuple values represent optimizable variables
        * None values represent non-optimized, unknown variables
    initial_guess: initial guess for the parameters to optimize, if using a non-global
        optimization method. The length of the array should match the number of tuples
        in the `device_characteristics` dictionary.
    opt: Underlying optimization method to use.
    """
    optimizable_func = OptimizableFunction(func_type, device_characteristics)
    bounds = device_characteristics.get_bounds_on_unknown_params();
    match opt:
        case "global":
            optim_params, objective_value, result = fit_parameters_global_optimization(
                optimizable_func, 
                x, y,
                bounds=bounds
            )
        case "minimize":
            optim_params, objective_value, result = fit_parameters_scipy_minimize(
                optimizable_func, 
                x, y,
                initial_guess,
                bounds=bounds
            )
        case "curve":
            optim_params, objective_value, result = fit_parameters_curve_fit(
                optimizable_func, 
                x, y,
                initial_guess,
                bounds=bounds
            )
        case "least_squares":
            optim_params, objective_value, result = fit_parameters_least_squares(
                optimizable_func, 
                x, y,
                initial_guess,
                bounds=bounds
            )
    return optim_params, objective_value, result, optimizable_func

    