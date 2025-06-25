from dataclasses import dataclass

@dataclass
class DeviceCharacteristics:
    """Device characteristics to model the fidelity.

    Each attribute can either be a float, a tuple or a None value. If the attribute is a:
    * `float`: the value is fixed and will not be optimized
    * `tuple`: the value will be optimized over. The 2 elements in the tuple represent the bounds
        on the variable, which will be the search space for the optimization routine.
    * `None`: the value will not be optimized, and is not required for calculating any value downstream.

    Ti_j represent the natural decay timescales for the qubit `j`, in terms of the underlying gate time. For
    example, if X and CZ gates are applied, T1_2 will be the T1 time for qubit 2 in multiples of `tg_X + tg_CZ`.
    ZZ, ZI, IZ represent the overrotation error in the CZ gate (over the duration of the CZ gate) in terms of radians.
    """

    T1_1: float | tuple[float, float] | None
    T1_2: float | tuple[float, float] | None
    T2_1: float | tuple[float, float] | None
    T2_2: float | tuple[float, float] | None
    ZZ: float | tuple[float, float] | None
    ZI: float | tuple[float, float] | None
    IZ: float | tuple[float, float] | None
    infinite_time_spam: float | tuple[float, float] | None
    zero_time_spam: float | tuple[float, float] | None

    def __init__(self, device_dict: dict) -> None:
        for k, v in device_dict.items():
            self.__dict__[k] = v
    
    def get_not_none_keys(self) -> list[str]:
        """Get the values that are not None."""
        return [k for k, v in self.__dict__.items() if v is not None]
    
    def get_bounds_on_unknown_params(self) -> list[str]:
        """Get the values that are lists or tuples."""
        return [v for v in self.__dict__.values() if type(v) in [list, tuple]]

    def get_fixed_value_sub_dict(self) -> dict[str, float]:
        """Get the sub-dictionary with values that are not None."""
        return {k: v for k, v in self.__dict__.items() if v is not None and type(v) not in [list, tuple]}
    
    def get_optimizee_keys(self) -> list[str]:
        """Get the keys which represent parameters subject to optimization."""
        return [k for k, v in self.__dict__.items() if v is not None and type(v) in [list, tuple]]
    
    def update_dict(self, new_dict) -> None:
        self.__dict__.update(new_dict)
    
    def get_dict(self) -> dict[str, float | None]:
        return self.__dict__