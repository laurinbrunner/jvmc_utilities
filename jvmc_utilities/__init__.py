from .operators import initialisation_operators, higher_order_M_T_inv, aqi_model_operators

try:
    from .version import __version__
except ImportError:
    __version__ = "0.0.0"