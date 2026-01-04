"""Primitive set definition for Genetic Programming."""

import math
import operator
import random as random_module

from deap import gp


def _random_constant() -> float:
    """Generate random constant for ephemeral terminals."""
    return round(random_module.uniform(-1, 1), 3)


def protected_div(left: float, right: float) -> float:
    """Protected division that returns 1.0 on division by zero.
    
    Args:
        left: Numerator.
        right: Denominator.
        
    Returns:
        left/right or 1.0 if right is close to zero.
    """
    if abs(right) < 1e-6:
        return 1.0
    return left / right


def protected_log(x: float) -> float:
    """Protected logarithm that handles non-positive values.
    
    Args:
        x: Input value.
        
    Returns:
        log(abs(x)) or 0.0 if x is close to zero.
    """
    if abs(x) < 1e-6:
        return 0.0
    return math.log(abs(x))


def protected_sqrt(x: float) -> float:
    """Protected square root that handles negative values.
    
    Args:
        x: Input value.
        
    Returns:
        sqrt(abs(x)).
    """
    return math.sqrt(abs(x))


def protected_exp(x: float) -> float:
    """Protected exponential that prevents overflow.
    
    Args:
        x: Input value.
        
    Returns:
        exp(x) clamped to prevent overflow.
    """
    try:
        return math.exp(min(x, 100))  # Clamp to prevent overflow
    except OverflowError:
        return 1e43


def if_then_else(condition: float, true_val: float, false_val: float) -> float:
    """Conditional operator for GP trees.
    
    Args:
        condition: If > 0, return true_val, else false_val.
        true_val: Value to return if condition > 0.
        false_val: Value to return otherwise.
        
    Returns:
        true_val or false_val based on condition.
    """
    return true_val if condition > 0 else false_val


def neg(x: float) -> float:
    """Negation operator."""
    return -x


def create_primitive_set(n_features: int, feature_names: list[str] | None = None) -> gp.PrimitiveSet:
    """Create a DEAP primitive set for GP classification.
    
    Args:
        n_features: Number of input features.
        feature_names: Optional list of feature names for terminals.
        
    Returns:
        Configured PrimitiveSet for genetic programming.
    """
    # Create primitive set with n_features input arguments
    pset = gp.PrimitiveSet("MAIN", n_features)
    
    # Arithmetic operators
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addPrimitive(protected_div, 2, name="div")
    pset.addPrimitive(neg, 1, name="neg")
    
    # Mathematical functions
    pset.addPrimitive(protected_sqrt, 1, name="sqrt")
    pset.addPrimitive(protected_exp, 1, name="exp")
    pset.addPrimitive(math.sin, 1, name="sin")
    pset.addPrimitive(math.cos, 1, name="cos")
    pset.addPrimitive(abs, 1, name="abs")
    
    # Conditional
    pset.addPrimitive(if_then_else, 3, name="if_then_else")
    
    # Comparison (return 1.0 or -1.0 for true/false)
    pset.addPrimitive(lambda x, y: 1.0 if x > y else -1.0, 2, name="gt")
    pset.addPrimitive(lambda x, y: 1.0 if x < y else -1.0, 2, name="lt")
    
    # Ephemeral constants (random values that become fixed in individuals)
    # Use named function instead of lambda for pickle compatibility
    pset.addEphemeralConstant("rand_const", _random_constant)
    
    # Rename arguments to feature names if provided
    if feature_names:
        for i, name in enumerate(feature_names[:n_features]):
            # Sanitize name for DEAP (no special chars)
            safe_name = name.replace("-", "_").replace(".", "_")
            pset.renameArguments(**{f"ARG{i}": safe_name})
    
    return pset


def get_primitive_info() -> dict:
    """Get information about available primitives.
    
    Returns:
        Dictionary with primitive categories and their operators.
    """
    return {
        "arithmetic": ["add", "sub", "mul", "div", "neg"],
        "mathematical": ["sqrt", "exp", "sin", "cos", "abs"],
        "conditional": ["if_then_else"],
        "comparison": ["gt (>)", "lt (<)"],
        "terminals": ["features", "ephemeral constants [-1, 1]"],
    }
