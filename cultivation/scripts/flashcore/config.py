"""
Configuration and constants for flashcore.
"""

# Paths and FSRS defaults will be added here.
# cultivation/flashcore/config.py

# Default FSRS parameters (weights 'w')
# Sourced from: py-fsrs library (specifically fsrs.scheduler.DEFAULT_PARAMETERS)
# These parameters are used by the FSRS algorithm to schedule card reviews.
# Each parameter influences a specific aspect of the memory model.
# For detailed explanations of each parameter, refer to FSRS documentation and the optimizer source.
DEFAULT_PARAMETERS = (
    0.2172,  # w[0]
    1.1771,  # w[1]
    3.2602,  # w[2]
    16.1507, # w[3]
    7.0114,  # w[4]
    0.57,    # w[5]
    2.0966,  # w[6]
    0.0069,  # w[7]
    1.5261,  # w[8]
    0.112,   # w[9]
    1.0178,  # w[10]
    1.849,   # w[11]
    0.1133,  # w[12]
    0.3127,  # w[13]
    2.2934,  # w[14]
    0.2191,  # w[15]
    3.0004,  # w[16]
    0.7536,  # w[17]
    0.3332,  # w[18]
    0.1437,  # w[19]
    0.2,     # w[20]
)
# Sourced from: https://github.com/open-spaced-repetition/py-fsrs/blob/main/fsrs/scheduler.py (DEFAULT_PARAMETERS)

# It's also good practice to define any other scheduler-related constants here.
# For example, default desired retention rate if not specified elsewhere.
DEFAULT_DESIRED_RETENTION = 0.9
