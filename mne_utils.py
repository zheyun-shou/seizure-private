"""
Utility functions for configuring MNE verbosity and other MNE-related settings.
"""

import mne
import warnings

def configure_mne_verbosity(verbose=False):
    """
    Configure MNE verbosity settings globally.
    This function is safe to call in multiprocessing workers.
    
    Args:
        verbose (bool): If False, disable all MNE output. If True, use default MNE verbosity.
    """
    if not verbose:
        # Use a simpler approach that works with multiprocessing
        mne.set_log_level('ERROR')
        # Disable warnings that might be annoying
        warnings.filterwarnings('ignore', category=UserWarning, module='mne')
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')
    else:
        # Use default MNE verbosity
        mne.set_log_level('INFO')

def setup_mne_for_processing(verbose=False):
    """
    Setup MNE for processing with proper verbosity settings.
    This function should be called at the beginning of any script that uses MNE.
    
    Args:
        verbose (bool): If False, disable all MNE output. If True, use default MNE verbosity.
    """
    configure_mne_verbosity(verbose) 