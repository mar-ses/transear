"""
Package for taking care of **individual transit search algorithms,
meaning for a single target.

To contain:
- BLS search algorithms
- detection rankings and threshold determination - on the large scale
	products. In other words, handle the distributed transit search, but
	potentially do the ranking of the results in here (or at least
	have the skeleton algorithm to be use and utilities here).


Distribution is handled separately by k2gp_dist, or a dedicated
package.
"""

import os
import socket
#from . import k2gp, analysis
#from .k2gp import detrend

# K2GP FILE STRUCTURE
HOME_DIR = os.environ['HOME']
PACKAGES_DIR = HOME_DIR  # TODO: update when this changes
K2GP_DIR = '%s/k2gp'%HOME_DIR
K2GP_DATA = '%s/data'%K2GP_DIR
TRANSEAR_DIR = "{}/transear".format(HOME_DIR)

# PERSONAL DATA FILE STRUCTURE
LOCAL_DATA = "{}/data".format(HOME_DIR)
LOCAL_K2 = "{}/k2".format(LOCAL_DATA)

# UBELIX DATA FILE STRUCTURE
# TODO: switch to a home/data/... structure
# NOTE: MAYBE NOT - keep /data/ for very specific lightcurves
# the following mast files are actually project files, though perhaps
# simplify it (REMOVE mast_lightcurves, mast_tpfs)
UB_DATA_DIR = "{}/data".format(HOME_DIR)		# TODO: redundant with above
UB_MANUAL = "{}/k2_manual_lightcurves".format(HOME_DIR)
UB_MAST_LCF = "{}/k2_mast_lightcurves".format(HOME_DIR)
UB_MAST_TPF = "{}/k2_mast_tpfs".format(HOME_DIR)
UB_DETRENDED = "{}/k2_detrended".format(HOME_DIR)

# MAST VARIABLES
LATEST_CAMPAIGN = 18		# the latest fully calibrated campaign.

# Add this to BKJD to turn it into BJD - 2450000
BKJD_REF = 2454833-2450000

# Choose which to use as DATA_DIR
pc_name = socket.gethostname()
if 'ubelix' in pc_name:
	DATA_DIR = UB_DATA_DIR
elif pc_name in ('telesto', 'Marko-PC'):
	DATA_DIR = LOCAL_DATA
else:
	DATA_DIR = LOCAL_DATA	# just an assumption
	