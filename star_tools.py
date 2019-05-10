"""Library for functions and tools for dealing with stellar parameters."""

import re
import numpy as np
import pandas as pd

from .__init__ import HOME_DIR, K2GP_DATA

catalog_path_m = "{}/data/catalogs/list_m6-m9_dwarfs_gagne.csv".format(HOME_DIR)
catalog_path_uc = "{}/data/catalogs/list_ultracool_brown_dwarfs_gagne.csv".format(HOME_DIR)

# Catalogue parsing
# -----------------

def get_catalog():
	catalog_md = pd.read_csv(catalog_path_m)
	catalog_ucd = pd.read_csv(catalog_path_uc)

	# Flag source
	catalog_md['catalog_source'] = 'm_dwarf'
	catalog_ucd['catalog_source'] = 'b_dwarf'
	catalog_md = catalog_md.rename(columns={'μα':'μα (mas/yr)',
											'μδ':'μδ (mas/yr)',
											'D_Trig':'D_Trig (pc)'})

	catalog = pd.concat([catalog_md, catalog_ucd],
						ignore_index=True,
						sort=False)
	catalog.rename(columns={'Decl. (deg)':'DEC', 'R.A. (deg)':'RA'},
				   inplace=True)
	return catalog

def get_tl(clean=True):
	tl = pd.read_csv(
		"{}/aggregated_target_lists/aggregated_ultracool.csv".format(K2GP_DATA)
	)
	tl['RA'] = pd.to_numeric(tl.RA, errors='coerce')
	tl['DEC'] = pd.to_numeric(tl.DEC, errors='coerce')

	if clean:
		tl = tl[~tl.RA.isnull()]
	return tl

def match_epic(ra, dec, catalog):
	"""Matches the closest star in the catalog.

	Args:
		ra, dec (floats)
		catalog (pd.DataFrame): with RA and REC columns

	Returns:
		catalog_index, distance.
	"""

	ra_diff = catalog.RA - ra
	dec_diff = catalog.DEC - dec

	dist = np.sqrt(ra_diff**2 + dec_diff**2)
	i = dist.idxmin()			# dist must be a Series not np.array
	return i, dist[i]


def enhance_target_list(save=False, tl_loc=None):

	catalog = get_catalog()
	
	if tl_loc is None:
		tl = get_tl()
	else:
		tl = pd.read_pickle(tl_loc)

	add_cols = ['2MASS-J', '2MASS-H', '2MASS-K', 'WISE-W1', 'WISE-W2',
				'WISE-W3', 'WISE-W4', 'SDSS-u’', 'SDSS-g’',
				'SDSS-r’', 'SDSS-i’', 'SDSS-z’', 'UCAC4-R', 'UCAC4-I',
				'μα (mas/yr)', 'μδ (mas/yr)']

	tl = tl.assign(**{col:np.nan for col in add_cols})

	# Force cast the types
	tl = tl.assign(cat_found=np.zeros_like(tl.epic, dtype=bool))
	tl = tl.assign(cat_index=np.zeros_like(tl.epic, dtype=int))

	for i in tl.index:
		targ = tl.loc[i]
		cidx, dist = match_epic(targ.RA, targ.DEC, catalog)
		tl.loc[i, 'cat_index'] = cidx
		tl.loc[i, 'cat_dist'] = dist
		tl.loc[i, 'cat_pixdist'] = dist / (0.000278 * 3.98)
		tl.loc[i, 'cat_found'] = tl.loc[i, 'cat_pixdist'] < 5

		cati = catalog.loc[cidx]
		if cati['SpT_OPT'] not in (np.nan, '...', ''):
			spt = cati['SpT_OPT']
		elif cati['SpT_NIR'] not in (np.nan, '...', ''):
			spt = cati['SpT_NIR']
		else:
			spt = np.nan		

		tl.loc[i, 'cat_spectral_type'] = spt

		tl.loc[i, add_cols] = cati[add_cols]

	tl = read_spectral_type(tl)

	tl.rename(columns={'μα (mas/yr)':'cat_mua', 'μδ (mas/yr)':'cat_mud'},
			  inplace=True)

	if save:
		if tl_loc is None:
			tl.to_csv("{}/aggregated_target_lists/"
					  "aggregated_ultracool.csv".format(K2GP_DATA),
					  index=False)
		else:
			tl.to_pickle(tl_loc)

	return tl

def read_spectral_type(df):
	# Columns stellar_class
	#		  stellar_number

	for i in df.index:
		if 'M' in df.loc[i, 'cat_spectral_type']:
			df.loc[i, 'stellar_class'] = 'M'
		elif 'L' in df.loc[i, 'cat_spectral_type']:
			df.loc[i, 'stellar_class'] = 'L'
		elif 'T' in df.loc[i, 'cat_spectral_type']:
			df.loc[i, 'stellar_class'] = 'T'
		else:
			df.loc[i, 'stellar_class'] = np.nan

		num_str = df.loc[i, 'cat_spectral_type']
		num_str = num_str.replace(':', ' ')
		num_str = num_str.replace('+', ' ')
		num_str = num_str.replace('(', '')
		num_str = num_str.replace(')', '')
		num_str = num_str[1:]
		num = float(num_str.split(' ')[0])

		df.loc[i, 'stellar_number'] = num

	return df

# Stellar modelling
# -----------------

basic_m_table = pd.DataFrame(
	columns=['stellar_number', 'mass', 'radius', 'teff'],
	data=[[0, 0.60, 0.62, 3800],
		  [1, 0.49, 0.49, 3600],
		  [2, 0.44, 0.44, 3400],
		  [3, 0.36, 0.39, 3250],
		  [4, 0.20, 0.26, 3100],
		  [5, 0.14, 0.20, 2800],
		  [6, 0.10, 0.15, 2600],
		  [7, 0.09, 0.12, 2500],
		  [8, 0.08, 0.11, 2400],
		  [9, 0.075, 0.08, 2300]]
)

basic_l_table = pd.DataFrame(
	columns=['stellar_number', 'teff', 'radius', 'mass'],
	data=[[0, np.nan, np.nan, np.nan],		# currently guess-work
		  [1, 2100, 1.01],
		  [2, np.nan, np.nan, np.nan],		# currently guess-work
		  [3, np.nan, np.nan, np.nan],		# currently guess-work
		  [4, 2000, 0.88],
		  [5, 1800, 0.79],
		  [6, 1500, 0.76],
		  [7, 1500, 0.74],
		  [8, 1600, 0.65],
		  [9, 1600, 0.66]]
)

# Y-dwarfs are basically 13Mj or even below apparently
# Radius = 0.8 - 1.1 according to sorahama

def undo_stupidity():
	idiot_file_loc = ('{}/data/baraffe_stellar_model/' 
					  'raw_stellar_model_tables.txt'.format(HOME_DIR))

	# A list of table-strings
	lot = []

	with open(idiot_file_loc, 'r') as idiot_file:
		for i, oddity in enumerate(idiot_file.readlines()):
			if not oddity.startswith(('!', '\n')):
				# Data lines
				oddity = oddity.strip()
				oddity = re.sub(r"\s+", r",", oddity)
				lot[-1].append(oddity)
			elif oddity.startswith('! '):
				# Headers have spaces both between elements,
				# and WITHIN THE FUCKING ELEMENTS
				oddity = oddity[2:]
				oddity = oddity.strip()
				oddity = oddity.replace('log ', 'log')
				oddity = re.sub(r"\s+", r",", oddity)
				lot.append([oddity])
			elif oddity.startswith(('!-', '\n')):
				continue

	for i, table in enumerate(lot):
		for j, line in enumerate(table):
			table[j] = line.split(',')

		lot[i] = pd.DataFrame(data=table[1:], columns=table[0])

	full_table = pd.concat(lot, ignore_index=True)

	# Convert data types
	for col in full_table.columns:
		full_table[col] = full_table[col].astype(float)

	full_table.rename(columns={'M/Ms':'M', 'logt(yr)':'logt', 'L/Ls':'L',
							   'R/Rs':'R', 'Log(Li/Li0)':'logL'},
					  inplace=True)
	full_table.to_pickle('{}/data/baraffe_stellar_model/' 
						 'model_table.pickle'.format(HOME_DIR))
	return full_table

def get_baraffe_table():
	return pd.read_pickle('{}/data/baraffe_stellar_model/' 
						  'model_table.pickle'.format(HOME_DIR))







# Main function
# -------------

def main():
	import matplotlib.pyplot as plt
	import seaborn as sns
	from IPython.terminal import embed

	catalog = get_catalog()
	tl = get_tl()

	# Force cast the types
	tl = tl.assign(found=np.zeros_like(tl.epic, dtype=bool))
	tl = tl.assign(catalog=np.zeros_like(tl.epic, dtype=int))

	for i in tl.index:
		targ = tl.loc[i]
		cidx, dist = match_epic(targ.RA, targ.DEC, catalog)
		tl.loc[i, 'catalog_index'] = cidx
		tl.loc[i, 'dist'] = dist
		tl.loc[i, 'pixdist'] = dist / (0.000278 * 3.98)
		tl.loc[i, 'found'] = tl.loc[i, 'pixdist'] < 5

	log_dists = np.log10(tl['pixdist'])
	log_dists = log_dists[np.isfinite(log_dists)]
	ax = sns.distplot(log_dists, hist=True, kde=False, rug=True)

	number_not_found = np.sum(~tl['found'])
	print("Number not found: {}".format(number_not_found))
	plt.show()

	number_in_tpf = np.sum(tl['found'] & (tl['pixdist'] > 1))
	print("Number nearby: {}".format(number_in_tpf))

	embed.embed()




# def match_epic(ra, dec, catalog):
# 	if not np.shape(ra) == np.shape(dec):
# 		raise ValueError("ra and dec don't have the same shape.")
# 	if not isinstance(ra, np.ndarray) or not isinstance(dec, np.ndarray):
# 		ra = np.array(ra)
# 		dec = np.array(dec)

# 	len_ra = len(ra) if len(np.shape(ra)) > 0 else 1
# 	cra = np.array([catalog.RA]*len_ra).T
# 	cdec = np.array([catalog.DEC]*len_ra).T

# 	ra_diff = cra - ra
# 	dec_diff = cdec - dec

# 	dists = np.sqrt(ra_diff**2 + dec_diff**2)
# 	indexes = np.argmin(dists, axis=0)

# 	print(np.min(dists[indexes, :]))
# 	import pdb; pdb.set_trace()
