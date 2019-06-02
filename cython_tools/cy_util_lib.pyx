



def bin_model(f_model, bin_indices, bin_multiplicity):
	"""Performs a binning from model to data.

	Args:
		f_model (np.array): the values of the model at a set of points
		bin_indices (np.array): same shape as f_model, for each point
			in f_model, gives the index of the bin to which the point
			goes.
		bin_multiplicity (np.array): number of points per bin

	Returns:
		f_binned (np.array): the flux, binned to the data grid

	TODO: speed it up in cases where multiplicity is 1 for some bins,
	by ignoring those bins.
	"""

	raise NotImplementedError
