import matplotlib
import numpy
import pylab as plt
import math
import random
#
import sklearn
#
from  matplotlib.mlab import PCA
#
#def lzip(*X):
#	return list(zip(X))
#
#
def pca1(theta=math.pi/6., a=1.0, b=.5, x0=0., y0=0., N=1000):
	#
	Rx = random.Random()
	Ry = random.Random()
	#
	# make a blob of data; the PCA should find the center, axes, etc.
	#
	#XY = [rotate_ccw([x0 + a*Rx.random(), y0 + b*Ry.random()], theta, x0=x0, y0=y0) for n in xrange(N)]
	XY = [[x0 + a*Rx.random(), y0 + b*Ry.random()] for n in range(N)]
	print(("variances on raw matrix: ", numpy.var(list(zip(*XY))[0]), numpy.var(list(zip(*XY))[1])))
	print(("std on raw matrix: ", numpy.std(list(zip(*XY))[0]), numpy.std(list(zip(*XY))[1])))
	XY = [rotate_ccw(rw, theta, x0=x0, y0=y0) for rw in XY]
	# first, let's just visualize the rotation matrix:
	#
	pcas = PCA(numpy.array(XY))
	#
	# and do it manually:
	x_mean, y_mean = numpy.mean(list(zip(*XY))[0]), numpy.mean(list(zip(*XY))[1])
	my_mu = numpy.array([x_mean, y_mean])
	print(("x_mean, y_mean: ", x_mean, y_mean))		# these check with pcas.mu
	dXY = [[(x-x_mean), (y-y_mean)] for x,y in XY]
	my_cov = numpy.dot(numpy.array(list(zip(*dXY))),numpy.array(dXY))/(float(N)-1.)
	# use eigh() for symmetric or hermitian matrices. it's faster and avoids some round-off errors that can result in complex/imaginary valued elements.
	# ... and sort by eigen-values
	eig_vals, eig_vecs = numpy.linalg.eigh(my_cov)		# ... and there's something weird about eig() vs eigh() they return somewhat different eigen values.
	#eig_vecs = numpy.array(zip(*eig_vecs))	# transpose so each 'row' is an eigen vector.
	eig_vals, eig_vecs = list(zip(*sorted(zip(eig_vals, eig_vecs.transpose()), key=lambda x:x[0])))		# and we can use reverse=True, or just remember that we're reverse order sorted.
	eig_vals_norm = numpy.linalg.norm(eig_vals)
	#axis_weights = [math.sqrt(e/max(eig_vals)) for e in eig_vals]		# i think this is pretty standard, but it seems that normalizing these vectors so that e1*82 + e2**2 = 1
	#																						# also makes sense.
	axis_weights = [e/eig_vals_norm for e in eig_vals]
	print(("normed(eig_vals): %f (normed-check (should be 1.0): %f)" % (eig_vals_norm, numpy.linalg.norm(axis_weights))))
	#	
	print(("eigs: ", eig_vals, eig_vecs))
	frac_x, frac_y = [x/(numpy.linalg.norm(eig_vals)**2.) for x in eig_vals]
	#
	# primed axes?:
	x_prime = numpy.dot(numpy.array(eig_vecs), numpy.array([1., 0.]))*frac_x**.5 #+ numpy.array([x_mean, y_mean])
	y_prime = numpy.dot(numpy.array(eig_vecs), numpy.array([0., 1.]))*frac_y**.5 #+ numpy.array([x_mean, y_mean])
	#
	plt.figure(0)
	plt.clf()
	#
	plt.plot([x for x,y in XY], [y for x,y in XY], 'b.')
	plt.plot([x for x,y in pcas.a], [y for x,y in pcas.a], 'r.')
	print(("means again: ", x_mean, y_mean))
	plt.plot(x_mean, y_mean, 'co', ms=9, zorder=4)
	#plt.plot(*pcas.mu, color='r', ms=5, marker='o',zorder=5)
	#
	#plt.plot(numpy.array([0., x_prime[0]]) + x_mean, numpy.array([0., x_prime[1]]) + y_mean, 'o-')
	#plt.plot(numpy.array([0., y_prime[0]]) + x_mean, numpy.array([0., y_prime[1]]) + y_mean, 's-')
	#
	
	#
	ax_fact = -.5*axis_weights[0]
	plt.plot([x_mean, x_mean+eig_vecs[0][0]*ax_fact], [y_mean, y_mean+eig_vecs[1][0]*ax_fact], 'ro-', zorder=5)
	ax_fact = -.5*axis_weights[1]
	plt.plot([x_mean, x_mean+eig_vecs[0][1]*ax_fact], [y_mean, y_mean+eig_vecs[1][1]*ax_fact], 'gs-')
	#
	plt.figure(1)
	plt.clf()
	plt.plot(*list(zip(*pcas.Y)), marker='.', ls='', color='g')

	#return pcas
	return [eig_vals, eig_vecs]

class PCA_transform(object):
	# Comments:
	# while yes, this could be tuned up a bit, it looks about right for what it was intended to do...
	# which is to use PCA to do a rotate-and-stretch transformation (a slightly non-standard application of PCA).
	# accordingly, the use of normalized eigen-vectors/eigen-values as axis lengths makes sense.
	#
	# - use numpy.cov() instead of manually calculating cov (numpy should be compiled)
	# - the stretching transform should probalby be coded as a dot-product with a diagonal matrix, D = numpy.diag(eig_vals)
	# - ... and ingeneral, separate the salient matrices:
	#   - input data, say X
	#   - transformed data, U
	#   - to_pca rotation R_to
	#   - from_pca rotation R_from = inv(R_to)
	#   - stretch = D = numpy.diag(eig_vals)
	#
	# ... and then figure out if we can use a fully compiled, one-stop-shopping for PCA, like scikit-learn.
	# also, of course, double-check applications to be sure they're using the transform correctly.
	def __init__(self, data_in=None, N=None, theta=None):
		if data_in==None:
			N=(1000 or N)
			theta = (theta or math.pi/6.)
			#
			data_in = make_test_data(theta=theta, N=100, x0=0., y0=0., a=2., b=1.)
		#
		self.data = data_in
		self.calc_pca(data_in=data_in, do_return=False)
	
	def calc_pca(self, data_in=None, do_return=False):
		'''
		# calc pca and assign class scope variables.
		# assume data are in rows, X = [[x0,x1,x2,x3], [x0,x1,x2,x3],...], or for example: X = [[x0,y0,z0], [x1,y1,z1], ...]
		#
		'''
		data = (data_in or self.data)
		data = numpy.array(data)
		#
		# first, get means:
		mus = [numpy.mean(col) for col in zip(*data)]		# mean values for each column. note, this will also constitute a vector 
															# between our transoformed origin and the origina origin.
		# note: we can get this with numpy.cov(list(zip(*A)))
		data_centered = [[x-mus[j] for j,x in enumerate(rw)] for rw in data]
		#
		# TODO:use::
		#data_centered = data - numpy.mean(data, axis=0)
		#
		# NOTE: conventional wisdom would suggest using numpy.cov(). in general, we might want to do our own covariance calculation, since we can then choose our
		#  axis of rotation (as opposed to necessarily subtracting off the algebraic mean). in this case, we are subtractingoff the algebraic mean, so we should
		#  probably either 1) use numpy.cov() or 2) allow [mus] to be passed as a parameter.
		#
		cov_normed = numpy.dot(numpy.array(list(zip(*data_centered))),numpy.array(data_centered))/(float(len(data_centered))-1.)
		# TODO: use::
		# cov_normed = numpy.cov(data_centered.T) 
		#
		# get eigen-values/vectors:
		# note: for a symmetric or hermitian matrix, we should use numpy.linalg.eigh(); it's faster, more accurate, and will quash complex value rounding errors.
		#       it might be necessary, down the road, to just spin through these and dump any complex components.
		eig_vals, eig_vecs = numpy.linalg.eigh(cov_normed)
		#
		if data_in==None or data_in==self.data:			
			#
			self.max_eig = max(eig_vals)
			self.max_eig_vec_index = eig_vals.tolist().index(self.max_eig)		# this might be a stupid way to do this;
			#
			# this is just wrong, so let's correct it. it shouldn't make any significant difference in the applcations that are using it.
			# but note that we're doing this relative stretching... and there was a reason for that (aka, e/max(E) instead of just e).
			# ... so let's get to this after we finish these paper revisions??? during???
			self.axis_weights = [math.sqrt(e/self.max_eig) for e in eig_vals]
			#self.axis_weights = [e/self.max_eig for e in eig_vals]
			#
			# assign stuff.
			self.mus = numpy.array(mus)
			self.data_centered = data_centered
			self.eig_vals = eig_vals
			self.eig_vecs = eig_vecs
			#
			#self.axis_weights = axis_weights
			#self.max_eig = max_eig
			#self.max_eig_vec_index = max_eig_vec_index
			self.cov_normed =cov_normed
			#
			# what we actually want here is more elegantly just something like:
			# X' = x V^-1 E^-1
			# where V is the transform of eigen-vectors-as-columns and E is diag(lambdas).
			# ... and note we've been square-rooting our axis lengths, which is fine for the linear transformation applications
			# we've been doing, but not proper PCA, strictly speaking... so we'll fix it.
			self.to_PCA_rotation = numpy.array([[x*self.axis_weights[j] for j,x in enumerate(rw)] for rw in eig_vecs])
		#
		if do_return: return [eig_vals, eig_vecs]
		#
	def to_PCA(self, v):
		#x0 = self.M_to_PCA.dot(v)
		#x1 = self.mus
		#print "adding: ", x0, x1, x0+x1
		return numpy.array(self.to_PCA_rotation.dot(v)) + numpy.array(self.mus)
	#
	@property
	def primary_axis(self):
		return zip(*self.to_PCA_rotation)[self.max_eig_vec_index]
	#
	def plot_data(self, fignum=0, axes=[0,1]):
		'''
		# some diagnostic plotting...
		# for now, just 2d plotting.
		'''
		#
		plt.figure(fignum)
		plt.clf()
		#
		plt.plot(zip(*self.data)[axes[0]], zip(*self.data)[axes[1]], marker='.', color='b', zorder=2, ls='')
		#
		#print "vx: ", self.to_PCA([1.,0.])
		vx = numpy.array([self.mus, self.to_PCA([1.,0.])])
		vy = numpy.array([self.mus, self.to_PCA([0.,1.])])
		
		print(("vx, vy: ", vx, vy))
		#
		plt.plot(*self.mus, marker='*', color='r', ms=15)
		plt.plot(*list(zip(*vx)), color='r', lw=1.5, ls='-', marker='s')
		plt.plot(*list(zip(*vy)), color='g', lw=1.5, ls='-', marker='s')
		#
		#[plt.plot(*(self.mus + v), color='m', ls='--', marker='*') for v in zip(*self.eig_vecs)]		# needs to be corrected to use axes[]
		[plt.plot(*(self.mus + v), color='m', ls='--', ms=10, marker='*') for v in zip(*self.to_PCA_rotation)]
#		
#
class PCA_cross_section(list):
	def __init__(self, XYW, x_min=None, x_max=None, y_min=None, y_max=None, n_NN=4, n_points_xc=None):
		# compute the covariance of [[x*w, y*w], ...], get eigen-value/vectors,
		#  construct a cross section vector, then compute cross-section values via a weighted
		#  average (which we can show is a Bayes maximum-likelihood value) from each point's NN.
		# TODO: add an option for a distance calculation? use a spherical distance formula in
		#  both the NN finder (presumably from sklearn, which i think uses an r-tree index) and
		#  the weighted average.
		#
		# TODO: do we need to keep the original inputs? this is more memory and compute intensive. how often
		#  do we re-use these objects? are we typically better off just recomputing the whole class for variations
		#  on PCA bounds, etc?
		XYW = numpy.array(XYW)
		#
		if x_min is None: x_min = min(XYW.T[0])
		if x_max is None: x_max = max(XYW.T[0])
		if y_min is None: y_min = min(XYW.T[1])
		if y_max is None: y_max = max(XYW.T[1])
		#
		#XYW_pca = XYW[XYW.T[0]>=x_min and XYW.T[0]<=x_max and XYW.T[1]>=y_min and XYW.T[1]<=y_max]
		f_between = lambda x, y, x1, x2, y1, y2: (x>=x1 and x<=x2 and y>=y1 and y<=y2)
		#
		# trying to use numpy indexing here, so we end up keeping the indices of the array where these
		#  criteria are met. having trouble getting it to take a multi-valued condition. maybe the better
		#  approach is to pass an array of indices that satisfy the "between" condition?
		#
		# this requires multiple passes throught the array (which is most likely not terribly costly)
		#XYW_pca = XYW[XYW.T[0]>=x_min]
		#XYW_pca = XYW_pca[XYW.T[0]<=x_max]
		# TODO: maybe, instead of copying this, we define the index and implement it as a @property function?
		# so idx_pca = numpy.array([k for k,(x,y,w) in... ]) and then self.XYW_pca 
		# returns self.XYW[self.idx_pca] ??
		#
		XYW_pca = XYW[numpy.array([k for k,(x,y,w) in enumerate(XYW) 
								   if f_between(x,y, x_min, x_max, y_min, y_max)])]
		#
		#print('***DEBUG lens: ', len(XYW_pca), len(XYW))
		#n_points_xc = n_points_xc or len(XYW_pca)
		# ... so i don't recall the logic here. we want approximately the length of the x or y axis:
		# ???
		n_points_xc = n_points_xc or max(len(set(XYW_pca.T[0])), len(XYW_pca))
		#n_points_xc = n_points_xc or max([len(list(set(XYW_pca[:,0]))), len(list(set(XYW_pca[:,1])))])
		#
		#XYw_pca = numpy.array([[x*w,y*w] for x,y,w in XYW_pca ])
		# TODO?: interpolate Y,w onto a regulaized X axis, or assume valid inputs?
		#
		#w_cov = numpy.cov(XYw_pca, rowvar=False)
		# TODO: will matrices be properly aligned if we just skip all of the A.T ? probably at least
		#      one layer of this to revise...
		#
		# as a sanity check, do a line-fit to the local data to get an approximate b-value:
		xy_w = XYW_pca.T[0:2].T*numpy.atleast_2d(XYW_pca.T[2]).T
		lsq_xyz = numpy.linalg.lstsq([[1.,x] for x,y in xy_w], [y for x,y in xy_w])
		#print('** DEBUG lsq: ', lsq_xyz[0])
		#
		w_cov = numpy.cov(XYW_pca.T[0:2].T*numpy.atleast_2d(XYW_pca.T[2]).T, rowvar=False)
		# Note: leave eig_vecs matrix intact, so we can use it for rotation transformations.
		eig_vals, eig_vecs = numpy.linalg.eig(w_cov)
		#print('*** Debug (prelim): ', eig_vals, eig_vecs)
		#
		# now, sort by eigenvalues:
		idx = (eig_vals**2.).argsort()[::-1]   
		#
		#print('eigs (idx, vals, vecs): ', idx, len(eig_vals), len(eig_vecs))
		#
		#eig_vals = eig_vals.T[idx].T
		# TODO: sort out (equivalent) syntax for (not) fancy inxing... this is from SourceForge, or
		#    or something, but it is not quite right (i never use this syntax, but it's probably fast)
		#e1,e2 = eig_vecs[:,idx]
		# this looks correct (ish):
		e1, e2 = eig_vecs.T[idx]
		# note: we need to be careful about how we define the vectors of the data vs the axes. are we
		#   rotating the axes to the data or the data to the axes (which are identical/inverse) operations
		#   but it is important to be clear with respect too making rotations vs drawing cross-secgtions.
		#
		#print('*** e1, e2: ', e1, e2)
		#print('eigs (idx, vals, vecs): ', idx, len(eig_vals), len(eig_vecs))
		#
		# Compute linear slope factors, for y' = a + bx type transformations, as opposed to the
		#  X' = dot(e_k, X) or X' = L_x*e_1, y' = L_y * e_2 approaches (just multiplying the eigen-vectors).
		b_major = e1[1]/e1[0] 
		b_minor = e2[1]/e2[0]
		#
		# stash inputs:
		self.__dict__.update({key:val for key,val in locals().items() if not key in ('self', '__class__')})
		#
		# etas.ETAS_array['x'], y0 + b_major*(etas.ETAS_array['x'] - x0)
		# NOTE: we can optionally replace this next line with, X_pca = self.X_pca, etc.; self.XYW_pca has been assigned
		#     and we have definde @property functions for X_, Y_, W_ _pca. leaving the XYW_pca.T call is probaby faster,
		#     but might lead to confusion down the road...
		X_pca, Y_pca, W_pca = XYW_pca.T
		#
		X = numpy.array(sorted(set(X_pca)))
		# XY = [pca_cross_2.e1*x for x in numpy.linspace(-5., 5., 100)]
		#
		dx = max(X_pca)-min(X_pca)
		x0 = numpy.mean(X_pca)
		y0 = numpy.mean(Y_pca)
		Xs = numpy.linspace(-dx, dx, n_points_xc)
		# numpy.mean(X_pca)
		#
		# TODO (DEPRICATED): maybe revisit the default cross-section vector.
		#super(PCA_cross_section, self).__init__(numpy.array([X, 
		#                                            numpy.mean(Y_pca) + b_major*(X-numpy.mean(X_pca))]).T)
		# TODO (DONE):
		#   set self with dot-product transform: numpy.dot(XY, self.eig_vecs_inv) + numpy.array([x_mu, y_mu]
		super(PCA_cross_section, self).__init__(self.get_cross_section_xy(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, n_points=n_points_xc))
		#
		# probably the 'right' way to compute the cross-section vector is to just multiply (scale) and 
		#  translate (add to) the eigen-vector:
		#super(PCA_cross_section, self).__init__([[e1[0]*x + x0, e1[1]*x + y0]
		#                                for x in numpy.linspace(-dx, dx, n_points_xc)])
		del X_pca, Y_pca, W_pca, X
		#
		
	#
	def get_cross_section_xy_lin_func(self, x_min=None, x_max=None, y_min=None, y_max=None, n_points=None, b=None):
		'''
		# **DO NOT **DEPRICATE WARNING: this computes cross section cooridnates via a linear transformation... it looks like a sort of 
		#  silly way to do it, and can be problematic for steep slopes. the newer get_cross_section_xy() version uses a rotation on
		#  the PCA axes, rather than a liner function y' = a + b*x, which is of course still a linear transformation, just a smarter
		#  way of doing it. THAT SAID, this is a simple interface to get an arbitray cross section. we can give some (x,y) boundaries,
		#  specify a slope, and get a cross section, so keep this.
		#
		# note: this function can be removed as soon as we are happy with its replacement.
		#
		# returns [[x,y], ...] coordinates of the cross section, to (presumabley) be used to compute mean cross-section
		#   weights (z-values) via a NN method.
		#
		# TODO: this should work -- it will return a  cross-section, but i think we need to be more thoughtful
		#	 about how we go about this, specifically how we choose the middle. we'll need some variationos
		#	 of this to use the center, weighted center, etc. and then a smart way to draw the cross-section
		#	 axis through it (y = a + bx vs (x',y') = x*v1)
		#
		'''
		#
		n_points = n_points or self.n_points_xc
		#
		if b is None: b = self.b_major
		#
		if x_min is None:
			x_min = min(self.X)
		if x_max is None:
			x_max = max(self.X)
		if y_min is None:
			y_min = min(self.Y)
		if y_max is None:
			y_max = max(self.Y)
		#
		# 
		X = numpy.linspace(x_min, x_max, n_points)
		#return numpy.array([X, numpy.mean(Y_pca) + b*(X-numpy.mean(self.X_pca))]).T
		#
		# this will use an unweighted mean to center:
		#return numpy.array([[x,y] for x,y in 
		#                numpy.array([X, numpy.mean(self.Y_pca) + b*(X-numpy.mean(self.X_pca))]).T
		#                    if (y>=y_min and y<=y_max)])
		#
		# ... and a weighted mean:
		#  this is a bit difficult to follow, and the outer array() cast can probalby be simplified... 
		#  the basics are:
		#    - return all values for x_min < x < x_max and y_min < y < y_max (outer array() cast)
		#    - the X axis is n_points between x_min < x x_max
		#    - then just a linear function along X. 
		#    - note: we don't do a PCA here, but we use the slope computed in the PCA. probably a better appraoch
		#      is to take even spaced points along [X,0] and then PCA-rotate them into the new coordinate system.
		#   
		return numpy.array([[x,y] for x,y in 
		                numpy.array([X, numpy.average(self.Y_pca, weights=self.W_pca) +
		                             b*(X-numpy.average(self.X_pca, weights=self.W_pca))]).T
		                    if (y>=y_min and y<=y_max)])
		#
	def get_cross_section_xy(self, x_min=None, x_max=None, y_min=None, y_max=None, n_points=None, minor=False):
		'''
		# returns [[x,y], ...] coordinates of the cross section, to (presumabley) be used to compute mean cross-section
		#   weights (z-values) via a NN method.
		# @minor: if true, return the track of the minor,not major, axis.
		#
		# TODO: this should work -- it will return a  cross-section, but i think we need to be more thoughtful
		#	 about how we go about this, specifically how we choose the middle. we'll need some variationos
		#	 of this to use the center, weighted center, etc. and then a smart way to draw the cross-section
		#	 axis through it (y = a + bx vs (x',y') = x*v1)
		#
		'''
		n_points = n_points or self.n_points_xc
		#if b is None: b = self.b_major
		#
		if x_min is None:
			x_min = min(self.X)
		if x_max is None:
			x_max = max(self.X)
		if y_min is None:
			y_min = min(self.Y)
		if y_max is None:
			y_max = max(self.Y)
		#
		# 
		#X = numpy.linspace(x_min, x_max, n_points)
		x_mu, y_mu = numpy.average(self.XYW_pca[:,0:2], weights = self.XYW_pca[:,2], axis=0)
		#x_mu =  numpy.average(self.X_pca, weights=self.W_pca)
		#y_mu =  numpy.average(self.Y_pca, weights=self.W_pca)
		#
		if not minor:
			XY = numpy.array([numpy.linspace(x_min, x_max, n_points) - x_mu, numpy.zeros(n_points)]).T
		else:
			XY = numpy.array([numpy.zeros(n_points), numpy.linspace(y_min, y_max, n_points) - y_mu]).T
		#
		return numpy.dot(XY, self.eig_vecs_inv) + numpy.array([x_mu, y_mu])
	#
	
	def get_cross_section_zs(self, XY_xc=None, XYZ=None, n_NN=None):
		# TODO: reconfigure this to pass n_points and to use get_cross_secition_xy() as default behavior. the self.XY_xc behavior may have a different number
		#   of points. also, we need to compute an approximate distance along the cross-section coordinates for the "X" axis. 
		#TODO: this is not working. so maybe take it off-line to work out the code, then put back into
		#	the class.
		#
		n_NN = n_NN or self.n_NN
		if XY_xc is None: XY_xc = numpy.array(self)
		#if XYZ is None:   XYZ = self.XYW_pca
		if XYZ is None:   XYZ = self.XYW
		#
		# get NN:
		# TODO: look at "fancy" indexing version of this X.T[0:2].T operation, something like:
		#   X[0:2, :]
		#nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_NN, algorithm='ball_tree').fit(XYZ.T[0:2].T)
		nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_NN, algorithm='ball_tree').fit(XYZ[:,0:2])
		#
		# TODO: i think a better and more efficient way to do this is to just assign the weights, from
		#  the distances, as w_jk = 1/(r_jk + <r>_k) or 1.
		#  where "or" kicks in if the denominator (r_jk + <r>_k) == 0
		# we will want to carefully evaluate the corner cases and evaluate how the fraction addition
		# is affected when we have these singular cases.
		
		distances, indices = nbrs.kneighbors(XY_xc)
		#
		# TODO: are distances always positive?
		mean_distances = numpy.mean(distances, axis=1)
		#denom = numpy.array([1./r if r!=0 else 1. for r in mean_distances])
		weights        = numpy.array([[1./(r+mu) if (r+mu)!=0. else 1. for r in rw]
									  for rw, mu in zip(distances, mean_distances)])
		#
		# we want a weighted average of the z values, based on NN distances,
		# z_xc_k = sum_j(z_j/r_jk)/sum(1/r_jk)
		# but we need to handle 1/0 cases in a generalized way. maybe something like:
		# w = 1/(r_jk + a*<r>_k)
		# where <r>_k is the mean nn distance and a is a tuning parameter;
		#  if <r>==0, evenly weight all elements?
		#
		# like this:
		# (but the main thing is to efficiently handle the x/0 cases).
		#
		z_xc = [numpy.dot(XYZ.T[2][js], ws)/numpy.sum(ws)
				for js, ws, mu in zip(indices, weights, mean_distances)]
		#z_xc = [numpy.sum([XYZ[j][2] for j in js]) for js, ws, mu in zip(indices, weights, mean_distances)]
		#
		return z_xc
		
	#@property
	#def XYW_pca(self):
	#	numpy.array([[x,y,w] for x,y,w in XYw if (x>=x_min and x<=x_max and y>=y_min and y<=y_max) ])
	#
	def dist_axis(self, XY=None):
		if XY is None:
			XY = self
		#
		return numpy.sqrt([ (x - XY[0][0])**2. + (y - XY[0][1])**2. for x,y in XY])
	#
	# TODO: use numpy fancy indexing...
	@property
	def X(self):
		#return self.XYW.T[0]
		return self.XYW[:,0]
	@property
	def Y(self):
		#return self.XYW.T[1]
		return self.XYW[:,1]
	@property
	def w(self):
		#return self.XYW.T[2]
		return self.XYW[:,2]
	#
	@property
	def X_pca(self):
		return self.XYW_pca.T[0]
		#return self.XYW_pca[:,0]
	@property
	def Y_pca(self):
		#return self.XYW_pca.T[1]
		return self.XYW_pca[:,1]
	@property
	def W_pca(self):
		#return self.XYW_pca.T[2]
		return self.XYW_pca[:,2]
	#
	@property
	def eig_vecs_inv(self):
		return numpy.linalg.inv(self.eig_vecs)
	#		
#		
def pca_test2(theta=math.pi/6., N=1000, x0=0., y0=0., fignum=0):
	#
	my_data = make_test_data(theta=theta, N=N, x0=x0, y0=y0)
	my_pca = yoda_pca(my_data)
	#
	print(("my_pca: ", my_pca))
	#return my_pca
	ax_x, ax_y = list(zip(*my_pca[1]))
	axis_weights = [math.sqrt(e/max(my_pca[0])) for e in my_pca[0]]
	ax_x = numpy.array(ax_x)*axis_weights[0]
	ax_y = numpy.array(ax_y)*axis_weights[1]
	#
	plt.figure(fignum)
	plt.clf()
	plt.plot(*list(zip(*my_data)), marker='.', color='b', zorder=2, ls='')
	plt.plot(*list(zip([0,.0], ax_x)), marker='s', ls='-', lw=2, color='r', label='ax_x')
	plt.plot(*list(zip([0,.0], ax_y)), marker='s', ls='-', lw=2, color='g', label='ax_y')
	#
	xprime = my_pca[1].dot([1.,0.]) + numpy.array([.5,0.])
	yprime = my_pca[1].dot([0.,1.]) + numpy.array([.5,0.])
	#
	plt.plot(*list(zip([.5,0.], xprime)), marker='s', ls='--', lw=2, color='m', label='x_prime')
	plt.plot(*list(zip([.5,0.], yprime)), marker='s', ls='--', lw=2, color='c', label='y_prime')
	#
	print('basically, axes should be parallel/perpendicualar to the data-cloud')
	#
	plt.legend(loc=0)
#
def yoda_pca(data_in):
	# we'll rename this later. for now, this is just a super simple PCA approach to finding principal axes. we'll leave them in their original order
	# and leave them all in tact. it's basically a fitting algorithm so we can construct a transformation matrix between two frames.
	# so, basically, do normal PCA (mean value, eigen-vals/vecs, etc.
	# assume data are like [[x0, x1, x2, x3], [x0,x1,x2,x3],...]
	#
	#mus = [numpy.mean(col) for col in zip(*data_in)]
	#print(mus)
	#centered = [[x-mus[j] for j,x in enumerate(rw)] for rw in data_in]
	#centered = numpy.array(centered)-numpy.mean(centered, axis=0)
	#my_cov = numpy.dot(numpy.array(list(zip(*centered))),numpy.array(centered))/(float(len(data_in))-1.)
	my_cov = numpy.cov(numpy.array(data_in).T)
	#
	# for now, we can't assume hermitian or symmetric matrices, so use eig(), not eigh()
	eig_vals, eig_vecs = numpy.linalg.eig(my_cov)
	#axis_weights = [math.sqrt(e/max(eig_vals)) for e in eig_vals]
	#
	# note: the transformed axes are the column vectors in eig_vecs; the main axis of interest is the one with the largest eigen vector, but we also want to preserve the
	# geometry of the system.
	#
	
	#
	#return centered
	return [eig_vals, eig_vecs]

def rotate_ccw(v,theta, x0=0., y0=0.):
	# rotates a vector about it's tail. if no tail is given, vector is in [x,y] format, we effectively assume vector at origin.
	# note that this is not a good generalized format; if we input v like v=[[0.,0.], [0.,1]], it returns a v'=[.866,.5] (aka, drops the tail).
	# however, it will be a useful format for these little testing scripts.
	#
	tail = [x0, y0]
	tip = v
	if hasattr(v[0], '__len__'):
		# we have a vector-vector, aka, not from origin. assume we're rotating about the tail
		tail = v[0]
		tip  = [v[-1][j] - v[0][j] for j in range(len(v[0]))]
		#
		print(("tip, tail: ", tip, tail))
	v_prime = numpy.dot([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]], tip)
	#
	return [x+tail[j] for j,x in enumerate(v_prime)]

def make_test_data_gen(thetas=[math.pi/6.], N=100, X0=[0., 0.], delta_X=[2.,1.]):
	# eventually, interpret thetas to be a rotation in the [x_j, x_j+1] plane about [x_j+2]. but we'll get to that later. for now, just produce a generalized arary.
	#
	while len(delta_X)<len(X0): delta_X += [1.]
	while len(thetas)<(len(X0)-1): thetas+=0.
	#
	Rs = [random.Random() for x in X0]
	XY = [[x0 + dx*r.random() for x0,dx,r in zip(X0, delta_X, Rs)] for n in range(N)]
	#
	for k,theta in enumerate(thetas):
		for rw in XY:
			xprime, yprime = rotate_ccw(v=[rw[k], rw[k+1]], theta=theta, x0=X0[k], y0=X0[k+1])
			#
		#
	#
	return XY

def make_test_data(theta=math.pi/6., N=100, x0=0., y0=0., a=2., b=1.):
	#
	# for now, limit to 2D.
	Rx,Ry = [random.Random() for x in [0,1]]
	#
	#XY = [rotate_ccw([x0 + a*Rx.random(), y0 + b*Ry.random()], theta, x0=x0, y0=y0) for n in xrange(N)]
	XY = [[x0 + a*Rx.random(), y0 + b*Ry.random()] for n in range(N)]
	#print "variances on raw matrix: ", numpy.var(zip(*XY)[0]), numpy.var(zip(*XY)[1])
	#print "std on raw matrix: ", numpy.std(zip(*XY)[0]), numpy.std(zip(*XY)[1])
	XY = [rotate_ccw(rw, theta, x0=x0, y0=y0) for rw in XY]
	#
	return XY

def lzip(X):
	return list(zip(X))

if __name__=='__main__':
	pass
else:
	plt.ion()	
