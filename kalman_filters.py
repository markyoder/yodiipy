'''
#################################################################
# Kalman and Bayes-Markov type filters.
#
# Principal author(s): Mark R. Yoder, Ph.D.
#
#################################################################
#
'''
#
import numpy
#
################################
################################

##
class Kalman_Filter(list):
	# container object plus relevant member functions 
	# input raw data; compute Kalman means and variances.
	# maybe inherit from list or something, so the default representation is (maybe) the processed seqeunce.
	# but in the end, because of how we call the member arrays (xs, vars, etc., we likely don't gain much by inheriting a default list
	# behavior
	#
	# use var_b(a)_factor variables to compute var_b, var_ba from one or more training subdomains of the input sequence.
	#
	def __init__(self, Zs, x0=None, var0=None, var_b=None, var_ba=None, var_ba_factor=1., var_b_factor=1., training_domain=None):
		#
		# "Zs, Z, z": measurements (and lists of measurements)
		# "X, Xs, x": estimates (and lists thereof)
		# var_ba, var_a, var_b: variances (implicit probabilities) associated with the Bayes components, A, B, (B|A), etc.
		#
		# first, handle some default conditions.
		if training_domain is None:
			training_domain = (0,len(Zs))
		if not hasattr(training_domain, '__iter__'):
			training_domain = [training_domain]
		if len(training_domain)<2:
			training_domain = list(training_domain) + [len(Zs)]
		j0,j1 = training_domain
		#		
		var_Zs = numpy.var(Zs[j0:j1])
		var_ba = (var_ba or var_Zs*var_ba_factor)
		var_b  = (var_b or var_Zs*var_b_factor)
		#
		# TODO: this needs to be expanded to handle the case where we get multiple (exactly) identical measurements at the begining of the sequence.
		#   ... in fact, this case probably need to be handled in general, during the evaluation of the sequence, since there could be some scenario
		# that freakishly produces a zero-variance pair.
		x0 = (x0 or numpy.mean(Zs[0:2]))
		#
		# initial variance: if we get (bogus?) data with the exact same value, we get zero variance, so initialize with something bigger.
		# the Kalman filter should be pretty stable, so we should be able to give it pretty much anything and get a useful response within a few
		# measurements, but a smart approach is to use the variance between the first two not-equal values in the sequence.
		# var0 = (var0 or numpy.var(Zs[0:2]))
		if var0 is None:
			for z in Zs[1:]:
				var0 = numpy.var([Zs[0], z])
				if var0!=0: break
		#
		#
		# eventually, include  change-prediction step.
		#dx_x = 0.   # delta_x value (change, predict, etc.)
		#dx_var = 1.   # variance of prediction step.
		#delta_x = (dx_x, dx_var)
		#
		# initialize the array of filtered values. initialixe with two copies so we get len(in) = len(out)
		self.Xs = [[x0, var0],[x0, var0]]
		#
		self.__dict__.update(locals())
		#
		self.process_sequence()
		#
		super(Kalman_Filter, self).__init__(self.Xs)
	#
	@property
	def xs(self):
		return numpy.array([x for x,v in self.Xs])
	@property
	def vars(self):
		return numpy.array([v for x,v in self.Xs])
	#
	def process_sequence(self):
		#self
		for z in self.Zs[2:]: self.kalman_step(z)
	#
	#def kalman_step(self, z=None, x_var=None, dx_dxvar=[0.,0.], var_ba=None, var_b=None, do_update=True, var_b_type=None):
	def kalman_step(self, z=None, x_var=None, dx_dxvar=[0.,0.], var_ba=None, var_b=None, do_update=True):
		#
		# a Kalman step.
		# @z: current measurement
		# @x_var: (x, var) of previous measurement
		# @dx_dxvar: (dx, var(dx)) of a predicted change (of previous value since last measurement)
		#		   note: in the future, environmental corrections can go here.
		# @var_ba: the variance of the P(B|A) distribution (measurement variance)
		# @var_b : variance of the data
		#
		#var_b_type = int(var_b_type or self.var_b_type)
		#var_b_type = (var_b_type or 1)
		#
		var_ba = (var_ba or self.var_ba)
		var_b  = (var_b  or self.var_b)
		#
		x_var = (x_var or self.Xs[-1])
		z = (z or self.Zs[len(self.Xs)])
		#
		# input value and variance:
		#x,var = x_var
		#
		# Change Prediction step (whihc may be zero):
		x_var = gauss_add(x_var, dx_dxvar)
		#
		# get the mean value and preliminar variance from the joint probability distribution of our (adjusted)
		# previous step and the new measurement:
		# make a copy of x_var; we may need it to evaluate our belief in our estimate -- aka, adjust the variance based on the expcted change (see below).
		# that said (and we'll get to it again below), maybe we did it (accidentally) better before. if we use (z-x), instead of (z-x_prev), we punish an estimate
		# for being far from the measured/observed value (which we've presumably already accounted for in the joint probability?). 
		x_var_prev = numpy.array(x_var).copy()
		x_var = gauss_multiply(x_var, [z,var_ba])
		#
		# this gives the mean value (position) and variance of the new distribution, but how well do we believe it?
		# large changes are considered suspect; compute a variance that expresses this. use, more or less, the ratio
		# of the change to a reference variance:
		# TODO: is this correct? should this be x_var_before_gauss_multiply[0] ?? aka, the prediction-adjusted previous value.
		#      so... from a Bayes perspective, the second or third options are probably more appropriate (reduce confidence based on the difference between the previous
		# estimate and the the current measurement or the previous and current estimates (aka, large changes reduce confidence). The first version effectively applies the
		# measurement uncertainty twice -- which seems wrong. however, we made this change (from the first to the second) and Salasky insists that the output was 'better'
		# before -- and we believe him, so we leave this to be evaluated later. because this is a Markov process, the effect shold be relatively minor one way or another, and
		# should be observed primarily as a single-step lag one direction or the other.
		# So, in practice, this is an unresolved question and should be optimized later. 
		# 1)
		#
		# this is probably correct. see #4 below. if we approximate the true Bayes relationship, P(A)/P(B), we get approximately this.
		var_R_scatter = ((z-x_var[0])**4)/var_b
		#
		x_var[1]+=var_R_scatter
		#
		if do_update:
			self.Xs += [x_var]
		#
		return x_var
#
# Gauss math:
def gauss_multiply(g1, g2):
	# g1, g1 are tuples/lists of the gauss mu,var
	mu1, var1 = g1
	mu2, var2 = g2
	mean = (var1*mu2 + var2*mu1) / (var1 + var2)
	variance = (var1 * var2) / (var1 + var2)
	#
	return [mean, variance]
def gauss_add(g1,g2):
	return [g1[0]+g2[0], g1[1]+g2[1]]
	#return [sum(cl) for cl in zip(g1, g2)]
#
# some helper functions:
def gauss_pdf(x=0., mu=0., sigma=1.):
	return (1./numpy.sqrt(2.*sigma*sigma*numpy.pi))*numpy.exp(-((x-mu)**2.)/(2.*sigma*sigma))

def gauss_char_sigma(x=0., mu=0., sigma=1.):
	# return a "characteristic" stdev; assuming gauss, given x,mu,sigma, return sigma for a gaussian
	# centered at (x-mu)=0.
	return sigma*numpy.exp(((x-mu)**2.)/(2*sigma*sigma))
def p_to_sigma(p):
	# convert a probability to a characteristic sigma, assuming gaussian and x=0.
	return 1./p*numpy.sqrt(2.*math.pi)

