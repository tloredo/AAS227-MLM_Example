"""
PyStan gamma-Poisson demo for AAS 227 astrostatistics session

For PyStan info:

https://pystan.readthedocs.org/en/latest/getting_started.html

Created 2014-11-04 by Tom Loredo for IAC Winter School
Adapted for AAS227 2016-01-06
"""

import numpy as np
import scipy
from scipy import stats
import matplotlib as mpl

# Pollute the namespace!
from matplotlib.pyplot import *
from scipy import *

from stanfitter import StanFitter
from shrinkage_plot import shrinkage_plot

# Get TL's interactive plotting customizations if availalbe.
try:
    import myplot
    from myplot import close_all, csavefig
    ion()
    # myplot.tex_on()
    csavefig.save = False
except:
    pass


# Stan code defining a gamma-Poisson MLM for number counts (log N - log S)
# fitting:
code = """
data {
    int<lower=0> N; 
    int<lower=0> counts[N];
    real  exposures[N]; 
} 

parameters {
    real<lower=0> alpha; 
    real<lower=0> beta;
    real<lower=0> fluxes[N];
}

transformed parameters {
    real<lower=0> flux_cut;
    flux_cut <- 1./beta;
}

model {
    alpha ~ exponential(1.0);
    beta ~ gamma(0.1, 0.1);
    for (i in 1:N){
        fluxes[i] ~ gamma(alpha, beta);
        counts[i] ~ poisson(fluxes[i] * exposures[i]);
  }
}
"""


# Setup "true" model and design parameters for stellar observations:
if False:
    # Define gamma dist'n parameters alpha & F_cut:
    Jy_V0 = 3640.  # V=0 energy flux in Jy
    phi_V0 = 1.51e7 * 0.16 * Jy_V0  # V=0 photon number flux (s m^2)^{-1}
    V_cut = 24.  # V magnitude at rollover
    F_cut = phi_V0 * 10.**(-0.4*V_cut)  # flux at rollover
    alpha = .4  # power law part has exponent alpha-1; requires alpha > 0

    # Variables describing the data sample:
    n_s = 25
    area = pi*(8.4**2 - 5**2)  # LSST primary area (m^2)
    exposures = 10.*area*ones(n_s)  # LSST single-image integration time * area
    mid = n_s//2
    exposures[mid:] *= 10  # last half use 10x default exposure

# Setup "true" model and design parameters for GRB observations:
if True:
    # Define gamma dist'n parameters alpha & F_cut:
    F_cut = 10.  # peak flux of bright BATSE GRB, photons/s/cm^2
    alpha = .4  # power law part has exponent alpha-1; requires alpha > 0

    # Variables describing the data sample:
    n_s = 20
    area = 335.  # Single BATSE LAD effective area, cm^2
    # Fake projected areas for a triggered detector:
    areas = area*stats.uniform(loc=.5, scale=.5).rvs(n_s)
    exposures = .064*areas  # use 64 ms peak flux time scale


# Define the true flux dist'n as a gamma dist'n.
beta = 1./F_cut  # Stan uses the inverse scale
ncdistn = stats.gamma(a=alpha, scale=F_cut)

# Sample some source fluxes from the flux population dist'n.
fluxes = ncdistn.rvs(n_s)


# Generate observations of the flux sample.
def gen_data():
    """
    Simulate photon count data from the Poisson distribution, gathering
    the data and descriptive information in a dict as needed by Stan.
    """
    n_exp = fluxes*exposures  # expected counts for each source
    counts = stats.poisson.rvs(n_exp)
    return dict(N=n_s, exposures=exposures, counts=counts)

data = gen_data()


# Invoke Stan to build the model.
fitter = StanFitter(code, data)  # use code string in this script file
# fit = StanFit('gamma-poisson.stan', data)  # use source in a .stan file

# Stan will "write" and compile a C++ executable implementing the model, a
# posterior sampler, and an optimizer.  StanFit will cache the Stan products so
# subsequent runs of the script need not rebuild the model from scratch.


# Run 4 chains of length 2000 (Stan will use 1/2 of each for burn-in).
fit = fitter.sample(2000, 4)

# Print a quick textual summary of the MCMC results for all parameters and
# the log posterior density, log_p.  For the vector parameter, `fluxes`, a
# summary is printed for *every* element, which may not be desired if there
# are many such parameters.
# print fit

# After sampling, the fit object has attributes for each parameter and
# transformed parameter in the model; the attributes provide access to
# the chains, the pooled collection of samples from all chains, and
# various summary statistics for each parameter (including MCMC output
# diagnostics).

# Verify convergence by looking at the Gelman-Rubin R statistic for every
# parameter of interest; it should be within a few % of 1.
# Also check mixing by looking at the effective sample size.

# For scalars, just make a table.
scalars = [fit.alpha, fit.beta, fit.flux_cut]
print '*** Checks for convergence, mixing ***'
print 'Rhat, ESS for scalar params:'
for param in scalars:
    print '    {0:12s}:  {1:6.3f}  {2:6.0f}'.format(param.name, param.Rhat, param.ess)

# For the vector of latent fluxes, make a plot.
flux_Rhats = [fit.fluxes[i].Rhat for i in range(n_s)]
figure()
subplots_adjust(top=.925, right=.875)  # make room for title, right ESS labels
ax_left = subplot(111)  # left axis for Rhat
plot(range(n_s), flux_Rhats, 'ob')
ylim(0.8, 1.2)
axhline(y=1., ls='--', color='k')
title('Rhat, ESS for fluxes')
xlabel('Source #')
ylabel(r'$\hat R$', color='b')
ax_right = twinx()  # right axis for ESS
flux_ess = [fit.fluxes[i].ess for i in range(n_s)]
plot(range(n_s), flux_ess, 'og')
title('Rhat, ESS for fluxes')
ylabel('ESS', color='g')


# Also check mixing by examining trace plots of parameters of interest,
# making sure there are no obvious trends or strong, long-range correlations.
# Here we look at the scalars alpha, beta (flux_cut is derived from beta so
# needn't be separately checcked).
f=figure(figsize=(10,8))
ax=f.add_subplot(2,1,1)
fit.alpha.trace(axes=ax,alpha=.6)  # without `axes`, this will make its own fig
ax=f.add_subplot(2,1,2)
fit.beta.trace(axes=ax,alpha=.6)

# Look at traces for some fluxes.
f=figure(figsize=(10,8))
ax=f.add_subplot(3,1,1)
fit.fluxes[0].trace(axes=ax,alpha=.6)
ax=f.add_subplot(3,1,2)
fit.fluxes[3].trace(axes=ax,alpha=.6)
ax=f.add_subplot(3,1,3)
fit.fluxes[8].trace(axes=ax,alpha=.6)

# Look at the log_p trace plot.
fit.log_p.trace()  # creates a new fig by default

# Now, *after* the checks, we're ready to make some inferences.

# Show the joint distribution for (alpha,flux_cut) as a scatterplot, and
# marginals as histograms.
f = figure(figsize=(10,8))
subplots_adjust(bottom=.1, left=.1, right=.975, wspace=.24, )

# subplot(232)  # joint at mid-top
f.add_axes([.25, .6, .5, .36])
plot(fit.alpha.thinned, log10(fit.flux_cut.thinned), 'b.', alpha=.4)
# crosshair showing true values:
xhair = { 'color' : 'r', 'linestyle' : ':' , 'linewidth' : '2'}
axvline(alpha, **xhair)
axhline(log10(F_cut), **xhair)
xlabel(r'$\alpha$')
ylabel(r'$\log_{10}F_{c}$')

subplot(223)  # marginal for alpha bottom-left
hist(fit.alpha.thinned, 20, alpha=.4)
axvline(alpha, **xhair)
xlabel(r'$\alpha$')
ylabel(r'$p(\alpha|D)$')

subplot(224)  # marginal for F_cut bottom-right
hist(log10(fit.flux_cut.thinned), 20, alpha=.4)
axvline(log10(F_cut), **xhair)
xlabel(r'$\log_{10}F_{c}$')
ylabel(r'$p(\log_{10}F_c|D)$')


# Make a plot illustrating shrinkage of point estimates.

# Max likelihood estimates:
F_ml = data['counts']/data['exposures']
# Means of marginal posteriors:
F_post = array([fit.fluxes[i].mean for i in range(n_s)])
#F_vals = linspace(.0001, 1.1*fluxes.max(), 200)  # fluxes for PDF plot
u = max(1.1*fluxes.max(), 4*F_cut)
F_vals = logspace(-4, log10(u), 200)  # fluxes for PDF plot
pdf_vals = ncdistn.pdf(F_vals)  # true number count dist'n over F_vals
ax_pdf, ax_pts = shrinkage_plot(F_vals, pdf_vals, fluxes, F_ml, F_post, r'$F$',
                 log_x=True, log_y=False)

# Get a 'best fit' set of parameters by finding the sample with highest
# posterior density; this would be a bad idea in high dimensions (use Stan's
# optimizer in such cases).
i = fit.log_p.thinned.argmax()
a, F = fit.alpha.thinned[i], fit.flux_cut.thinned[i]
# Plot the PDF for the best-fit model.
best = stats.gamma(a=a, scale=F)
ax_pdf.semilogx(F_vals, F_vals*best.pdf(F_vals), 'g--', lw=2, label='Est.')
ax_pdf.legend(frameon=False)

# Show the PDFs for some posterior samples.
for i in 49*linspace(1,10,10,dtype=int):  # every 49th sample
    a, F = fit.alpha.thinned[i], fit.flux_cut.thinned[i]
    distn = stats.gamma(a=a, scale=F)
    ax_pdf.semilogx(F_vals, F_vals*distn.pdf(F_vals), 'k', lw=1, alpha=.5, label=None)
