"""
Function for creating a 2-panel plot illustrating shrinkage of point estimates
in a simple hierarchical Bayesian model.

Created Nov 11, 2014 by Tom Loredo
"""

from numpy import ones_like
from matplotlib.pyplot import *

import myplot
from myplot import close_all, csavefig

ion()
#myplot.tex_on()
csavefig.save = False

dkred = '#882222'

def shrinkage_plot(x_vals, pdf_vals, x_true, x_ml, x_post, xlabel,
                   log_x=False, log_y=False, legend=True):
    """
    Make a plot showing a population distribution as a PDF in a top pane,
    with line plots beneath showing true subject values and the maximum
    likelihood and marginal posterior point estimates.

    Return the PDF and point estimate axes instances.
    """
    est_fig = figure(figsize=(10,8))

    # Axis rectangles:  left, bottom, width, height
    ax_pdf = est_fig.add_axes([.11, .5, .86, .47])
    ax_pts = est_fig.add_axes([.11, .05, .86, .35], frameon=False)
    ax_pts.autoscale_view(scaley=False) # *** seems to not work (or .plot overrides)

    # Plot the hyperprior.
    # True:
    if log_x and log_y:
        ax_pdf.loglog(x_vals, pdf_vals, 'b-', lw=2, label='True')
    elif log_x:
        ax_pdf.semilogx(x_vals, x_vals*pdf_vals, 'b-', lw=2, label='True')
    else:
        ax_pdf.plot(x_vals, pdf_vals, 'b-', lw=2, label='True')

    ax_pdf.set_xlabel(xlabel)
    ax_pdf.set_ylabel('PDF')
    if legend:
        ax_pdf.legend(frameon=False)

    # Plot true values and estimates:
    # Draw horizontal axes:
    y_true = .96
    y_ml = .5
    y_post = 0.04
    ax_pts.axhline(y_true, color='k')
    ax_pts.axhline(y_ml, color='k')
    ax_pts.axhline(y_post, color='k')
    # Don't plot ticks (marjor or minor)
    ax_pts.tick_params(bottom=False, top=False, left=False, right=False,
                       which='both')

    if log_x:
        pt_plot = ax_pts.semilogx
    else:
        pt_plot = ax_pts.plot

    # First draw links between estimates for the same subject:
    for xt, ml, post in zip(x_true, x_ml, x_post):
        pt_plot([xt, ml], [y_true, y_ml], 'k-', lw=1)
        pt_plot([ml, post], [y_ml, y_post], 'k-', lw=1)

    # Then the points:
    ms = 8
    msb = 10
    mew = 0.5
    u = ones_like(x_true)  # unit y values to scale
    pt_plot(x_true, y_true*u, 'bo', mew=mew, ms=ms)
    pt_plot(x_ml, y_ml*u, 'o', mew=mew, mfc=dkred, ms=ms)
    pt_plot(x_post, y_post*u, 'o', mew=mew, mfc='c', ms=ms)

    # Match the PDF and estimate plot limits.
    ax_pts.set_xlim(*ax_pdf.get_xlim())
    ax_pts.set_ylim(0,1)

    # Label the pt axes:
    tdict = { 'fontsize':20, 'verticalalignment':'bottom', 'horizontalalignment':'left',\
        'transform':ax_pts.transAxes }
    ax_pts.text(.02, y_true+.015, 'True', **tdict)
    ax_pts.text(.02, y_ml+.015, 'ML', **tdict)
    ax_pts.text(.02, y_post+.015, 'Post.', **tdict)

    return ax_pdf, ax_pts

