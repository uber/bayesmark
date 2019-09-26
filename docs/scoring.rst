.. _how-scoring-works:

How scoring works
=================

The scoring system is about aggregating the function evaluations of the optimizers. We represent :math:`F_{pmtn}` as the function evaluation of objective function :math:`p` (``TEST_CASE``) from the suggestion of method :math:`m` (``METHOD``) at batch :math:`t` (``ITER``) under repeated trial :math:`n` (``TRIAL``). In the case of batch sizes greater than 1, :math:`F_{pmtn}` is the minimum function evaluation across the suggestions in batch :math:`t`. The first transformation is that we consider the *cumulative minimum* over batches :math:`t` as the performance of the optimizer on a particular trial:

.. math::

   S_{pmtn} = \textrm{cumm-min}_t F_{pmtn}\,.

All of the aggregate quantities described here are computed by :func:`.experiment_analysis.compute_aggregates` (which is called by `bayesmark-anal <#analyze-and-summarize-results>`_) in either the ``agg_result`` or ``summary`` xarray datasets. Additionally, the baseline performances are in the xarray dataset ``baseline_ds`` from :func:`.experiment_baseline.compute_baseline`. The baseline dataset can be generated via the ``bayesmark-baseline`` command, but it is called automatically by ``bayesmark-anal`` if needed.

Median scores
-------------

The more robust, but less decision-theoretically appealing method for aggregation is to look at median scores. On a per problem basis we simply consider the median (``agg_result[PERF_MED]``):

.. math::

   \textrm{med-perf}_{pmt} = \textrm{median}_n \, S_{pmtn} \,.

However, this score is not very comparable across different problems as the objectives are all on different scales with possible different units. Therefore, we decide the *normalized score* (``agg_result[NORMED_MED]``) in a way that is *invariant* to linear transformation of the objective function:

.. math::

   \textrm{norm-med-perf}_{pmt} = \frac{\textrm{med-perf}_{pmt}  - \textrm{opt}_p}{\textrm{rand-med-perf}_{pt} - \textrm{opt}_p} \,,

where :math:`\textrm{opt}_p` (``baseline_ds[PERF_BEST]``) is an estimate of the global minimum of objective function :math:`p`; and :math:`\textrm{rand-med-perf}_{pt}` is the median performance of random search at batch :math:`t` on objective function :math:`p`. This means that, on any objective, an optimizer has score 0 after converging to the global minimum; and random search performs as a straight line at 1 for all :math:`t`. Conceptually, the median random search performance (``baseline_ds[PERF_MED]``) is computed as:

.. math::

   \textrm{rand-med-perf}_{pt} = \textrm{median}_n \, S_{pmtn} \,,

with :math:`m=` random search. However, every observation of :math:`F_{pmtn}` is iid in the case of random search. There is no reason to break the samples apart into trials :math:`n`. Instead, we use the function :func:`.quantiles.min_quantile_CI` to compute a more statistically efficient pooled estimator using the pooled random search samples over :math:`t` and :math:`n`. This pooled method is a nonparametric estimator of the quantiles of the minimum over a batch of samples, which is distribution free.

To further aggregate the performance over all objectives for a single optimizer we can consider the median-of-medians (``summary[PERF_MED]``):

.. math::

   \textrm{med-perf}_{mt} = \textrm{median}_p \, \textrm{norm-med-perf}_{pmt} \,.

Combining scores across different problems is sensible here because we have transformed them all onto the same scale.

Mean scores
-----------

From a decision theoretical perspective it is more sensible to consider the mean (possible warped) score. The median score can hide a high percentage of runs that completely fail. However, when we look at the mean score we first take the clipped score with a baseline value:

.. math::

   S'_{pmtn} = \min(S_{pmtn}, \textrm{clip}_p) \,.

This is largely because there may be a non-zero probably of :math:`F = \infty` (as in when the objective function crashes), which means that mean random search performance is infinite loss. We set :math:`\textrm{clip}_p` (``baseline_ds[PERF_CLIP]``) to the median score after a single function evaluation, which is :math:`\textrm{rand-med-perf}_{p0}` for a batch size of 1. The mean performance on a single problem (``agg_result[PERF_MEAN]``) then becomes:

.. math::

   \textrm{mean-perf}_{pmt} = \textrm{mean}_n \, S'_{pmtn} \,.

Which then becomes a normalized performance (``agg_result[NORMED_MEAN]``) of:

.. math::

   \textrm{norm-mean-perf}_{pmt} = \frac{\textrm{mean-perf}_{pmt}  - \textrm{opt}_p}{\textrm{clip}_p  - \textrm{opt}_p} \,.

Note there that the random search performance is only 1 at the first batch unlike for :math:`\textrm{norm-med-perf}_{pmt}`.

Again we can aggregate this into all objective function performance with (``summary[PERF_MEAN]``):

.. math::

   \textrm{mean-perf}_{mt} = \textrm{mean}_p \, \textrm{norm-mean-perf}_{pmt} \,,

which is a mean-of-means (or *grand mean*), which is much more sensible in general than a median-of-medians. We can again obtain the property of random search having a constant performance of 1 for all :math:`t` using (``summary[NORMED_MEAN]``):

.. math::

   \textrm{norm-mean-perf}_{mt} = \frac{\textrm{mean-perf}_{mt}}{\textrm{rand-mean-perf}_{t}} \,,

where the random search baseline has been determined with the same sequence of equations as the other methods. These all collapse down to:

.. math::

   \textrm{rand-mean-perf}_{t} = \textrm{mean}_p \, \frac{\textrm{rand-mean-perf}_{pt} - \textrm{opt}_p}{\textrm{clip}_p  - \textrm{opt}_p} \,.

Conceptually, we compute this random search baseline (``baseline_ds[PERF_MEAN]``) as:

.. math::

   \textrm{rand-mean-perf}_{pt} = \textrm{mean}_n \, S'_{pmtn} \,,

with :math:`m=` random search. However, because all function evaluations for random search are iid across :math:`t`, we can use a more statistically efficient pooled estimator :func:`.expected_max.expected_min`, which is an unbiased distribution free estimator on the expected minimum of :math:`m` samples from a distribution.

Note that :math:`\textrm{norm-mean-perf}_{mt}` is, in aggregate, a linear transformation on the expected loss :math:`S'`. This makes it more justified in a decision theory framework than the median score. However, to view it as a linear transformation we are considering the values in ``baseline_ds`` to be fixed reference losses values and not the output from the experiment.

Error bars
----------

The datasets ``agg_result`` and ``summary`` also compute error bars in the form of ``LB_`` and ``UB_`` variables. These error bars do not consider the random variation in the baseline quantities from ``baseline_ds`` like ``opt`` and ``clip``. They are instead treated as fixed constant reference points. Therefore, they are computed by a different command ``bayesmark-baseline``. The user can generate the baselines when they want, but since they are not considered a random quantity in the statistics they are not automatically generated from the experimental data (unless the baseline file ``derived/baseline.json`` is missing).

Additionally, the error bars on the grand mean (``summary[PERF_MEAN]``) are computed by simply using t-statistic based error bars on the individual means. Under a "random effects" model, this does not actually lose any statistical power. However, this is computing the mean on the loss over sampling from new problems under the "same distribution" of benchmark problems. These error bars will be wider than if we computed the error bars on the grand mean over this particular set of benchmark problems.
