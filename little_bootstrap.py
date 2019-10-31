import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split

# todo: consider setting up so default missing is Y rather than X

class LBOB():

    def __init__(self):
        self.x = None
        self.y = None
        # Sample size in outer loop, b, n**0.67 used in literature.
        # If an integer it is taken as the number of rows to sample;
        # if less than 1, the sample size is taken as n**subsample_size.
        self.subsample_size = .67
        # Number of sub-samples in outer loop, s, >3 for simple funcs.
        self.n_subsamples = 20
        # Iterations for each subsample, r, >50 for simple funcs.
        self.n_trials = 30
        # Whether the scoring function uses value frequencies.
        self.use_freqs = False
        # Function used to score sampled values.
        # Is required to have three parameters even if not used.
        self.score_func = lambda x, y, freq: np.average(y, axis=0, weights=freq)
        # Function used to aggregate separate estimates made in
        # the inner loop. Needs to operate on an array_like object.
        self.agg_func = np.mean
        # Estimates generated on each iteration of the outer loop.
        # These are used to estimate the upper and lower bounds.
        self.scores = None
        # Final value calculated in the outer loop, mean of self.scores.
        self.mean = None
        self.lo_bound = None
        self.hi_bound = None


def lbob_get_multinomial_sample(X=None, Y=None, sample_size=None):
    """Draw a multinomial sample from a vector or feature matrix, vector combination.

    Samples with replacement. Does not return a set of frequencies.

    :param array_like X: Feature matrix.
    :param 1-D array_like Y: Dependent variables or values.
    :param int or float sample_size: Number of rows in returned arrays. Defaults to size of input vector Y.
    :rtype: tuple of arrays
    :return: Rows in X and Y corresponding to samples drawn.
    """
    if sample_size is None:
        sample_size = Y.size

    # rows = np.linspace(0, Y.size-1, Y.size)
    # print rows.size, rows
    rows = np.random.randint(0, Y.size, sample_size)
    # rows = list(np.sort())

    Ysamp = Y[rows]
    if X is not None:
        Xsamp = X[rows, :]
    else:
        Xsamp = None
    return Xsamp, Ysamp


def lbob_get_multinomial_sample_with_freqs(X=None, Y=None, sample_size=None):
    """Draw a multinomial sample from a vector or feature matrix, vector combination.

    Returns the rows sampled and frequencies with which they were sampled.
    Unsampled rows are not returned.

    :param 1-D array_like X: Feature matrix.
    :param array_like Y: Dependent variables or values.
    :param int or float sample_size: Number of rows in returned arrays. Defaults to size of input vector Y.
    :rtype: tuple of arrays
    :return: Rows in X and Y corresponding to samples drawn.
    """
    if sample_size is None:
        sample_size = Y.size
    n_rows = Y.size

    freq_by_row = np.random.multinomial(sample_size, [1./n_rows]*n_rows)
    rows = np.nonzero(freq_by_row)
    freq = freq_by_row[rows]

    Ysamp = Y[rows]
    if X is not None:
        Xsamp = X[rows, :][0]
    return Xsamp, Ysamp, freq


# todo: check that frequencies are provided when use_freq is true?
def lbob_score_func(Xeval=None, Yeval=None, freq=None):
    """Default scoring function. Returns the average of input array.

    If freq array is supplied it is interpreted as the frequencies
    of the corresponding array elements. Three parameters are
    maintained in all scoring functions.


    :param None Xeval: None
    :param array_like Yeval: Vector of values.
    :param array_like freq: Vector of frequencies corresponding to the values.
    :rtype : float
    :return: Average of input values.
    """
    return np.average(Yeval, axis=0, weights=freq)


def lbob_agg_func(scores):
    """Default aggregation function. Returns mean of input vector.

    :param array_like scores: Array of values to aggregate.
    :return: Mean of input array.
    """
    return scores.mean()


def lbob_little_boot(X, Y, n_trials, sample_size, use_freqs, score_func=None, agg_func=None):
    """Implements the inner loop of little-bag-of-bootstraps.

    Carries out n_trials of resampling with replacement and calculates
    a score on each sample using the scoring function. Applies the
    aggregation function to the scores, usually a mean, and returns
    a single value.

    :param array_like X: Feature matrix.
    :param array_like Y: Vector of values or dependent variables.
    :param int n_trials: Number of times to sample.
    :param int sample_size: Effective sample size of each sample.
    :param bool use_freqs: If True then frequencies are used.
    :param func score_func:
    :param func agg_func:
    :rtype float
    :return: result of aggregating the score on each sample.
    """
    freq = None
    scores = np.zeros((n_trials, 1))
    if score_func is None:
        score_func = lbob_score_func
    if agg_func is None:
        agg_func = lbob_agg_func

    for t in range(n_trials):
        if use_freqs:
            Xeval, Yeval, freq = lbob_get_multinomial_sample_with_freqs(X, Y, sample_size)
        else:
            Xeval, Yeval = lbob_get_multinomial_sample(X, Y, sample_size)
        scores[t, ] = score_func(Xeval, Yeval, freq)
    subsample_score = agg_func(scores)
    return subsample_score

# todo: check that samples requested are not larger than total samples
def lbob_big_boot(lbob, X=None, Y=None):
    """Implements the outer loop of little-bag-of-bootstraps.

    See the LOB initialization for a list of parameters.

    :param LBOB lbob: An LBOB object containing necessary data and parameters.
    :return: Unnecessarily returns the modified LBOB.
    """
    if X is not None:
        lbob.x = X
    if Y is not None:
        lbob.y = Y
    if lbob.x is None:
        lbob.x = np.zeros_like(lbob.y)
    Yunused = lbob.y
    Xunused = lbob.x

    subsample_size = lbob.subsample_size
    if subsample_size < 1:
        subsample_size = int(np.power(lbob.y.size, subsample_size))
    n_trials = lbob.n_trials
    sample_size = lbob.y.size
    use_freqs = lbob.use_freqs

    scores = np.zeros((lbob.n_subsamples, 1))

    # Call inner bootstrap (little_boot) with different data n_subsample times.
    for subset in range(lbob.n_subsamples):
        if Yunused.size <= subsample_size:
            print('Warning: Reusing original data in lbob_big_boot. Sub-samples have {} rows'.format(subsample_size))
            Xunused = lbob.x
            Yunused = lbob.y
        Xsamp, Xunused, Ysamp, Yunused = train_test_split(Xunused, Yunused, test_size=(Yunused.size - subsample_size))
        scores[subset, 0] = lbob_little_boot(Xsamp, Ysamp, n_trials, sample_size, use_freqs,
                                             score_func=lbob.score_func, agg_func=lbob.agg_func)
    lbob.scores = scores
    lbob.mean = scores.mean()
    lbob.lo_bound = np.percentile(scores, 5)
    lbob.hi_bound = np.percentile(scores, 95)
    return lbob

def lbob_histogram(lbob, actual=None):
    score_vec = lbob.scores[:, ]
    score_mean = lbob.mean
    cnt, ins, patch = plt.hist(score_vec, bins=15, histtype='stepfilled', normed=True,
                               color='b', label='Scores')
    p5 = np.percentile(score_vec, 5)
    p95 = np.percentile(score_vec, 95)

    # These require the seaborn libraries
    #sns.rugplot(score_vec[:, 0], color='red', linewidth=3)###############
    #sns.kdeplot(score_vec[:, 0], shade=False)############################
    plt.plot([p5, p5], [0, cnt.max()*1.1], color='grey', ls='dotted', linewidth=3)
    plt.plot([p95, p95], [0, cnt.max()*1.1], color='grey', ls='dotted', linewidth=3)
    plt.plot([score_mean, score_mean], [0, cnt.max()*1.1], color='blue', ls='dotted',
             linewidth=3)
    if actual:
        plt.plot([actual, actual], [0, cnt.max()+1.1], color='red', ls='solid', linewidth=5)
    plt.title("Estimated Value at Each Iteration")
    plt.xlabel("Values Calculated for Individual Iterations")
    plt.ylabel("Frequency")
    plt.show()


def lbob_convergence_plot(lbob, title='Convergence of Est Mean', actual=None, n_trials=20):
    n_subsamples = lbob.n_subsamples
    x = np.array(np.linspace(1, n_subsamples, n_subsamples))

    avg_arr = np.empty((n_trials, 1))
    for trial in range(n_trials):
        lbob_big_boot(lbob)
        convergence = lbob.scores[:, ].cumsum()/x
        plt.plot(x, convergence, color='blue')
        avg_arr[trial, 0] = lbob.mean

    print('\n\nConvergence Trial')
    print('True mean and standard deviation of the DISTRIBUTION are: {:.6}, {:.6}'.format(lbob.y.mean(), lbob.y.std()))
    print('\nMean and standard deviation of separate TRIALS are: {:.6}, {:.6}'.format(avg_arr.mean(), avg_arr.std()))

    # Add reference line for true value to plot
    if actual:
        plt.plot([1, lbob.n_subsamples], [actual, actual], linestyle='dotted', c='black')
    plt.title('{} with {:,d} Sub-samples, and {:,d} Trials'.format(title, lbob.n_subsamples, n_trials))
    plt.xlabel('Sample Iteration')
    plt.ylabel('Estimated Value at Each Iteration')
    plt.show()


def lbob_check_bumpy_mean(n_samples=100000):
    n1 = int(n_samples/2)
    n2 = n_samples - n1
    zeros = np.zeros((n_samples, 1))
    bump = np.zeros((n_samples, 1))
    bump[0:n1, ] = np.array(np.random.normal(100, 40, n1), ndmin=2).T
    bump[n1:n_samples, ] = np.array(np.random.triangular(650, 700, 710, n2), ndmin=2).T
    bump_mean = bump.mean()

    plt.hist([bump], bins=150, histtype='stepfilled', normed=True, alpha=0.5)
    plt.title('Histogram for the Very Bumpy Distribution')
    plt.show()
    print('\nBootstrap the Mean of a Pathologic Distribution')
    print('The true mean is {:.8}'.format(bump_mean))

    l = LBOB()
    l.x = zeros
    l.y = bump
    l.use_freqs = True
    lbob_convergence_plot(lbob=l, title='Convergence and Estimated Mean of Bumpy Data', actual=bump_mean)


def lbob_check_norms(n_samples=10000):
    # lbob_check_norms generates an approximately standard normal distribution
    # Initially it calculates the mean and standard deviation on the sampled
    # distribution. It then uses the little-bag-of-bootstraps technique to
    # estimate the mean, standard deviation, and 97.5th percentile of the
    # sampled distribution.

    # Generate the normal distribution and calculate the 'true' stats
    norm_dist = np.array(np.random.normal(0, 1, n_samples), ndmin=2).T
    norm_mean = norm_dist.mean()
    norm_std = norm_dist.std()
    norm_p975 = np.percentile(norm_dist, 97.5)

    # Plot a simple histogram to look at the distribution
    plt.hist([norm_dist], bins=100, histtype='stepfilled', normed=True, alpha=0.5)
    plt.title('Approx Std Norm Dist, N={:,d}  Mean={:.4}  Std Dev={:.4}  97.5th Percentile={:.5}'.format(
                n_samples, norm_mean, norm_std, norm_p975))
    plt.show()

    # Define the data and parameters
    l = LBOB()
    l.y = norm_dist
    l.n_subsamples = 25
    l.subsample_size = 0.6
    l.n_trials = 50

    # Get estimates for the distribution stats repetitively and plot
    # how they converge. Normally only one series would be calculated.
    # The difference in the estimates on the last iteration gives an
    # indication of how much variance one can expect in an estimated
    # answer.

    lbob_convergence_plot(lbob=l, title='Std Normal, Mean, no Freqs',
                          actual=norm_mean)
    l.score_func = lambda x, y, freq: np.std(y)
    lbob_convergence_plot(lbob=l, title='Std Normal, Std, no Freqs',
                          actual=norm_std)
    l.score_func = lambda x, y, freq: np.percentile(y, 97.5)
    lbob_convergence_plot(lbob=l, title='Std Normal, 97.5th%, no Freqs',
                          actual=norm_p975)


def lbob_check_basic(n_samples=100000, score_func=lambda x, y, freq: y.mean()):
    # lbob_check_basic generates an approximate standard normal
    # distribution and estimates the mean sampling. Histograms
    # are plotted so that one can visually check that the
    # results are reasonable.

    # Generate an approximately standard normal distribution
    # Calculate and print the resulting mean and standard deviation
    norm_dist = np.array(np.random.normal(0, 1, n_samples), ndmin=2).T
    out = 'Approx Std Norm Dist as Sampled, N={} Mean={:.5} Std Dev={:.5}'.format(
            n_samples, norm_dist.mean(), norm_dist.std())
    print(out)
    # Specify the parameters and data to use
    l = LBOB()
    l.y = norm_dist
    l.use_freqs = False
    l.subsample_size = 0.67
    l.n_subsamples = 50
    l.n_trials = 30
    if score_func:
        l.score_func = score_func

    # Run the bootstrap and then print the results
    lbob_big_boot(l)
    print('\nResults of Sampling from an Approximately Standard Normal Distribution')
    print('The estimated metric is: {:.5}'.format(l.mean))
    #lbob_histogram(l, actual=norm_dist.mean())


#if __name__ == '__main__':
    import timeit
    #lbob_check_basic(1000, score_func=lambda x, y, freq: y.mean())
    #lbob_check_basic(1000, score_func=lambda x, y, freq: np.std(y))
    #lbob_check_basic(1000, score_func=lambda x, y, freq: np.percentile(y, 95))
    #lbob_check_norms(n_samples=10000)
    #lbob_check_bumpy_mean()

