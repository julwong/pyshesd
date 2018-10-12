# pylint: disable-msg=E1101
"""
python clone of the s-h-esd R implementation and a wrapper of STL

STL:
Initial Fortran code available at:
http://netlib.bell-labs.com/netlib/a/stl.gz
Initial Authors: R. B. Cleveland, W. S. Cleveland, J. E. McRae, and
I. Terpenning, 1990.
Simple-to-double precision conversion of the Fortran code by Pierre
Gerard-Marchant, 2007/03.

s-h-esd:
Initial Fortran code available at:
https://github.com/twitter/AnomalyDetection
"""
from numpy import array
import pandas
from scipy import stats
from math import sqrt
import _stl

#####---------------------------------------------------------------------------
#--- --- STL ---
#####---------------------------------------------------------------------------
def stl(y, np=12, ns=7, nt=None, nl=13, isdeg=1, itdeg=1, ildeg=1,
        nsjump=None,ntjump=None,nljump=None, robust=True, ni=None,no=None,periodic=False,index=None):
    """Decomposes a time series into seasonal and trend  components.

:Parameters:
    y : Numerical array
        Time Series to be decomposed.
    np : Integer *[12]*
        Period of the seasonal component.
        For example, if  the  time series is monthly with a yearly cycle, then
        np=12.
    ns : Integer *[7]*
        Length of the seasonal smoother.
        The value of  ns should be an odd integer greater than or equal to 3.
        A value ns>6 is recommended. As ns  increases  the  values  of  the
        seasonal component at a given point in the seasonal cycle (e.g., January
        values of a monthly series with  a  yearly cycle) become smoother.
    nt : Integer *[None]*
        Length of the trend smoother.
        The  value  of  nt should be an odd integer greater than or equal to 3.
        A value of nt between 1.5*np and 2*np is  recommended. As nt increases,
        the values of the trend component become  smoother.
        If nt is None, it is estimated as the smallest odd integer greater
        or equal to (1.5*np)/[1-(1.5/ns)]
    nl : Integer *[None]*
        Length of the low-pass filter.
        The value of nl should  be an odd integer greater than or equal to 3.
        The smallest odd integer greater than or equal to np is used by default.
    isdeg : Integer *[1]*
        Degree of locally-fitted polynomial in seasonal smoothing.
        The value is 0 or 1.
    itdeg : Integer *[1]*
        Degree of locally-fitted polynomial in trend smoothing.
        The value is 0 or 1.
    ildeg : Integer *[1]*
        Degree of locally-fitted polynomial in low-pass smoothing.
        The value is 0 or 1.
    nsjump : Integer *[None]*
        Skipping value for seasonal smoothing.
        The seasonal smoother skips ahead nsjump points and then linearly
        interpolates in between.  The value  of nsjump should be a positive
        integer; if nsjump=1, a seasonal smooth is calculated at all n points.
        To make the procedure run faster, a reasonable choice for nsjump is
        10%-20% of ns. By default, nsjump= 0.1*ns.
    ntjump : Integer *[1]*
        Skipping value for trend smoothing. If None, ntjump= 0.1*nt
    nljump : Integer *[1]*
        Skipping value for low-pass smoothing. If None, nljump= 0.1*nl
    robust : Boolean *[True]*
        Flag indicating whether robust fitting should be performed.
    ni : Integer *[None]*
        Number of loops for updating the seasonal and trend  components.
        The value of ni should be a positive integer.
        See the next argument for advice on the  choice of ni.
        If ni is None, ni is set to 1 for robust fitting, to 5 otherwise.
    no : Integer *[0]*
        Number of iterations of robust fitting. The value of no should
        be a nonnegative integer. If the data are well behaved without
        outliers, then robustness iterations are not needed. In this case
        set no=0, and set ni=2 to 5 depending on how much security
        you want that  the seasonal-trend looping converges.
        If outliers are present then no=3 is a very secure value unless
        the outliers are radical, in which case no=5 or even 10 might
        be better.  If no>0 then set ni to 1 or 2.
        If None, then no is set to 15 for robust fitting, to 0 otherwise.

Returns:
    A recarray of estimated trend values ('trend'), estimated seasonal
    components ('seasonal'), local robust weights ('weights') and fit
    residuals ('residuals').
    The final local robust weights are all 1 if no=0.

Reference
---------

    R. B. Cleveland, W. S. Cleveland, J. E. McRae and I. Terpenning.
    1990. STL: A Seasonal-Trend Decomposition Procedure Based on LOESS
    (with Discussion). Journal of Official Statistics, 6:3-73.

    """
    ns = max(ns, 3)
    n = len(y)
    if ns is None:
        ns = 7
    if periodic:
        ns = 10 * n + 1
        isdeg = 0
    ns = max(3, ns)
    if ns%2 == 0:
        ns += 1
    np = max(2, np)
    if nt is None:
        nt = max(int((1.5*np/(1.-1.5/ns))+0.5), 3)
        if not nt%2:
            nt += 1
    if nl is None:
        nl = max(3,np)
        if not nl%2:
            nl += 1
    if nsjump is None:
        nsjump = int(0.1*ns + 0.9)
    if ntjump is None:
        ntjump = int(0.1*nt + 0.9)
    if nljump is None:
        nljump = int(0.1*nl + 0.9)
    if robust:
        if ni is None:
            ni = 1
        if no is None:
            no = 15
    else:
        if ni is None:
            ni = 5
        if no is None:
            no = 0

    #if hasattr(y,'_mask') and numpy.any(y._mask):
    #    raise ValueError("Missing values should first be filled !")
    y = array(y, subok=True, copy=False, order="Fortran").ravel(order="F")
    (rw,szn,trn,work) = _stl.stl(y,np,ns,nt,nl,isdeg,itdeg,ildeg,
                                 nsjump,ntjump,nljump,ni,no,)
    if periodic:
        for i in range(np):
            szn[i::np] = szn[i::np].mean()
    result = {
        "trend":trn,
        "seasonal":szn,
        "resid":y-trn-szn,
        "weight":rw
    }
    return pandas.DataFrame(result,index=index)

def shesd(data, k=0.05, alpha=0.05, num_obs_per_period=None,
                         use_esd=False, kind='two_tail',
                         verbose=False):
    """
    Detects anomalies in a time series using S-H-ESD.
 
    :Parameters:
        data: Pandas Time series to perform anomaly detection on.
             s = pd.Series(data=data['CR'].values,index=data['ATS'].apply(pd.Timestamp).values) e.g.
 	k: Float *[0.49]*
            Maximum number of anomalies that S-H-ESD will detect as a percentage of the data.
 	alpha: Float *[0.05]*
            The level of statistical significance with which to accept or reject anomalies.
 	num_obs_per_period: Integer *[None]*
            Defines the number of observations in a single period, and used during seasonal decomposition.
 	use_esd: Boolean *[True]*
            Uses regular ESD instead of hybrid-ESD. Note hybrid-ESD is more statistically robust.
 	kind: String *[two_tail]*
            Wich kind of test to be perform, should be exactly one of two_tail/upper_tail/lower_tail.
 	verbose: Boolean *[False]*
            Additionally printing for debugging.
    Returns:
        A list containing the anomalies (anoms) and decomposition components (stl).
    """
    if num_obs_per_period is None:
        raise ValueError("must supply period length for time series decomposition")
    
    num_obs = len(data)

    # Check to make sure we have at least two periods worth of data for anomaly context
    if num_obs < num_obs_per_period * 2:
        raise ValueError("Anom detection needs at least 2 periods worth of data")

    if kind not in ('upper_tail', 'lower_tail', 'two_tail'):
        raise ValueError("kind should be one of upper_tail/lower_tail/two_tail")

    # TODO: Handle NAs
    
    # -- Step 1: Decompose data. This returns a univarite remainder which will be used for anomaly detection. Optionally, we might NOT decompose.
    data_decomp = stl(data.values, np=num_obs_per_period, periodic=True, robust=True, index=data.index)
    
    # Remove the seasonal component, and the median of the data to create the univariate remainder
    if use_esd:
        data = data - data_decomp["trend"] - data_decomp["seasonal"]
    else:
        data = data - data.median() - data_decomp["seasonal"]
    
    # Store the smoothed seasonal component, plus the trend component for use in determining the "expected values" option
    data_decomp = pandas.Series((data_decomp["trend"]+data_decomp["seasonal"]).values, index=data.index)

    # Maximum number of outliers that S-H-ESD can detect (e.g. 49% of data)
    max_outliers = int(num_obs*k)

    if max_outliers == 0:
      raise ValueError("With longterm=TRUE, AnomalyDetection splits the data into 2 week periods by default. You have %s observations in a period, which is too few. Set a higher piecewise_median_period_weeks." % num_obs)

    ## Define values and vectors.
    n = len(data)
    R_idx = [None for _ in range(max_outliers)]
    num_anoms = 0
    latest_idx = data.index[-1]

    if use_esd:
        def ma_func(data):
            return data.mean()
        def sigma_func(data):
            return data.std()
    else:
        def ma_func(data):
            return data.median()
        def sigma_func(data):
            return data.mad()

    # Compute test statistic until r=max_outliers values have been
    # removed from the sample.
    for i in range(max_outliers):
        if verbose:
            print("%s/%s completed" % (i, max_outliers))

        # if kind == 'upper_tail':
        #     ares = data - data.median()
        # elif kind == 'lower_tail':
        #     ares = data.median() - data
        # else:
        #     ares = (data - data.median()).abs()
        if kind == 'upper_tail':
            ares = data - ma_func(data)
        elif kind == 'lower_tail':
            ares = ma_func(data) - data
        else:
            ares = (data - ma_func(data)).abs()
        

        # protect against constant time series
        # data_sigma = data.mad()
        data_sigma = sigma_func(data)
        if data_sigma == 0:
            break

        ares = ares/data_sigma

        #TODO: speed up with duplicated value R
        temp_max_idx = ares.idxmax()
        R = ares[temp_max_idx]
        R_idx[i] = temp_max_idx
        data = data.drop(index=temp_max_idx)
      
        ## Compute critical value.
        if kind == 'tow_tail':
            p = 1 - alpha/(2*(n-i+1))
        else:
            p = 1 - alpha/(n-i+1)
           
        t = stats.t.ppf(p, n-i-1)
        lam = t*(n-i) / sqrt((n-i-1+pow(t,2))*(n-i+1))
        if R > lam:
            num_anoms = i
    
    if num_anoms > 0:
      R_idx =  R_idx[:num_anoms]
      latest_obs_outlier = latest_idx in R_idx
      if verbose:
          print("anomaly rate:%s" % (len(R_idx)/len(data)))
    else:
      R_idx = None
      latest_obs_outlier = False
      if verbose:
          print("timeseries is clean")

    return dict(anoms = R_idx, stl = data_decomp, latest_obs_outlier = latest_obs_outlier)
