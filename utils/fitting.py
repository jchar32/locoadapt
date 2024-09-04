import numpy as np
from scipy import optimize
from scipy import stats


def exponential(x, a, b, c):
    """Calculate the output of a three-term exponential given the input x and parameters a, b, and c.

    Parameters
    ----------
    x : ndarray
        x values to be evaluated based on the function
    a : float
        coefficient for direction of  growth or decay with respect to y axis
    b : float
        coefficient for growth rate or decay rate
    c : float
        coefficient for offset in y axis

    Returns
    -------
    ndarray
        y = f(x) = a * exp(-b * x) + c
    """
    return a * np.exp(-b * x) + c


def inverse_exponential(y, a, b, c):
    """Inverse exponential function to solve for x given y and parameteres a, b, and c.

    Parameters
    ----------
    y : ndarray
        predicted values from an exponential function to be evaluated for their corresponding x values
    a : float
        coefficient for direction of  growth or decay with respect to y axis
    b : float
        coefficient for growth rate or decay rate
    c : float
        coefficient for offset in y axis

    Returns
    -------
    ndarray
        x = f(y) = -ln((y - c) / a) / b
    """
    return -1 * np.log((y - c) / a) / b


def linear(x, m, b):
    """Simple linear function to calculate y given x and parameters m and b.

    Parameters
    ----------
    x : ndarray
        x values to be evaluated based on the function
    m : _type_
        coefficient for slope
    b : _type_
        coefficient for offset in y axis

    Returns
    -------
    ndarray
        y = f(x) = m * x + b
    """
    return m * x + b


def piecewise_linear(x, x0, y0, k1, k2):
    """
    Compute the piecewise linear function.

    Parameters
    ----------
    x : array_like
        The input array of x values for fitting.
    x0 : float
        The x-coordinate of the breakpoint.
    y0 : float
        The y-coordinate of the breakpoint.
    k1 : float
        The slope of the line segment before the breakpoint.
    k2 : float
        The slope of the line segment after the breakpoint.

    Returns
    -------
    array_like
        The computed values of the piecewise linear function.

    """
    return np.piecewise(
        x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0]
    )


def fit_piecewise_linear(x, y):
    """Wrapper for scipy.optimize.curve_fit to fit a piecewise linear function to the input x and y data.
    This uses the self-starter method to estimate the initial parameters (x0, y0, k1, k2) for the piecewise linear function.

    Parameters
    ----------
    x : ndarray
        x values to evaluate the function on
    y : ndarray
        y values to optimize the function parameters on

    Returns
    -------
    tuple[ndarray, ndarray, dict, str]
        A tuple is returned containing:
        popt- ndarray: optimal parameters for the piecewise linear function
        pcov- ndarray: covariance matrix for the parameters
        infodict- dict: information about the optimization
        mesg- str: message about the optimization
    """
    popt, pcov, infodict, mesg, _ = optimize.curve_fit(
        piecewise_linear, x, y, full_output=True, p0=[0, 0, 0, 0], maxfev=10000
    )
    return popt, pcov, infodict, mesg


def piecewise_breakpoint(y):
    """Compute the breakpoint of a piecewise linear function.

    Parameters
    ----------
    x : array_like
        The input array of x values for fitting.
    y : array_like
        The input array of y values for fitting.

    Returns
    -------
    float
        The computed breakpoint of the piecewise linear function.

    """
    x0 = np.argmin(np.abs(np.diff(y)))
    return x0


def fit_exp(x, y):
    """Wrapper for scipy.optimize.curve_fit to fit an exponential function to the input x and y data.
    This uses the self-starter method to estimate the initial parameters (a, b, c) for the exponential function.

    Parameters
    ----------
    x : ndarray
        x values to evaluate the function on
    y : ndarray
        y values to optimize the function parameters on

    Returns
    -------
    tuple[ndarray, ndarray, dict, str]
        A tuple is returned containing:
        popt- ndarray: optimal parameters for the exponential function
        pcov- ndarray: covariance matrix for the parameters
        infodict- dict: information about the optimization
        mesg- str: message about the optimization
    """
    popt, pcov, infodict, mesg, _ = optimize.curve_fit(
        exponential, x, y, full_output=True, p0=[1, 1e-6, 1], maxfev=10000
    )
    return popt, pcov, infodict, mesg


def fit_lin(x, y):
    """Wrapper for scipy.optimize.curve_fit to fit an linear function to the input x and y data.
    This uses the self-starter method to estimate the initial parameters (m, b) for the linear function.

    Parameters
    ----------
    x : ndarray
        x values to evaluate the function on
    y : ndarray
        y values to optimize the function parameters on

    Returns
    -------
    tuple[ndarray, ndarray, dict, str]
        A tuple is returned containing:
        popt- ndarray: optimal parameters for the linear function
        pcov- ndarray: covariance matrix for the parameters
        infodict- dict: information about the optimization
        mesg- str: message about the optimization
    """
    popt, pcov, infodict, mesg, _ = optimize.curve_fit(linear, x, y, full_output=True)
    return popt, pcov, infodict, mesg


def rsquared(y, predy):
    """Calculate the R-squared value for the input y and predicted y values of a function.

    Parameters
    ----------
    y : ndarray
        measured y values
    predy : ndarray
        predicted y values of a function

    Returns
    -------
    float
        R-squared 1- (residual sum of squares / total sum of squares)
    """
    ss_res = np.sum((y - predy) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def adjusted_rsquared(y, predy, n, k):
    """
    Calculate the adjusted R-squared value.

    Parameters:
    -----------
    y (array-like): The observed values.
    predy (array-like): The predicted values.
    n (int): The number of observations.
    k (int): The number of predictors.

    Returns:
    --------
    float: The adjusted R-squared value.
    """
    rsqrd = rsquared(y, predy)
    adj_rsqrd = 1 - (((1 - rsqrd) * (n - 1)) / (n - k - 1))
    return adj_rsqrd


def aic(data, fitted_params, distribution="norm"):
    if distribution == "norm":
        logLik = np.sum(stats.norm.logpdf(data, fitted_params))
    elif distribution == "gamma":
        logLik = np.sum(stats.gamma.logpdf(data, fitted_params))
    elif distribution == "exponential":
        logLik = np.sum(stats.expon.logpdf(data, fitted_params))

    k = len(fitted_params)
    aic = 2 * k - 2 * logLik
    return aic


def rad_of_curve(xdata, ydata):
    """Calculate the radius of a curvature. This was implemented to calculate the radius of curvature of the exponential fit to the data with the goal of finding the inflection point.

    Parameters
    ----------
    xdata : ndarray
        X values that the function was evaluated on
    ydata : ndarray
        predicted y values of the function

    Returns
    -------
    float
        radius of the curvature (1 + y'**2)**1.5 / |y''| where y' and y'' are the first and second derivatives of the function evaluated at xdata and ydata
    """
    yd = derivative(xdata, ydata)
    ydd = derivative(xdata, yd)
    return ((1 + yd**2) ** 1.5) / np.abs(ydd)


def derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Wrapper for numpy.gradient to calculate the first derivative of the input y values with respect to the input x values.

    Parameters
    ----------
    x : ndarray
        x values
    y : ndarray
        corresponding y values

    Returns
    -------
    ndarray
        the first derivative of y with respect to x
    """
    return np.gradient(y, x)


def fit_inv_exp(query_pts: list, fit_params: dict) -> dict:
    x_at_q = dict.fromkeys(query_pts, None)
    for q in query_pts:
        x_at_q[q] = inverse_exponential(q, *fit_params)
    return x_at_q


def fit_inv_lin(query_pts: list, fit_params: dict) -> dict:
    x_at_q = dict.fromkeys(query_pts, None)
    for q in query_pts:
        x_at_q[q] = (q - fit_params[1]) / fit_params[0]
    return x_at_q
