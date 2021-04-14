"""Python program for golden section search.  This implementation
   reuses function evaluations, saving 1/2 of the evaluations per
   iteration, and returns a bounding interval.
   Source: https://en.wikipedia.org/wiki/Golden-section_search#Iterative_algorithm
   """
import math


inv_phi = (math.sqrt(5) - 1) / 2  # 1 / phi
inv_phi_sq = (3 - math.sqrt(5)) / 2  # 1 / phi^2


def min_gss(f, a, b, tol=1e-12):
    """Golden-section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    Example:
    >>> f = lambda x: (x-2)**2
    >>> a = 1
    >>> b = 5
    >>> tol = 1e-5
    >>> (c,d) = min_gss(f, a, b, tol)
    >>> print(c, d)
    1.9999959837979107 2.0000050911830893
    """

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return a, b

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(inv_phi)))

    c = a + inv_phi_sq * h
    d = a + inv_phi * h
    yc = f(c)
    yd = f(d)

    for k in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = inv_phi * h
            c = a + inv_phi_sq * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = inv_phi * h
            d = a + inv_phi * h
            yd = f(d)

    if yc < yd:
        return a, d
    else:
        return c, b
