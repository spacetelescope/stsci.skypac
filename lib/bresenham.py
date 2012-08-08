"""
This module has an implementation of the Bresenham's Line Algorithm.

http://www.barricane.com/2009/08/28/bresenhams-line-algorithm-in-python.html
    
"""
# STDLIB
import sys

__all__ = ['line', 'lines']

def _irange(start, finish):
    """
    This is a naive inclusive range function.

    Parameters
    ----------
    start, finish : int
        Range limits, inclusive.

    Returns
    -------
    Iterator for the range.

    Examples
    --------
    >>> list(_irange(1, 1))
    [1]

    >>> list(_irange(1, 4))
    [1, 2, 3, 4]

    >>> list(_irange(4, 1))
    [4, 3, 2, 1]

    """
    if start <= finish:
        i = 1
    else:
        i = -1

    return xrange(start, finish + i, i)


def line((x0, y0), (x1, y1)):
    """
    Bresenham's Line Algorithm.

    Parameters
    ----------
    (x0, y0), (x1, y1) : tuples of int

    Returns
    -------
    points : list of tuple of int
        All the points between input points, including the inputs.

    Examples
    --------
    >>> line((0, 0), (4, 4))
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    >>> line((0, 0), (3, 5))
    [(0, 0), (1, 1), (1, 2), (2, 3), (2, 4), (3, 5)]

    >>> line((1, 1), (4, 6))
    [(1, 1), (2, 2), (2, 3), (3, 4), (3, 5), (4, 6)]

    Test lines go from start to finish.
    
    >>> line((-1, -1), (-6, -4))
    [(-1, -1), (-2, -2), (-3, -2), (-4, -3), (-5, -3), (-6, -4)]

    >>> line((-6, -4), (-1, -1))
    [(-6, -4), (-5, -3), (-4, -3), (-3, -2), (-2, -2), (-1, -1)]

    Test that the same points are generated in the opposite direction.
    
    >>> a = line((0, 0), (29, 43))
    >>> b = line((0, 0), (-29, -43))
    >>> c = [(-x, -y) for (x, y) in b]
    >>> a == c
    True

    Test that the the same points are generated when the line is mirrored
    on the x=y line.

    >>> d = line((0, 0), (43, 29))
    >>> e = [(y, x) for (x, y) in d]
    >>> a == e
    True

    """
    points = []
    orig_x0 = x0
    orig_y0 = y0

    if abs(y1 - y0) > abs(x1 - x0):
        steep = True
    else:
        steep = False

    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = deltax / 2
    y = y0
    if y0 < y1:
        ystep = 1
    else:
        ystep = -1

    for x in _irange(x0,x1):
        if steep:
            points.append((y,x))
        else:
            points.append((x,y))
        error -= deltay
        
        if error < 0:
            y += ystep
            error += deltax

    # If the points go in the wrong direction, reverse them.
    if points[0] != (orig_x0, orig_y0):
        points.reverse()

    return points


def lines(*args):
    """
    Like `line` but for many input points.

    Parameters
    ----------
    *args : Multiple (x,y) points

    Returns
    -------
    all_points : list of tuple of int
        All the points between input points, including the inputs.

    Examples
    --------
    >>> lines((-1,0), (1,1), (1,4))
    [(-1, 0), (0, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    
    """
    argc = len(args)
    assert argc > 0

    all_points = [args[0]]
    
    for i in xrange(argc - 1):
        cur_points = line(args[i], args[i+1])
        all_points += cur_points[1:]  # Skip first elem to avoid repeat

    return all_points
