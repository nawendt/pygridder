# Copyright (c) 2025 Patrick Marsh.
# Distributed under the terms of the BSD 3-Clause License.
"""Simple Python class to grid points, lines, and polygons."""

import numpy as np
import scipy.spatial as ss
from skimage import draw as skdraw


class Gridder:
    """Class for gridding points,lines, and polygons.

    A simple class that uses a KDTree to allow for gridding of points, lines, and polygons on
    a regular grid.
    """

    def __init__(self, tx, ty, dx=np.inf, centered=False):
        """Create a KDTree lookup object.

        Parameters
        ----------
        tx : array_like
            2-d  x-coordiantes of the grid.

        ty : array_like
            2-d  y-coordiantes of the grid.

        dx : float
            The delta between grid grid coordinates [assumed regular grid].
            Default is infinity.

        centered : bool
            Flag indicating whether gridpoints denote center or lower-left corner of grid if
            not center, move grid points to center to facility lookup. Default is False.
        """
        tx = np.array(tx, copy=True, subok=True)
        ty = np.array(ty, copy=True, subok=True)
        if not centered:
            x = tx.copy()
            y = ty.copy()
            x[:, :-1] = tx[:, :-1] + (tx[:, 1:] - tx[:, :-1]) / 2.
            x[:, -1] += (tx[:, -1] - tx[:, -2]) / 2.
            y[:-1, :] = ty[:-1, :] + (ty[1:, :] - ty[:-1, :]) / 2.
            y[-1, :] += (ty[-1, :] - ty[-2, :]) / 2.
            tx = x
            ty = y
            del x
            del y
        self.tx = tx
        self.ty = ty
        self.dx = dx
        self.tpoints = np.asarray(list(zip(self.tx.ravel(), self.ty.ravel(), strict=True)))
        self.tree = ss.KDTree(self.tpoints)

    def _kdtree_query(self, x, y):
        """Spatial lookup method.

        Parameters
        ----------
        x : array_like
            x-coordinates to be gridded.

        y : array_like
            y-coordinates to be gridded.

        Returns
        -------
        Tuple containing x-indices [first return] and y-indices [second return] denoting which
        grid points have been mapped.
        """
        try:
            points = np.asarray(list(zip(x, y, strict=True)))
        except TypeError:
            points = np.asarray(list(zip([x], [y], strict=True)))
        _dists, inds = self.tree.query(points, k=1, distance_upper_bound=self.dx)
        bad_inds = np.where(inds >= len(self.tpoints))[0]
        inds = np.delete(inds, bad_inds)
        return np.unravel_index(inds, self.tx.shape)

    def make_empty_grid(self, dtype='int'):
        """Create a grid of zeros of the size of the grid used to initialize the gridder.

        Parameters
        ----------
        dtype : str or `numpy.dtype`
            The data type to use in the construction of the numpy grid. Default is integer.

        Returns
        -------
        A `numpy.ndarray` of zeros of the type provided.
        """
        return np.zeros(self.tx.shape, dtype=dtype)

    def grid_points(self, xs, ys):
        """Take a single point or a list of points and return the grid indices that are hits.

        Parameters
        ----------
        xs : scalar or array_like
            x-coordinate(s)

        ys : scalar or array_like
            y-coordinate(s)

        Returns
        -------
        A list of grid indices corresponding to points that were mapped to the grid.
        """
        xinds, yinds = self._kdtree_query(x=xs, y=ys)
        points = list(zip(xinds, yinds, strict=True))
        return points

    def grid_lines(self, sxs, sys, exs, eys):
        """Take a single line or list of lines and return the grid indices that are hits.

        Parameters
        ----------
        sxs : scalara or array_like
            Starting x-coordinates.

        sys : scalar or array_like
            Starting y-coordinates.

        exs : scalar or array_like
            Ending x-coordinates.

        eys : scalar or array_like
            Ending y-coordinates.

        Returns
        -------
        A list of grid indices corresponding to lines mapped to the grid.
        """
        sxinds, syinds = self._kdtree_query(x=sxs, y=sys)
        exinds, eyinds = self._kdtree_query(x=exs, y=eys)
        lines = [skdraw.line(sx, sy, ex, ey) for sx, sy, ex, ey in zip(sxinds, syinds,
                                                                       exinds, eyinds,
                                                                       strict=True)]
        return lines

    def grid_polygons(self, xs, ys, fill=True):
        """Grid a polygon or list of polygons.

        Parameters
        ----------
        xs : scalar or array_like
            x-coordinates of polygon(s).

        ys : scalar or array_like
            y-coordinates of polygon(s).

        ill : bool
            Flag to determine whether or not to fill polygons. I.e., grid the polygon
            exterior (False) or grid the entire filled polygon (True, default).

        Returns
        -------
        A list of grid indices corresponding to the hits by the polygons.
        """
        xinds = []
        yinds = []
        for x, y in zip(xs, ys, strict=True):
            _xinds, _yinds = self._kdtree_query(x=x, y=y)
            xinds.append(_xinds)
            yinds.append(_yinds)
        if fill:
            polys = []
            for _x, _y in zip(xinds, yinds, strict=True):
                try:
                    polys.append(skdraw.polygon(_x, _y))
                except ValueError:
                    continue
        else:
            polys = []
            for _x, _y in zip(xinds, yinds, strict=True):
                try:
                    polys.append(skdraw.polygon_perimeter(_x, _y))
                except ValueError:
                    continue
        return polys
