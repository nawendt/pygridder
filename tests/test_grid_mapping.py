# Copyright (c) 2025 Patrick Marsh.
# Distributed under the terms of the BSD 3-Clause License.
"""Test mapping points, lines, and polygons to a grid."""

import json
from pathlib import Path

import numpy as np
import pyproj
import pytest

from pygridder import Gridder


@pytest.fixture(scope='class')
def projection():
    """Projection for tests."""
    return pyproj.Proj({
            'proj': 'lcc',
            'datum': 'NAD83',
            'lat_0': 39,
            'lon_0': -96,
            'lat_1': 33,
            'lat_2': 45
        })


@pytest.fixture(scope='class')
def base_grid(projection):
    """Create base grid for tests."""
    lonmin = -122
    latmin = 23
    lonmax = -57.5
    latmax = 52

    xmin, ymin = projection(lonmin, latmin)
    xmax, ymax = projection(lonmax, latmax)

    dx = dy = 80 * 1000

    xaxis = np.arange(xmin, xmax + 1, dx)
    yaxis = np.arange(ymin, ymax + 1, dy)

    x, y = np.meshgrid(xaxis, yaxis)

    grid = Gridder(x, y)

    return grid


@pytest.fixture(scope='class')
def path_data():
    """Load tornado path data."""
    _path = Path(__file__).parent / 'data' / 'tornado_paths.json'
    with open(_path) as data:
        paths = json.load(data)
    return paths


@pytest.fixture(scope='class')
def area_data():
    """Load discussion area data."""
    _area = Path(__file__).parent / 'data' / 'discussion_areas.json'
    with open(_area) as data:
        areas = json.load(data)
    return areas


@pytest.mark.usefixtures('base_grid', 'path_data', 'area_data')
class TestGridder:
    """Test class for Gridder class."""

    point_ref = Path(__file__).parent / 'data' / 'mapped_points.npz'
    line_ref = Path(__file__).parent / 'data' / 'mapped_lines.npz'
    poly_ref = Path(__file__).parent / 'data' / 'mapped_polygons.npz'

    def test_points(self, base_grid, path_data, projection):
        """Test mapping points to grid."""
        grid = base_grid.make_empty_grid()

        start_x = []
        start_y = []

        for path in path_data:
            px, py = projection(*path['start'])
            start_x.append(px)
            start_y.append(py)

        points = base_grid.grid_points(start_x, start_y)

        for point in points:
            grid[point] += 1

        with np.load(self.point_ref) as _ref:
            np.testing.assert_equal(grid, _ref['grid'])

    def test_lines(self, base_grid, path_data, projection):
        """Test mapping lines to grid."""
        grid = base_grid.make_empty_grid()

        start_x = []
        start_y = []
        end_x = []
        end_y = []

        for path in path_data:
            spx, spy = projection(*path['start'])
            epx, epy = projection(*path['end'])
            start_x.append(spx)
            end_x.append(epx)
            start_y.append(spy)
            end_y.append(epy)

        lines = base_grid.grid_lines(start_x, start_y, end_x, end_y)

        for line in lines:
            grid[line] += 1

        with np.load(self.line_ref) as _ref:
            np.testing.assert_equal(grid, _ref['grid'])

    def test_polygons(self, base_grid, area_data, projection):
        """Test mapping polygons to grid."""
        grid = base_grid.make_empty_grid()

        x = []
        y = []

        for area in area_data:
            px, py = projection(area['lon'], area['lat'])
            x.append(px)
            y.append(py)

        polygons = base_grid.grid_polygons(x, y)

        for polygon in polygons:
            grid[polygon] += 1

        with np.load(self.poly_ref) as _ref:
            np.testing.assert_equal(grid, _ref['grid'])
