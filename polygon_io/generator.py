#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
from random import randint
import numpy as np
from scipy.spatial import ConvexHull


class PolygonGenerator:
    """ Convex hull generator.
    Random generate ${sample_point_num} points as point pool.
    Each call to self.random() randomly choice ${vertex_num} points, generates
    its convex hull.
    """

    def __init__(self, sample_point_num) -> None:
        """
        Args:
            sample_point_num: how many points available in the point pool.
        """
        rng = np.random.default_rng()
        # ${sample_points} random points in 2-D
        self.sample_points = rng.random((sample_point_num, 2))

    def random(self, vertex_num):
        """
        Args:
            vertex_num: how many points to choose and generate its convex hull.
        """
        vertex_indices = np.random.choice(
            range(len(self.sample_points)), vertex_num, replace=False)
        assert len(vertex_indices.tolist()) == len(set(vertex_indices.tolist()))
        return ConvexHull(self.sample_points[vertex_indices])

    @staticmethod
    def dump(filename, polygons, end='\n'):
        """
        format:
        ```
        # line1
        polygon_num
        # rest lines
        point_num
        point_list

        2
        4
        [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0),]
        5
        [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.5,1.5),(0.0,1.0),]
        ```
        """
        def __list_formater(points):
            poly_str = str([tuple(i) for i in points])
            return poly_str.replace(' ', '')

        with open(filename, 'w') as f:
            f.write(str(len(polygons)))
            for poly in polygons:
                f.write(end)
                points = poly.points[poly.vertices]
                # NOTE that len(points) can be 3 at lowest, indicating that all
                # other points are inside the polygon.
                f.write(str(len(points)))
                f.write(end)
                f.write(__list_formater(points.tolist()))


@click.command()
@click.option('--count', type=int, help='Number of polygons.')
@click.option('--point', type=int, help='Number of sample points.')
@click.option('--vmin', type=int, help='Minimal number of vertices.')
@click.option('--vmax', type=int, help='Maximal number of vertices.')
@click.option('--output', type=str, help='Output text file.')
def main(count, point, vmin, vmax, output):
    """
    e.g.
    python generator.py --count 5 --point 30 --vmin 3 --vmax 6 --output polygon_input.txt
    """
    poly_generator = PolygonGenerator(point)
    polygons = [poly_generator.random(randint(vmin, vmax))
                for _ in range(count)]
    PolygonGenerator.dump(output, polygons)


if __name__ == '__main__':
    main()
