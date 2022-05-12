#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
from random import random
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib
from numpy import poly
matplotlib.use('agg')
# matplotlib.use('TkAgg')


class SATPolygon:
    def __init__(self, vertices) -> None:
        self.polygon = Polygon(vertices, True)
        self.n_vertex = len(vertices)

    def profile(self):
        n_edge = self.n_vertex
        return n_edge


def load(input_file):
    polygons = []
    with open(input_file, 'r') as f:
        # total number of polygons
        n = int(f.readline())
        print(n)
        for i in range(n):
            # number of vertices
            n_vertex = int(f.readline())
            # list of vertices
            vertices = eval(f.readline())
            assert n_vertex == len(vertices), \
                f"vertices length not match, please check ouput of polygon {i+1} (count from 1)"

            polygons.append(SATPolygon(vertices))
    return polygons


def draw(polygon_group, output):
    fig, axes = plt.subplots()

    # draw polygon
    patches = []
    colors = []
    for satpolygon in polygon_group:
        patches.append(satpolygon.polygon)
        colors.append(random() * 100)
    # print(colors)

    p = PatchCollection(patches, alpha=0.15)
    p.set_array(colors)
    axes.add_collection(p)

    fig.colorbar(p, ax=axes)
    # convex_hull_plot_2d(hulls[0])
    plt.savefig(output)
    # plt.show()


def profile(polygons):
    """ Output the statistics profile of the polygons.
    Metric:
        counter of n-edge polygon
    """
    print(Counter([p.profile() for p in polygons]))


@click.command()
@click.option('--input', type=str, help='Input text file.')
@click.option('--output', type=str, help='Output png file.')
def main(input, output):
    print(input, output)
    polygons = load(input)

    profile(polygons)
    draw(polygons, output)


if __name__ == '__main__':
    main()
