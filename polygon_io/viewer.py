#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

N = 3
patches = []

points1 = hulls[0].points[hulls[0].vertices]
points2 = np.random.rand(N, 2)
print(points1)
print(points2)

polygon = Polygon(points1, True)
patches.append(polygon)
polygon = Polygon(points2, True)
patches.append(polygon)

colors = 100 * np.random.rand(len(patches))
p = PatchCollection(patches, alpha=0.4)
p.set_array(colors)
ax.add_collection(p)
fig.colorbar(p, ax=ax)

convex_hull_plot_2d(hulls[0])
plt.show()