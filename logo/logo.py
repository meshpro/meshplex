# -*- coding: utf-8 -*-
import numpy
import meshplex
import matplotlib.pyplot as plt


def _main():
    points = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.4, 0.8]])
    cells = numpy.array([[0, 1, 2]])

    mesh = meshplex.mesh_tri.MeshTri(points, cells)

    plt.plot(
        [points[0, 0], points[1, 0], points[2, 0], points[0, 0]],
        [points[0, 1], points[1, 1], points[2, 1], points[0, 1]],
        color="k",
    )
    ax = plt.gca()

    circle1 = plt.Circle(
        mesh.cell_circumcenters[0], mesh.circumradius[0], color="k", fill=False
    )
    ax.add_artist(circle1)

    circle2 = plt.Circle(
        mesh.cell_incenters[0], mesh.inradius[0], color="k", fill=False
    )
    ax.add_artist(circle2)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.4, 0.9)
    ax.set_aspect("equal")
    plt.axis('off')
    plt.savefig("logo.svg")
    # plt.show()

    return


if __name__ == "__main__":
    _main()
