import matplotlib.pyplot as plt
import numpy

import meshplex


def _main():
    points = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.8]])
    # points = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.5, numpy.sqrt(3) / 2]])
    cells = numpy.array([[0, 1, 2]])

    mesh = meshplex.mesh_tri.MeshTri(points, cells)

    lw = 5.0
    col = "0.6"

    ax = plt.gca()

    # circumcircle
    circle1 = plt.Circle(
        mesh.cell_circumcenters[0],
        mesh.cell_circumradius[0],
        color=col,
        fill=False,
        linewidth=lw,
    )
    ax.add_artist(circle1)

    # perpendicular bisectors
    for i, j in [[0, 1], [1, 2], [2, 0]]:
        m1 = (points[i] + points[j]) / 2
        v1 = m1 - mesh.cell_circumcenters[0]
        e1 = (
            mesh.cell_circumcenters[0]
            + v1 / numpy.linalg.norm(v1) * mesh.cell_circumradius[0]
        )
        plt.plot(
            [mesh.cell_circumcenters[0, 0], e1[0]],
            [mesh.cell_circumcenters[0, 1], e1[1]],
            col,
            linewidth=lw,
        )

    # heights
    for i, j, k in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
        p = points - points[i]
        v1 = p[j] / numpy.linalg.norm(p[j])
        m1 = points[i] + numpy.dot(p[k], v1) * v1
        plt.plot(
            [points[k, 0], m1[0]], [points[k, 1], m1[1]], col, linewidth=lw, color=col
        )

    # # incircle
    # circle2 = plt.Circle(
    #     mesh.cell_incenters[0], mesh.inradius[0], color=col, fill=False, linewidth=lw
    # )
    # ax.add_artist(circle2)

    # # angle bisectors
    # for i, j, k in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
    #     p = points - points[i]
    #     v1 = p[j] / numpy.linalg.norm(p[j])
    #     v2 = p[k] / numpy.linalg.norm(p[k])
    #     alpha = numpy.arccos(numpy.dot(v1, v2))
    #     c = numpy.cos(alpha / 2)
    #     s = numpy.sin(alpha / 2)
    #     beta = numpy.linalg.norm(mesh.cell_incenters[0] - points[i]) + mesh.inradius[0]
    #     m1 = points[i] + numpy.dot([[c, -s], [s, c]], v1) * beta
    #     plt.plot(
    #         [points[i, 0], m1[0]],
    #         [points[i, 1], m1[1]],
    #         col,
    #         linewidth=lw,
    #         color=col,
    #     )

    # triangle
    plt.plot(
        [points[0, 0], points[1, 0], points[2, 0], points[0, 0]],
        [points[0, 1], points[1, 1], points[2, 1], points[0, 1]],
        color="#d62728",
        linewidth=lw,
    )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.4, 0.9)
    ax.set_aspect("equal")
    plt.axis("off")
    plt.savefig("logo.svg")
    # plt.show()
    return


if __name__ == "__main__":
    _main()
