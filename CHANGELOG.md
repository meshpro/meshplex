# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.22.0] - 2026-xx

### Added

- Edge collapses

### Changed

- Better support for non-manifold meshes
- Better support for line meshes
- Better caching
- More consistent data structure
- Fixes for line and triangle plots
- Consistent point -> vertex renaming

## [0.21.9] - 2026-03-19

### Changed

- Better accuracy of q_radius_ratio for nearly degenerate triangles
- More stable angle computation

## [0.21.8] - 2026-03-19

### Added

- Lemoine points
- Spieker centers
- Nagel points
- Monge points
- Center of gravity of the entire mesh
- Volume of entire mesh
- Surface area of entire mesh
- Euler characteristic for all simplex meshes
- Outer normals
- Outside boundary angles

### Changed

- Better accuracy for near-degenerate simplices
- Bugfix for edge flips
- Bugfix for boundaries in higher dimensions
- Bugfix for incenters

## [0.16.3] - 2021-07-13

### Changed

- Fixed computation of `genus` and `euler_characteristic`

## [0.16.0] - 2021-04-15

### Changed

- `mesh.cells` is now a function; e.g., `mesh.cells["points"]` is now
  `mesh.cells("points")`
- `mesh.idx_hierarchy` is deprecated in favor of `mesh.idx[-1]` (the new `idx` list
  contains more index magic)

## [0.14.0] - 2020-11-05

### Changed

- `node_coords` is now `points`
- `mesh_tri`: fixed inconsistent state after setting the points
