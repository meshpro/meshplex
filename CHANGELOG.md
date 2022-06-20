# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
