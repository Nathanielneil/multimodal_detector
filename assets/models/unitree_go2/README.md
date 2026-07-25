# Unitree Go2 Demo Mesh

Source: https://github.com/unitreerobotics/unitree_mujoco

Original asset path: `unitree_robots/go2`

License: BSD-3-Clause, Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")

The OBJ files in this directory are generated demo meshes converted from the official Go2 MuJoCo XML/OBJ assets. The conversion bakes the visual body hierarchy into pre-aligned OBJ parts and simplifies the mesh for the PyQtGraph acceptance-demo scene.

Current UI note: the midterm-demo renderer uses a solid low-poly Go2-style proxy in `ui/ground_vehicles.py` by default, because this source mesh looks too fragmented in the embedded PyQtGraph OpenGL scene on the target desktop setup. These OBJ files are retained as licensed source/reference assets for later offline retopology or a higher-fidelity renderer.

Current runtime mesh budget:

- `go2_black.obj`: 5,931 vertices / 3,744 faces
- `go2_gray.obj`: 18,000 vertices / 4,634 faces
- `go2_metal.obj`: 10,903 vertices / 2,513 faces
- `go2_white.obj`: 978 vertices / 646 faces
- Total: 35,812 vertices / 11,537 faces

Simplification method: 2mm vertex clustering with degenerate and duplicate triangles removed. Runtime loading does not require ROS, xacro, URDF, or MuJoCo.
