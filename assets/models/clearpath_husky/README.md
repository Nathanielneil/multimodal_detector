# Clearpath Husky Demo UGV

Source: https://github.com/husky/husky

Original asset path: `husky_description/meshes`

License: BSD-3-Clause, Copyright 2021 Clearpath Robotics Inc.

The OBJ files in this directory are generated demo meshes converted from the official Husky COLLADA visual meshes. The conversion bakes only the stable demo geometry into dependency-free OBJ assets. Runtime loading does not require ROS, xacro, URDF, or COLLADA support.

Current runtime mesh budget:

- `husky_base.obj`: 570 vertices / 1,220 faces
- `husky_top_chassis.obj`: 645 vertices / 1,452 faces
- `husky_bumper.obj`: 428 vertices / 840 faces, reused for front and rear
- `husky_top_plate.obj`: 16 vertices / 28 faces
- `husky_user_rail.obj`: 346 vertices / 791 faces
- `husky_wheel.obj`: 2,192 vertices / 4,544 faces, reused for four wheels

Effective rendered budget is about 23k faces, but disk storage keeps only one wheel mesh. Simplification method: 5mm vertex clustering for chassis, bumper, and rails; 10mm vertex clustering for wheels; exact deduplication for the top plate.
