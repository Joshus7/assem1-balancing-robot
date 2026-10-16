#!/usr/bin/env python3
"""
make_road_world.py - writes road_hills.world (Gazebo classic): grass field, trees, houses, cloudy sky and a
road along +x (the robot's forward direction, robot spawns at x = 0) with two hills:

   x: -6 ........ 2 | 5 deg up | plateau | 5 deg down | flat | 3 deg up | top | 3 deg down | flat ... 30
   Edit ROAD below and re-run:  python3 make_road_world.py   (then colcon build is not needed: symlink install)

5 deg needs ~11 deg of lean (centre of mass 5.2 cm above the axle); 7 deg and more is hard (see balance_core.py).
"""
import math
import os

ROAD_W = 1.4           # m
ROAD_Z = 0.01          # top of the flat road
# (x_start, x_end, height at start, height at end)   heights above ROAD_Z
HILL1 = 0.306          # 3.5 m at 5 deg
HILL2 = 0.157          # 3.0 m at 3 deg
ROAD = [(-6.0, 2.0, 0, 0), (2.0, 5.5, 0, HILL1), (5.5, 8.0, HILL1, HILL1), (8.0, 11.5, HILL1, 0),
        (11.5, 15.0, 0, 0), (15.0, 18.0, 0, HILL2), (18.0, 19.5, HILL2, HILL2), (19.5, 22.5, HILL2, 0),
        (22.5, 30.0, 0, 0)]
SLAB = 0.4             # thickness of the sloped slabs (their lower part is hidden under the ground)


def box(name, pose, size, material, collide=True):
    x, y, z, r, p, yw = pose
    col = ('<collision name="c"><geometry><box><size>%.4f %.4f %.4f</size></box></geometry>'
           '<surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>' % size) if collide else ''
    return ('<model name="%s"><static>true</static><pose>%.4f %.4f %.4f %.4f %.4f %.4f</pose><link name="l">%s'
            '<visual name="v"><cast_shadows>%s</cast_shadows><geometry><box><size>%.4f %.4f %.4f</size></box></geometry>'
            '<material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>%s</name></script></material>'
            '</visual></link></model>\n' % (name, x, y, z, r, p, yw, col, 'true' if collide else 'false', *size, material))


def segment(i, x0, x1, h0, h1):
    """road piece + centre dashes + edge lines"""
    out = ''
    z0, z1 = ROAD_Z + h0, ROAD_Z + h1
    a = math.atan2(z1 - z0, x1 - x0)
    L = math.hypot(x1 - x0, z1 - z0)
    xm, zm = (x0 + x1) / 2, (z0 + z1) / 2
    if h0 == h1 == 0:                                   # flat road: thin layer on the grass
        out += box('road_%d' % i, (xm, 0, ROAD_Z / 2, 0, 0, 0), (L, ROAD_W, ROAD_Z), 'Gazebo/Road')
    elif h0 == h1:                                      # plateau: solid block
        out += box('road_%d' % i, (xm, 0, z0 / 2, 0, 0, 0), (L, ROAD_W, z0), 'Gazebo/Road')
    else:                                               # ramp: thick slab, top face is the road
        nx, nz = -math.sin(a), math.cos(a)
        out += box('road_%d' % i, (xm - nx * SLAB / 2, 0, zm - nz * SLAB / 2, 0, -a, 0), (L, ROAD_W, SLAB), 'Gazebo/Road')
    # painted lines (visual only, 2 mm above the surface)
    def on_road(s, y, length, width, mat, name):
        px = x0 + (x1 - x0) * s
        pz = z0 + (z1 - z0) * s + 0.002 / math.cos(a)
        return box(name, (px, y, pz, 0, -a, 0), (length, width, 0.002), mat, collide=False)
    n = max(1, int(L / 1.0))
    for k in range(n):
        out += on_road((k + 0.5) / n, 0.0, 0.5, 0.06, 'Gazebo/White', 'dash_%d_%d' % (i, k))
    for side, y in (('l', ROAD_W / 2 - 0.06), ('r', -ROAD_W / 2 + 0.06)):
        out += on_road(0.5, y, L, 0.05, 'Gazebo/Yellow', 'edge_%d_%s' % (i, side))
    return out


def tree(i, x, y, s=1.0):
    return ('<model name="tree_%d"><static>true</static><pose>%.2f %.2f 0 0 0 0</pose><link name="l">'
            '<collision name="c"><pose>0 0 %.2f 0 0 0</pose><geometry><cylinder><radius>%.2f</radius><length>%.2f</length></cylinder></geometry></collision>'
            '<visual name="trunk"><pose>0 0 %.2f 0 0 0</pose><geometry><cylinder><radius>%.2f</radius><length>%.2f</length></cylinder></geometry>'
            '<material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Wood</name></script></material></visual>'
            '<visual name="leaves"><pose>0 0 %.2f 0 0 0</pose><geometry><sphere><radius>%.2f</radius></sphere></geometry>'
            '<material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Green</name></script></material></visual>'
            '</link></model>\n' % (i, x, y, 0.75 * s, 0.12 * s, 1.5 * s, 0.75 * s, 0.12 * s, 1.5 * s, 1.9 * s, 0.8 * s))


def house(i, x, y, yaw):
    return (box('house_%d' % i, (x, y, 1.2, 0, 0, yaw), (4.0, 3.0, 2.4), 'Gazebo/Bricks') +
            box('roof_%d' % i, (x, y, 2.75, 0, 0, yaw), (4.4, 3.4, 0.7), 'Gazebo/Red'))


def world():
    road = ''.join(segment(i, *seg) for i, seg in enumerate(ROAD))
    trees = ''
    k = 0
    for x in range(-5, 30, 3):
        for y in (-3.0, 3.2):
            trees += tree(k, x + (0.7 if y > 0 else 0.0), y + 0.4 * math.sin(x), 0.8 + 0.3 * abs(math.cos(x)))
            k += 1
    houses = house(0, 4.0, -8.0, 0.1) + house(1, 13.0, 8.5, -0.15) + house(2, 22.0, -8.5, 0.05) + house(3, -3.0, 8.0, 0.0)
    return '''<?xml version="1.0"?>
<!-- GENERATED by make_road_world.py - edit that file, not this one -->
<sdf version="1.6">
  <world name="road_hills">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <scene>
      <ambient>0.5 0.5 0.5 1</ambient>
      <background>0.6 0.75 0.95 1</background>
      <shadows>true</shadows>
      <sky><clouds><speed>4</speed></clouds></sky>
    </scene>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 30 0 0 0</pose>
      <diffuse>0.9 0.9 0.85 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.85</direction>
    </light>
    <model name="grass">
      <static>true</static>
      <link name="l">
        <collision name="c"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
        <visual name="v"><cast_shadows>false</cast_shadows><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Grass</name></script></material></visual>
      </link>
    </model>
''' + road + trees + houses + '''    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>-2.2 -2.6 1.1 0 0.25 0.85</pose>
      </camera>
    </gui>
  </world>
</sdf>
'''


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'road_hills.world')
    with open(out, 'w') as f:
        f.write(world())
    print('wrote', out)
