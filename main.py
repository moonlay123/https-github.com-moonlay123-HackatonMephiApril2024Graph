import json
import time
import numpy as np
import taichi as ti
import warnings
warnings.filterwarnings("ignore")


# ========== INITIALIZATION ==========

ti.init()
IS_BUILD = True
NEAREST_BLOCKS_COUNT = 30000
EPS = 0.00000001
vec2 = ti.types.vector(2, float)


def log(info):
    k = 50
    info = str(info)
    print('==========', ' ' * ((k - len(info)) // 2), info, ' ' * ((k - len(info) + 1) // 2), '==========')


# ========== READING ==========

if IS_BUILD:
    blue_road_file_name = input('Введите путь до файла с синей дорогой: ')
    red_road_file_name = input('Введите путь до файла с красной дорогой: ')
    green_road_file_name = input('Введите путь до файла, в который нужно сохранить зеленую дорогу: ')
    precision = float(input('Введите погрешность в метрах: '))
else:
    blue_road_file_name = 'C:/Users/User/Desktop/Graph/Graph/kaliningrad_blue_WGS84.geojson'
    red_road_file_name = 'C:/Users/User/Desktop/Graph/Graph/kaliningrad_red_WGS84.geojson'
    green_road_file_name = 'C:/Users/User/Desktop/Graph/Graph/res.geojson'
    precision = 25

log('READING FILES')

blue_road_file = open(blue_road_file_name, encoding="utf8")
red_road_file = open(red_road_file_name, encoding="utf8")

blue_road_info = json.load(blue_road_file)
red_road_info = json.load(red_road_file)

blue_road_file.close()
red_road_file.close()


# ========== BLUE ROAD TO SEGMENTS ==========

log('CONVERTING BLUE ROAD TO SEGMENTS')

blue_points_x, blue_points_y, is_blue_segment_end = [], [], []
blue_segments = []
sorted_blue_points = []

for road_info in blue_road_info['features']:

    road_geometry = road_info['geometry']
    road_coordinates = road_geometry['coordinates'][0]

    for i, (x, y) in enumerate(road_coordinates):

        blue_points_x.append(x)
        blue_points_y.append(y)

        sorted_blue_points.append((x, y))

        is_blue_segment_end.append(
            i == 0 or i == len(road_coordinates) - 1
        )

        if i >= 1:
            blue_segments.append([
                len(blue_points_x) - 2, len(blue_points_x) - 1
            ])

blue_segments.sort(
    key=lambda i: (blue_points_x[i[0]] + blue_points_x[i[1]]) / 2
)
sorted_blue_points.sort(key=lambda i: i[0])

log(f'BLUE SEGMENTS COUNT: {len(blue_segments)}')


# ========== GETTING POINTS FROM RED ROAD ==========

log('GETTING POINTS FROM RED ROAD')

red_points_x, red_points_y = [], []

for road_info in red_road_info['features']:

    road_geometry = road_info['geometry']
    road_coordinates = road_geometry['coordinates'][0]

    for x, y in road_coordinates:

        red_points_x.append(x)
        red_points_y.append(y)

log(f'RED POINTS COUNT: {len(red_points_x)}')


# ========== CONVERTING TO TAICHI ==========

log('CONVERTING TO TAICHI')

# BLUE
ti_blue_points_x = ti.field(float, shape=(len(blue_points_x)))
ti_blue_points_y = ti.field(float, shape=(len(blue_points_y)))
ti_is_blue_segment_end = ti.field(bool, shape=(len(is_blue_segment_end)))
ti_blue_segments = ti.field(int, shape=(len(blue_segments), 2))
ti_sorted_blue_points = ti.field(float, shape=(len(sorted_blue_points), 2))

ti_blue_points_x.from_numpy( np.array( blue_points_x ) )
ti_blue_points_y.from_numpy( np.array( blue_points_y ) )
ti_is_blue_segment_end.from_numpy( np.array( is_blue_segment_end ) )
ti_blue_segments.from_numpy( np.array( blue_segments ) )
ti_sorted_blue_points.from_numpy( np.array( sorted_blue_points ) )

# RED
ti_red_points_x = ti.field(float, shape=(len(red_points_x)))
ti_red_points_y = ti.field(float, shape=(len(red_points_y)))

ti_red_points_x.from_numpy( np.array( red_points_x ) )
ti_red_points_y.from_numpy( np.array( red_points_y ) )

# ========== FREAKING MATH ==========

log('PROCESSING RED POINTS')


# @ti.func
def dist_from_point_to_point(px1: float, py1: float, px2: float, py2: float) -> float:
    dx = px1 - px2
    dy = py1 - py2
    return ti.sqrt(dx * dx + dy * dy)


# @ti.func
def dist_from_point_to_line(px: float, py: float, lx1: float, ly1: float, lx2: float, ly2: float) -> float:
    # ========== LINE FORMULA ==========

    a = ly1 - ly2
    b = lx2 - lx1
    c = lx1 * ly2 - lx2 * ly1

    return ti.abs(a * px + b * py + c) / ti.sqrt(a * a + b * b)


# @ti.func
def dist_from_point_to_segment(px: float, py: float, lx1: float, ly1: float, lx2: float, ly2: float) -> float:
    # ========== LINE FORMULA ==========

    a = ly1 - ly2
    b = lx2 - lx1
    c = lx1 * ly2 - lx2 * ly1

    # ========== NORMALIZATION ==========

    normal_len = ti.sqrt(a * a + b * b)
    a, b, c = a / normal_len, b / normal_len, c / normal_len

    # ========== CHECK FIRST NORMAL DIRECTION ========

    dist = ti.abs(a * px + b * py + c)
    npx, npy = px + a * dist, py + b * dist

    if dist_from_point_to_line(npx, npy, lx1, ly1, lx2, ly2) < EPS:

        # ========== CHECK IF PROJECTION IS OUT OF SEGMENT ========

        if not (ti.min(lx1, lx2) <= npx <= ti.max(lx1, lx2)):
            dist = ti.min(
                dist_from_point_to_point(px, py, lx1, ly1),
                dist_from_point_to_point(px, py, lx2, ly2),
            )

    else:
        # ========== ANOTHER NORMAL DIRECTION ========
        npx, npy = px - a * dist, py - b * dist

        if not (ti.min(lx1, lx2) <= npx <= ti.max(lx1, lx2)):
            dist = ti.min(
                dist_from_point_to_point(px, py, lx1, ly1),
                dist_from_point_to_point(px, py, lx2, ly2),
            )
    return dist


# @ti.func
def project_point_on_line(px: float, py: float, lx1: float, ly1: float, lx2: float, ly2: float) -> vec2:
    # ========== LINE FORMULA ==========

    a = ly1 - ly2
    b = lx2 - lx1
    c = lx1 * ly2 - lx2 * ly1

    # ========== NORMALIZATION ==========

    normal_len = ti.sqrt(a * a + b * b)
    a, b, c = a / normal_len, b / normal_len, c / normal_len

    # ========== CHECK FIRST NORMAL DIRECTION ========

    dist = ti.abs(a * px + b * py + c)
    npx, npy = px + a * dist, py + b * dist

    # ========== ANOTHER NORMAL DIRECTION ========
    if dist_from_point_to_line(npx, npy, lx1, ly1, lx2, ly2) > EPS:
        npx, npy = px - a * dist, py - b * dist

    return vec2(npx, npy)


def distance(lat1: float, lon1: float, lat2: float, lon2: float):
    radius = 6371

    dlat = (lat2 - lat1) * 3.14159 / 180
    dlon = (lon2 - lon1) * 3.14159 / 180
    a = (ti.sin(dlat / 2) * ti.sin(dlat / 2) +
         ti.cos(lat1 * 3.14159 / 180) * ti.cos(lat2 * 3.14159 / 180) *
         ti.sin(dlon / 2) * ti.sin(dlon / 2))
    c = 2 * ti.atan2(ti.sqrt(a), ti.sqrt(1 - a))
    d = radius * c

    return d * 1000


@ti.kernel
def correct_red_points(red_points_count: int, blue_segments_count: int, blue_points_count: int):
    for red_point in range(red_points_count):

        px, py = ti_red_points_x[red_point], ti_red_points_y[red_point]

        # ========== FINDING NEAREST POINT ==========

        min_dist, nearest_point = 99999999999.9, 0

        left_point, right_point = 0, blue_points_count - 1
        for _ in range(40):
            mid_point = (left_point + right_point) // 2
            bpx = ti_sorted_blue_points[mid_point, 0]
            if bpx < px:
                left_point = mid_point
            else:
                right_point = mid_point

        for blue_point in range(
            max(0, left_point - NEAREST_BLOCKS_COUNT),
            min(blue_points_count, left_point + NEAREST_BLOCKS_COUNT)
        ):
            bpx, bpy = ti_sorted_blue_points[blue_point, 0], ti_sorted_blue_points[blue_point, 1]
            dist = distance(px, py, bpx, bpy)

            if dist < min_dist:
                min_dist, nearest_point = dist, blue_point

        bpx, bpy = ti_sorted_blue_points[nearest_point, 0], ti_sorted_blue_points[nearest_point, 1]

        if min_dist <= precision:
            ti_red_points_x[red_point], ti_red_points_y[red_point] = bpx, bpy

        else:
            # ========== FINDING NEAREST SEGMENT ==========

            min_dist, nearest_segment = 99999999999.9, 0

            left_segment, right_segment = 0, blue_segments_count - 1
            for _ in range(40):
                mid_segment = (left_segment + right_segment) // 2
                point1 = ti_blue_segments[mid_segment, 0]
                point2 = ti_blue_segments[mid_segment, 1]
                x1, x2 = ti_blue_points_x[point1], ti_blue_points_x[point2]
                if (x1 + x2) / 2 < px:
                    left_segment = mid_segment
                else:
                    right_segment = mid_segment

            for blue_segment in range(
                max(0, left_segment - NEAREST_BLOCKS_COUNT),
                min(blue_segments_count, left_segment + NEAREST_BLOCKS_COUNT)
            ):
                point1 = ti_blue_segments[blue_segment, 0]
                point2 = ti_blue_segments[blue_segment, 1]

                lx1, ly1 = ti_blue_points_x[point1], ti_blue_points_y[point1]
                lx2, ly2 = ti_blue_points_x[point2], ti_blue_points_y[point2]

                dist = dist_from_point_to_segment(px, py, lx1, ly1, lx2, ly2)
                if dist < min_dist:
                    min_dist, nearest_segment = dist, blue_segment

            # ========== PROJECTING POINT ON THE NEAREST SEGMENT ==========

            point1 = ti_blue_segments[nearest_segment, 0]
            point2 = ti_blue_segments[nearest_segment, 1]

            lx1, ly1 = ti_blue_points_x[point1], ti_blue_points_y[point1]
            lx2, ly2 = ti_blue_points_x[point2], ti_blue_points_y[point2]

            projection = project_point_on_line(px, py, lx1, ly1, lx2, ly2)
            ti_red_points_x[red_point], ti_red_points_y[red_point] = projection[0], projection[1]


start_projecting_time = time.time()

correct_red_points(len(red_points_x), len(blue_segments), len(sorted_blue_points))

log('RED POINTS CORRECTION TIME: {:.1f} SECONDS'.format((time.time() - start_projecting_time)))


# ========== SAVING TO FILE ==========

log('SAVING')

cur_point_in_red_points = 0

for road_info_i in range(len(red_road_info['features'])):

    for point_i in range(len(
        red_road_info['features'][road_info_i]['geometry']['coordinates'][0]
    )):
        red_road_info['features'][road_info_i]['geometry']['coordinates'][0][point_i] = [
            ti_red_points_x[cur_point_in_red_points],
            ti_red_points_y[cur_point_in_red_points]
        ]
        cur_point_in_red_points += 1

with open(green_road_file_name, "w") as file:
    json.dump(red_road_info, file)

log('FINISHED')
