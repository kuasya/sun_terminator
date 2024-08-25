"""
Солнечный терминатор

старт: 31.07.2024
место: Иркутский радар НР, ИСЗФ СО РАН, Иркутск
"""

import uvicorn
from fastapi import FastAPI, Depends
from shemas import ResponseCoord, QueryDateTime, Coordinate

from skyfield.api import load, wgs84
from skyfield.positionlib import Geocentric

import numpy as np
import matplotlib.pyplot as plt

app = FastAPI(docs_url='/docs')


def run_terminator(t):
    """
    Вычисление координат терминатора

    :param t: время в формате <Time tt=...>
    :return: координаторы терминатора в формате двумерного массива [[широта, долгота],..]
    """
    eph = load('de421.bsp')  # набор данных для объектов Солнечной системы
    earth = eph['earth']
    sun = eph['sun']
    vec_to_sun = sun - earth  # вектор от Земли к Солнцу

    ts = load.timescale()

    sun_zenit = wgs84.subpoint(vec_to_sun.at(t))  # субточка (зенит) Солнца

    print('Точка зенита latitude: ', sun_zenit.latitude.degrees)
    print('Точка зенита longitude: ', sun_zenit.longitude.degrees)

    sun_angular_position = 90.833  # зенитный угол Солнца на восходе/закате

    sun_vec = vec_to_sun.at(t).position.au  # солнечный вектор
    normal_vec = np.cross(sun_vec, np.array([1, 0, 0]))  # вектор нормали к положению Солнца и оси x

    # Вычисление матрицы вращения
    rotation_radians = np.radians(sun_angular_position)

    cos_theta = np.cos(rotation_radians)
    sin_theta = np.sin(rotation_radians)

    outer_product = np.outer(normal_vec, normal_vec)
    identity_matrix = np.identity(3)

    matrix_rot = (cos_theta * identity_matrix +
                  sin_theta * np.cross(normal_vec, identity_matrix * -1) +
                  (1 - cos_theta) * outer_product)

    first_terminator_vec = np.dot(matrix_rot, sun_vec)  # умножение матриц: расчет первой позиции на терминаторе

    terminator_latitudes = []
    terminator_longitudes = []

    num_points_on_terminator = 100
    for angle in np.linspace(0, 360, num_points_on_terminator):
        rotation_radians_angle = np.radians(angle)
        cos_angle = np.cos(rotation_radians_angle)
        sin_angle = np.sin(rotation_radians_angle)

        rotation_matrix_angle = (cos_angle * identity_matrix +
                                 sin_angle * np.cross(sun_vec, identity_matrix * -1) +
                                 (1 - cos_angle) * np.outer(sun_vec, sun_vec))

        terminator_vector = np.dot(rotation_matrix_angle, first_terminator_vec)
        terminator_position = Geocentric(terminator_vector, t=t)
        geographic_position = wgs84.subpoint(terminator_position)

        terminator_latitudes.append(geographic_position.latitude.degrees)
        terminator_longitudes.append(geographic_position.longitude.degrees)

    terminator_latitudes = np.array(terminator_latitudes)
    terminator_longitudes = np.array(terminator_longitudes)

    terminator_coordinates = []
    for i in range(len(terminator_latitudes)):
        terminator_coordinates += [[terminator_longitudes[i], terminator_latitudes[i]]]

    result = {
        'terminator': terminator_coordinates,
        'sun': [sun_zenit.longitude.degrees, sun_zenit.latitude.degrees]
    }

    return result


@app.get("/get_coordinates", response_model=ResponseCoord)
def get_coordinates(query: QueryDateTime=Depends()):

    """
    Получение координат солнечного терминатора

    :param query: дата и время
    :return: координаты в формате широта-долгота
    """

    data = query.data
    time = query.time

    ts = load.timescale()

    # переход к времени для skyfield
    year, month, day = map(int, data.split('-'))
    hour, minute, second = map(int, time.split('-'))

    t1 = ts.utc(year, month, day, hour, minute)

    # положение Солнца
    sun_position = []  # далее извлекаем из файла

    our_terminator = run_terminator(t1)
    print(our_terminator)

    terminator_coordinates = our_terminator['terminator']
    sun_position = [our_terminator['sun']]
    print(sun_position)
    # print(run_terminator(t1))

    terminator_coordinates2 = np.array(terminator_coordinates)

    # Визуализация линии терминатора
    plt.scatter(terminator_coordinates2[:, 0], terminator_coordinates2[:, 1], color='green', s=20,
                label='граница тени')  # s - размер точек
    plt.title(f"Солнечный терминатор на {data} {time}")
    # plt.xlim(-180, 180)
    # plt.ylim(-90, 90)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.grid()
    plt.legend()
    plt.show()
    sun_position_itog = [Coordinate(latitude=lat, longitude=lon) for lon, lat in sun_position]
    terminator_coordinates_itog = [Coordinate(latitude=lat, longitude=lon) for lon, lat in terminator_coordinates]

    return ResponseCoord(sun_position=sun_position_itog, terminator_coordinates=terminator_coordinates_itog)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
