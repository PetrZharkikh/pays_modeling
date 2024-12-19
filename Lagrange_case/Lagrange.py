import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def lagrange_with_gravity(t, y, I1, I3, m, g, l):
    phi, theta, psi, w1, w2, w3 = y
    
    # Моменты силы тяжести
    M1 = -m * g * l * np.sin(theta) * np.cos(phi)
    M2 = -m * g * l * np.sin(theta) * np.sin(phi)
    M3 = 0  # Момент вокруг оси симметрии равен нулю
    
    # Уравнения Лагранжа-Эйлера
    dw1 = ((I3 - I1) / I1) * w2 * w3 + M1 / I1
    dw2 = -((I3 - I1) / I1) * w1 * w3 + M2 / I1
    dw3 = ((I1 - I1) / I3) * w1 * w2 + M3 / I3
    
    # Кинематические уравнения для углов Эйлера
    dphi = w1 + w2 * np.sin(phi) * np.tan(theta) + w3 * np.cos(phi) * np.tan(theta)
    dtheta = w2 * np.cos(phi) - w3 * np.sin(phi)
    dpsi = (w2 * np.sin(phi) + w3 * np.cos(phi)) / np.cos(theta)
    
    return [dphi, dtheta, dpsi, dw1, dw2, dw3]

# Параметры волчка
I1, I3 = 1.0, 2.0  # I1 = I2
m = 1.0            
g = 9.81           
l = 0.5            # расстояние от оси до центра масс
w0 = [0.5, 0.5, 5.0]  # Начальная угловая скорость
angles0 = [0.0, np.pi / 6, 0.0]  # Начальные углы Эйлера (φ, θ, ψ)

# Начальные условия: углы Эйлера и угловая скорость
y0 = [*angles0, *w0]

# Временные параметры
t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)

# Решение ОДУ
sol = solve_ivp(
    lagrange_with_gravity,
    t_span,
    y0,
    t_eval=t_eval,
    args=(I1, I3, m, g, l),
    rtol=1e-9,
    atol=1e-9
)

phi, theta, psi = sol.y[0], sol.y[1], sol.y[2]

# Создаем фигуру твердого тела (куб)
cube_vertices = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1]
]).T

# Строим ребра куба
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# Визуализаця вращения и построение матрицы поворота
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

lines = [ax.plot([], [], [], 'b')[0] for _ in cube_edges]

def euler_to_rotation_matrix(phi, theta, psi):
    Rz_psi = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi),  0],
        [0,           0,           1]
    ])
    Ry_theta = np.array([
        [np.cos(theta),  0, np.sin(theta)],
        [0,              1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rx_phi = np.array([
        [1, 0,           0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    return Rz_psi @ Ry_theta @ Rx_phi

def update(frame):
    R = euler_to_rotation_matrix(phi[frame], theta[frame], psi[frame])
    rotated_vertices = R @ cube_vertices
    for edge, line in zip(cube_edges, lines):
        x = rotated_vertices[0, list(edge)]
        y = rotated_vertices[1, list(edge)]
        z = rotated_vertices[2, list(edge)]
        line.set_data(x, y)
        line.set_3d_properties(z)
    return lines

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=20, blit=True)
plt.show()