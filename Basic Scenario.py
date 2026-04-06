import numpy as np
import heapq
import cvxpy as cp
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- GÖRSEL YÜKLEME ---
def get_image(path, zoom=0.08, angle=0):
    try:
        img = Image.open(path).convert("RGBA")
        if angle != 0:
            img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
        data = np.array(img)
        # Beyaz arka plan temizleme
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        mask = (r > 200) & (g > 200) & (b > 200)
        data[:,:,3][mask] = 0
        return OffsetImage(data, zoom=zoom, resample=True)
    except:
        return None

# --- A* ALGORİTMASI ---
def a_star(start, goal, obstacles, radius):
    def dist(a, b): return np.linalg.norm(np.array(a) - np.array(b))
    step = 3.0
    close_set = set()
    came_from = {}
    start_t = tuple(np.round(start, 2))
    gscore = {start_t: 0}
    fscore = {start_t: dist(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start_t], start_t))
    neighbors = [(step,0),(-step,0),(0,step),(0,-step),(step,step),(step,-step),(-step,step),(-step,-step)]
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        if dist(current, goal) < 5.0:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return np.array(path[::-1])
        close_set.add(current)
        for dx, dy in neighbors:
            neighbor = (np.round(current[0]+dx, 2), np.round(current[1]+dy, 2))
            if any(dist(neighbor, obs) < radius + 0.5 for obs in obstacles) or neighbor in close_set:
                continue
            tg = gscore[current] + dist(current, neighbor)
            if neighbor not in gscore or tg < gscore[neighbor]:
                came_from[neighbor] = current
                gscore[neighbor] = tg
                fscore[neighbor] = tg + dist(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return None

# --- SCP FONKSİYONU ---
def solve_scp(start, goal, obstacles, radius, initial_guess_path, lmb=100):
    N = 50
    indices = np.linspace(0, len(initial_guess_path)-1, N, dtype=int)
    p_curr = initial_guess_path[indices]
    for i in range(8):
        p = cp.Variable((N, 2))
        cost = cp.sum_squares(p[1:] - p[:-1]) + lmb * cp.sum_squares(p[2:] - 2*p[1:-1] + p[:-2])
        constraints = [p[0] == start, p[-1] == goal]
        for k in range(N):
            for obs in obstacles:
                diff_vec = p_curr[k] - obs
                d_prev = np.linalg.norm(diff_vec)
                constraints.append(d_prev + (diff_vec/d_prev) @ (p[k] - p_curr[k]) >= radius + 0.1)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS)
        if p.value is not None: p_curr = p.value
        else: break
    return p_curr

# --- PARAMETRELER ---
start_p = np.array([90, 60])
goal_p = np.array([50, 90])
obs_centers = np.array([[40, 40], [40, 60], [65, 80], [75, 75], [70, 40], [70, 60]])
d = 10

# --- HESAPLAMALAR ---
a_star_path = a_star(start_p, goal_p, obs_centers, d)
hybrid_path = solve_scp(start_p, goal_p, obs_centers, d, a_star_path, lmb=100)
straight_line = np.linspace(start_p, goal_p, 50)
pure_scp_path = solve_scp(start_p, goal_p, obs_centers, d, straight_line, lmb=100)

# --- GÖRSELLEŞTİRME ---
fig, ax = plt.subplots(figsize=(12, 9))

# Önce Grid ve Oranlar
ax.set_aspect('equal', adjustable='box') # plt.axis('equal') yerine daha kesin çözüm
ax.grid(True, alpha=0.2)

# Engeller
for obs in obs_centers:
    ax.add_patch(plt.Circle(obs, d, color='red', alpha=0.3, edgecolor='darkred'))
    tank = get_image("tank_a.png", zoom=d * 0.015)
    if tank: ax.add_artist(AnnotationBbox(tank, obs, frameon=False))

# Rotalar
ax.plot(a_star_path[:,0], a_star_path[:,1], 'b--', alpha=0.6, label='A* Route')
ax.plot(hybrid_path[:,0], hybrid_path[:,1], 'g-', linewidth=3, label='A* Driven SCP')
ax.plot(pure_scp_path[:,0], pure_scp_path[:,1], 'm-', linewidth=2, label='Pure SCP')

# İkonlar
s_img = get_image("start.png", zoom=0.1, angle=60)
g_img = get_image("base.png", zoom=0.15)
if s_img: ax.add_artist(AnnotationBbox(s_img, start_p, frameon=False, zorder=10))
if g_img: ax.add_artist(AnnotationBbox(g_img, goal_p, frameon=False, zorder=10))

# SINIRLAR (En son ayarlanmalı)
ax.set_xlim(20, 110)
ax.set_ylim(20, 110)

# Tek ve Büyük Lejant
ax.legend(prop={'size': 10}, markerscale=1.5, loc='upper right', frameon=True, shadow=True)

plt.show()