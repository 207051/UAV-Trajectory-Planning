from PIL import Image
import numpy as np
import heapq
import cvxpy as cp
import time 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- GÖRSEL YÜKLEME FONKSİYONU ---
def get_image(path, zoom=0.08, angle=0): # angle parametresi eklendi
    try:
        img = Image.open(path).convert("RGBA")
        
        # --- DÖNDÜRME İŞLEMİ ---
        if angle != 0:
            # expand=True resmin köşelerinin kesilmesini önler
            img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
        
        data = np.array(img)
        # ... (maskeleme işlemleri aynı kalıyor) ...
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        mask = (r > 200) & (g > 200) & (b > 200)
        data[:,:,3][mask] = 0
        
        return OffsetImage(data, zoom=zoom, resample=True)
    except Exception as e:
        print(f"Hata: {e}")
        return None

# --- 1. ADIM: A* ALGORİTMASI (Dinamik Yarıçaplı) ---
def a_star(start, goal, obstacles, radii, grid_size=120):
    def dist(a, b): return np.linalg.norm(np.array(a) - np.array(b))
    
    close_set = set()
    came_from = {}
    gscore = {tuple(start): 0}
    fscore = {tuple(start): dist(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[tuple(start)], tuple(start)))
    
    # Komşu hareketleri (8 yönlü)
    neighbors = [(0,2),(0,-2),(2,0),(-2,0),(2,2),(2,-2),(-2,2),(-2,-2)]
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        if dist(current, goal) < 4.0:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return np.array(path[::-1])
        
        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if 0 <= neighbor[0] <= grid_size and 0 <= neighbor[1] <= grid_size:
                is_blocked = False
                # Her engelin kendi yarıçapını kontrol et
                for idx, obs in enumerate(obstacles):
                    if dist(neighbor, obs) < radii[idx] + 1.5: # 1.5 güvenlik payı
                        is_blocked = True
                        break
                
                if is_blocked or neighbor in close_set: continue
                
                tg = gscore[current] + dist(current, neighbor)
                if neighbor not in gscore or tg < gscore[neighbor]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tg
                    fscore[neighbor] = tg + dist(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return None

# --- 2. ADIM: SCP ALGORİTMASI (Dinamik Yarıçaplı) ---
def solve_scp_hybrid(start, goal, obstacles, radii, initial_path):
    start_total_time = time.time()
    # Yolu örnekle (N nokta)
    indices = np.linspace(0, len(initial_path)-1, 50, dtype=int)
    p_curr = initial_path[indices]
    N = len(p_curr)
    p = cp.Variable((N, 2))
    
    tolerance = 0.05
    max_iters = 5
    
    for i in range(max_iters):
        # Maliyet: Kısa yol + Yumuşak dönüşler (Smoothness)
        cost = cp.sum_squares(p[1:] - p[:-1]) + 50 * cp.sum_squares(p[2:] - 2*p[1:-1] + p[:-2])
        
        constraints = [p[0] == start, p[-1] == goal]
        
        # Dinamik Engel Kısıtlamaları (Linearized Distance Constraints)
        for k in range(N):
            for idx, obs in enumerate(obstacles):
                diff_vec = p_curr[k] - obs
                d_prev = np.linalg.norm(diff_vec)
                # Taylor serisi açılımı ile dışbükeyleştirme
                constraints.append(d_prev + (diff_vec/d_prev) @ (p[k] - p_curr[k]) >= radii[idx] + 1.0)
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        
        if p.value is None: break # Çözüm bulunamazsa çık
        
        p_new = p.value
        diff_magnitude = np.linalg.norm(p_new - p_curr, axis=1).max()
        p_curr = p_new
        if diff_magnitude < tolerance: break
        
    return p_curr, time.time() - start_total_time

# --- 3. ADIM: SENARYO VE ÇALIŞTIRMA ---
grid_limit = 120
start_node = np.array([10, 60])
goal_node = np.array([110, 60])

# Engeller ve Yarıçapları: [x, y, radius]
obs_data = np.array([
    [35, 65, 12.0], 
    [60, 50, 6.0],  
    [85, 70, 15.0], 
    [60, 80, 8.0],  
    [45, 35, 7.0],
    [20, 100, 15.0],  
    [60, 100, 15.0], 
    [60, 20, 10.0], 
    [60, 60, 10.0], 
    [60, 35, 10.0], 
    [20, 35, 10.0], 
    [90, 30, 15.0]
])
complex_obs = obs_data[:, :2]
radii = obs_data[:, 2]

# Hesaplamalar
raw_path = a_star(start_node, goal_node, complex_obs, radii, grid_limit)

if raw_path is not None:
    final_path, scp_duration = solve_scp_hybrid(start_node, goal_node, complex_obs, radii, raw_path)
    path_dist = np.sum(np.linalg.norm(np.diff(final_path, axis=0), axis=1))

    # --- GÖRSELLEŞTİRME ---
    fig, ax = plt.subplots(figsize=(12, 7))
    start_img = get_image("start.png", zoom=0.1, angle=-130)
    base_img = get_image("base.png", zoom=0.11)

    # Engelleri Çiz (Her biri kendi yarıçapıyla)
    for i, o in enumerate(complex_obs):
        circle = plt.Circle(o, radii[i], color='red', alpha=0.15, label="Obstacle" if i==0 else "")
        ax.add_patch(circle)
        # Tank görsellerini ekle (Dosyalar mevcutsa)
        # Mevcut satırı şununla değiştir:
        tank_img = get_image("tank_a.png", zoom=radii[i] * 0.009)
        if tank_img:
            ax.add_artist(AnnotationBbox(tank_img, o, frameon=False))
        else:
            ax.scatter(o[0], o[1], c='darkred', marker='X')

    # Rotalar
    ax.plot(raw_path[:,0], raw_path[:,1], 'b--', alpha=0.4, label="Route A*")
    ax.plot(final_path[:,0], final_path[:,1], 'm-', linewidth=2.5, label="A* Driven SCP")

    # Başlangıç ve Bitiş
    # Başlangıç (Drone) ve Bitiş (Üs) İkonları
    if start_img:
        ax.add_artist(AnnotationBbox(start_img, (start_node[0], start_node[1]), frameon=False, zorder=6))
    if base_img:
        ax.add_artist(AnnotationBbox(base_img, (goal_node[0], goal_node[1]), frameon=False, zorder=6))

    # Bilgi
    #ax.set_title("Farklı Yarıçaplı Engeller Arasında Yol Planlama", fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(0, grid_limit); ax.set_ylim(0, grid_limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    plt.show()
else:
    print("Yol bulunamadı!")