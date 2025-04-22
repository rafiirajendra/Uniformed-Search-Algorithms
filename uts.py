import heapq
import time
from collections import deque
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import math
from functools import lru_cache

# Data jarak asli (adjacency matrix)
distance_matrix = {
    'A1': {'A2': 8.8, 'A3': 7.4, 'A4': 7.7, 'A5': 12.2, 'A6': 10.1, 'A7': 15, 'A8': 8.2, 'A9': 13, 'A10': 18, 'A11': 16.5, 'A12': 25.8, 'A13': 27.1},
    'A2': {'A1': 8.8, 'A3': 14.5, 'A4': 15.9, 'A5': 19.4, 'A6': 13.1, 'A7': 16.1, 'A8': 7.5, 'A9': 8.7, 'A10': 16, 'A11': 18.6, 'A12': 25.9, 'A13': 18.6},
    'A3': {'A1': 7.4, 'A2': 14.5, 'A4': 13.3, 'A5': 18.1, 'A6': 16.7, 'A7': 25.5, 'A8': 14.9, 'A9': 18.1, 'A10': 25.1, 'A11': 23.8, 'A12': 32, 'A13': 38.8},
    'A4': {'A1': 7.7, 'A2': 15.9, 'A3': 13.3, 'A5': 8.8, 'A6': 9, 'A7': 11.7, 'A8': 12.2, 'A9': 17.1, 'A10': 13.4, 'A11': 13.9, 'A12': 26.2, 'A13': 19},
    'A5': {'A1': 12.2, 'A2': 19.4, 'A3': 18.1, 'A4': 8.8, 'A6': 5.8, 'A7': 9.7, 'A8': 10.2, 'A9': 14.5, 'A10': 13.5, 'A11': 7.1, 'A12': 18.7, 'A13': 10.6},
    'A6': {'A1': 10.1, 'A2': 13.1, 'A3': 16.7, 'A4': 9, 'A5': 5.8, 'A7': 7, 'A8': 5.4, 'A9': 9.8, 'A10': 10.1, 'A11': 7.8, 'A12': 17, 'A13': 18.4},
    'A7': {'A1': 15, 'A2': 16.1, 'A3': 25.5, 'A4': 11.7, 'A5': 9.7, 'A6': 7, 'A8': 8, 'A9': 9.7, 'A10': 7, 'A11': 5.4, 'A12': 11.1, 'A13': 14.4},
    'A8': {'A1': 8.2, 'A2': 7.5, 'A3': 14.9, 'A4': 12.2, 'A5': 10.2, 'A6': 5.4, 'A7': 8, 'A9': 4.3, 'A10': 8.6, 'A11': 9.3, 'A12': 18.1, 'A13': 20},
    'A9': {'A1': 13, 'A2': 8.7, 'A3': 18.1, 'A4': 17.1, 'A5': 14.5, 'A6': 9.8, 'A7': 9.7, 'A8': 4.3, 'A10': 10.1, 'A11': 12.1, 'A12': 18.4, 'A13': 17.1},
    'A10': {'A1': 18, 'A2': 16, 'A3': 25.1, 'A4': 13.4, 'A5': 13.5, 'A6': 10.1, 'A7': 7, 'A8': 8.6, 'A9': 10.1, 'A11': 11.1, 'A12': 11.6, 'A13': 16.7},
    'A11': {'A1': 16.5, 'A2': 18.6, 'A3': 23.8, 'A4': 13.9, 'A5': 7.1, 'A6': 7.8, 'A7': 5.4, 'A8': 9.3, 'A9': 12.1, 'A10': 11.1, 'A12': 14.1, 'A13': 5.9},
    'A12': {'A1': 25.8, 'A2': 25.9, 'A3': 32, 'A4': 26.2, 'A5': 18.7, 'A6': 17, 'A7': 11.1, 'A8': 18.1, 'A9': 18.4, 'A10': 11.6, 'A11': 14.1, 'A13': 12.7},
    'A13': {'A1': 27.1, 'A2': 18.6, 'A3': 38.8, 'A4': 19, 'A5': 10.6, 'A6': 18.4, 'A7': 14.4, 'A8': 20, 'A9': 17.1, 'A10': 16.7, 'A11': 5.9, 'A12': 12.7}
}

# Koordinat untuk visualisasi (untuk layout sederhana)
# Koordinat ini dapat disesuaikan untuk visualisasi yang lebih baik
node_positions = {
    'A1': (1, 1),
    'A2': (3, 2),
    'A3': (0, 3),
    'A4': (2, 4),
    'A5': (4, 5),
    'A6': (5, 3),
    'A7': (7, 4),
    'A8': (5, 1),
    'A9': (6, 0),
    'A10': (8, 2),
    'A11': (7, 6),
    'A12': (9, 5),
    'A13': (9, 7)
}

# Konstanta waktu
DEFAULT_SPEED = 30  # km/jam
UNLOADING_TIME = 15  # menit
LOADING_TIME = 30  # menit

# Optimasi: Caching jarak dengan LRU cache untuk mengurangi perhitungan berulang
@lru_cache(maxsize=1024)
def get_distance(node1, node2):
    return distance_matrix[node1][node2]

def bfs(start, goals):
    if not goals:
        return [start], 0, 0
    
    nodes_visited = 0
    current = start
    full_path = [current]
    total_distance = 0
    
    for goal in goals:
        queue = deque([[current]])
        visited = set()
        
        path_found = False
        while queue and not path_found:
            path = queue.popleft()
            node = path[-1]
            nodes_visited += 1
            
            if node == goal:
                full_path.extend(path[1:])
                for i in range(len(path)-1):
                    total_distance += get_distance(path[i], path[i+1])
                current = goal
                path_found = True
                break
            
            if node not in visited:
                for neighbor in distance_matrix[node]:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                visited.add(node)
        
        if not path_found:
            return None, float('inf'), nodes_visited
    
    return full_path, total_distance, nodes_visited

def dfs(start, goals):
    if not goals:
        return [start], 0, 0
    
    nodes_visited = 0
    current = start
    full_path = [current]
    total_distance = 0
    
    for goal in goals:
        stack = [[current]]
        visited = set()
        
        path_found = False
        while stack and not path_found:
            path = stack.pop()
            node = path[-1]
            nodes_visited += 1
            
            if node == goal:
                full_path.extend(path[1:])
                for i in range(len(path)-1):
                    total_distance += get_distance(path[i], path[i+1])
                current = goal
                path_found = True
                break
            
            if node not in visited:
                # Optimasi: Urutkan tetangga berdasarkan jarak terpendek
                neighbors = sorted(distance_matrix[node].items(), key=lambda x: x[1])
                for neighbor, _ in neighbors:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)
                visited.add(node)
        
        if not path_found:
            return None, float('inf'), nodes_visited
    
    return full_path, total_distance, nodes_visited

def ucs(start, goals):
    if not goals:
        return [start], 0, 0
    
    nodes_visited = 0
    current = start
    full_path = [current]
    total_distance = 0
    
    for goal in goals:
        heap = []
        heapq.heappush(heap, (0, [current]))
        visited = set()
        
        path_found = False
        while heap and not path_found:
            cost, path = heapq.heappop(heap)
            node = path[-1]
            nodes_visited += 1
            
            if node == goal:
                full_path.extend(path[1:])
                total_distance += cost
                current = goal
                path_found = True
                break
            
            if node not in visited:
                for neighbor, distance in distance_matrix[node].items():
                    new_cost = cost + distance
                    new_path = list(path)
                    new_path.append(neighbor)
                    heapq.heappush(heap, (new_cost, new_path))
                visited.add(node)
        
        if not path_found:
            return None, float('inf'), nodes_visited
    
    return full_path, total_distance, nodes_visited

def dls(start, goals, depth_limit):
    if not goals: 
        return [start], 0, 0
    
    nodes_visited = 0
    current = start
    full_path = [current]
    total_distance = 0
    
    for goal in goals:
        stack = [(current, [current], 0)]
        visited = set()
        
        path_found = False
        while stack and not path_found:
            node, path, depth = stack.pop()
            nodes_visited += 1
            
            if node == goal:
                full_path.extend(path[1:])
                for i in range(len(path)-1):
                    total_distance += get_distance(path[i], path[i+1])
                current = goal
                path_found = True
                break
            
            if depth < depth_limit:
                if node not in visited:
                    # Optimasi: Urutkan tetangga berdasarkan jarak terpendek
                    neighbors = sorted(distance_matrix[node].items(), key=lambda x: x[1])
                    for neighbor, _ in neighbors:
                        new_path = list(path)
                        new_path.append(neighbor)
                        stack.append((neighbor, new_path, depth + 1))
                    visited.add(node)
        
        if not path_found:
            return None, float('inf'), nodes_visited
    
    return full_path, total_distance, nodes_visited

def calculate_path_distance(path):
    distance = 0
    for i in range(len(path)-1):
        distance += get_distance(path[i], path[i+1])
    return distance

def calculate_travel_time(distance, speed=DEFAULT_SPEED):
    return (distance / speed) * 60

def calculate_total_duration(path, speed=DEFAULT_SPEED):
    distance = calculate_path_distance(path)
    travel_time = calculate_travel_time(distance, speed)
    total_time = LOADING_TIME
    for i in range(1, len(path)):
        total_time += UNLOADING_TIME
    total_time += travel_time
    return total_time

def is_within_operating_hours(start_time, duration, end_time_limit):
    end_time = start_time + datetime.timedelta(minutes=duration)
    return end_time <= end_time_limit

def format_duration(minutes):
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours} jam {mins} menit"

def visualize_graph(path=None, start=None, goals=None, save_path=None):
    G = nx.Graph()
    
    # Tambahkan semua node dan edge
    for node in distance_matrix:
        G.add_node(node)
        for neighbor, distance in distance_matrix[node].items():
            G.add_edge(node, neighbor, weight=distance)
    
    # Ukuran gambar
    plt.figure(figsize=(12, 10))
    
    # Warna dan ukuran node
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if node == start:
            node_colors.append('green')  # Titik awal berwarna hijau
            node_sizes.append(700)
        elif node in goals:
            node_colors.append('red')    # Tujuan berwarna merah
            node_sizes.append(700)
        else:
            node_colors.append('skyblue')
            node_sizes.append(500)
    
    # Gambar graf dengan node position yang telah ditentukan
    pos = node_positions
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Gambar jalur yang ditemukan
    path_edges = []
    if path and len(path) > 1:
        for i in range(len(path)-1):
            path_edges.append((path[i], path[i+1]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='red', alpha=0.7)
    
    # Gambar edge lainnya
    other_edges = [(u, v) for u, v in G.edges() if (u, v) not in path_edges and (v, u) not in path_edges]
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, width=1, alpha=0.5)
    
    # Tambahkan label jarak
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Tambahkan label node
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title("Visualisasi Graf Jaringan Pengiriman")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualisasi graf disimpan di: {save_path}")
    
    plt.show()

def print_path(path, start_time=None, end_time_limit=None, nodes_visited=0):
    if not path:
        print("Tidak ditemukan jalur")
        return
    
    print("Rute:", " -> ".join(path))
    distance = calculate_path_distance(path)
    print(f"Total jarak: {distance:.2f} km")
    
    duration = calculate_total_duration(path)
    print(f"Estimasi waktu perjalanan: {format_duration(duration)}")
    
    print(f"Jumlah node yang dikunjungi: {nodes_visited}")
    
    if start_time and end_time_limit:
        end_time = start_time + datetime.timedelta(minutes=duration)
        print(f"Waktu mulai: {start_time.strftime('%H:%M')}")
        print(f"Perkiraan waktu selesai: {end_time.strftime('%H:%M')}")
        
        if is_within_operating_hours(start_time, duration, end_time_limit):
            print("Status: Perjalanan dapat diselesaikan dalam jam operasional")
        else:
            print("Status: PERINGATAN! Perjalanan melewati batas jam operasional")
            overtime = (end_time - end_time_limit).total_seconds() / 60
            print(f"Perkiraan keterlambatan: {format_duration(overtime)}")

def parse_time(time_str):
    try:
        hours, minutes = map(int, time_str.split(':'))
        now = datetime.datetime.now()
        return datetime.datetime(now.year, now.month, now.day, hours, minutes)
    except ValueError:
        print("Format waktu tidak valid! Gunakan format HH:MM (contoh: 08:30)")
        return None

def compare_algorithms(start, goals, depth_limit=None):
    """Membandingkan performa semua algoritma dan menampilkan hasilnya."""
    results = []
    
    # BFS
    start_time = time.time()
    bfs_path, bfs_distance, bfs_visited = bfs(start, goals)
    bfs_time = time.time() - start_time
    if bfs_path:
        bfs_duration = calculate_total_duration(bfs_path)
        results.append(("BFS", bfs_distance, bfs_duration, bfs_visited, bfs_time, bfs_path))
    
    # DFS
    start_time = time.time()
    dfs_path, dfs_distance, dfs_visited = dfs(start, goals)
    dfs_time = time.time() - start_time
    if dfs_path:
        dfs_duration = calculate_total_duration(dfs_path)
        results.append(("DFS", dfs_distance, dfs_duration, dfs_visited, dfs_time, dfs_path))
    
    # UCS
    start_time = time.time()
    ucs_path, ucs_distance, ucs_visited = ucs(start, goals)
    ucs_time = time.time() - start_time
    if ucs_path:
        ucs_duration = calculate_total_duration(ucs_path)
        results.append(("UCS", ucs_distance, ucs_duration, ucs_visited, ucs_time, ucs_path))
    
    # DLS
    if depth_limit:
        start_time = time.time()
        dls_path, dls_distance, dls_visited = dls(start, goals, depth_limit)
        dls_time = time.time() - start_time
        if dls_path:
            dls_duration = calculate_total_duration(dls_path)
            results.append((f"DLS (depth={depth_limit})", dls_distance, dls_duration, dls_visited, dls_time, dls_path))
    
    # Sort by total distance
    results.sort(key=lambda x: x[1])
    
    print("\n" + "="*70)
    print("PERBANDINGAN ALGORITMA PENCARIAN".center(70))
    print("="*70)
    print(f"{'Algoritma':<20} {'Jarak (km)':<15} {'Waktu Tempuh':<20} {'Node Dikunjungi':<15} {'Waktu Komputasi':<15}")
    print("-"*70)
    
    for algo, distance, duration, visited, comp_time, _ in results:
        print(f"{algo:<20} {distance:<15.2f} {format_duration(duration):<20} {visited:<15} {comp_time:<15.4f}")
    
    # Visualisasi perbandingan algoritma
    visualize_comparison(results)
    
    return results

def visualize_comparison(results):
    """Visualisasi perbandingan algoritma dalam bentuk grafik."""
    if not results:
        return
    
    algos = [r[0] for r in results]
    distances = [r[1] for r in results]
    durations = [r[2] for r in results]
    nodes_visited = [r[3] for r in results]
    comp_times = [r[4] for r in results]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot jarak
    axs[0, 0].bar(algos, distances, color='skyblue')
    axs[0, 0].set_title('Perbandingan Jarak (km)')
    axs[0, 0].set_ylabel('Jarak (km)')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    plt.setp(axs[0, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot waktu tempuh
    durations_hours = [d/60 for d in durations]  # Convert to hours for better visualization
    axs[0, 1].bar(algos, durations_hours, color='salmon')
    axs[0, 1].set_title('Perbandingan Waktu Tempuh (jam)')
    axs[0, 1].set_ylabel('Waktu (jam)')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    plt.setp(axs[0, 1].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot node dikunjungi
    axs[1, 0].bar(algos, nodes_visited, color='lightgreen')
    axs[1, 0].set_title('Perbandingan Node Dikunjungi')
    axs[1, 0].set_ylabel('Jumlah Node')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    plt.setp(axs[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot waktu komputasi
    axs[1, 1].bar(algos, comp_times, color='mediumpurple')
    axs[1, 1].set_title('Perbandingan Waktu Komputasi (detik)')
    axs[1, 1].set_ylabel('Waktu (detik)')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def get_algorithm_menu():
    """Menu pilihan algoritma"""
    print("\nPilih metode pencarian:")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Uniform Cost Search (UCS)")
    print("4. Depth-Limited Search (DLS)")
    print("5. Bandingkan Semua Algoritma")
    print("6. Ganti titik awal/tujuan")
    print("7. Keluar")
    return input("Masukkan pilihan (1-7): ")

def main():
    while True:
        print("\n╔══════════════════════════════════════════════════╗")
        print("║       PENCARIAN JALUR OPTIMAL PENGIRIMAN         ║")
        print("║        DENGAN VISUALISASI DAN OPTIMASI           ║")
        print("╚══════════════════════════════════════════════════╝")
        print("\nTitik yang tersedia:", list(distance_matrix.keys()))
        
        start = input("\nMasukkan titik awal (atau 'exit' untuk keluar): ").upper()
        
        if start.lower() == 'exit':
            print("Keluar dari program...")
            break
            
        if start not in distance_matrix:
            print("Titik awal tidak valid!")
            continue
        
        print("\nMasukkan titik tujuan (pisahkan dengan koma, contoh: A2,A5,A8)")
        goals_input = input("Tujuan: ").upper()
        goals = [g.strip() for g in goals_input.split(',')]
        
        invalid_goals = [g for g in goals if g not in distance_matrix]
        if invalid_goals:
            print(f"Titik tujuan tidak valid: {', '.join(invalid_goals)}")
            continue
        
        use_time_limit = input("\nGunakan batas waktu operasional? (y/n): ").lower() == 'y'
        
        start_time = None
        end_time_limit = None
        
        if use_time_limit:
            print("Masukkan waktu mulai operasional (format HH:MM, contoh: 08:00):")
            start_time_str = input("Waktu mulai: ")
            start_time = parse_time(start_time_str)
            
            print("Masukkan batas akhir waktu operasional (format HH:MM, contoh: 17:00):")
            end_time_str = input("Batas waktu: ")
            end_time_limit = parse_time(end_time_str)
            
            if not start_time or not end_time_limit:
                continue
                
            if end_time_limit <= start_time:
                print("Error: Waktu akhir harus lebih besar dari waktu mulai!")
                continue
        
        # Visualisasi graf awal dengan titik awal dan tujuan
        visualize_graph(start=start, goals=goals)
        
        while True:
            choice = get_algorithm_menu()
            
            if choice == '6':
                break
            elif choice == '7':
                print("Keluar dari program...")
                exit()
            elif choice in ['1', '2', '3', '4', '5']:
                if choice == '5':
                    depth_limit = int(input("Masukkan kedalaman maksimum untuk DLS: "))
                    results = compare_algorithms(start, goals, depth_limit)
                    
                    # Menanyakan kepada user apakah ingin memvisualisasikan salah satu jalur
                    viz_choice = input("\nApakah ingin memvisualisasikan salah satu jalur? (y/n): ").lower()
                    if viz_choice == 'y':
                        for i, (algo, _, _, _, _, _) in enumerate(results, 1):
                            print(f"{i}. {algo}")
                        viz_algo = int(input("Pilih algoritma (nomor): ")) - 1
                        if 0 <= viz_algo < len(results):
                            visualize_graph(path=results[viz_algo][5], start=start, goals=goals)
                else:
                    search_start_time = time.time()
                    
                    if choice == '1':
                        print("\n--- Hasil BFS ---")
                        path, distance, nodes_visited = bfs(start, goals)
                    elif choice == '2':
                        print("\n--- Hasil DFS ---")
                        path, distance, nodes_visited = dfs(start, goals)
                    elif choice == '3':
                        print("\n--- Hasil UCS ---")
                        path, distance, nodes_visited = ucs(start, goals)
                    elif choice == '4':
                        depth_limit = int(input("Masukkan kedalaman maksimum: "))
                        print(f"\n--- Hasil DLS (kedalaman {depth_limit}) ---")
                        path, distance, nodes_visited = dls(start, goals, depth_limit)
                    
                    search_end_time = time.time()
                    
                    print("\n" + "="*50)
                    print("HASIL PENCARIAN RUTE")
                    print("="*50)
                    
                    if path:
                        print_path(path, start_time, end_time_limit, nodes_visited)
                        print(f"\nWaktu komputasi pencarian: {search_end_time - search_start_time:.4f} detik")
                        
                        # Visualisasi jalur hasil pencarian
                        save_option = input("\nApakah ingin menyimpan visualisasi? (y/n): ").lower()
                        save_path = None
                        if save_option == 'y':
                            save_path = input("Masukkan nama file (contoh: rute.png): ")
                        
                        visualize_graph(path=path, start=start, goals=goals, save_path=save_path)
                    else:
                        print("Tidak ditemukan jalur untuk semua tujuan")
                
                another_method = input("\nCoba metode lain untuk rute yang sama? (y/n): ").lower()
                if another_method != 'y':
                    break
            else:
                print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()