import os
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # python-louvain
from networkx.algorithms import bipartite
 
random.seed(42)
 
# ==========================================
# 0. KONFIGURASI
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CANDIDATES = [
    os.path.join(BASE_DIR, "act-mooc", "mooc_actions.tsv"),
    os.path.join(BASE_DIR, "act-mooc", "act-mooc", "mooc_actions.tsv"),
]
DATASET_PATH = next((p for p in DATASET_CANDIDATES if os.path.exists(p)), None)
OUTPUT_DIR = "."  # ganti jika ingin menyimpan hasil di folder lain
 
if DATASET_PATH is None:
    raise FileNotFoundError(
        f"File dataset tidak ditemukan di lokasi yang dicoba: {DATASET_CANDIDATES}."
    )
 
# ==========================================
# 1. MEMBACA DATASET (dengan skiprows=1 untuk membuang baris header)
# ==========================================
print("Membaca dataset MOOC...")
df = pd.read_csv(DATASET_PATH, sep="\t", header=None, skiprows=1)
df.columns = ["ACTIONID", "USERID", "TARGETID", "TIMESTAMP"][: df.shape[1]]
 
print(f"  Jumlah baris (aksi)      : {len(df)}")
print(f"  Unique USERID            : {df['USERID'].nunique()}")
print(f"  Unique TARGETID          : {df['TARGETID'].nunique()}")
print(f"  Irisan USERID ∩ TARGETID : "
      f"{len(set(df['USERID'].astype(str)) & set(df['TARGETID'].astype(str)))}")
 
# ==========================================
# 2. NAMESPACE ID + BOBOT EDGE (frekuensi interaksi)
# ==========================================
print("\nMemberi namespace pada ID dan menghitung bobot edge...")
df["USERID"] = "U_" + df["USERID"].astype(str)
df["TARGETID"] = "T_" + df["TARGETID"].astype(str)
 
# Bobot = jumlah aksi berulang pada pasangan (user, target) yang sama.
# Ini mempertahankan informasi frekuensi yang hilang jika edge langsung
# dibangun tanpa agregasi (lihat catatan di README/diskusi review).
df_weighted = (
    df.groupby(["USERID", "TARGETID"])
    .size()
    .reset_index(name="weight")
)
print(f"  Jumlah aksi total        : {len(df)}")
print(f"  Jumlah pasangan unik     : {len(df_weighted)} (ini yang jadi edge)")
 
# ==========================================
# 3. BANGUN GRAF (DIRECTED, BERBOBOT)
# ==========================================
print("\nMembangun graf NetworkX...")
G = nx.from_pandas_edgelist(
    df_weighted,
    source="USERID",
    target="TARGETID",
    edge_attr="weight",
    create_using=nx.DiGraph(),
)
 
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print(f"  Jumlah Node : {num_nodes}  (target: 7144 = 7047 user + 97 target)")
print(f"  Jumlah Edge : {num_edges}")
 
user_nodes = {n for n in G.nodes() if n.startswith("U_")}
target_nodes = {n for n in G.nodes() if n.startswith("T_")}
print(f"  -> Node bertipe USER   : {len(user_nodes)}")
print(f"  -> Node bertipe TARGET : {len(target_nodes)}")
 
# ==========================================
# 4. METRIK DASAR
# ==========================================
print("\n=== METRIK DASAR ===")
 
degree_cent = nx.degree_centrality(G)
top_5_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 Node berdasarkan Degree Centrality:")
for node, val in top_5_degree:
    tipe = "USER" if node.startswith("U_") else "TARGET"
    print(f"  {node} [{tipe}]: {val:.4f}")
 
density = nx.density(G)
print(f"\nDensity Jejaring : {density:.6f}")
 
G_undirected = G.to_undirected()
 
avg_clustering = nx.average_clustering(G_undirected)
print(f"Average Clustering Coefficient : {avg_clustering:.6f}  "
      f"(harus mendekati 0 untuk graf bipartit murni)")
if avg_clustering > 0.01:
    print("  !! PERINGATAN: ACC jauh dari 0 - masih ada anomali yang perlu "
          "ditelusuri sebelum lanjut ke tahap berikutnya.")
 
is_bip = bipartite.is_bipartite(G_undirected)
print(f"Graf terverifikasi bipartit (nx.bipartite.is_bipartite) : {is_bip}")
 
# ==========================================
# 5. GIANT COMPONENT, DIAMETER, AVERAGE PATH LENGTH
# ==========================================
print("\n=== METRIK JARINGAN (GIANT COMPONENT) ===")
if nx.is_connected(G_undirected):
    G_sub = G_undirected
    print("Jejaring terhubung penuh.")
else:
    giant_component = max(nx.connected_components(G_undirected), key=len)
    G_sub = G_undirected.subgraph(giant_component).copy()
    print("Jejaring tidak terhubung sepenuhnya.")
    print(f"Ukuran giant component: {G_sub.number_of_nodes()} node "
          f"({G_sub.number_of_nodes() / num_nodes * 100:.1f}% dari total)")
 
print("Menghitung diameter (mohon tunggu)...")
diameter = nx.diameter(G_sub)
print(f"Diameter Komponen Terbesar : {diameter}")
 
print("Menghitung average shortest path length (approx, sample 500 node)...")
sample_nodes = random.sample(list(G_sub.nodes()), min(500, G_sub.number_of_nodes()))
lengths = []
for n in sample_nodes:
    sp = nx.single_source_shortest_path_length(G_sub, n)
    lengths.extend(sp.values())
avg_path_length = sum(lengths) / len(lengths)
print(f"Average Path Length (approx, sample 500) : {avg_path_length:.4f}")
 
# ==========================================
# 6. CENTRALITY LANJUTAN (pada giant component)
# ==========================================
print("\nMenghitung Betweenness Centrality (approx, k=500)...")
betweenness = nx.betweenness_centrality(G_sub, k=min(500, G_sub.number_of_nodes()), seed=42)
top_5_betw = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 Betweenness Centrality:")
for node, val in top_5_betw:
    tipe = "USER" if node.startswith("U_") else "TARGET"
    print(f"  {node} [{tipe}]: {val:.4f}")
 
print("\nMenghitung Closeness Centrality...")
closeness = nx.closeness_centrality(G_sub)
top_5_close = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 Closeness Centrality:")
for node, val in top_5_close:
    tipe = "USER" if node.startswith("U_") else "TARGET"
    print(f"  {node} [{tipe}]: {val:.4f}")
 
print("\nMenghitung Eigenvector Centrality...")
eigenvector = nx.eigenvector_centrality_numpy(G_sub)
top_5_eigen = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 Eigenvector Centrality:")
for node, val in top_5_eigen:
    tipe = "USER" if node.startswith("U_") else "TARGET"
    print(f"  {node} [{tipe}]: {val:.4f}")
 
# ==========================================
# 7. DETEKSI KOMUNITAS — VIA PROYEKSI ONE-MODE (BUKAN LOUVAIN LANGSUNG)
# ==========================================
# Louvain pada graf bipartit mentah tidak bermakna secara teoretis
# (tidak ada segitiga untuk dioptimalkan modularitasnya secara wajar).
# Pendekatan baku: proyeksikan ke graf one-mode (user-user, terhubung
# jika berbagi >=1 target yang sama), baru jalankan Louvain di situ.
print("\n=== DETEKSI KOMUNITAS (PROYEKSI USER-USER) ===")
print("Membangun proyeksi user-user (bisa memakan waktu)...")
 
G_bip_user_target = nx.Graph()
G_bip_user_target.add_nodes_from(user_nodes, bipartite=0)
G_bip_user_target.add_nodes_from(target_nodes, bipartite=1)
G_bip_user_target.add_edges_from(G_undirected.edges())
 
user_projection = bipartite.weighted_projected_graph(G_bip_user_target, user_nodes)
print(f"  Node pada proyeksi user-user : {user_projection.number_of_nodes()}")
print(f"  Edge pada proyeksi user-user : {user_projection.number_of_edges()}")
 
partition = community_louvain.best_partition(user_projection, random_state=42)
modularity_q = community_louvain.modularity(partition, user_projection)
num_communities = len(set(partition.values()))
print(f"  Jumlah Komunitas Terdeteksi : {num_communities}")
print(f"  Modularity (Q)              : {modularity_q:.4f}")
 
# Distribusi ukuran komunitas -- dihitung eksak, bukan estimasi visual
from collections import Counter
komunitas_sizes = Counter(partition.values())
print("  Ukuran tiap komunitas (eksak):")
for kom_id, size in sorted(komunitas_sizes.items(), key=lambda x: -x[1]):
    print(f"    Komunitas {kom_id}: {size} user")
 
# ==========================================
# 8. VISUALISASI
# ==========================================
print("\nMembuat visualisasi graf bipartit (user vs target)...")
plt.figure(figsize=(14, 14))
pos = nx.spring_layout(G_undirected, k=0.15, seed=42)
 
node_colors = ["#e74c3c" if n.startswith("U_") else "#3498db" for n in G_undirected.nodes()]
node_sizes = [degree_cent[n] * 2000 for n in G_undirected.nodes()]
 
nx.draw_networkx_nodes(G_undirected, pos, node_size=node_sizes,
                        node_color=node_colors, alpha=0.7)
nx.draw_networkx_edges(G_undirected, pos, alpha=0.02, edge_color="gray")
plt.axis("off")
plt.title(
    "Visualisasi Jejaring MOOC (Bipartit)\n"
    "Merah = User, Biru = Target Aktivitas | Ukuran = Degree Centrality",
    fontsize=16,
)
 
output_filename = os.path.join(OUTPUT_DIR, "visualisasi_mooc_jejaring_v2.png")
plt.savefig(output_filename, format="PNG", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Visualisasi disimpan sebagai '{output_filename}'.")
 
# ==========================================
# 9. RINGKASAN AKHIR
# ==========================================
print("\n=== RINGKASAN AKHIR ===")
print(f"Jumlah Node (user+target)   : {num_nodes}")
print(f"Jumlah Edge (unik, berbobot): {num_edges}")
print(f"Total aksi (sebelum agregasi bobot): {len(df)}")
print(f"Density                     : {density:.6f}")
print(f"Average Clustering Coeff.   : {avg_clustering:.6f}")
print(f"Bipartit terverifikasi      : {is_bip}")
print(f"Diameter (giant component)  : {diameter}")
print(f"Average Path Length (approx): {avg_path_length:.4f}")
print(f"Jumlah Komunitas (proyeksi) : {num_communities}")
print(f"Modularity (Q)              : {modularity_q:.4f}")

import random
from collections import defaultdict

random.seed(42)

def simulate_SI(G, seed_node, beta=0.1, steps=10):
    infected = {seed_node}
    susceptible = set(G.nodes()) - infected
    history = [len(infected)]
    for _ in range(steps):
        new_infected = set()
        for node in list(infected):
            for neighbor in G.neighbors(node):
                if neighbor in susceptible:
                    if random.random() < beta:
                        new_infected.add(neighbor)
        infected |= new_infected
        susceptible -= new_infected
        history.append(len(infected))
    return history

print("\n=== SIMULASI DIFUSI SI (CONTENT-EXPOSURE) ===")
print("Titik injeksi: T_1, T_3, T_8 | beta=0.1, steps=10\n")

seed_nodes = ["T_1", "T_3", "T_8"]
results = {}
for seed in seed_nodes:
    random.seed(42)  # reset seed di tiap run agar ketiganya independen & reproducible
    results[seed] = simulate_SI(G_undirected, seed_node=seed, beta=0.1, steps=10)

time_steps = list(range(len(results["T_1"])))
print(f"{'Time Step':<12}{'T_1':<10}{'T_3':<10}{'T_8':<10}")
for t in time_steps:
    print(f"{t:<12}{results['T_1'][t]:<10}{results['T_3'][t]:<10}{results['T_8'][t]:<10}")

print("\n=== RINGKASAN AKHIR SIMULASI ===")
for seed in seed_nodes:
    total_terekspos = results[seed][-1]
    persen = total_terekspos / G_undirected.number_of_nodes() * 100
    print(f"{seed}: {total_terekspos} node terekspos setelah 10 step "
          f"({persen:.1f}% dari total jejaring)")

print("\n✅ Selesai.")