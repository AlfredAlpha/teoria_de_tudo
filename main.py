import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# 1. Constante de Estrutura Fina (α ≈ 1/137)
alpha = 1 / 137
print(f"Constante de Estrutura Fina (α): {alpha}")

# 2. Simulação de uma Estrutura Fractal (Teia Cósmica e Rede Neural)
def generate_fractal_structure(n_points=1000):
    # Gera pontos aleatórios em 3D para simular uma estrutura fractal
    np.random.seed(42)
    points = np.random.rand(n_points, 3)
    
    # Calcula distâncias entre pontos
    distances = squareform(pdist(points))
    
    # Conecta pontos próximos para formar uma rede
    threshold = np.percentile(distances, 5)  # Conecta os 5% mais próximos
    adjacency_matrix = distances < threshold
    
    # Cria um grafo a partir da matriz de adjacência
    graph = nx.from_numpy_array(adjacency_matrix)
    return graph, points

# 3. Visualização da Estrutura Fractal
def plot_fractal_structure(graph, points, title="Estrutura Fractal: Teia Cósmica e Rede Neural"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plota os nós
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, c='blue', alpha=0.6)
    
    # Plota as arestas
    for edge in graph.edges():
        start, end = edge
        ax.plot([points[start, 0], points[end, 0]],
                [points[start, 1], points[end, 1]],
                [points[start, 2], points[end, 2]], c='black', alpha=0.1)
    
    ax.set_title(title)
    plt.show()

# 4. Simulação de Buracos Negros como "Portais"
def simulate_black_holes(points, n_black_holes=5):
    # Seleciona aleatoriamente alguns pontos como "buracos negros"
    black_holes = np.random.choice(len(points), n_black_holes, replace=False)
    
    # Marca os buracos negros no gráfico
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plota todos os pontos
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, c='blue', alpha=0.6)
    
    # Plota os buracos negros
    ax.scatter(points[black_holes, 0], points[black_holes, 1], points[black_holes, 2],
               s=100, c='red', marker='X', label="Buracos Negros")
    
    ax.set_title("Buracos Negros como 'Portais' na Estrutura Fractal")
    ax.legend()
    plt.show()

# 5. Física de Buracos Negros (Simulação Simplificada)
def black_hole_physics(points, black_holes):
    # Simula o efeito gravitacional de buracos negros (atração de pontos próximos)
    for bh in black_holes:
        distances = np.linalg.norm(points - points[bh], axis=1)
        influence = 0.1 / (distances + 1e-6)  # Lei do inverso do quadrado (simplificada)
        points += influence[:, None] * (points[bh] - points) * 0.01  # Atualiza posições
    return points

# 6. Simulação de Multiverso (Múltiplas Estruturas Fractais)
def simulate_multiverse(n_universes=3, n_points=500):
    multiverse = []
    for i in range(n_universes):
        graph, points = generate_fractal_structure(n_points)
        multiverse.append((graph, points))
        plot_fractal_structure(graph, points, title=f"Universo {i+1}")
    return multiverse

# 7. Machine Learning: Identificação de Padrões na Estrutura Fractal
def analyze_patterns_with_ml(points):
    # Gera dados sintéticos para treinamento (exemplo: distâncias como características)
    X = squareform(pdist(points))
    y = np.random.rand(len(points))  # Rótulos aleatórios (exemplo)
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treina uma rede neural para prever padrões
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    
    # Avalia o modelo
    score = model.score(X_test, y_test)
    print(f"Score do modelo de ML: {score:.2f}")

# 8. Execução do Código
if __name__ == "__main__":
    # Gera e plota a estrutura fractal
    graph, points = generate_fractal_structure()
    plot_fractal_structure(graph, points)
    
    # Simula buracos negros na estrutura fractal
    black_holes = np.random.choice(len(points), 5, replace=False)
    simulate_black_holes(points, black_holes)
    
    # Aplica física de buracos negros
    points = black_hole_physics(points, black_holes)
    plot_fractal_structure(graph, points, title="Estrutura Fractal com Efeito de Buracos Negros")
    
    # Simula um multiverso
    multiverse = simulate_multiverse(n_universes=3, n_points=500)
    
    # Usa machine learning para analisar padrões
    analyze_patterns_with_ml(points)