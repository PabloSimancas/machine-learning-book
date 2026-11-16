# ================================
# Perceptrón: visualizaciones 2D y 3D
# ================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necesario para proyección 3d en mpl)

# ---------- Utilidades comunes ----------
def decision_boundary_2d_x2(x1, w, b):
    # Frontera: w1*x1 + w2*x2 + b = 0  => x2 = -(w1/w2)*x1 - b/w2
    print(-(w[0]/w[1]) * x1 - b / w[1])
    return -(w[0]/w[1]) * x1 - b / w[1]

def perceptron_sign(X, w, b):
    # Devuelve signo de z = X @ w + b
    return np.sign(X @ w + b)

# ---------- 1) Plot 2D (dos features) ----------
def plot_perceptron_2d(w=np.array([1.0, -0.6]), b=-0.2, seed=7):
    np.random.seed(seed)
    # Datos de ejemplo (dos "nubes")
    X_pos = np.random.uniform(2.5, 5.0, (20, 2))
    X_neg = np.random.uniform(0.0, 2.5, (20, 2))

    # Punto para mostrar proyección sobre w
    x_sample = np.array([3.2, 1.3])

    # Unitarizamos w para dibujar flecha de dirección
    w_unit = w / np.linalg.norm(w)
    proj_len = np.dot(x_sample, w_unit)
    x_proj = proj_len * w_unit

    fig, ax = plt.subplots(figsize=(7, 7))

    # Clasificación con el perceptrón (solo para colorear)
    pos_mask = perceptron_sign(X_pos, w, b) >= 0
    neg_mask = perceptron_sign(X_neg, w, b) < 0

    ax.scatter(X_pos[pos_mask, 0], X_pos[pos_mask, 1], marker='s', label='σ(z) ≥ 0 (pred. Clase 1)')
    ax.scatter(X_neg[neg_mask, 0], X_neg[neg_mask, 1], marker='o', label='σ(z) < 0 (pred. Clase 0)')

    # Frontera de decisión
    x1_vals = np.linspace(-0.5, 5.5, 200)
    x2_vals = decision_boundary_2d_x2(x1_vals, w, b)
    ax.plot(x1_vals, x2_vals, '--', lw=2, label='Frontera: wᵀx + b = 0')

    # Flecha de w (normal a la recta), anclada en el intercepto con eje x2 cuando x1=0
    x1_anchor = 0.0
    x2_anchor = decision_boundary_2d_x2(x1_anchor, w, b)
    anchor = np.array([x1_anchor, x2_anchor])
    arrow_len = 1.8
    ax.arrow(anchor[0], anchor[1], w_unit[0]*arrow_len, w_unit[1]*arrow_len,
             head_width=0.12, length_includes_head=True, color='k')
    ax.text(anchor[0] + w_unit[0]*arrow_len*1.05,
            anchor[1] + w_unit[1]*arrow_len*1.05, "w (normal)", fontsize=10)

    # Punto x y su proyección sobre w
    ax.plot([0, x_sample[0]], [0, x_sample[1]], lw=1.8, color='C2')
    ax.scatter([x_sample[0]], [x_sample[1]], s=50, color='C2')
    ax.text(x_sample[0]+0.05, x_sample[1]+0.05, "x (punto)", fontsize=10)
    ax.plot([0, x_proj[0]], [0, x_proj[1]], lw=1.8, color='C3')
    ax.scatter([x_proj[0]], [x_proj[1]], s=45, color='C3')
    ax.text(x_proj[0]+0.05, x_proj[1]+0.05, "proy_w(x)", fontsize=10)
    ax.plot([x_sample[0], x_proj[0]], [x_sample[1], x_proj[1]], ':', lw=1.5, color='C3')

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Perceptrón 2D: frontera, vector normal w y proyección")
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 5.5)
    ax.grid(True, linestyle=':')
    ax.legend(loc='upper left') 
    plt.tight_layout()

# ---------- 2) Plot 3D (tres features) ----------
def plot_perceptron_3d(w=np.array([1.0, -0.6, 0.8]), b=-0.2, seed=0):
    """
    Caso con 3 características (x1, x2, x3). La frontera de decisión es un PLANO:
    w1*x1 + w2*x2 + w3*x3 + b = 0  =>  x3 = -(w1*x1 + w2*x2 + b) / w3
    """
    if len(w) != 3:
        raise ValueError("w debe tener 3 componentes para el plot 3D.")

    np.random.seed(seed)
    X = np.random.uniform(-1.0, 2.0, (100, 3))
    yhat = perceptron_sign(X, w, b)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[yhat>=0, 0], X[yhat>=0, 1], X[yhat>=0, 2], marker='s', label='σ(z) ≥ 0')
    ax.scatter(X[yhat<0, 0],  X[yhat<0, 1],  X[yhat<0, 2],  marker='o', label='σ(z) < 0')

    # Plano de decisión
    x1 = np.linspace(-1.2, 2.2, 25)
    x2 = np.linspace(-1.2, 2.2, 25)
    X1, X2 = np.meshgrid(x1, x2)
    X3 = -(w[0]*X1 + w[1]*X2 + b) / w[2]
    ax.plot_surface(X1, X2, X3, alpha=0.25, edgecolor='none')

    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
    ax.set_title("Perceptrón 3D: plano de decisión (solo si tienes 3 features)")
    ax.legend()
    plt.tight_layout()

# ------------------ Ejemplos de uso ------------------
plot_perceptron_2d()   # <- Descomenta para 2D
plot_perceptron_3d()   # <- Descomenta para 3D (solo con 3 features)
plt.show()
