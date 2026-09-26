"""
plotting.py — Funciones de dibujo sobre un Axes3D YA EXISTENTE.
Pensadas para reusarse dentro de una GUI embebida (Tkinter), no para
crear su propia ventana como hacía visualizer.py.
"""
import numpy as np

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def plane_z(x_grid, y_grid, w1, w2, w3, bias):
    w3_safe = w3 if abs(w3) > 1e-6 else 1e-6
    return -(w1 * x_grid + w2 * y_grid + bias) / w3_safe


def make_grid(X, margin=1.5, n=15):
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    return gx, gy


def _draw_points(ax, X, y, n_classes):
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                      c=COLORS[c % len(COLORS)], label=f"Clase {c}",
                      s=35, edgecolor="k", linewidth=0.3)


def redraw_empty(ax, mensaje="Sin datos todavía.\nGenera datos o agrega puntos manuales."):
    ax.clear()
    ax.text2D(0.5, 0.5, mensaje, transform=ax.transAxes,
              ha="center", va="center", fontsize=11, color="gray")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def redraw_points_only(ax, X, y, n_classes, title="Datos cargados (sin entrenar)"):
    ax.clear()
    _draw_points(ax, X, y, n_classes)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(title)
    ax.legend()


def redraw_static(ax, X, y, model, n_classes, title):
    """Puntos + plano(s) final(es) entrenado(s)."""
    ax.clear()
    _draw_points(ax, X, y, n_classes)

    gx, gy = make_grid(X)
    if n_classes == 2:
        w1, w2, w3 = model.weights
        gz = plane_z(gx, gy, w1, w2, w3, model.bias)
        ax.plot_surface(gx, gy, gz, alpha=0.35, color="gray")
    else:
        for c in model.classes:
            p = model.perceptrons[c]
            w1, w2, w3 = p.weights
            gz = plane_z(gx, gy, w1, w2, w3, p.bias)
            ax.plot_surface(gx, gy, gz, alpha=0.25,
                            color=COLORS[int(c) % len(COLORS)])

    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(title)
    ax.legend()


def get_animation_histories(model, n_classes):
    """Devuelve [(weight_history, bias_history, color), ...] y n_frames."""
    if n_classes == 2:
        histories = [(model.weight_history, model.bias_history, "gray")]
        n_frames = len(model.weight_history)
    else:
        histories = [(model.perceptrons[c].weight_history,
                      model.perceptrons[c].bias_history,
                      COLORS[int(c) % len(COLORS)]) for c in model.classes]
        n_frames = max(len(h[0]) for h in histories)
    return histories, n_frames


def redraw_frame(ax, X, y, n_classes, histories, frame, n_frames, title):
    """Un cuadro de la animación de entrenamiento (plano en la época `frame`)."""
    ax.clear()
    _draw_points(ax, X, y, n_classes)
    gx, gy = make_grid(X)
    for wh, bh, color in histories:
        idx = min(frame, len(wh) - 1)
        w1, w2, w3 = wh[idx]
        b = bh[idx]
        gz = plane_z(gx, gy, w1, w2, w3, b)
        ax.plot_surface(gx, gy, gz, alpha=0.30, color=color)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(f"{title} — Época {frame + 1}/{n_frames}")
    ax.legend()


# ----------------------------------------------------------------------
# Curvas de entrenamiento: error por época y evolución de los pesos.
# Todo calculado a partir de error_history / weight_history / bias_history
# del propio Perceptron — sin ninguna librería de redes neuronales.
# ----------------------------------------------------------------------
def redraw_curves_empty(fig, mensaje="Entrena el modelo para ver\nel error y los pesos por época."):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, mensaje, transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color="gray")
    ax.set_xticks([]); ax.set_yticks([])


def _plot_weights_de_un_perceptron(ax, p, titulo):
    epochs = range(1, len(p.weight_history) + 1)
    ax.plot(epochs, [w[0] for w in p.weight_history], label="w1", marker=".")
    ax.plot(epochs, [w[1] for w in p.weight_history], label="w2", marker=".")
    ax.plot(epochs, [w[2] for w in p.weight_history], label="w3", marker=".")
    ax.plot(epochs, p.bias_history, label="bias", linestyle="--", color="black")
    ax.set_xlabel("Época")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def redraw_curves(fig, model, n_classes):
    """
    Dibuja, en la misma figura:
      - fila 1: error (número de puntos mal clasificados) por época.
      - fila 2: evolución de w1, w2, w3 y bias por época
                (un subplot por perceptrón si hay más de 2 clases).
    """
    fig.clear()

    if n_classes == 2:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3])
        ax_err = fig.add_subplot(gs[0])
        ax_err.plot(range(1, len(model.error_history) + 1), model.error_history,
                   marker="o", color="gray")
        ax_err.set_title("Error por época")
        ax_err.set_xlabel("Época"); ax_err.set_ylabel("Puntos mal clasificados")
        ax_err.grid(alpha=0.3)

        ax_w = fig.add_subplot(gs[1])
        _plot_weights_de_un_perceptron(ax_w, model, "Evolución de pesos")
        ax_w.set_ylabel("Valor")

    else:
        classes = list(model.classes)
        n_p = len(classes)
        gs = fig.add_gridspec(2, n_p, height_ratios=[1, 1.3])

        ax_err = fig.add_subplot(gs[0, :])
        for c in classes:
            p = model.perceptrons[c]
            ax_err.plot(range(1, len(p.error_history) + 1), p.error_history,
                       marker="o", label=f"Clase {c}",
                       color=COLORS[int(c) % len(COLORS)])
        ax_err.set_title("Error por época (uno vs. resto, por clase)")
        ax_err.set_xlabel("Época"); ax_err.set_ylabel("Puntos mal clasificados")
        ax_err.legend(fontsize=8)
        ax_err.grid(alpha=0.3)

        for i, c in enumerate(classes):
            axw = fig.add_subplot(gs[1, i])
            _plot_weights_de_un_perceptron(axw, model.perceptrons[c], f"Pesos — Clase {c}")
            if i == 0:
                axw.set_ylabel("Valor")

    fig.tight_layout()
