"""
app.py — Simulador de Perceptrón: TODO en una sola ventana
(controles a la izquierda, gráfica 3D en vivo a la derecha).

Ejecutar:
    python app.py
"""
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from data_generator import generate_data
from perceptron import Perceptron, OneVsRestPerceptron
import plotting as plot


class PerceptronApp:
    def __init__(self, root):
        self.root = root
        root.title("Simulador de Perceptrón 3D")
        root.geometry("1200x720")

        # ---------------- Estado ----------------
        self.X = None
        self.y = None
        self.model = None
        self.n_classes = 2
        self.mode = None
        self.last_acc = None
        self._anim_frame = 0
        self._anim_job = None

        # ---------------- Layout general ----------------
        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)

        panel = ttk.Frame(main, width=340)
        panel.pack(side="left", fill="y", padx=8, pady=8)

        plot_frame = ttk.Frame(main)
        plot_frame.pack(side="right", fill="both", expand=True)

        notebook = ttk.Notebook(plot_frame)
        notebook.pack(fill="both", expand=True)

        tab_3d = ttk.Frame(notebook)
        tab_curvas = ttk.Frame(notebook)
        notebook.add(tab_3d, text="Plano 3D")
        notebook.add(tab_curvas, text="Error y pesos por época")

        # ---------------- Pestaña 1: plano 3D ----------------
        self.fig = plt.Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        plot.redraw_empty(self.ax)

        self.canvas = FigureCanvasTkAgg(self.fig, master=tab_3d)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, tab_3d)
        toolbar.update()
        self.canvas.draw()

        # ---------------- Pestaña 2: curvas de entrenamiento ----------------
        self.fig_curvas = plt.Figure(figsize=(6, 6))
        plot.redraw_curves_empty(self.fig_curvas)

        self.canvas_curvas = FigureCanvasTkAgg(self.fig_curvas, master=tab_curvas)
        self.canvas_curvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar_curvas = NavigationToolbar2Tk(self.canvas_curvas, tab_curvas)
        toolbar_curvas.update()
        self.canvas_curvas.draw()

        # ---------------- Panel de control ----------------
        self._build_panel_generar(panel)
        self._build_panel_manual(panel)
        self._build_panel_entrenar(panel)
        self._build_panel_clasificar(panel)

        ttk.Button(panel, text="Limpiar todo", command=self.limpiar_todo).pack(
            fill="x", pady=(10, 4))

        self.status_var = tk.StringVar()
        ttk.Label(panel, textvariable=self.status_var, foreground="#444",
                 wraplength=320, justify="left").pack(fill="x", pady=(8, 0))
        self._actualizar_status()

    # ------------------------------------------------------------------
    # Bloque 1: generar datos automáticos
    # ------------------------------------------------------------------
    def _build_panel_generar(self, parent):
        box = ttk.LabelFrame(parent, text="1) Generar datos automáticos")
        box.pack(fill="x", pady=4)

        self.var_n_classes = tk.IntVar(value=2)
        fila = ttk.Frame(box); fila.pack(fill="x", pady=2)
        ttk.Label(fila, text="Clases:").pack(side="left")
        ttk.Radiobutton(fila, text="2", variable=self.var_n_classes, value=2).pack(side="left")
        ttk.Radiobutton(fila, text="3", variable=self.var_n_classes, value=3).pack(side="left")

        self.entry_n_samples = self._fila_entry(box, "Muestras por clase:", "50")
        self.entry_mean = self._fila_entry(box, "Media:", "0.0")
        self.entry_std = self._fila_entry(box, "Desviación estándar:", "1.0")

        self.var_separable = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Clases separables", variable=self.var_separable).pack(
            anchor="w", pady=2)

        ttk.Button(box, text="Generar datos", command=self.generar_datos).pack(
            fill="x", pady=4)

    # ------------------------------------------------------------------
    # Bloque 2: agregar dato manual
    # ------------------------------------------------------------------
    def _build_panel_manual(self, parent):
        box = ttk.LabelFrame(parent, text="2) Agregar dato manual")
        box.pack(fill="x", pady=4)

        self.entry_x1 = self._fila_entry(box, "x1:", "0.0")
        self.entry_x2 = self._fila_entry(box, "x2:", "0.0")
        self.entry_x3 = self._fila_entry(box, "x3:", "0.0")
        self.entry_clase = self._fila_entry(box, "Clase (0,1,2...):", "0")

        ttk.Button(box, text="Agregar punto", command=self.agregar_punto_manual).pack(
            fill="x", pady=4)

    # ------------------------------------------------------------------
    # Bloque 3: entrenar / animar
    # ------------------------------------------------------------------
    def _build_panel_entrenar(self, parent):
        box = ttk.LabelFrame(parent, text="3) Entrenamiento")
        box.pack(fill="x", pady=4)

        self.entry_lr = self._fila_entry(box, "Tasa de aprendizaje:", "0.1")
        self.entry_epochs = self._fila_entry(box, "Épocas máximas:", "100")

        fila = ttk.Frame(box); fila.pack(fill="x", pady=4)
        ttk.Button(fila, text="Entrenar", command=self.entrenar).pack(
            side="left", expand=True, fill="x", padx=2)
        ttk.Button(fila, text="Animar entrenamiento", command=self.animar_entrenamiento).pack(
            side="left", expand=True, fill="x", padx=2)

    # ------------------------------------------------------------------
    # Bloque 4: clasificar punto
    # ------------------------------------------------------------------
    def _build_panel_clasificar(self, parent):
        box = ttk.LabelFrame(parent, text="4) Clasificar un punto nuevo")
        box.pack(fill="x", pady=4)

        self.entry_cx1 = self._fila_entry(box, "x1:", "0.0")
        self.entry_cx2 = self._fila_entry(box, "x2:", "0.0")
        self.entry_cx3 = self._fila_entry(box, "x3:", "0.0")

        ttk.Button(box, text="Clasificar", command=self.clasificar_punto).pack(
            fill="x", pady=4)

        self.resultado_var = tk.StringVar(value="Resultado: -")
        ttk.Label(box, textvariable=self.resultado_var, foreground="#005500").pack(anchor="w")

    # ------------------------------------------------------------------
    # Helper de UI
    # ------------------------------------------------------------------
    def _fila_entry(self, parent, etiqueta, valor_default):
        fila = ttk.Frame(parent)
        fila.pack(fill="x", pady=2)
        ttk.Label(fila, text=etiqueta, width=18).pack(side="left")
        entry = ttk.Entry(fila)
        entry.insert(0, valor_default)
        entry.pack(side="left", fill="x", expand=True)
        return entry

    def _float(self, entry, default=0.0):
        try:
            return float(entry.get())
        except ValueError:
            return default

    def _int(self, entry, default=0):
        try:
            return int(float(entry.get()))
        except ValueError:
            return default

    def _actualizar_status(self):
        n = len(self.y) if self.y is not None else 0
        entrenado = "sí" if self.model is not None else "no"
        acc = f"{self.last_acc:.2f}%" if self.last_acc is not None else "-"
        self.status_var.set(
            f"Puntos: {n}   |   Clases: {self.n_classes}   |   Modo: {self.mode or '-'}\n"
            f"Entrenado: {entrenado}   |   Precisión: {acc}"
        )

    # ------------------------------------------------------------------
    # Acciones
    # ------------------------------------------------------------------
    def generar_datos(self):
        self._detener_animacion()
        n_classes = self.var_n_classes.get()
        n_samples = self._int(self.entry_n_samples, 50)
        mean = self._float(self.entry_mean, 0.0)
        std = self._float(self.entry_std, 1.0)
        separable = self.var_separable.get()

        X, y = generate_data(n_samples, mean, std, n_classes, separable)
        self.X, self.y = X, y
        self.n_classes = n_classes
        self.mode = "auto"
        self.model = None
        self.last_acc = None

        plot.redraw_points_only(self.ax, self.X, self.y, self.n_classes,
                               title=f"{len(y)} puntos generados ({n_classes} clases)")
        self.canvas.draw()
        plot.redraw_curves_empty(self.fig_curvas)
        self.canvas_curvas.draw()
        self._actualizar_status()

    def agregar_punto_manual(self):
        self._detener_animacion()
        try:
            x1 = self._float(self.entry_x1)
            x2 = self._float(self.entry_x2)
            x3 = self._float(self.entry_x3)
            clase = self._int(self.entry_clase)
        except ValueError:
            messagebox.showerror("Error", "Valores inválidos.")
            return

        if self.mode != "manual" and self.X is not None:
            if not messagebox.askyesno(
                "Confirmar",
                "Ya hay datos generados automáticamente.\n"
                "¿Reemplazarlos y empezar una colección manual nueva?"):
                return
            self.X, self.y = None, None
            self.model, self.last_acc = None, None

        self.mode = "manual"
        nuevo_punto = np.array([[x1, x2, x3]])
        nuevo_label = np.array([clase])
        if self.X is None:
            self.X, self.y = nuevo_punto, nuevo_label
        else:
            self.X = np.vstack([self.X, nuevo_punto])
            self.y = np.append(self.y, nuevo_label)

        self.n_classes = int(np.max(self.y)) + 1
        self.model = None
        self.last_acc = None

        plot.redraw_points_only(self.ax, self.X, self.y, self.n_classes,
                               title=f"{len(self.y)} puntos (manual)")
        self.canvas.draw()
        plot.redraw_curves_empty(self.fig_curvas)
        self.canvas_curvas.draw()
        self._actualizar_status()

    def entrenar(self):
        self._detener_animacion()
        if self.X is None or len(self.X) < 2:
            messagebox.showwarning("Aviso", "Necesitas al menos 2 puntos antes de entrenar.")
            return

        lr = self._float(self.entry_lr, 0.1)
        epochs = self._int(self.entry_epochs, 100)

        if self.n_classes == 2:
            y_bin = np.where(self.y == 0, -1, 1)
            model = Perceptron(lr, epochs)
            model.fit(self.X, y_bin)
            acc = float(np.mean(model.predict(self.X) == y_bin)) * 100
        else:
            model = OneVsRestPerceptron(lr, epochs)
            model.fit(self.X, self.y)
            acc = float(np.mean(model.predict(self.X) == self.y)) * 100

        self.model = model
        self.last_acc = acc

        titulo = f"Perceptrón — {self.n_classes} clases — Precisión {acc:.1f}%"
        plot.redraw_static(self.ax, self.X, self.y, self.model, self.n_classes, titulo)
        self.canvas.draw()

        plot.redraw_curves(self.fig_curvas, self.model, self.n_classes)
        self.canvas_curvas.draw()

        self._actualizar_status()

    def animar_entrenamiento(self):
        if self.model is None:
            messagebox.showwarning("Aviso", "Primero entrena el modelo.")
            return
        self._detener_animacion()
        self.histories, self.n_frames = plot.get_animation_histories(self.model, self.n_classes)
        self._anim_frame = 0
        self._titulo_anim = f"Perceptrón — {self.n_classes} clases — Precisión {self.last_acc:.1f}%"
        self._paso_animacion()

    def _paso_animacion(self):
        plot.redraw_frame(self.ax, self.X, self.y, self.n_classes,
                          self.histories, self._anim_frame, self.n_frames,
                          self._titulo_anim)
        self.canvas.draw()
        self._anim_frame += 1
        if self._anim_frame < self.n_frames:
            self._anim_job = self.root.after(250, self._paso_animacion)

    def _detener_animacion(self):
        if self._anim_job is not None:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None

    def clasificar_punto(self):
        if self.model is None:
            messagebox.showwarning("Aviso", "Primero entrena el modelo.")
            return
        x1 = self._float(self.entry_cx1)
        x2 = self._float(self.entry_cx2)
        x3 = self._float(self.entry_cx3)
        x = np.array([x1, x2, x3])
        cls = self.model.predict_single(x)
        if self.n_classes == 2:
            cls = 0 if cls == -1 else 1
        self.resultado_var.set(f"Resultado: Clase {cls}")

    def limpiar_todo(self):
        self._detener_animacion()
        self.X = self.y = self.model = self.last_acc = self.mode = None
        self.n_classes = 2
        self.resultado_var.set("Resultado: -")
        plot.redraw_empty(self.ax)
        self.canvas.draw()
        plot.redraw_curves_empty(self.fig_curvas)
        self.canvas_curvas.draw()
        self._actualizar_status()


def main():
    root = tk.Tk()
    PerceptronApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
