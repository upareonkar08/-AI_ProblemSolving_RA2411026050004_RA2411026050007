"""
Student Exam Score Predictor
=============================
A Tkinter GUI application that trains a Linear Regression model to predict
student exam scores, with CSV upload, manual entry, and synthetic data support.

Dependencies:
    pip install scikit-learn pandas numpy

Features:
    ① Load a CSV file  OR  generate a synthetic dataset
    ② Add / edit individual student rows manually
    ③ Train the model and review metrics (R², MAE, RMSE)
    ④ Predict exam scores for new students
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import math
import random
import csv
import io

# ── ML / Data ─────────────────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Design Tokens
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#0d1117"
SURFACE   = "#161b22"
CARD      = "#1e2430"
BORDER    = "#30363d"
INPUT_BG  = "#0d1117"

BLUE      = "#58a6ff"
GREEN     = "#3fb950"
AMBER     = "#d29922"
RED       = "#f85149"
PURPLE    = "#bc8cff"
TEAL      = "#39d353"

TEXT      = "#e6edf3"
TEXT_DIM  = "#7d8590"
TEXT_MUTE = "#484f58"

FONT_H1   = ("Georgia", 20, "bold")
FONT_H2   = ("Georgia", 13, "bold")
FONT_H3   = ("Georgia", 11, "bold")
FONT_MONO = ("Courier New", 10)
FONT_MONO_SM = ("Courier New", 9)
FONT_BTN  = ("Georgia", 10, "bold")
FONT_BODY = ("Courier New", 10)

FEATURES  = ["Hours Studied", "Attendance %", "Previous Score", "Assignments Done"]
TARGET    = "Exam Score"
COL_NAMES = FEATURES + [TARGET]

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(n: int = 120) -> pd.DataFrame:
    """
    Generate a realistic synthetic student dataset.
    Exam score is a weighted combination of the 4 features plus noise.
    """
    rng = np.random.default_rng(42)
    hours      = rng.uniform(1, 12, n)
    attendance = rng.uniform(40, 100, n)
    prev_score = rng.uniform(30, 95, n)
    assignments = rng.integers(0, 11, n).astype(float)

    # Weighted formula + Gaussian noise
    score = (
        4.0  * hours
        + 0.25 * attendance
        + 0.45 * prev_score
        + 2.5  * assignments
        + rng.normal(0, 4, n)
    )
    score = np.clip(score, 0, 100).round(1)

    return pd.DataFrame({
        "Hours Studied":     hours.round(1),
        "Attendance %":      attendance.round(1),
        "Previous Score":    prev_score.round(1),
        "Assignments Done":  assignments,
        "Exam Score":        score,
    })


# ─────────────────────────────────────────────────────────────────────────────
# ML Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class StudentModel:
    """Encapsulates data processing, training, and prediction."""

    def __init__(self):
        self.df       = None
        self.model    = None
        self.scaler   = StandardScaler()
        self.metrics  = {}
        self.trained  = False

    # ── Data Loading ──────────────────────────────────────────────────────────
    def load_csv(self, path: str) -> tuple[bool, str]:
        """Load and validate a CSV file."""
        try:
            df = pd.read_csv(path)
            return self._validate_and_set(df)
        except Exception as e:
            return False, f"Failed to read CSV: {e}"

    def load_synthetic(self, n: int = 120) -> tuple[bool, str]:
        """Generate and load synthetic data."""
        self.df = generate_synthetic_data(n)
        return True, f"Synthetic dataset with {n} rows generated."

    def load_from_rows(self, rows: list[list]) -> tuple[bool, str]:
        """Load data from manually entered rows."""
        try:
            df = pd.DataFrame(rows, columns=COL_NAMES)
            df = df.apply(pd.to_numeric, errors="coerce")
            return self._validate_and_set(df)
        except Exception as e:
            return False, str(e)

    def _validate_and_set(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Validate columns, handle missing values, store df."""
        missing_cols = [c for c in COL_NAMES if c not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {', '.join(missing_cols)}"

        before = len(df)
        df = df[COL_NAMES].copy()
        df = df.apply(pd.to_numeric, errors="coerce")

        # Fill numeric NaNs with column median
        df.fillna(df.median(numeric_only=True), inplace=True)
        # Drop any rows still containing NaN
        df.dropna(inplace=True)
        after = len(df)

        if after < 10:
            return False, f"Only {after} valid rows — need at least 10."

        self.df = df
        note = f" ({before - after} rows removed)" if before != after else ""
        return True, f"Loaded {after} rows{note}."

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, test_size: float = 0.2) -> tuple[bool, str]:
        """Split data, scale, fit Linear Regression, compute metrics."""
        if self.df is None:
            return False, "No data loaded."
        if len(self.df) < 10:
            return False, "Need at least 10 rows to train."

        X = self.df[FEATURES].values
        y = self.df[TARGET].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        # Scale features for numerical stability
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model = LinearRegression()
        self.model.fit(X_train_s, y_train)

        y_pred = self.model.predict(X_test_s)
        y_pred = np.clip(y_pred, 0, 100)

        self.metrics = {
            "r2":   r2_score(y_test, y_pred),
            "mae":  mean_absolute_error(y_test, y_pred),
            "rmse": math.sqrt(mean_squared_error(y_test, y_pred)),
            "n_train": len(X_train),
            "n_test":  len(X_test),
            "coefs": dict(zip(FEATURES, self.model.coef_)),
        }
        self.trained = True
        return True, "Model trained successfully."

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict_one(self, values: list[float]) -> tuple[bool, float | str]:
        """Predict exam score for a single student."""
        if not self.trained:
            return False, "Model is not trained yet."
        try:
            x = np.array(values, dtype=float).reshape(1, -1)
            x_s = self.scaler.transform(x)
            pred = float(self.model.predict(x_s)[0])
            return True, round(min(max(pred, 0), 100), 2)
        except Exception as e:
            return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# GUI Application
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Score Predictor  ·  ML Dashboard")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(960, 680)

        self.model = StudentModel()
        self._manual_rows: list[list] = []   # rows entered manually

        if not ML_AVAILABLE:
            messagebox.showerror(
                "Missing dependencies",
                "Please install: pip install scikit-learn pandas numpy"
            )
            self.destroy()
            return

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=SURFACE, pady=0)
        hdr.pack(fill="x")
        tk.Frame(hdr, bg=BLUE, height=3).pack(fill="x")
        inner_hdr = tk.Frame(hdr, bg=SURFACE)
        inner_hdr.pack(fill="x", padx=28, pady=14)
        tk.Label(inner_hdr, text="🎓  Student Score Predictor",
                 font=FONT_H1, bg=SURFACE, fg=TEXT).pack(side="left")
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(inner_hdr, textvariable=self.status_var,
                 font=FONT_MONO_SM, bg=SURFACE, fg=TEXT_DIM).pack(
            side="right", pady=4)

        # ── Notebook tabs ─────────────────────────────────────────────────────
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("Dark.TNotebook",
                        background=BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure("Dark.TNotebook.Tab",
                        background=SURFACE, foreground=TEXT_DIM,
                        font=FONT_BTN, padding=[18, 8])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", CARD)],
                  foreground=[("selected", BLUE)])

        nb = ttk.Notebook(self, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self._tab_data(nb)
        self._tab_train(nb)
        self._tab_predict(nb)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1 — Data
    # ─────────────────────────────────────────────────────────────────────────
    def _tab_data(self, nb):
        page = tk.Frame(nb, bg=BG)
        nb.add(page, text="  ① Dataset  ")

        # Two-column layout
        page.columnconfigure(0, weight=1)
        page.columnconfigure(1, weight=2)
        page.rowconfigure(0, weight=1)

        # ── Left: load options ────────────────────────────────────────────────
        left = self._card(page, "Load Data", col=0, padx=(20, 8), pady=20)

        # CSV upload
        self._section_label(left, "① CSV File")
        csv_row = tk.Frame(left, bg=CARD)
        csv_row.pack(fill="x", padx=14, pady=(4, 12))
        self.csv_path_var = tk.StringVar(value="No file selected")
        tk.Label(csv_row, textvariable=self.csv_path_var,
                 font=FONT_MONO_SM, bg=CARD, fg=TEXT_DIM,
                 anchor="w", width=28).pack(side="left", fill="x", expand=True)
        self._btn(csv_row, "Browse", self._load_csv, color=BLUE, side="left")

        # Synthetic data
        self._section_label(left, "② Synthetic Dataset")
        syn_row = tk.Frame(left, bg=CARD)
        syn_row.pack(fill="x", padx=14, pady=(4, 12))
        tk.Label(syn_row, text="Rows:", font=FONT_BODY, bg=CARD, fg=TEXT_DIM
                 ).pack(side="left")
        self.syn_n_var = tk.StringVar(value="120")
        tk.Entry(syn_row, textvariable=self.syn_n_var, width=6,
                 font=FONT_MONO, bg=INPUT_BG, fg=TEXT, insertbackground=BLUE,
                 relief="flat", highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=BLUE
                 ).pack(side="left", padx=8, ipady=4)
        self._btn(syn_row, "Generate", self._load_synthetic, color=GREEN, side="left")

        # Manual entry
        self._section_label(left, "③ Manual Entry")
        manual_frame = tk.Frame(left, bg=CARD)
        manual_frame.pack(fill="x", padx=14, pady=(4, 0))

        self.manual_entries = {}
        for feat in COL_NAMES:
            row = tk.Frame(manual_frame, bg=CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=feat, font=FONT_BODY, bg=CARD, fg=TEXT_DIM,
                     width=18, anchor="w").pack(side="left")
            v = tk.StringVar()
            tk.Entry(row, textvariable=v, width=8,
                     font=FONT_MONO, bg=INPUT_BG, fg=TEXT,
                     insertbackground=BLUE, relief="flat",
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=BLUE).pack(side="left", ipady=4)
            self.manual_entries[feat] = v

        btn_row = tk.Frame(left, bg=CARD)
        btn_row.pack(fill="x", padx=14, pady=10)
        self._btn(btn_row, "+ Add Row", self._add_manual_row,
                  color=AMBER, side="left")
        self._btn(btn_row, "Use Manual Data", self._use_manual_data,
                  color=PURPLE, side="left")

        # Manual row count
        self.manual_count_var = tk.StringVar(value="0 rows entered")
        tk.Label(left, textvariable=self.manual_count_var,
                 font=FONT_MONO_SM, bg=CARD, fg=TEXT_DIM).pack(pady=(0, 12))

        # ── Right: data preview ───────────────────────────────────────────────
        right = self._card(page, "Data Preview", col=1, padx=(8, 20), pady=20)
        self._build_preview_table(right)

    def _build_preview_table(self, parent):
        """Scrollable preview of loaded data."""
        style = ttk.Style()
        style.configure("Dark.Treeview",
                        background=INPUT_BG, foreground=TEXT,
                        fieldbackground=INPUT_BG, rowheight=22,
                        font=FONT_MONO_SM)
        style.configure("Dark.Treeview.Heading",
                        background=SURFACE, foreground=BLUE,
                        font=("Courier New", 9, "bold"), relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", BLUE)],
                  foreground=[("selected", BG)])

        frame = tk.Frame(parent, bg=CARD)
        frame.pack(fill="both", expand=True, padx=14, pady=8)

        self.tree = ttk.Treeview(frame, columns=COL_NAMES, show="headings",
                                 style="Dark.Treeview", height=18)
        for col in COL_NAMES:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=110, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)

        vsb = tk.Scrollbar(frame, orient="vertical",
                           command=self.tree.yview, bg=CARD)
        vsb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=vsb.set)

        self.data_info_var = tk.StringVar(value="No data loaded")
        tk.Label(parent, textvariable=self.data_info_var,
                 font=FONT_MONO_SM, bg=CARD, fg=TEXT_DIM).pack(pady=(0, 10))

    def _refresh_preview(self):
        """Populate the treeview from self.model.df."""
        self.tree.delete(*self.tree.get_children())
        if self.model.df is None:
            return
        df = self.model.df
        for _, row in df.head(200).iterrows():
            self.tree.insert("", "end", values=[
                f"{row[c]:.1f}" for c in COL_NAMES])
        n = len(df)
        shown = min(n, 200)
        self.data_info_var.set(
            f"{n} rows loaded  (showing first {shown})")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2 — Train
    # ─────────────────────────────────────────────────────────────────────────
    def _tab_train(self, nb):
        page = tk.Frame(nb, bg=BG)
        nb.add(page, text="  ② Train Model  ")

        page.columnconfigure(0, weight=1)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(0, weight=1)

        # ── Left: controls ────────────────────────────────────────────────────
        left = self._card(page, "Training Controls", col=0,
                          padx=(20, 8), pady=20)

        self._section_label(left, "Algorithm")
        tk.Label(left, text="Linear Regression  (scikit-learn)",
                 font=FONT_BODY, bg=CARD, fg=TEXT).pack(
            anchor="w", padx=14, pady=(4, 12))

        self._section_label(left, "Test Split")
        split_row = tk.Frame(left, bg=CARD)
        split_row.pack(fill="x", padx=14, pady=(4, 12))
        self.split_var = tk.StringVar(value="20")
        tk.Entry(split_row, textvariable=self.split_var, width=5,
                 font=FONT_MONO, bg=INPUT_BG, fg=TEXT,
                 insertbackground=BLUE, relief="flat",
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=BLUE).pack(side="left", ipady=4)
        tk.Label(split_row, text="% of data as test set",
                 font=FONT_BODY, bg=CARD, fg=TEXT_DIM).pack(
            side="left", padx=8)

        self._section_label(left, "Features Used")
        for f in FEATURES:
            tk.Label(left, text=f"  ✓  {f}", font=FONT_MONO,
                     bg=CARD, fg=GREEN).pack(anchor="w", padx=14, pady=1)
        tk.Label(left, text=f"  →  Target: {TARGET}", font=FONT_MONO,
                 bg=CARD, fg=BLUE).pack(anchor="w", padx=14, pady=(4, 12))

        self._btn(left, "🚀  Train Model", self._train_model,
                  color=GREEN, fill="x", padx=14, pady=(8, 4))

        self.train_msg_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.train_msg_var,
                 font=FONT_MONO_SM, bg=CARD, fg=TEXT_DIM,
                 wraplength=220).pack(padx=14, pady=4)

        # ── Right: metrics ────────────────────────────────────────────────────
        right = self._card(page, "Model Metrics", col=1,
                           padx=(8, 20), pady=20)
        self._build_metrics_panel(right)

    def _build_metrics_panel(self, parent):
        """Metric cards + coefficient table."""
        metrics_row = tk.Frame(parent, bg=CARD)
        metrics_row.pack(fill="x", padx=14, pady=8)

        self.metric_vars = {}
        defs = [
            ("R² Score", "r2", BLUE, "Closer to 1.0 is better"),
            ("MAE", "mae", AMBER, "Mean Absolute Error (pts)"),
            ("RMSE", "rmse", RED, "Root Mean Squared Error"),
        ]
        for label, key, color, tip in defs:
            box = tk.Frame(metrics_row, bg=INPUT_BG, bd=1,
                           highlightthickness=1, highlightbackground=BORDER)
            box.pack(side="left", fill="both", expand=True, padx=4, pady=4)
            v = tk.StringVar(value="—")
            self.metric_vars[key] = v
            tk.Label(box, textvariable=v, font=("Georgia", 18, "bold"),
                     bg=INPUT_BG, fg=color, pady=6).pack()
            tk.Label(box, text=label, font=FONT_BTN,
                     bg=INPUT_BG, fg=TEXT).pack()
            tk.Label(box, text=tip, font=FONT_MONO_SM,
                     bg=INPUT_BG, fg=TEXT_MUTE, pady=4).pack()

        # Train/test split info
        split_row = tk.Frame(parent, bg=CARD)
        split_row.pack(fill="x", padx=14, pady=4)
        self.split_info_var = tk.StringVar(value="")
        tk.Label(split_row, textvariable=self.split_info_var,
                 font=FONT_MONO_SM, bg=CARD, fg=TEXT_DIM).pack(anchor="w")

        # Coefficients
        self._section_label(parent, "Feature Coefficients (scaled)")
        coef_frame = tk.Frame(parent, bg=INPUT_BG,
                              highlightthickness=1, highlightbackground=BORDER)
        coef_frame.pack(fill="x", padx=14, pady=(4, 12))
        self.coef_labels = {}
        for f in FEATURES:
            row = tk.Frame(coef_frame, bg=INPUT_BG)
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=f, font=FONT_BODY, bg=INPUT_BG,
                     fg=TEXT_DIM, width=20, anchor="w").pack(side="left")
            v = tk.StringVar(value="—")
            tk.Label(row, textvariable=v, font=FONT_MONO,
                     bg=INPUT_BG, fg=TEAL, anchor="e").pack(side="right")
            self.coef_labels[f] = v

    def _refresh_metrics(self):
        m = self.model.metrics
        self.metric_vars["r2"].set(f"{m['r2']:.4f}")
        self.metric_vars["mae"].set(f"{m['mae']:.2f}")
        self.metric_vars["rmse"].set(f"{m['rmse']:.2f}")
        self.split_info_var.set(
            f"Training rows: {m['n_train']}   Test rows: {m['n_test']}")
        for f, v in m["coefs"].items():
            self.coef_labels[f].set(f"{v:+.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3 — Predict
    # ─────────────────────────────────────────────────────────────────────────
    def _tab_predict(self, nb):
        page = tk.Frame(nb, bg=BG)
        nb.add(page, text="  ③ Predict  ")

        page.columnconfigure(0, weight=1)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(0, weight=1)

        # ── Left: input form ──────────────────────────────────────────────────
        left = self._card(page, "Student Details", col=0,
                          padx=(20, 8), pady=20)

        self._section_label(left, "Enter Feature Values")
        self.predict_entries = {}
        hints = {
            "Hours Studied":    "(e.g. 6.5)",
            "Attendance %":     "(e.g. 85.0)",
            "Previous Score":   "(e.g. 72.0)",
            "Assignments Done": "(0 – 10)",
        }
        for feat in FEATURES:
            row = tk.Frame(left, bg=CARD)
            row.pack(fill="x", padx=14, pady=4)
            tk.Label(row, text=feat, font=FONT_BODY, bg=CARD,
                     fg=TEXT, width=18, anchor="w").pack(side="left")
            v = tk.StringVar()
            tk.Entry(row, textvariable=v, width=10,
                     font=FONT_MONO, bg=INPUT_BG, fg=TEXT,
                     insertbackground=BLUE, relief="flat",
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=BLUE).pack(side="left", ipady=5)
            tk.Label(row, text=hints[feat], font=FONT_MONO_SM,
                     bg=CARD, fg=TEXT_MUTE).pack(side="left", padx=8)
            self.predict_entries[feat] = v

        self._btn(left, "⚡  Predict Score", self._predict,
                  color=BLUE, fill="x", padx=14, pady=(16, 4))
        self._btn(left, "  Clear Fields", self._clear_predict,
                  color=SURFACE, fill="x", padx=14, pady=(0, 4))

        # Ranges reminder
        self._section_label(left, "Feature Ranges")
        ranges = [
            ("Hours Studied",    "1 – 12 hrs"),
            ("Attendance %",     "40 – 100 %"),
            ("Previous Score",   "30 – 95 pts"),
            ("Assignments Done", "0 – 10"),
        ]
        for name, rng in ranges:
            r = tk.Frame(left, bg=CARD)
            r.pack(fill="x", padx=14, pady=1)
            tk.Label(r, text=name, font=FONT_MONO_SM,
                     bg=CARD, fg=TEXT_DIM, width=20, anchor="w").pack(side="left")
            tk.Label(r, text=rng, font=FONT_MONO_SM,
                     bg=CARD, fg=TEXT_MUTE).pack(side="left")

        # ── Right: result ─────────────────────────────────────────────────────
        right = self._card(page, "Prediction Result", col=1,
                           padx=(8, 20), pady=20)
        self._build_result_panel(right)

    def _build_result_panel(self, parent):
        # Big score display
        score_frame = tk.Frame(parent, bg=INPUT_BG,
                               highlightthickness=1, highlightbackground=BORDER)
        score_frame.pack(fill="x", padx=14, pady=(8, 4))

        tk.Label(score_frame, text="Predicted Exam Score",
                 font=FONT_H3, bg=INPUT_BG, fg=TEXT_DIM, pady=8).pack()
        self.score_var = tk.StringVar(value="—")
        tk.Label(score_frame, textvariable=self.score_var,
                 font=("Georgia", 48, "bold"),
                 bg=INPUT_BG, fg=BLUE).pack(pady=4)
        self.grade_var = tk.StringVar(value="")
        tk.Label(score_frame, textvariable=self.grade_var,
                 font=("Georgia", 14, "bold"),
                 bg=INPUT_BG, fg=GREEN, pady=8).pack()

        # Interpretation bar
        self._section_label(parent, "Score Interpretation")
        bands = [
            ("90 – 100", "Excellent", TEAL),
            ("75 – 89",  "Good",      GREEN),
            ("60 – 74",  "Average",   AMBER),
            ("40 – 59",  "Below Avg", RED),
            (" 0 – 39",  "At Risk",   "#f85149"),
        ]
        for rng, label, color in bands:
            brow = tk.Frame(parent, bg=CARD)
            brow.pack(fill="x", padx=14, pady=2)
            tk.Frame(brow, bg=color, width=10).pack(side="left", fill="y")
            tk.Label(brow, text=f"  {rng}  →  {label}",
                     font=FONT_BODY, bg=CARD, fg=TEXT).pack(
                side="left", padx=6)

        # History log
        self._section_label(parent, "Prediction History")
        log_frame = tk.Frame(parent, bg=INPUT_BG,
                             highlightthickness=1, highlightbackground=BORDER)
        log_frame.pack(fill="both", expand=True, padx=14, pady=(4, 14))

        self.log_text = tk.Text(
            log_frame, font=FONT_MONO_SM,
            bg=INPUT_BG, fg=TEXT_DIM, state="disabled",
            relief="flat", wrap="none", padx=8, pady=6)
        self.log_text.pack(fill="both", expand=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────────────
    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV Files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        ok, msg = self.model.load_csv(path)
        self.csv_path_var.set(path.split("/")[-1])
        self._set_status(msg, ok)
        if ok:
            self._refresh_preview()

    def _load_synthetic(self):
        try:
            n = int(self.syn_n_var.get())
            if n < 10:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Rows must be an integer ≥ 10.")
            return
        ok, msg = self.model.load_synthetic(n)
        self._set_status(msg, ok)
        if ok:
            self._refresh_preview()

    def _add_manual_row(self):
        """Validate and store one manually entered row."""
        vals = []
        for col in COL_NAMES:
            raw = self.manual_entries[col].get().strip()
            try:
                vals.append(float(raw))
            except ValueError:
                messagebox.showerror(
                    "Input Error",
                    f'Invalid value for "{col}": "{raw}"\nPlease enter a number.')
                return
        self._manual_rows.append(vals)
        self.manual_count_var.set(f"{len(self._manual_rows)} rows entered")
        for v in self.manual_entries.values():
            v.set("")

    def _use_manual_data(self):
        if len(self._manual_rows) < 5:
            messagebox.showwarning(
                "Not Enough Data",
                f"Enter at least 5 rows (currently {len(self._manual_rows)}).")
            return
        ok, msg = self.model.load_from_rows(self._manual_rows)
        self._set_status(msg, ok)
        if ok:
            self._refresh_preview()

    def _train_model(self):
        if self.model.df is None:
            messagebox.showwarning("No Data", "Load or generate a dataset first.")
            return
        try:
            split = int(self.split_var.get())
            if not (5 <= split <= 50):
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Test split must be 5 – 50.")
            return

        self.train_msg_var.set("Training…")
        self.update_idletasks()

        ok, msg = self.model.train(test_size=split / 100)
        self.train_msg_var.set(msg)
        self._set_status(msg, ok)
        if ok:
            self._refresh_metrics()

    def _predict(self):
        if not self.model.trained:
            messagebox.showwarning("Not Trained", "Train the model first (Tab ②).")
            return
        vals = []
        for feat in FEATURES:
            raw = self.predict_entries[feat].get().strip()
            try:
                vals.append(float(raw))
            except ValueError:
                messagebox.showerror(
                    "Input Error",
                    f'Invalid value for "{feat}": "{raw}"')
                return

        ok, result = self.model.predict_one(vals)
        if not ok:
            messagebox.showerror("Prediction Error", str(result))
            return

        score = result
        self.score_var.set(f"{score:.1f}")

        # Grade label
        if score >= 90:   grade = "🏆  Excellent"
        elif score >= 75: grade = "✅  Good"
        elif score >= 60: grade = "📘  Average"
        elif score >= 40: grade = "⚠️  Below Average"
        else:             grade = "🔴  At Risk"
        self.grade_var.set(grade)

        # Append to log
        log_line = (
            f"Score={score:.1f}  |  "
            + "  ".join(f"{f[:4]}={v}" for f, v in zip(FEATURES, vals))
            + "\n"
        )
        self.log_text.config(state="normal")
        self.log_text.insert("end", log_line)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_predict(self):
        for v in self.predict_entries.values():
            v.set("")
        self.score_var.set("—")
        self.grade_var.set("")

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Widgets
    # ─────────────────────────────────────────────────────────────────────────
    def _card(self, parent, title, col, padx=(0, 0), pady=0):
        outer = tk.Frame(parent, bg=BORDER)
        outer.grid(row=0, column=col, sticky="nsew", padx=padx, pady=pady)
        inner = tk.Frame(outer, bg=CARD)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        hdr = tk.Frame(inner, bg=SURFACE)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, font=FONT_H2,
                 bg=SURFACE, fg=TEXT, pady=10, padx=14,
                 anchor="w").pack(fill="x")
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x")
        return inner

    def _section_label(self, parent, text):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=14, pady=(10, 0))
        tk.Label(row, text=text.upper(), font=("Courier New", 8, "bold"),
                 bg=CARD, fg=TEXT_MUTE).pack(side="left")
        tk.Frame(row, bg=BORDER, height=1).pack(
            side="left", fill="x", expand=True, padx=(8, 0), pady=6)

    def _btn(self, parent, text, command, color=BLUE,
             side=None, fill=None, padx=0, pady=0):
        b = tk.Button(parent, text=text, command=command,
                      font=FONT_BTN, bg=color, fg=BG,
                      relief="flat", activebackground=TEXT,
                      activeforeground=BG, cursor="hand2",
                      pady=7, padx=12, bd=0)
        if side:
            b.pack(side=side, padx=padx, pady=pady)
        else:
            b.pack(fill=fill, padx=padx, pady=pady)
        return b

    def _set_status(self, msg, ok=True):
        self.status_var.set(("✓  " if ok else "✗  ") + msg)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()

