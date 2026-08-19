"""
TSP Tourist Travel Planner
==========================
A Tkinter GUI application that solves the Travelling Salesman Problem
using a brute-force (permutation-based) approach for tourist route planning.

Features:
  - Add/remove city destinations
  - Enter distances between each pair of cities
  - Compute optimal route with minimum total distance
  - Display results with path and distance
"""

import tkinter as tk
from tkinter import ttk, messagebox
from itertools import permutations
import math


# ─────────────────────────────────────────────────────────────────────────────
# Color Palette & Styling
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK     = "#0f1117"
BG_CARD     = "#1a1d27"
BG_INPUT    = "#242736"
ACCENT      = "#f5a623"
ACCENT2     = "#e8556d"
TEXT_MAIN   = "#e8eaf0"
TEXT_SUB    = "#8b90a4"
TEXT_ACCENT = "#f5a623"
BORDER      = "#2e3248"
SUCCESS     = "#4caf8c"
FONT_HEAD   = ("Georgia", 22, "bold")
FONT_SUB    = ("Georgia", 11, "italic")
FONT_LABEL  = ("Courier New", 10, "bold")
FONT_BODY   = ("Courier New", 10)
FONT_RESULT = ("Courier New", 12, "bold")
FONT_BTN    = ("Georgia", 11, "bold")


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Class
# ─────────────────────────────────────────────────────────────────────────────
class TSPTravelPlanner:
    def __init__(self, root):
        self.root = root
        self.root.title("✈  TSP Tourist Travel Planner")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)
        self.root.minsize(820, 680)

        # ── Data Structures ──────────────────────────────────────────────────
        self.cities = []                 # Ordered list of city names
        self.distances = {}              # dict: {(cityA, cityB): distance}
        self.dist_entries = {}           # dict: {(cityA, cityB): Entry widget}

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        """Build the full application layout."""
        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=BG_DARK, pady=20)
        header.pack(fill="x", padx=30)

        tk.Label(header, text="✈  Tourist Route Optimizer",
                 font=FONT_HEAD, bg=BG_DARK, fg=ACCENT).pack()
        tk.Label(header, text="Solve the Travelling Salesman Problem for your journey",
                 font=FONT_SUB, bg=BG_DARK, fg=TEXT_SUB).pack(pady=(2, 0))

        # Divider
        tk.Frame(self.root, bg=ACCENT, height=2).pack(fill="x", padx=30)

        # ── Three-column body ─────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=30, pady=20)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_city_panel(body)
        self._build_distance_panel(body)
        self._build_result_panel(body)

    # ── Section 1: City Panel ────────────────────────────────────────────────
    def _build_city_panel(self, parent):
        card = self._card(parent, "① Add Destinations", col=0)

        # Entry row
        entry_row = tk.Frame(card, bg=BG_CARD)
        entry_row.pack(fill="x", padx=16, pady=(4, 8))

        self.city_var = tk.StringVar()
        city_entry = tk.Entry(entry_row, textvariable=self.city_var,
                              font=FONT_BODY, bg=BG_INPUT, fg=TEXT_MAIN,
                              insertbackground=ACCENT, relief="flat",
                              highlightthickness=1, highlightbackground=BORDER,
                              highlightcolor=ACCENT)
        city_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        city_entry.bind("<Return>", lambda e: self._add_city())

        self._btn(entry_row, "Add", self._add_city,
                  color=ACCENT, side="left", padx=0)

        # Listbox with scrollbar
        list_frame = tk.Frame(card, bg=BG_CARD)
        list_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        scrollbar = tk.Scrollbar(list_frame, bg=BG_CARD, troughcolor=BG_INPUT)
        scrollbar.pack(side="right", fill="y")

        self.city_listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            font=FONT_BODY, bg=BG_INPUT, fg=TEXT_MAIN,
            selectbackground=ACCENT, selectforeground=BG_DARK,
            relief="flat", highlightthickness=0, borderwidth=0,
            activestyle="none"
        )
        self.city_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.city_listbox.yview)

        # Remove button
        self._btn(card, "✕  Remove Selected", self._remove_city,
                  color=ACCENT2, fill="x", padx=16, pady=(0, 12))

        # City count badge
        self.city_count_var = tk.StringVar(value="0 cities added")
        tk.Label(card, textvariable=self.city_count_var,
                 font=FONT_BODY, bg=BG_CARD, fg=TEXT_SUB).pack(pady=(0, 10))

    # ── Section 2: Distance Panel ────────────────────────────────────────────
    def _build_distance_panel(self, parent):
        card = self._card(parent, "② Enter Distances (km)", col=1)

        # Scrollable inner frame
        canvas_frame = tk.Frame(card, bg=BG_CARD)
        canvas_frame.pack(fill="both", expand=True, padx=16, pady=8)

        self.dist_canvas = tk.Canvas(canvas_frame, bg=BG_CARD,
                                     highlightthickness=0)
        self.dist_canvas.pack(side="left", fill="both", expand=True)

        v_scroll = tk.Scrollbar(canvas_frame, orient="vertical",
                                command=self.dist_canvas.yview)
        v_scroll.pack(side="right", fill="y")
        self.dist_canvas.configure(yscrollcommand=v_scroll.set)

        self.dist_inner = tk.Frame(self.dist_canvas, bg=BG_CARD)
        self.dist_window = self.dist_canvas.create_window(
            (0, 0), window=self.dist_inner, anchor="nw"
        )
        self.dist_inner.bind("<Configure>", self._on_dist_configure)
        self.dist_canvas.bind("<Configure>", self._on_canvas_configure)

        # Placeholder label
        self.dist_placeholder = tk.Label(
            self.dist_inner,
            text="Add at least 2 cities\nto enter distances.",
            font=FONT_BODY, bg=BG_CARD, fg=TEXT_SUB, justify="center"
        )
        self.dist_placeholder.grid(row=0, column=0, pady=30, padx=30)

    # ── Section 3: Result Panel ──────────────────────────────────────────────
    def _build_result_panel(self, parent):
        card = self._card(parent, "③ Optimal Route", col=2)

        # Solve button
        solve_frame = tk.Frame(card, bg=BG_CARD)
        solve_frame.pack(fill="x", padx=16, pady=(4, 12))
        self._btn(solve_frame, "🗺  Find Optimal Route",
                  self._solve_tsp, color=SUCCESS, fill="x")

        # Separator
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=16, pady=4)

        # Route display
        tk.Label(card, text="ROUTE", font=FONT_LABEL,
                 bg=BG_CARD, fg=TEXT_SUB).pack(anchor="w", padx=16, pady=(8, 2))

        route_frame = tk.Frame(card, bg=BG_INPUT, bd=0,
                               highlightthickness=1, highlightbackground=BORDER)
        route_frame.pack(fill="x", padx=16, pady=(0, 12))

        self.route_text = tk.Text(
            route_frame, font=FONT_RESULT,
            bg=BG_INPUT, fg=ACCENT,
            relief="flat", height=4,
            wrap="word", state="disabled",
            padx=10, pady=8
        )
        self.route_text.pack(fill="x")

        # Distance display
        tk.Label(card, text="TOTAL DISTANCE", font=FONT_LABEL,
                 bg=BG_CARD, fg=TEXT_SUB).pack(anchor="w", padx=16, pady=(0, 2))

        dist_frame = tk.Frame(card, bg=BG_INPUT,
                              highlightthickness=1, highlightbackground=BORDER)
        dist_frame.pack(fill="x", padx=16)

        self.dist_label = tk.Label(
            dist_frame, text="— km",
            font=("Georgia", 20, "bold"),
            bg=BG_INPUT, fg=SUCCESS, pady=10
        )
        self.dist_label.pack()

        # Status message
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=16, pady=12)

        self.status_var = tk.StringVar(value="Awaiting input…")
        self.status_label = tk.Label(
            card, textvariable=self.status_var,
            font=FONT_BODY, bg=BG_CARD, fg=TEXT_SUB,
            wraplength=180, justify="center"
        )
        self.status_label.pack(padx=16, pady=(0, 12))

        # Reset button
        self._btn(card, "↺  Reset All", self._reset_all,
                  color=ACCENT2, fill="x", padx=16, pady=(0, 12))

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Widgets
    # ─────────────────────────────────────────────────────────────────────────
    def _card(self, parent, title, col):
        """Create a styled card frame placed in the given grid column."""
        outer = tk.Frame(parent, bg=BORDER, bd=1)
        outer.grid(row=0, column=col, sticky="nsew",
                   padx=(0 if col == 0 else 8, 0 if col == 2 else 8))

        inner = tk.Frame(outer, bg=BG_CARD)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        # Card header
        header = tk.Frame(inner, bg=BG_DARK)
        header.pack(fill="x")
        tk.Label(header, text=title, font=FONT_BTN,
                 bg=BG_DARK, fg=TEXT_MAIN, pady=10, padx=16,
                 anchor="w").pack(fill="x")
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x")

        return inner

    def _btn(self, parent, text, command, color=ACCENT,
             side=None, fill=None, padx=0, pady=0):
        """Create a styled flat button."""
        btn = tk.Button(
            parent, text=text, command=command,
            font=FONT_BTN, bg=color, fg=BG_DARK,
            relief="flat", activebackground=TEXT_MAIN,
            activeforeground=BG_DARK, cursor="hand2",
            pady=7, padx=12, bd=0
        )
        if side:
            btn.pack(side=side, padx=padx, pady=pady)
        else:
            btn.pack(fill=fill, padx=padx, pady=pady)
        return btn

    # ─────────────────────────────────────────────────────────────────────────
    # Distance Grid (Canvas scroll helpers)
    # ─────────────────────────────────────────────────────────────────────────
    def _on_dist_configure(self, event):
        self.dist_canvas.configure(
            scrollregion=self.dist_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.dist_canvas.itemconfig(
            self.dist_window, width=event.width)

    # ─────────────────────────────────────────────────────────────────────────
    # City Management
    # ─────────────────────────────────────────────────────────────────────────
    def _add_city(self):
        """Add a new city destination."""
        name = self.city_var.get().strip().title()

        # Validation
        if not name:
            messagebox.showwarning("Input Error", "City name cannot be empty.")
            return
        if name in self.cities:
            messagebox.showwarning("Duplicate", f'"{name}" is already in the list.')
            return
        if len(self.cities) >= 10:
            messagebox.showwarning(
                "Limit Reached",
                "Maximum 10 cities supported (brute-force is O(n!))."
            )
            return

        self.cities.append(name)
        self.city_listbox.insert("end", f"  {name}")
        self.city_var.set("")
        self._update_city_count()
        self._rebuild_distance_grid()

    def _remove_city(self):
        """Remove the selected city and clean up distances."""
        selection = self.city_listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select a city to remove.")
            return

        idx = selection[0]
        city = self.cities[idx]

        # Remove from data structures
        self.cities.pop(idx)
        self.city_listbox.delete(idx)

        # Remove all distances involving this city
        keys_to_del = [k for k in self.distances if city in k]
        for k in keys_to_del:
            del self.distances[k]

        self._update_city_count()
        self._rebuild_distance_grid()

    def _update_city_count(self):
        n = len(self.cities)
        self.city_count_var.set(f"{n} {'city' if n == 1 else 'cities'} added")

    # ─────────────────────────────────────────────────────────────────────────
    # Distance Grid
    # ─────────────────────────────────────────────────────────────────────────
    def _rebuild_distance_grid(self):
        """Rebuild the distance-entry grid based on current cities."""
        # Clear old widgets
        for widget in self.dist_inner.winfo_children():
            widget.destroy()
        self.dist_entries.clear()

        n = len(self.cities)
        if n < 2:
            lbl = tk.Label(
                self.dist_inner,
                text="Add at least 2 cities\nto enter distances.",
                font=FONT_BODY, bg=BG_CARD, fg=TEXT_SUB, justify="center"
            )
            lbl.grid(row=0, column=0, pady=30, padx=30)
            return

        # Generate all unique pairs (undirected)
        pairs = [(self.cities[i], self.cities[j])
                 for i in range(n) for j in range(i + 1, n)]

        # Header
        tk.Label(self.dist_inner, text="From → To",
                 font=FONT_LABEL, bg=BG_CARD, fg=TEXT_SUB,
                 width=20, anchor="w").grid(
            row=0, column=0, padx=(12, 4), pady=(8, 4), sticky="w")
        tk.Label(self.dist_inner, text="Distance (km)",
                 font=FONT_LABEL, bg=BG_CARD, fg=TEXT_SUB,
                 anchor="w").grid(
            row=0, column=1, padx=4, pady=(8, 4), sticky="w")

        for r, (a, b) in enumerate(pairs, start=1):
            # Route label
            tk.Label(
                self.dist_inner,
                text=f"{a} → {b}",
                font=FONT_BODY, bg=BG_CARD, fg=TEXT_MAIN,
                anchor="w", width=20
            ).grid(row=r, column=0, padx=(12, 4), pady=3, sticky="w")

            # Entry — prefill if distance already known
            var = tk.StringVar()
            existing = self.distances.get((a, b)) or self.distances.get((b, a))
            if existing is not None:
                var.set(str(existing))

            entry = tk.Entry(
                self.dist_inner, textvariable=var,
                font=FONT_BODY, bg=BG_INPUT, fg=TEXT_MAIN,
                insertbackground=ACCENT, relief="flat",
                highlightthickness=1, highlightbackground=BORDER,
                highlightcolor=ACCENT, width=10
            )
            entry.grid(row=r, column=1, padx=4, pady=3, ipady=5, sticky="w")

            # Bind live-update of distances dict
            var.trace_add("write", lambda *_, a=a, b=b, v=var: self._update_distance(a, b, v))
            self.dist_entries[(a, b)] = var

        # Bottom padding
        tk.Label(self.dist_inner, text="", bg=BG_CARD).grid(
            row=len(pairs) + 1, column=0)

    def _update_distance(self, a, b, var):
        """Validate and store a distance entry in real time."""
        raw = var.get().strip()
        if not raw:
            self.distances.pop((a, b), None)
            self.distances.pop((b, a), None)
            return
        try:
            val = float(raw)
            if val < 0:
                raise ValueError
            self.distances[(a, b)] = val
            self.distances[(b, a)] = val
        except ValueError:
            pass  # Ignore invalid input silently; caught at solve time

    # ─────────────────────────────────────────────────────────────────────────
    # TSP Solver (Brute-Force)
    # ─────────────────────────────────────────────────────────────────────────
    def _solve_tsp(self):
        """Solve TSP using brute-force permutation search."""
        # ── Pre-checks ────────────────────────────────────────────────────────
        n = len(self.cities)
        if n < 2:
            messagebox.showwarning("Not Enough Cities",
                                   "Please add at least 2 cities.")
            return
        if n > 10:
            messagebox.showwarning("Too Many Cities",
                                   "Brute-force supports up to 10 cities.")
            return

        # Verify all distances are entered and valid
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.cities[i], self.cities[j]
                d = self.distances.get((a, b)) or self.distances.get((b, a))
                if d is None:
                    missing.append(f"{a} ↔ {b}")
                else:
                    try:
                        float(d)
                    except (TypeError, ValueError):
                        missing.append(f"{a} ↔ {b} (invalid)")

        if missing:
            messagebox.showerror(
                "Missing Distances",
                "Please enter valid distances for:\n" + "\n".join(missing)
            )
            return

        # ── Brute-Force TSP ───────────────────────────────────────────────────
        start_city = self.cities[0]
        other_cities = self.cities[1:]

        best_distance = math.inf
        best_route = None

        # Fix the starting city; permute all others to cut redundant rotations
        for perm in permutations(other_cities):
            route = [start_city] + list(perm) + [start_city]
            total = 0
            valid = True

            for k in range(len(route) - 1):
                a, b = route[k], route[k + 1]
                d = self.distances.get((a, b)) or self.distances.get((b, a))
                if d is None:
                    valid = False
                    break
                total += float(d)

            if valid and total < best_distance:
                best_distance = total
                best_route = route

        # ── Display Results ───────────────────────────────────────────────────
        if best_route is None:
            messagebox.showerror("Error", "Could not compute a valid route.")
            return

        route_str = " → ".join(best_route)
        self._set_route_text(route_str)
        self.dist_label.config(
            text=f"{best_distance:,.1f} km")
        self.status_var.set(
            f"✓ Optimal route found!\n{n} cities, {n} legs.")
        self.status_label.config(fg=SUCCESS)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _set_route_text(self, text):
        """Write text into the read-only route Text widget."""
        self.route_text.config(state="normal")
        self.route_text.delete("1.0", "end")
        self.route_text.insert("end", text)
        self.route_text.config(state="disabled")

    def _reset_all(self):
        """Clear everything and start fresh."""
        if self.cities:
            if not messagebox.askyesno(
                    "Reset", "Clear all cities and distances?"):
                return

        self.cities.clear()
        self.distances.clear()
        self.dist_entries.clear()

        self.city_listbox.delete(0, "end")
        self._update_city_count()
        self._rebuild_distance_grid()
        self._set_route_text("")
        self.dist_label.config(text="— km")
        self.status_var.set("Awaiting input…")
        self.status_label.config(fg=TEXT_SUB)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = TSPTravelPlanner(root)
    root.mainloop()

