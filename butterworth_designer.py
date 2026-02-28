"""
Butterworth Filter Designer
A standalone GUI application for analog Butterworth filter design.
"""

# DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ---------------------------------------------------------------------------
# Catppuccin Mocha palette
# ---------------------------------------------------------------------------
BG       = "#1e1e2e"   # base
BG2      = "#181825"   # mantle
SURFACE  = "#313244"   # surface0
OVERLAY  = "#45475a"   # overlay0
TEXT     = "#cdd6f4"   # text
SUBTEXT  = "#a6adc8"   # subtext1
LAVENDER = "#b4befe"   # lavender  (accent)
GREEN    = "#a6e3a1"   # green
RED      = "#f38ba8"   # red
YELLOW   = "#f9e2af"   # yellow
BLUE     = "#89b4fa"   # blue
MAUVE    = "#cba6f7"   # mauve


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def fmt_coeff(c: float) -> str:
    """Compact number string for one coefficient."""
    if c == 0:
        return "0"
    abs_c = abs(c)
    if abs_c >= 1e6 or (abs_c < 1e-3 and abs_c != 0):
        # scientific notation, strip trailing zeros
        s = f"{c:.3e}"
        # e.g. "1.000e+12" -> "1e+12", "2.828e+03" -> "2828" handled below
        mantissa, exp_part = s.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp_val = int(exp_part)
        if mantissa in ("1", "-1"):
            return f"{'−' if c < 0 else ''}1e{exp_val:+d}"
        return f"{mantissa}e{exp_val:+d}"
    if abs_c == int(abs_c) and abs_c < 1e6:
        return str(int(c))
    # up to 4 significant figures
    s = f"{c:.4g}"
    return s


def poly_to_str(coeffs: np.ndarray, var: str = "s") -> str:
    """
    Convert a coefficient array (highest power first) to a readable
    polynomial string, e.g. "s^4  +  2828*s^3  +  4e+06*s^2  +  ..."
    """
    coeffs = np.asarray(coeffs, dtype=float)
    n = len(coeffs) - 1  # degree
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        if c == 0:
            continue
        cs = fmt_coeff(abs(c))
        sign = "+" if c >= 0 else "−"
        if power == 0:
            term = cs
        elif power == 1:
            term = f"{cs}*{var}" if cs not in ("1", "−1") else var
        else:
            term = f"{cs}*{var}^{power}" if cs not in ("1", "−1") else f"{var}^{power}"
        terms.append((sign, term))

    if not terms:
        return "0"

    parts = []
    for idx, (sign, term) in enumerate(terms):
        if idx == 0:
            # leading sign: only show minus
            prefix = "−" if sign == "−" else ""
            parts.append(f"{prefix}{term}")
        else:
            parts.append(f"  {sign}  {term}")
    return "".join(parts)


def build_hs_display(b: np.ndarray, a: np.ndarray) -> str:
    """
    Build a 3-line ASCII fraction block showing H(s) = num/den.
    Returns a multi-line string ready for a Text widget.
    """
    num_str = poly_to_str(b)
    den_str = poly_to_str(a)
    width = max(len(num_str), len(den_str)) + 4
    bar = "─" * width
    num_pad = num_str.center(width)
    den_pad = den_str.center(width)
    return f"H(s)  =\n{num_pad}\n{bar}\n{den_pad}"


def build_poles_display(a: np.ndarray) -> str:
    """
    Compute and format the poles (roots of denominator polynomial).
    Returns a multi-line string.
    """
    try:
        poles = np.roots(a)
    except Exception:
        return "(poles unavailable)"

    lines = ["", "Poles:"]
    for i, p in enumerate(poles, 1):
        re = p.real
        im = p.imag
        if abs(im) < 1e-10 * max(1, abs(re)):
            lines.append(f"  p{i} = {fmt_coeff(re)}")
        else:
            sign = "+" if im >= 0 else "−"
            lines.append(f"  p{i} = {fmt_coeff(re)}  {sign}  j·{fmt_coeff(abs(im))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class ButterworthDesigner(tk.Tk):
    """Main application window."""

    FILTER_TYPES = ["Low Pass", "High Pass", "Band Pass", "Band Reject"]
    UNITS = ["rad/s", "Hz"]

    def __init__(self):
        super().__init__()
        self.title("Butterworth Filter Designer")
        self.configure(bg=BG)
        self.minsize(920, 640)

        self._setup_style()
        self._build_ui()

        # auto-design after window is fully rendered
        self.after(300, self._design)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        common = {
            "background": BG,
            "foreground": TEXT,
            "fieldbackground": SURFACE,
            "selectbackground": LAVENDER,
            "selectforeground": BG,
            "troughcolor": BG2,
            "bordercolor": OVERLAY,
            "darkcolor": BG2,
            "lightcolor": SURFACE,
            "relief": "flat",
        }

        style.configure("TFrame",      background=BG)
        style.configure("TLabel",      background=BG,      foreground=TEXT,     font=("Segoe UI", 10))
        style.configure("TLabelframe", background=BG,      foreground=LAVENDER, font=("Segoe UI", 10, "bold"),
                        bordercolor=OVERLAY, relief="groove")
        style.configure("TLabelframe.Label", background=BG, foreground=LAVENDER, font=("Segoe UI", 10, "bold"))
        style.configure("TButton",     background=LAVENDER, foreground=BG,       font=("Segoe UI", 10, "bold"),
                        bordercolor=LAVENDER, focuscolor=LAVENDER, padding=6)
        style.map("TButton",
                  background=[("active", MAUVE), ("pressed", BLUE)],
                  foreground=[("active", BG)])
        style.configure("TCombobox",   **{k: v for k, v in common.items()
                                           if k in ("background", "foreground", "fieldbackground",
                                                    "selectbackground", "selectforeground")})
        style.configure("TSpinbox",    **{k: v for k, v in common.items()
                                          if k in ("background", "foreground", "fieldbackground",
                                                   "selectbackground", "selectforeground",
                                                   "bordercolor", "troughcolor")})
        style.map("TCombobox",
                  fieldbackground=[("readonly", SURFACE)],
                  background=[("readonly", SURFACE)])

        style.configure("Error.TLabel", background=BG, foreground=RED,    font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=BG, foreground=GREEN, font=("Segoe UI", 9))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # Left panel
        left = ttk.LabelFrame(outer, text="Parameters", padding=12, width=220)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_propagate(False)
        self._build_params(left)

        # Right column
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # H(s) display box
        hs_frame = ttk.LabelFrame(right, text="Transfer Function  H(s)", padding=8)
        hs_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        hs_frame.columnconfigure(0, weight=1)

        self._hs_text = tk.Text(
            hs_frame, height=8, bg=BG2, fg=TEXT,
            font=("Courier New", 10), relief="flat",
            state="disabled", wrap="none",
            insertbackground=TEXT,
            selectbackground=LAVENDER, selectforeground=BG,
        )
        self._hs_text.pack(fill="both", expand=True)

        # Plot area
        plot_frame = ttk.LabelFrame(right, text="Frequency Response", padding=4)
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        self._build_plot(plot_frame)

    def _build_params(self, parent):
        parent.columnconfigure(1, weight=1)
        row = 0

        # Order N
        ttk.Label(parent, text="Order N:").grid(row=row, column=0, sticky="w", pady=3)
        self._var_n = tk.StringVar(value="4")
        spin = ttk.Spinbox(parent, from_=1, to=20, textvariable=self._var_n,
                           width=5, font=("Segoe UI", 10))
        spin.grid(row=row, column=1, sticky="ew", pady=3, padx=(6, 0))
        row += 1

        # Filter type
        ttk.Label(parent, text="Type:").grid(row=row, column=0, sticky="w", pady=3)
        self._var_type = tk.StringVar(value="Low Pass")
        cb_type = ttk.Combobox(parent, textvariable=self._var_type,
                               values=self.FILTER_TYPES, state="readonly", width=10)
        cb_type.grid(row=row, column=1, sticky="ew", pady=3, padx=(6, 0))
        cb_type.bind("<<ComboboxSelected>>", lambda _: self._on_type_change())
        row += 1

        # Unit
        ttk.Label(parent, text="Unit:").grid(row=row, column=0, sticky="w", pady=3)
        self._var_unit = tk.StringVar(value="rad/s")
        cb_unit = ttk.Combobox(parent, textvariable=self._var_unit,
                               values=self.UNITS, state="readonly", width=10)
        cb_unit.grid(row=row, column=1, sticky="ew", pady=3, padx=(6, 0))
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # fc1 label + entry
        self._lbl_fc1 = ttk.Label(parent, text="ωc:")
        self._lbl_fc1.grid(row=row, column=0, sticky="w", pady=3)
        self._var_fc1 = tk.StringVar(value="1000")
        self._entry_fc1 = ttk.Entry(parent, textvariable=self._var_fc1,
                                    font=("Segoe UI", 10))
        self._entry_fc1.grid(row=row, column=1, sticky="ew", pady=3, padx=(6, 0))
        row += 1

        # fc2 frame (hidden for LP/HP)
        self._fc2_frame = ttk.Frame(parent)
        self._fc2_frame.columnconfigure(1, weight=1)
        self._lbl_fc2 = ttk.Label(self._fc2_frame, text="ωc2:")
        self._lbl_fc2.grid(row=0, column=0, sticky="w", pady=3)
        self._var_fc2 = tk.StringVar(value="2000")
        ttk.Entry(self._fc2_frame, textvariable=self._var_fc2,
                  font=("Segoe UI", 10)).grid(row=0, column=1, sticky="ew",
                                               pady=3, padx=(6, 0))
        self._fc2_frame.grid(row=row, column=0, columnspan=2, sticky="ew")
        self._fc2_frame.grid_remove()  # hidden by default
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Design button
        btn = ttk.Button(parent, text="Design Filter", command=self._design)
        btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        row += 1

        # Status / error label
        self._lbl_status = ttk.Label(parent, text="", style="Status.TLabel",
                                     wraplength=200, justify="left")
        self._lbl_status.grid(row=row, column=0, columnspan=2, sticky="w")

    def _build_plot(self, parent):
        self._fig, (self._ax_mag, self._ax_ph) = plt.subplots(
            2, 1, figsize=(7, 5), facecolor=BG2)
        self._fig.subplots_adjust(hspace=0.45, left=0.1, right=0.97, top=0.95, bottom=0.09)

        self._init_ax(self._ax_mag)
        self._init_ax(self._ax_ph)
        self._ax_mag.set_title("Magnitude  |H(ω)|", color=LAVENDER, fontsize=10, pad=6)
        self._ax_ph.set_title("Phase  ∠H(ω)", color=LAVENDER, fontsize=10, pad=6)

        canvas = FigureCanvasTkAgg(self._fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas = canvas

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.config(background=BG2)
        for child in toolbar.winfo_children():
            try:
                child.config(background=BG2, foreground=TEXT)
            except Exception:
                pass
        toolbar.update()

    def _init_ax(self, ax):
        ax.set_facecolor(BG2)
        for spine in ax.spines.values():
            spine.set_edgecolor(OVERLAY)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.yaxis.label.set_color(SUBTEXT)
        ax.grid(True, which="both", color=OVERLAY, linestyle="--", linewidth=0.5, alpha=0.7)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_type_change(self):
        ft = self._var_type.get()
        is_two_freq = ft in ("Band Pass", "Band Reject")
        if is_two_freq:
            self._fc2_frame.grid()
            self._lbl_fc1.config(text="ωc1:")
        else:
            self._fc2_frame.grid_remove()
            self._lbl_fc1.config(text="ωc:")

    # ------------------------------------------------------------------
    # Design
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False):
        style = "Error.TLabel" if error else "Status.TLabel"
        self._lbl_status.config(text=msg, style=style)

    def _design(self):
        # --- parse inputs ---
        try:
            N = int(self._var_n.get())
            if N < 1 or N > 20:
                raise ValueError("Order N must be between 1 and 20.")
        except ValueError as e:
            self._set_status(str(e), error=True)
            return

        ft = self._var_type.get()
        unit = self._var_unit.get()

        try:
            fc1 = float(self._var_fc1.get())
            if fc1 <= 0:
                raise ValueError("Cutoff frequency must be positive.")
        except ValueError as e:
            self._set_status(f"ωc1: {e}", error=True)
            return

        # convert Hz → rad/s
        w1 = fc1 * 2 * np.pi if unit == "Hz" else fc1

        if ft in ("Band Pass", "Band Reject"):
            try:
                fc2 = float(self._var_fc2.get())
                if fc2 <= 0:
                    raise ValueError("Upper cutoff must be positive.")
            except ValueError as e:
                self._set_status(f"ωc2: {e}", error=True)
                return
            w2 = fc2 * 2 * np.pi if unit == "Hz" else fc2
            if w2 <= w1:
                self._set_status("ωc2 must be greater than ωc1.", error=True)
                return
            Wn = [w1, w2]
            wns = np.array([w1, w2])
        else:
            Wn = w1
            wns = np.array([w1])

        # scipy btype mapping
        btype_map = {
            "Low Pass":   "low",
            "High Pass":  "high",
            "Band Pass":  "bandpass",
            "Band Reject": "bandstop",
        }
        btype = btype_map[ft]

        try:
            b, a = signal.butter(N, Wn, btype=btype, analog=True)
        except Exception as e:
            self._set_status(f"Design error: {e}", error=True)
            return

        # --- H(s) display ---
        hs_text = build_hs_display(b, a)
        poles_text = build_poles_display(a)
        full_text = hs_text + "\n" + poles_text

        self._hs_text.config(state="normal")
        self._hs_text.delete("1.0", "end")
        self._hs_text.insert("end", full_text)
        self._hs_text.config(state="disabled")

        # --- frequency sweep ---
        w_min = wns.min() / 100
        w_max = wns.max() * 100
        w = np.logspace(np.log10(w_min), np.log10(w_max), 3000)

        try:
            _, H = signal.freqs(b, a, worN=w)
        except Exception as e:
            self._set_status(f"freqs error: {e}", error=True)
            return

        mag_db = 20 * np.log10(np.abs(H) + 1e-300)
        phase_deg = np.degrees(np.unwrap(np.angle(H)))

        self._plot(w, mag_db, phase_deg, wns, ft, N)
        actual_order = len(a) - 1
        self._set_status(
            f"OK — {ft}, order {actual_order} (N={N})",
            error=False,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot(self, w, mag, ph, wns, ft, N):
        ax_m = self._ax_mag
        ax_p = self._ax_ph
        ax_m.cla()
        ax_p.cla()
        self._init_ax(ax_m)
        self._init_ax(ax_p)

        # --- magnitude ---
        ax_m.semilogx(w, mag, color=BLUE, linewidth=1.8, label="|H(ω)| dB")
        ax_m.axhline(-3, color=RED, linestyle="--", linewidth=1, label="−3 dB")
        for wc in wns:
            ax_m.axvline(wc, color=GREEN, linestyle=":", linewidth=1.2,
                         label=f"ωc = {fmt_coeff(wc)}")
        ax_m.set_ylabel("dB", color=SUBTEXT, fontsize=9)
        ax_m.set_xlabel("ω (rad/s)", color=SUBTEXT, fontsize=9)
        ax_m.set_title("Magnitude  |H(ω)|", color=LAVENDER, fontsize=10, pad=6)
        ax_m.legend(fontsize=8, facecolor=SURFACE, edgecolor=OVERLAY,
                    labelcolor=TEXT, framealpha=0.85)

        # --- phase ---
        ax_p.semilogx(w, ph, color=MAUVE, linewidth=1.8, label="∠H(ω)")
        for wc in wns:
            ax_p.axvline(wc, color=GREEN, linestyle=":", linewidth=1.2)
        ax_p.set_ylabel("degrees", color=SUBTEXT, fontsize=9)
        ax_p.set_xlabel("ω (rad/s)", color=SUBTEXT, fontsize=9)
        ax_p.set_title("Phase  ∠H(ω)", color=LAVENDER, fontsize=10, pad=6)
        ax_p.legend(fontsize=8, facecolor=SURFACE, edgecolor=OVERLAY,
                    labelcolor=TEXT, framealpha=0.85)

        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = ButterworthDesigner()
    app.mainloop()
