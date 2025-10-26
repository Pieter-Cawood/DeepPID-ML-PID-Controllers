# ui_parts.py
from tkinter import ttk
from tkinter import font as tkfont


def build_summary_table(parent, controller_names):
    fixed = tkfont.nametofont("TkFixedFont")
    frame = ttk.LabelFrame(parent, text="MAE Summary (pp)", padding=8)
    frame.grid_columnconfigure(0, weight=0)
    frame.grid_columnconfigure(1, weight=1)

    header_style = {"font": fixed}
    ttk.Label(frame, text="Controller", **header_style).grid(row=0, column=0, sticky="w", padx=(2, 8))
    ttk.Label(frame, text="MAE (pp) / Max (pp)", **header_style).grid(row=0, column=1, sticky="w")

    rows = {}
    for i, name in enumerate(controller_names, start=1):
        name_lbl = ttk.Label(frame, text=name, font=fixed)
        val_lbl = ttk.Label(frame, text="-- / --", font=fixed)
        name_lbl.grid(row=i, column=0, sticky="w", padx=(2, 8))
        val_lbl.grid(row=i, column=1, sticky="w")
        rows[name] = (name_lbl, val_lbl)

    tail = ttk.Label(frame, text="Applied total --   | Step 0", font=fixed)
    tail.grid(row=len(controller_names) + 1, column=0, columnspan=2, sticky="e", pady=(6, 0))
    return frame, rows, tail


def build_cmp_table(parent, entity_title, controller_names, labels, fixed_font):
    cmpf = ttk.LabelFrame(parent, text="Controllers — Real-time Mix Error (pp)", padding=10)
    top = ttk.Frame(cmpf); top.pack(fill="x")
    MAT_W, TAR_W, ERR_W = 12, 10, 12
    col = 0
    ttk.Label(top, text=entity_title, width=MAT_W, font=fixed_font, anchor="w").grid(row=0, column=col, sticky="w"); col += 1
    ttk.Label(top, text="Target (%)", width=TAR_W, font=fixed_font, anchor="e").grid(row=0, column=col, sticky="e"); col += 1

    err_header_labels = {}
    err_cols = {}
    for name in controller_names:
        lbl = ttk.Label(top, text=f"{name}", width=ERR_W, font=fixed_font, anchor="e")
        lbl.grid(row=0, column=col, sticky="e")
        err_header_labels[name] = lbl
        err_cols[name] = col
        col += 1

    rows_cmp = []
    for i, name in enumerate(labels):
        r = ttk.Frame(cmpf); r.pack(fill="x", pady=2)
        ttk.Label(r, text=name, width=MAT_W, font=fixed_font, anchor="w").grid(row=0, column=0, sticky="w")
        t_lbl = ttk.Label(r, text="--", width=TAR_W, font=fixed_font, anchor="e")
        t_lbl.grid(row=0, column=1, sticky="e")
        err_labels = []
        for mname in controller_names:
            el = ttk.Label(r, text="--", width=ERR_W, font=fixed_font, anchor="e")
            el.grid(row=0, column=err_cols[mname], sticky="e")
            err_labels.append(el)
        rows_cmp.append((t_lbl, err_labels))

    return cmpf, err_header_labels, err_cols, rows_cmp
