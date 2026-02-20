import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scattnlay import scattnlay, fieldnlay
import optool
import pandas as pd
import seaborn as sns
from functions_test import *
import time
# global exe
exe = "/home/david/miniconda3/bin/optool"


#######

OPTOOL_SETTINGS['nl'] = 2
OPTOOL_SETTINGS['lmin'] = 1.65 
OPTOOL_SETTINGS['lmax'] = 850 


# D = 3
# kf = 0.3
D = 1.9 
kf = 1.04

a0 = 0.1
Rc = 100_000
N = 10
Material = "pyr-mg70 0.70 c 0.30"
Matrix = True

for i in [100_000]:
    p = Particle(D, kf, a0, i, N, Material, Matrix)
    print(f'calculating particle size: {i} Micron')
    print('Running EMT Calculation')
    p.RunMie()
    print('Running ML-EMT Calculation')
    p.RunScatt()
    print('Running MMF Calculation')
    p.RunMMF()

    save_particle(p, 'saved_particles', f'BCCA_{i}um_matrix')

    print('saved')


# #######
# legend_size = 18
# title_size = 25
# axis_size = 20
# ticksize = 15

# def plot_optool_vs_scatt(scatt_df, optool_df, title="Opacity comparison", save=None):
#     # read columns
#     lam_s = scatt_df["wavelength_um"].to_numpy()
#     a_s   = scatt_df["kabs_cm2g"].to_numpy()
#     s_s   = scatt_df["ksca_cm2g"].to_numpy()

#     lam_o = optool_df["wavelength_um"].to_numpy()
#     a_o   = optool_df["kabs_cm2g"].to_numpy()
#     s_o   = optool_df["ksca_cm2g"].to_numpy()

#     # interpolate optool onto scatt wavelengths so the residual uses same x grid
#     a_o_i = np.interp(np.log10(lam_s), np.log10(lam_o), a_o)
#     s_o_i = np.interp(np.log10(lam_s), np.log10(lam_o), s_o)

#     # percent difference: (scatt minus optool) over optool
#     da = (a_s - a_o_i) / a_o_i * 100.0
#     ds = (s_s - s_o_i) / s_o_i * 100.0

#     fig, (ax1, ax2) = plt.subplots(
#         2, 1, figsize=(10, 7), sharex=True,
#         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
#     )

#     # top panel
#     ax1.scatter(lam_s, a_s, color = 'black', s=35)
#     ax1.scatter(lam_s, s_s, color = 'black', s=35)
#     ax1.scatter(lam_s, a_s, label=r"ML-EMT $\kappa_{\text{abs}}$", s=25)
#     ax1.scatter(lam_s, s_s, label=r"ML-EMT $\kappa_{\text{sca}}$", s=25)
    
    
#     ax1.plot(lam_o, a_o, color ='black', lw = 2.5)
#     ax1.plot(lam_o, s_o, color ='black', lw = 2.5)
#     ax1.plot(lam_o, a_o, label=r"EMT $\kappa_{\text{abs}}$", lw = 2)
#     ax1.plot(lam_o, s_o, label=r"EMT $\kappa_{\text{sca}}$",lw = 2)
    

#     ax1.set_xscale("log")
#     ax1.set_yscale("log")
#     ax1.set_ylabel(r"$\kappa_\text{sca}, \kappa_\text{abs}$ ($\text{cm}^2 \text{g}^{-1}$)", fontsize = axis_size)
#     # ax1.set_title(title, fontsize = title_size)
#     ax1.legend(fontsize= legend_size, ncol=2, frameon=False)

#     # bottom panel
#     ax2.axhline(0, color = 'black', linewidth=1, zorder = -1)
    
#     ax2.scatter(lam_s, da, color = 'black', s=35)
#     ax2.scatter(lam_s, ds, color = 'black', s=35)
#     ax2.scatter(lam_s, da, s=20)
#     ax2.scatter(lam_s, ds, s=20)
    

#     ax2.set_xscale("log")
#     ax2.set_xlabel("Wavelength (Micron)", fontsize = axis_size)
#     ax2.set_ylabel("Residual (%)", fontsize = axis_size)
    
#     ax1.tick_params(axis="both", which="major", labelsize=ticksize)
#     ax2.tick_params(axis="both", which="major", labelsize=ticksize)


#     if save:
#         plt.savefig(save, dpi=300, bbox_inches="tight")
#     plt.show()

# def plot_g_optool_vs_scatt(scatt_df, optool_df,
#                            title="Asymmetry parameter comparison",
#                            save=None,
#                            zoom_min=700,
#                            zoom_max=1100,
#                            box_y_exaggeration=2.5,
#                            connector_lw=1.2,
#                            zoom = True):

#     lam_s = scatt_df["wavelength_um"].to_numpy()
#     g_s   = scatt_df["g"].to_numpy()

#     lam_o = optool_df["wavelength_um"].to_numpy()
#     g_o   = optool_df["g"].to_numpy()

#     dg = (g_s - g_o) / g_o * 100.0

#     fig, (ax1, ax2) = plt.subplots(
#         2, 1, figsize=(10, 7), sharex=True,
#         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
#     )

#     # ===== TOP PANEL =====
#     ax1.scatter(lam_s, g_s, color="black", s=35, zorder=3)
#     ax1.scatter(lam_s, g_s, s=25, label="ML-EMT g", zorder=4)

#     ax1.plot(lam_o, g_o, color="black", lw=2.5, zorder=2)
#     ax1.plot(lam_o, g_o, lw=2, label="EMT g", zorder=3)

#     ax1.set_xscale("log")
#     ax1.set_ylabel("Asymmetry parameter g", fontsize=axis_size)
#     ax1.legend(fontsize=legend_size, ncol=2, frameon=False)
    
#     if zoom:
#         # ===== ZOOM INSET (TOP) =====
#         mask_s = (lam_s >= zoom_min) & (lam_s <= zoom_max)
#         axins = inset_axes(ax1, width="27%", height="27%", loc="upper right", borderpad=0.8)

#         axins.scatter(lam_s[mask_s], g_s[mask_s], color="black", s=35, zorder=3)
#         axins.scatter(lam_s[mask_s], g_s[mask_s], s=25, zorder=4)

#         axins.plot(lam_o, g_o, color="black", lw=2.5, zorder=2)
#         axins.plot(lam_o, g_o, lw=2, zorder=3)

#         axins.set_xscale("log")
#         axins.set_xlim(zoom_min, zoom_max)

#         # tight y limits using only inset points
#         y_lo, y_hi = g_s[mask_s].min(), g_s[mask_s].max()
#         pad = 0.5 * (y_hi - y_lo) if (y_hi > y_lo) else 0.1 * max(abs(y_lo), 1e-6)
#         axins.set_ylim(y_lo - pad, y_hi + pad)

#         # inset frame style
#         for spine in axins.spines.values():
#             spine.set_color("black")
#             spine.set_linewidth(1.0)

#         # inset ticks: only 2 labels, no minor ticks
#         axins.xaxis.set_minor_locator(NullLocator())
#         tick_vals = [200,250, 1e3, 8e2]
#         tick_vals = [t for t in tick_vals if zoom_min <= t <= zoom_max]
#         if len(tick_vals) < 2:
#             tick_vals = [1e3]
#         axins.xaxis.set_major_locator(FixedLocator(tick_vals))
#         axins.xaxis.set_major_formatter(LogFormatterMathtext())
#         axins.tick_params(axis="both", which="major", labelsize=12)
        
#         # ===== BETTER BOX ON MAIN PLOT (EXAGGERATED Y) + CLEAN CONNECTORS =====
#         x0, x1 = zoom_min, zoom_max

#         # base y-range from actual inset y-limits
#         y0_in, y1_in = axins.get_ylim()
#         y_mid = 0.5 * (y0_in + y1_in)
#         half = 0.5 * (y1_in - y0_in)

#         # exaggerate box height around the middle
#         y0 = y_mid - box_y_exaggeration * half *20
#         y1 = y_mid + box_y_exaggeration * half *20

#         rect = Rectangle(
#             (x0, y0), x1 - x0, y1 - y0,
#             fill=False, edgecolor="black", linewidth=1.2, zorder=5
#         )
#         ax1.add_patch(rect)

#         # connect from rectangle corners to inset corners (axes coords)
#         # using inset corners keeps connectors away from inset tick labels
#         con1 = ConnectionPatch(
#             xyA=(x0, y1), coordsA=ax1.transData,
#             xyB=(0, 0),  coordsB=axins.transAxes,
#             color="black", linewidth=connector_lw
#         )
#         con2 = ConnectionPatch(
#             xyA=(x1, y1), coordsA=ax1.transData,
#             xyB=(1, 0),  coordsB=axins.transAxes,
#             color="black", linewidth=connector_lw
#         )
    
#         # make lines a bit cleaner
#         con1.set_zorder(4)
#         con2.set_zorder(4)
#         ax1.add_artist(con1)
#         ax1.add_artist(con2)

#     # ===== BOTTOM PANEL =====
#     ax2.axhline(0, color="black", linewidth=1, zorder=-1)
#     ax2.scatter(lam_s, dg, color="black", s=35, zorder=3)
#     ax2.scatter(lam_s, dg, s=20, zorder=4)

#     ax2.set_xscale("log")
#     ax2.set_xlabel("Wavelength (Micron)", fontsize=axis_size)
#     ax2.set_ylabel("Residual (%)", fontsize=axis_size)

#     ax1.tick_params(axis="both", which="major", labelsize=ticksize)
#     ax2.tick_params(axis="both", which="major", labelsize=ticksize)
    
#     # ax2.set_ylim(-1,1)

#     if save:
#         plt.savefig(save, dpi=300, bbox_inches="tight")

#     plt.show()



# folder = save_particle(BCCA, outdir="saved_particles", name="10cmBCCA")
# print("Saved to", folder)

# data = load_particle_results(folder)
# print(data["meta"])
# print(data["scattnlay_kappas"].head())

# t1 = time.perf_counter()
# print(f"Elapsed time: {t1 - t0:.3f} s")



