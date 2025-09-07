# Plots used for examples

import matplotlib.pyplot as plt
import numpy as np

# Define city population share (L_C) as x-axis

L_C = np.linspace(0, 100, 200)  # number of people in city (0 to 100)

# Define ΔP (price difference: city - rural)
# Assume ΔP decreases as more people move to the city (supply/demand effect)
# but also increases due to productivity premium (agglomeration effect)
DeltaP = 20 - 0.2 * L_C + 0.05 * (L_C**1.2) / 20

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(L_C, DeltaP, color="blue", linewidth=2)

# Labels
ax.set_title("Aggregate Demand for the City")
ax.set_xlabel("Number of People Living in the City ($L_C$)")
ax.set_ylabel("ΔP (City Price - Rural Price)")

# Remove axis numbers for stylistic consistency
ax.set_xticks([])
ax.set_yticks([])

# plt.show()

# Define range of people (x-axis, shared for city and rural)
L = np.linspace(0, 120, 500)

# Capacity limits
capacity_limit_city = 65
capacity_limit_rural = 85

# Shift total population line closer to rural capacity (50% of the way from rural to full sum)
true_total_population = capacity_limit_city + capacity_limit_rural
adjusted_total_population = capacity_limit_rural + 0.5 * (
    true_total_population - capacity_limit_rural
)

# Price functions
price_city = 12 + 0.3 * np.minimum(L, capacity_limit_city)
price_rural = 8 + 0.2 * np.minimum(L, capacity_limit_rural)

fig, ax = plt.subplots(figsize=(9, 6))

# City curve
mask_c = L < capacity_limit_city
ax.plot(L[mask_c], price_city[mask_c], color="blue", linewidth=2, label="City ($L_C$)")
ax.plot(
    [capacity_limit_city, capacity_limit_city],
    [price_city[mask_c][-1], price_city[mask_c][-1] + 20],
    color="blue",
    linewidth=2,
)
ax.scatter(capacity_limit_city, price_city[mask_c][-1], color="blue")

# Rural curve
mask_r = L < capacity_limit_rural
ax.plot(
    L[mask_r], price_rural[mask_r], color="green", linewidth=2, label="Rural ($L_R$)"
)
ax.plot(
    [capacity_limit_rural, capacity_limit_rural],
    [price_rural[mask_r][-1], price_rural[mask_r][-1] + 20],
    color="green",
    linewidth=2,
)
ax.scatter(capacity_limit_rural, price_rural[mask_r][-1], color="green")

# Vertical dashed lines at capacities
ax.axvline(capacity_limit_city, color="blue", linestyle="--", alpha=0.7)
ax.text(
    capacity_limit_city + 1,
    ax.get_ylim()[1] * 0.8,
    "City Capacity",
    rotation=90,
    color="blue",
)

ax.axvline(capacity_limit_rural, color="green", linestyle="--", alpha=0.7)
ax.text(
    capacity_limit_rural + 1,
    ax.get_ylim()[1] * 0.6,
    "Rural Capacity",
    rotation=90,
    color="green",
)

# Adjusted total population line (black dashed)
ax.axvline(adjusted_total_population, color="black", linestyle="--", alpha=0.9)
ax.text(
    adjusted_total_population + 1,
    ax.get_ylim()[1] * 0.4,
    "Total Population ($L = L_C + L_R$)",
    rotation=90,
    color="black",
)

# Labels
ax.set_title("Inverse Supply Curves with Capacity Limits")
ax.set_xlabel("Number of People Living in Area ($L_C$ or $L_R$)")
ax.set_ylabel("Price")
ax.set_xticks([])
ax.set_yticks([])
ax.legend()

plt.tight_layout()
# plt.show()


# Parameters
capacity_city = 65
capacity_rural = 85

# Use the SAME L (total population) as before
L = capacity_rural + 0.5 * ((capacity_city + capacity_rural) - capacity_rural)  # 117.5

# City inverse supply as a function of L_C
L_C = np.linspace(0, L, 600)
price_city = 12 + 0.3 * np.minimum(L_C, capacity_city)

# Rural inverse supply as a function of (L - L_C)
L_R = np.linspace(0, min(capacity_rural, L), 400)
price_rural = 8 + 0.2 * np.minimum(L_R, capacity_rural)
x_rural_as_LC = L - L_R  # reparam x-axis: L_C

fig, ax = plt.subplots(figsize=(10, 6))

# --- City curve (upward then vertical) ---
mask_c = L_C <= capacity_city
ax.plot(
    L_C[mask_c],
    price_city[mask_c],
    linewidth=2,
    color="blue",
    label="City supply vs $L_C$",
)
ax.plot(
    [capacity_city, capacity_city],
    [price_city[mask_c][-1], price_city[mask_c][-1] + 20],
    linewidth=2,
    color="blue",
)

# --- Rural curve (sloping then vertical) ---
mask_r = L_R <= capacity_rural
ax.plot(
    x_rural_as_LC[mask_r],
    price_rural[mask_r],
    linewidth=2,
    color="green",
    label="Rural supply vs $(L - L_C)$",
)
# add vertical at the start of the rural curve (L - capacity_rural)
ax.plot(
    [L - capacity_rural, L - capacity_rural],
    [price_rural[mask_r][-1], price_rural[mask_r][-1] + 20],
    linewidth=2,
    color="green",
)

# --- Total population line ---
ax.axvline(L, linestyle="--", color="black", alpha=0.9)
ax.text(
    L + 1, ax.get_ylim()[1] * 0.46, "Total Population ($L$)", rotation=90, color="black"
)

# Styling
ax.set_title("City Supply vs $L_C$ and Rural Supply vs $(L - L_C)$")
ax.set_xlabel("City Population $L_C$")
ax.set_ylabel("Price")
ax.set_xlim(0, L + 10)
ax.set_xticks([])
ax.set_yticks([])
ax.legend()

plt.tight_layout()
# plt.show()


# --- Parameters (same as before) ---
capacity_city = 65
capacity_rural = 85
L = capacity_rural + 0.5 * (
    (capacity_city + capacity_rural) - capacity_rural
)  # total population (same as earlier)

# Key x-locations
x_left = L - capacity_rural  # rural capacity in L_C units
x_right = capacity_city  # city capacity

# Visualization jump sizes for vertical capacity segments
VJUMP_RURAL = 25
VJUMP_CITY = 25

# ---------- Top plot: City & Rural inverse supply (prices), x = L_C ----------
# City: upward slope then vertical
Pc_slope_x = np.linspace(0, x_right, 200)
Pc_slope_y = 12 + 0.3 * Pc_slope_x
Pc_at_cap = Pc_slope_y[-1]

# Rural: vertical goes UP at x_left, then **downward** slope toward L (reverting to original shape)
Pr_start = 8 + 0.2 * capacity_rural  # price at the kink
Pr_slope_x = np.linspace(x_left, L - 10, 200)
Pr_slope_y = 8 + 0.2 * (L - Pr_slope_x)  # decreases with L_C

# ---------- Bottom plot: ΔPrice = P_C - P_R vs L_C (supply) + demand ----------
# Interior (both within capacity): x_left -> x_right
xs_interior = np.linspace(x_left, x_right, 250)
Pc_interior = 12 + 0.3 * xs_interior
Pr_interior = 8 + 0.2 * (L - xs_interior)
ys_interior = Pc_interior - Pr_interior  # ΔP

# Values at kink points
y_left = (12 + 0.3 * x_left) - (8 + 0.2 * (L - x_left))
y_right = (12 + 0.3 * x_right) - (8 + 0.2 * (L - x_right))

# Demand: choose intersection in the interior and ensure ΔP at eq ≠ 0
x_eq = 0.5 * (x_left + x_right)  # equilibrium L_C
y_eq = (12 + 0.3 * x_eq) - (
    8 + 0.2 * (L - x_eq)
)  # supply at eq (non-zero with this setup)

# Downward-sloping demand passing through (x_eq, y_eq)
b = 0.3  # positive magnitude; demand slope is -b
a = y_eq + b * x_eq


def demand(Lc):
    return a - b * Lc


xs_d = np.linspace(10, L - 10, 300)
ys_d = demand(xs_d)

plt.clf()
# ---------- Plot ----------
fig, axs = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

# Top: City & Rural supply (prices)
axs[0].plot(
    Pc_slope_x, Pc_slope_y, color="blue", linewidth=2, label="City supply vs $L_C$"
)
axs[0].plot(
    [x_right, x_right], [Pc_at_cap, Pc_at_cap + VJUMP_CITY], color="blue", linewidth=2
)

# Rural: vertical (UP) then downward slope
axs[0].plot(
    [x_left, x_left], [Pr_start, Pr_start + VJUMP_RURAL], color="green", linewidth=2
)
axs[0].plot(
    Pr_slope_x,
    Pr_slope_y,
    color="green",
    linewidth=2,
    label="Rural supply vs $(L - L_C)$",
)

# Total population and equilibrium verticals
axs[0].axvline(
    L, linestyle="--", color="black", alpha=0.9, label="Total Population (L)"
)
axs[0].axvline(
    x_eq, color="gray", linewidth=2, alpha=0.9
)  # equilibrium vertical (opaque gray)
axs[0].set_title("City Supply vs $L_C$ and Rural Supply vs $(L - L_C)$")
axs[0].set_ylabel("Price")
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].legend()

# Bottom: ΔPrice supply and demand
axs[1].plot(xs_interior, ys_interior, color="red", linewidth=2, label="Supply")
axs[1].plot(
    [x_left, x_left], [y_left - VJUMP_RURAL, y_left], color="red", linewidth=2
)  # vertical drop at rural kink
axs[1].plot(
    [x_right, x_right], [y_right, y_right + VJUMP_CITY], color="red", linewidth=2
)  # vertical rise at city kink
axs[1].plot(xs_d, ys_d, color="blue", linestyle="--", linewidth=2, label="Demand")
axs[1].scatter([x_eq], [y_eq], s=60, color="black", zorder=5)
axs[1].annotate(
    "Equilibrium",
    xy=(x_eq, y_eq),
    xytext=(x_eq + 5, y_eq - 8),
    fontsize=12,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", lw=1),
)

# Total population and equilibrium verticals
axs[1].axvline(L, linestyle="--", color="black", alpha=0.9, label="Total Population")
axs[1].axvline(
    x_eq, color="gray", linewidth=2, alpha=0.9
)  # equilibrium vertical (opaque gray)

axs[1].set_title("Demand and Supply")
axs[1].set_xlabel(r"City Population $L_C$")
axs[1].set_ylabel(r"$\Delta$ Price")
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].legend()

plt.tight_layout()
# plt.show()
fig.savefig("assets/full-model-example-eq.png")
# FULL MODEL

plt.clf()

# Parameters
capacity_city = 65
capacity_rural = 85
L = capacity_rural + 0.5 * ((capacity_city + capacity_rural) - capacity_rural)

# Boundaries
x_left = L - capacity_rural
x_right = capacity_city


# ΔP interior function
def delta_core(Lc):
    return (12 + 0.3 * Lc) - (8 + 0.2 * (L - Lc))


# Segments
VJUMP_RURAL = 25
VJUMP_CITY = 25

xs1 = np.linspace(x_left, x_right, 200)
ys1 = delta_core(xs1)
y_left = delta_core(x_left)
y_right = delta_core(x_right)

# Demand curve intersecting
x_star = 0.5 * (x_left + x_right)
y_star = delta_core(x_star)
b = 0.3
a = y_star + b * x_star


def demand(Lc):
    return a - b * Lc


xs_d = np.linspace(x_left - 10, L, 300)
ys_d = demand(xs_d)

# Figure with two panels
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# ---- LHS: Supply only ----

axs[0].plot(
    xs_d, ys_d, color="white", alpha=1.0, linestyle="--", linewidth=2
)  # this is just so plot are aligned on y-axis

axs[0].plot(xs1, ys1, color="red", linewidth=2, label="Supply")
axs[0].plot([x_left, x_left], [y_left, y_left - VJUMP_RURAL], color="red", linewidth=2)
axs[0].plot(
    [x_right, x_right], [y_right, y_right + VJUMP_CITY], color="red", linewidth=2
)
axs[0].axvline(L, linestyle="--", color="black", alpha=0.9, label="Total Population")
axs[0].set_title("Supply")
axs[0].set_xlabel(r"City Population $L_C$")
axs[0].set_ylabel(r"$\Delta$ Price")
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].legend()

# ---- RHS: Supply + Demand ----
axs[1].plot(xs1, ys1, color="red", linewidth=2, label="Supply")
axs[1].plot([x_left, x_left], [y_left, y_left - VJUMP_RURAL], color="red", linewidth=2)
axs[1].plot(
    [x_right, x_right], [y_right, y_right + VJUMP_CITY], color="red", linewidth=2
)
axs[1].plot(xs_d, ys_d, color="blue", linestyle="--", linewidth=2, label="Demand")
axs[1].scatter([x_star], [y_star], s=60, color="black", zorder=5)
axs[1].annotate(
    "Equilibrium",
    xy=(x_star, y_star),
    xytext=(x_star + 5, y_star - 8),
    fontsize=12,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", lw=1),
)
axs[1].axvline(L, linestyle="--", color="black", alpha=0.9, label="Total Population")
axs[1].set_title("Demand and Supply")
axs[1].set_xlabel(r"City Population $L_C$")
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].legend()

plt.tight_layout()
fig.savefig("assets/full-model-example.png")
# plt.show()
