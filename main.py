# %%
print("hey lars")
# %% Imports

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Optional


def inverse_demand_deltaP(L_C, e_low, e_high, K_C, L_tilde, alpha):
    """
    Inverse demand for relative price ΔP = P_C - P_R as a function of L_C.
    ΔP(L_C) = e_high + (K_C + L_tilde^alpha) - (e_high - e_low) * L_C
    """
    B = K_C + (L_tilde**alpha)
    return e_high + B - (e_high - e_low) * L_C


def inverse_city_price(L_C, slope_C, intercept_C, Q_C, *, DELTA, HIGH_SUPPLY_VALUE):
    LC = np.atleast_1d(np.asarray(L_C))
    QC = np.atleast_1d(np.asarray(Q_C))

    out = np.full(np.broadcast(LC, QC).shape, np.nan, dtype=float)

    mask1 = LC <= QC
    mask2 = (LC > QC) & (LC <= QC + DELTA)

    out[mask1] = intercept_C + slope_C * LC[mask1]
    out[mask2] = HIGH_SUPPLY_VALUE

    return out[0] if out.size == 1 else out


def inverse_rural_price_from_LC(
    L_C, L_total, slope_R, intercept_R, Q_R, *, DELTA, HIGH_SUPPLY_VALUE
):
    LC = np.atleast_1d(np.asarray(L_C))
    QR = np.atleast_1d(np.asarray(Q_R))

    LR = L_total - LC
    out = np.full(np.broadcast(LR, QR).shape, np.nan, dtype=float)

    mask1 = LR <= QR
    mask2 = (LR > QR) & (LR <= QR + DELTA)

    out[mask1] = intercept_R + slope_R * LR[mask1]
    out[mask2] = HIGH_SUPPLY_VALUE

    return out[0] if out.size == 1 else out


LOW_SUPPLY_VALUE_DEFAULT = 5.0  # example


def deltaP_supply(
    L_C,
    L_total,
    slope_C,
    intercept_C,
    Q_C,
    slope_R,
    intercept_R,
    Q_R,
    *,
    HIGH_SUPPLY_VALUE=35.0,
    LOW_SUPPLY_VALUE=-35.0,
    EPS=None,  # width of the “vertical” column; default = one grid step
):
    """
    ΔP_s(L_C) = P_C(L_C) - P_R(L_total - L_C)

    Shape:
      - Left vertical at L_C = L_total - Q_R (rural cap)
      - Upward sloped interior for LC ≤ Q_C and LR ≤ Q_R
      - Right vertical at L_C = Q_C (city cap)
      - NaN elsewhere

    EPS controls how many grid points get the vertical column (default: 1 step).
    """
    LC = np.asarray(L_C, dtype=float)
    LR = L_total - LC

    out = np.full(LC.shape, np.nan, dtype=float)

    # infer grid step for vertical "column" width if not given
    if EPS is None and LC.size >= 2:
        dLC = np.diff(np.unique(LC))
        dLC = dLC[dLC > 0]
        EPS = dLC.min() if dLC.size else 0.0
    elif EPS is None:
        EPS = 0.0

    # interior (both within capacity)
    interior = (LC <= Q_C) & (LR <= Q_R)
    out[interior] = (intercept_C + slope_C * LC[interior]) - (
        intercept_R + slope_R * LR[interior]
    )

    # left vertical (rural binds): around LC = L_total - Q_R
    LC_left = L_total - Q_R
    left_col = (LC >= LC_left) & (LC <= LC_left + EPS)
    out[left_col] = LOW_SUPPLY_VALUE

    # right vertical (city binds): around LC = Q_C
    right_col = (LC >= Q_C) & (LC <= Q_C + EPS)
    out[right_col] = HIGH_SUPPLY_VALUE

    return out[0] if out.size == 1 else out


def find_equilibrium(L_grid, dP_demand, dP_supply):
    """
    Find an interior equilibrium as the intersection of ΔP demand and ΔP supply curves.
    Returns (L_C_star, ΔP_star). Falls back to nearest point if no sign change is found.
    """
    valid = ~np.isnan(dP_supply)
    if not np.any(valid):
        return None
    L = L_grid[valid]
    y = dP_demand[valid] - dP_supply[valid]

    # Look for sign change
    sgn = np.sign(y)
    idx = np.where(np.diff(sgn) != 0)[0]
    if idx.size == 0:
        i = np.argmin(np.abs(y))
        return float(L[i]), float(dP_demand[valid][i])

    i0 = idx[0]
    x0, x1 = L[i0], L[i0 + 1]
    y0, y1 = y[i0], y[i0 + 1]
    # Linear interpolation
    L_star = x0 - y0 * (x1 - x0) / (y1 - y0)
    dP_star = float(
        np.interp(L_star, [x0, x1], [dP_demand[valid][i0], dP_demand[valid][i0 + 1]])
    )
    return float(L_star), dP_star


@dataclass
class DemandParams:
    e_low: float
    e_high: float
    K_C: float
    alpha: float
    L_tilde: float


@dataclass
class SupplyParams:
    sc: float
    ic: float
    Q_C: float
    sr: float
    ir: float
    Q_R: float
    L_total: float
    DELTA: float = 10.0
    HIGH_SUPPLY_VALUE: float = 35.0
    LOW_SUPPLY_VALUE: float = -35.0


@dataclass
class Scenario:
    name: str
    color: str
    show_demand: bool = True
    show_supply: bool | str = True
    demand_override: dict = field(default_factory=dict)  # keys of DemandParams
    supply_override: dict = field(default_factory=dict)  # keys of SupplyParams
    demand_style: dict = field(default_factory=lambda: {})  # e.g. dict(ls='-')
    supply_style: dict = field(default_factory=lambda: {})  # e.g. dict(ls='-')


def merge_dataclass(base, override: Dict):
    return replace(base, **override)


# --- Compute functions (top / bottom / eq) ---


def prices_at_x(LC: float, S: SupplyParams, *, tol: float = 0.0):
    """
    Piecewise-consistent prices at a given LC, with snapping to walls within `tol`.

    City wall  at LC in (Q_C, Q_C + DELTA]  -> P_C = HIGH_SUPPLY_VALUE
    Rural wall at LR in (Q_R, Q_R + DELTA]  -> P_R = HIGH_SUPPLY_VALUE
    Interior elsewhere; NaN if outside both interior and walls.
    """
    # --- snap LC to the city wall if very close
    if abs(LC - S.Q_C) <= tol:
        LC = S.Q_C

    # City side (price as function of LC)
    if LC <= S.Q_C:
        P_C = S.ic + S.sc * LC
    elif LC <= S.Q_C + S.DELTA:
        P_C = S.HIGH_SUPPLY_VALUE
    else:
        P_C = np.nan

    # Rural side works on L_R = L_total - LC; snap if close to wall
    LR = S.L_total - LC
    if abs(LR - S.Q_R) <= tol:
        LR = S.Q_R

    if LR <= S.Q_R:
        P_R = S.ir + S.sr * LR
    elif LR <= S.Q_R + S.DELTA:
        P_R = S.HIGH_SUPPLY_VALUE
    else:
        P_R = np.nan

    return P_C, P_R


def compute_top_curves(
    L_grid: np.ndarray, S: SupplyParams
) -> Tuple[np.ndarray, np.ndarray]:
    """Upper panel: prices vs L_C."""
    P_C_vals = inverse_city_price(
        L_grid, S.sc, S.ic, S.Q_C, DELTA=S.DELTA, HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE
    )
    P_R_vals = inverse_rural_price_from_LC(
        L_grid,
        S.L_total,
        S.sr,
        S.ir,
        S.Q_R,
        DELTA=S.DELTA,
        HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE,
    )
    return P_C_vals, P_R_vals


def compute_bottom_curves(
    L_grid: np.ndarray, D: DemandParams, S: SupplyParams
) -> Tuple[np.ndarray, np.ndarray]:
    """Lower panel: ΔP demand & supply vs L_C."""
    demand_vals = inverse_demand_deltaP(
        L_grid, D.e_low, D.e_high, D.K_C, D.L_tilde, D.alpha
    )
    supply_vals = deltaP_supply(
        L_grid,
        S.L_total,
        S.sc,
        S.ic,
        S.Q_C,
        S.sr,
        S.ir,
        S.Q_R,
        EPS=S.DELTA,
        HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE,
        LOW_SUPPLY_VALUE=S.LOW_SUPPLY_VALUE,
    )
    return demand_vals, supply_vals


def compute_equilibrium(
    L_grid: np.ndarray, D: DemandParams, S: SupplyParams
) -> Optional[Tuple[float, float]]:
    """Find interior equilibrium from bottom curves."""
    demand_vals, supply_vals = compute_bottom_curves(L_grid, D, S)
    return find_equilibrium(L_grid, demand_vals, supply_vals)


def grid_tol(L_grid: np.ndarray) -> float:
    """Half the smallest positive grid step (0 if degenerate)."""
    d = np.diff(np.unique(L_grid))
    d = d[d > 0]
    return float(d.min() / 2.0) if d.size else 0.0


def top_eq_prices_from_bottom(
    eq: Optional[Tuple[float, float]], S: SupplyParams, *, tol: float = 0.0
) -> Optional[Tuple[float, float]]:
    """
    Given (LC*, ΔP*) from the lower panel and supply params S,
    return (P_C*, P_R*) for the upper panel.

    Strategy:
      • If LC* is near/at the right (city) wall: anchor P_R from its piecewise rule,
        then set P_C = P_R + ΔP*.
      • If LC* is near/at the left (rural) wall: anchor P_C from its piecewise rule,
        then set P_R = P_C - ΔP*.
      • Otherwise (interior): anchor P_R on its linear rule and set P_C = P_R + ΔP*.

    This guarantees P_C - P_R == ΔP* (up to float noise).
    """
    if eq is None:
        return None

    LC, DP = float(eq[0]), float(eq[1])
    LC_left = S.L_total - S.Q_R

    # Which side is binding / should we anchor on?
    on_city_wall = (LC >= S.Q_C - tol) and (LC <= S.Q_C + S.DELTA + tol)
    on_rural_wall = (LC >= LC_left - tol) and (LC <= LC_left + S.DELTA + tol)

    if on_city_wall and not on_rural_wall:
        # Anchor rural price from its piecewise rule, then enforce ΔP
        P_R = inverse_rural_price_from_LC(
            LC,
            S.L_total,
            S.sr,
            S.ir,
            S.Q_R,
            DELTA=S.DELTA,
            HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE,
        )
        P_C = P_R + DP

    elif on_rural_wall and not on_city_wall:
        # Anchor city price from its piecewise rule, then enforce ΔP
        P_C = inverse_city_price(
            LC, S.sc, S.ic, S.Q_C, DELTA=S.DELTA, HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE
        )
        P_R = P_C - DP

    else:
        # Interior (or ambiguous): anchor on rural linear, enforce ΔP
        P_R = S.ir + S.sr * (S.L_total - LC)
        P_C = P_R + DP

    # Final sanity check (tolerate tiny FP error)
    assert np.isfinite(P_C) and np.isfinite(P_R)
    assert np.isclose((P_C - P_R), DP, rtol=1e-9, atol=1e-9), f"{(P_C - P_R)} vs {DP}"

    return float(P_C), float(P_R)


# --- Plotter (always two panels) ---


def plot_scenarios(
    L_grid: np.ndarray,
    base_D: DemandParams,
    base_S: SupplyParams,
    scenarios: List[Scenario],
    *,
    figsize=(10, 12),
):
    """
    Always shows both panels.
    Returns (fig, (ax_price, ax_delta), eq_points)
    """
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax_price, ax_delta = ax[0], ax[1]
    ax_price.set_title("Upper panel: Prices vs $L_C$")
    ax_delta.set_title("Lower panel: Relative price vs $L_C$")

    tol = grid_tol(L_grid)  # for near-wall detection
    eq_points: Dict[str, Optional[Tuple[float, float]]] = {}

    for scen in scenarios:
        # apply overrides
        D = replace(base_D, **scen.demand_override)
        S = replace(base_S, **scen.supply_override)

        # compute curves
        P_C_vals, P_R_vals = compute_top_curves(L_grid, S)
        demand_vals, supply_vals = compute_bottom_curves(L_grid, D, S)
        eq = find_equilibrium(L_grid, demand_vals, supply_vals)
        eq_points[scen.name] = eq

        # --- Upper panel (curves) ---
        if scen.show_supply in (True, "city", "rural"):
            if scen.show_supply in (True, "city"):
                ax_price.plot(
                    L_grid,
                    P_C_vals,
                    color=scen.color,
                    label=f"{scen.name}  $P_C$",
                    **({"ls": "-"} | scen.supply_style),
                )
            if scen.show_supply in (True, "rural"):
                ax_price.plot(
                    L_grid,
                    P_R_vals,
                    color=scen.color,
                    label=f"{scen.name}  $P_R$",
                    **({"ls": "--"} | scen.supply_style),
                )

        # --- Upper panel (EQ dots) — always plot, using ΔP on walls ---
        if eq is not None:
            prices = top_eq_prices_from_bottom(eq, S, tol=tol)
            if prices is not None:
                P_CITY_EQ, P_RURAL_EQ = prices
                LC_EQ, _ = eq
                ax_price.scatter(
                    [LC_EQ, LC_EQ],
                    [P_CITY_EQ, P_RURAL_EQ],
                    color=scen.color,
                    s=55,
                    zorder=4,
                    label=f"{scen.name} EQ (prices)",
                )

        # --- Lower panel ---
        if scen.show_demand:
            ax_delta.plot(
                L_grid,
                demand_vals,
                color=scen.color,
                label=f"{scen.name} demand",
                **({"ls": "--"} | scen.demand_style),
            )
        if scen.show_supply:
            ax_delta.plot(
                L_grid,
                supply_vals,
                color=scen.color,
                label=f"{scen.name} supply",
                **({"ls": "-"} | scen.supply_style),
            )
        if eq is not None:
            ax_delta.scatter(
                [eq[0]],
                [eq[1]],
                color=scen.color,
                s=60,
                zorder=3,
                label=f"{scen.name} EQ",
            )

    # cosmetics
    ax_price.set_ylabel("Price")
    ax_price.set_xlabel("$L_C$")
    ax_price.legend()

    ax_delta.set_ylabel("$\\Delta P = P_C - P_R$")
    ax_delta.set_xlabel("$L_C$")
    ax_delta.legend()

    fig.tight_layout()
    return fig, (ax_price, ax_delta), eq_points


# Scenario of supply shock, where we do not hit capacity in city

L_grid = np.linspace(0, 100, 801)

base_D = DemandParams(e_low=-0.05, e_high=0.05, K_C=12.0, alpha=0.30, L_tilde=50.0)
base_S = SupplyParams(
    sc=0.25, ic=10.0, Q_C=70.0, sr=0.20, ir=8.0, Q_R=65.0, L_total=100, DELTA=0.2
)

scenarios = [
    Scenario(name="Baseline", color="blue", show_demand=True, show_supply=True),
]

fig, (ax_price, ax_delta), eq_pts = plot_scenarios(L_grid, base_D, base_S, scenarios)
fig.savefig("assets/supply-shock-not-at-capacity-eq1.png")

# EQ 2
scenarios = [
    Scenario(name="Baseline", color="blue", show_demand=True, show_supply=True),
    Scenario(
        name="Supply shock",
        color="red",
        show_demand=False,
        show_supply="city",
        supply_override={"ic": 7.0, "Q_C": 80.0},
    ),
]

fig, (ax_price, ax_delta), eq_pts = plot_scenarios(L_grid, base_D, base_S, scenarios)
fig.savefig("assets/supply-shock-not-at-capacity-eq2.png")

# EQ 3
scenarios = [
    Scenario(name="Baseline", color="blue", show_demand=True, show_supply=True),
    Scenario(
        name="Supply shock",
        color="red",
        show_demand=False,
        show_supply="city",
        supply_override={"ic": 7.0, "Q_C": 80.0},
    ),
    Scenario(
        name="Demand shock",
        color="green",
        show_demand=True,
        show_supply=False,
        demand_override={"L_tilde": 2000.0},
        supply_override={"ic": 7.0, "Q_C": 80.0},
    ),
]

fig, (ax_price, ax_delta), eq_pts = plot_scenarios(L_grid, base_D, base_S, scenarios)
fig.savefig("assets/supply-shock-not-at-capacity-eq3.png")


# # %% BELOW ARE PLOTS
#
# L_grid = np.linspace(0, 100, 801)
#
# base_D = DemandParams(e_low=-0.05, e_high=0.05, K_C=22.0, alpha=0.30, L_tilde=50.0)
# base_S = SupplyParams(
#     sc=0.25,
#     ic=10.0,
#     Q_C=70.0,
#     sr=0.20,
#     ir=8.0,
#     Q_R=65.0,
#     L_total=100,
#     DELTA=0.2,
#     HIGH_SUPPLY_VALUE=40,
# )
#
# scenarios = [
#     Scenario(name="Baseline", color="blue", show_demand=True, show_supply=True),
#     Scenario(
#         name="Supply shock",
#         color="red",
#         show_demand=False,
#         show_supply=True,
#         supply_override={"ic": 7.0, "Q_C": 80.0},
#     ),
#     Scenario(
#         name="Demand shock",
#         color="green",
#         show_demand=True,
#         show_supply=False,
#         demand_override={"L_tilde": 2000.0},
#         supply_override={"ic": 7.0, "Q_C": 80.0},
#     ),
# ]
#
# fig, (ax_price, ax_delta), eq_pts = plot_scenarios(L_grid, base_D, base_S, scenarios)
# fig.savefig("assets/supply-shock1.pdf")
# %%
