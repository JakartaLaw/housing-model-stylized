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


def inverse_city_price(
    L_C, slope_C, intercept_C, Q_C, *, DELTA, HIGH_SUPPLY_VALUE
):
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
    slope_C, intercept_C, Q_C,
    slope_R, intercept_R, Q_R,
    *,
    HIGH_SUPPLY_VALUE=35.0,
    LOW_SUPPLY_VALUE=-35.0,
    EPS=None,              # width of the “vertical” column; default = one grid step
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
    out[interior] = (intercept_C + slope_C * LC[interior]) - \
                    (intercept_R + slope_R * LR[interior])

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
    show_supply: bool = True
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


def compute_top_curves(L_grid: np.ndarray, S: SupplyParams) -> Tuple[np.ndarray, np.ndarray]:
    """Upper panel: prices vs L_C."""
    P_C_vals = inverse_city_price(
        L_grid, S.sc, S.ic, S.Q_C,
        DELTA=S.DELTA, HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE
    )
    P_R_vals = inverse_rural_price_from_LC(
        L_grid, S.L_total, S.sr, S.ir, S.Q_R,
        DELTA=S.DELTA, HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE
    )
    return P_C_vals, P_R_vals

def compute_bottom_curves(L_grid: np.ndarray, D: DemandParams, S: SupplyParams) -> Tuple[np.ndarray, np.ndarray]:
    """Lower panel: ΔP demand & supply vs L_C."""
    demand_vals = inverse_demand_deltaP(L_grid, D.e_low, D.e_high, D.K_C, D.L_tilde, D.alpha)
    supply_vals = deltaP_supply(
        L_grid, S.L_total, S.sc, S.ic, S.Q_C, S.sr, S.ir, S.Q_R,
        EPS=S.DELTA, HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE, LOW_SUPPLY_VALUE=S.LOW_SUPPLY_VALUE
    )
    return demand_vals, supply_vals

def compute_equilibrium(L_grid: np.ndarray, D: DemandParams, S: SupplyParams) -> Optional[Tuple[float, float]]:
    """Find interior equilibrium from bottom curves."""
    demand_vals, supply_vals = compute_bottom_curves(L_grid, D, S)
    return find_equilibrium(L_grid, demand_vals, supply_vals)

def grid_tol(L_grid: np.ndarray) -> float:
    """Half the smallest positive grid step (0 if degenerate)."""
    d = np.diff(np.unique(L_grid))
    d = d[d > 0]
    return float(d.min() / 2.0) if d.size else 0.0


def top_eq_prices_from_bottom(
    eq: Optional[Tuple[float, float]],
    S: SupplyParams,
    *,
    tol: float = 0.0
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
    on_city_wall  = (LC >= S.Q_C - tol) and (LC <= S.Q_C + S.DELTA + tol)
    on_rural_wall = (LC >= LC_left - tol) and (LC <= LC_left + S.DELTA + tol)

    if on_city_wall and not on_rural_wall:
        # Anchor rural price from its piecewise rule, then enforce ΔP
        P_R = inverse_rural_price_from_LC(
            LC, S.L_total, S.sr, S.ir, S.Q_R,
            DELTA=S.DELTA, HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE
        )
        P_C = P_R + DP

    elif on_rural_wall and not on_city_wall:
        # Anchor city price from its piecewise rule, then enforce ΔP
        P_C = inverse_city_price(
            LC, S.sc, S.ic, S.Q_C,
            DELTA=S.DELTA, HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE
        )
        P_R = P_C - DP

    else:
        # Interior (or ambiguous): anchor on rural linear, enforce ΔP
        P_R = S.ir + S.sr * (S.L_total - LC)
        P_C = P_R + DP

    # Final sanity check (tolerate tiny FP error)
    assert np.isfinite(P_C) and np.isfinite(P_R)
    assert np.isclose((P_C - P_R), DP, rtol=1e-9, atol=1e-9), \
        f"{(P_C - P_R)} vs {DP}"

    return float(P_C), float(P_R)
# --- Plotter (always two panels) ---

def plot_scenarios(
    L_grid: np.ndarray,
    base_D: DemandParams,
    base_S: SupplyParams,
    scenarios: List[Scenario],
    *,
    figsize=(14, 12)
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
        if scen.show_supply:
            ax_price.plot(L_grid, P_C_vals, color=scen.color,
                          label=f"{scen.name}  $P_C$", **({"ls": "-"} | scen.supply_style))
            ax_price.plot(L_grid, P_R_vals, color=scen.color,
                          label=f"{scen.name}  $P_R$", **({"ls": "--"} | scen.supply_style))

        # --- Upper panel (EQ dots) — always plot, using ΔP on walls ---
        if eq is not None:
            prices = top_eq_prices_from_bottom(eq, S, tol=tol)
            if prices is not None:
                P_CITY_EQ, P_RURAL_EQ = prices
                LC_EQ, _ = eq
                ax_price.scatter([LC_EQ, LC_EQ], [P_CITY_EQ, P_RURAL_EQ],
                                 color=scen.color, s=55, zorder=4, label=f"{scen.name} EQ (prices)")

        # --- Lower panel ---
        if scen.show_demand:
            ax_delta.plot(L_grid, demand_vals, color=scen.color,
                          label=f"{scen.name} demand", **({"ls": "-"} | scen.demand_style))
        if scen.show_supply:
            ax_delta.plot(L_grid, supply_vals, color=scen.color,
                          label=f"{scen.name} supply", **({"ls": "-"} | scen.supply_style))
        if eq is not None:
            ax_delta.scatter([eq[0]], [eq[1]], color=scen.color, s=60, zorder=3, label=f"{scen.name} EQ")

    # cosmetics
    ax_price.set_ylabel("Price")
    ax_price.set_xlabel("$L_C$")
    ax_price.legend()

    ax_delta.set_ylabel("$\\Delta P = P_C - P_R$")
    ax_delta.set_xlabel("$L_C$")
    ax_delta.legend()

    fig.tight_layout()
    return fig, (ax_price, ax_delta), eq_points
# %% BELOW ARE PLOTS

L_grid = np.linspace(0, 100, 801)

base_D = DemandParams(e_low=-0.05, e_high=0.05, K_C=22.0, alpha=0.30, L_tilde=50.0)
base_S = SupplyParams(sc=0.25, ic=10.0, Q_C=70.0, sr=0.20, ir=8.0, Q_R=65.0, L_total=100, DELTA=0.2, HIGH_SUPPLY_VALUE=40)

scenarios = [
    Scenario(name="Baseline", color="blue", show_demand=True, show_supply=True),
    Scenario(name="Supply shock", color="red", show_demand=False, show_supply=True,
             supply_override={"ic": 7.0, "Q_C": 80.0}),
    Scenario(name="Demand shock", color="green", show_demand=True, show_supply=False,
             demand_override={"L_tilde": 2000.0}, supply_override={"ic": 7.0, "Q_C": 80.0}),
]

fig, (ax_price, ax_delta), eq_pts = plot_scenarios(L_grid, base_D, base_S, scenarios)

# %%
#
L_grid = np.linspace(0, 100, 801)

base_D = DemandParams(e_low=-0.05, e_high=0.05, K_C=12.0, alpha=0.30, L_tilde=50.0)
base_S = SupplyParams(sc=0.25, ic=10.0, Q_C=70.0, sr=0.20, ir=8.0, Q_R=65.0, L_total=100, DELTA=0.2)

scenarios = [
    Scenario(name="Baseline", color="blue", show_demand=True, show_supply=True),
    Scenario(name="Supply shock", color="red", show_demand=False, show_supply=True,
             supply_override={"ic": 7.0, "Q_C": 80.0}),
    Scenario(name="Demand shock", color="green", show_demand=True, show_supply=False,
             demand_override={"L_tilde": 2000.0}, supply_override={"ic": 7.0, "Q_C": 80.0}),
]

fig, (ax_price, ax_delta), eq_pts = plot_scenarios(L_grid, base_D, base_S, scenarios)
# %%
#
# from dataclasses import dataclass

@dataclass
class SubDemand:
    # Inverse demand for type i: ΔP_i(L_i) = A_i - m_i * L_i,  with 0 ≤ L_i ≤ cap_i
    A: float         # intercept (city benefit shift goes here)
    m: float         # slope (>0); equals (e_high - e_low)_i in your notation
    cap: float       # population / max city adoption for type i

def Li_of_DP(dp, A, m, cap):
    # L_i(ΔP) = (A - ΔP)/m, clipped
    Li = (A - dp) / m
    return np.clip(Li, 0.0, cap)

def aggregate_inverse_demand_two_types(LC, t1: SubDemand, t2: SubDemand):
    """
    Return ΔP_agg(L_C) for two linear sub-demands with caps, via piecewise closed-form.
    Vectorized over LC (scalar or array).
    """
    LC = np.asarray(LC, dtype=float)

    A1, m1, c1 = t1.A, t1.m, t1.cap
    A2, m2, c2 = t2.A, t2.m, t2.cap

    denom = (1.0/m1 + 1.0/m2)
    DP_both = (A1/m1 + A2/m2 - LC) / denom

    L1_b = (A1 - DP_both) / m1
    L2_b = (A2 - DP_both) / m2

    # Masks for interior validity
    both_ok = (L1_b >= 0) & (L1_b <= c1) & (L2_b >= 0) & (L2_b <= c2)

    # If type 1 would exceed cap -> cap1 binds; ΔP comes from type 2 on remaining L
    cap1 = L1_b > c1
    DP_cap1 = A2 - m2 * (LC - c1)

    # If type 2 would exceed cap -> cap2 binds
    cap2 = L2_b > c2
    DP_cap2 = A1 - m1 * (LC - c2)

    # If type 1 would be negative -> only type 2 participates
    neg1 = L1_b < 0
    DP_neg1 = A2 - m2 * LC

    # If type 2 would be negative -> only type 1 participates
    neg2 = L2_b < 0
    DP_neg2 = A1 - m1 * LC

    # Compose, prioritizing mutually exclusive regions; default to DP_both
    DP = DP_both.copy()
    DP[cap1] = DP_cap1[cap1]
    DP[cap2] = DP_cap2[cap2]
    DP[neg1] = DP_neg1[neg1]
    DP[neg2] = DP_neg2[neg2]
    # keep DP_both where both_ok; elsewhere above assignments override appropriately
    return DP if DP.shape != () else DP.item()

def subdemand_cumulative_curves(dp_grid: np.ndarray, t1: SubDemand, t2: SubDemand):
    """Given a ΔP grid, return (L1(ΔP), L1(ΔP)+L2(ΔP)) for plotting the stack."""
    L1 = Li_of_DP(dp_grid, t1.A, t1.m, t1.cap)
    L2 = Li_of_DP(dp_grid, t2.A, t2.m, t2.cap)
    Lcum1 = L1
    Lcum2 = L1 + L2
    return Lcum1, Lcum2

from dataclasses import dataclass, field

@dataclass
class Scenario:
    name: str
    color: str
    show_demand: bool = True
    show_supply: bool = True
    demand_override: dict = field(default_factory=dict)   # (kept for compatibility)
    supply_override: dict = field(default_factory=dict)   # SupplyParams overrides
    # NEW: per-scenario sub-demand overrides
    sub1_override: dict = field(default_factory=dict)     # fields of SubDemand, e.g. {"A": 30.0}
    sub2_override: dict = field(default_factory=dict)
    demand_style: dict = field(default_factory=dict)
    supply_style: dict = field(default_factory=dict)

def plot_scenarios_with_subdemand(
    L_grid: np.ndarray,
    base_D: DemandParams,
    base_S: SupplyParams,
    scenarios: List[Scenario],
    *,
    sub1_base: SubDemand,     # base type-1 sub-demand
    sub2_base: SubDemand,     # base type-2 sub-demand
    figsize=(14, 16),
):
    """
    Panels:
      (1) Prices vs L_C
      (2) ΔP vs L_C (AGGREGATE demand from sub-demands + supply) + EQ
      (3) Sub-demand decomposition (direct demand)
    """

    fig = plt.figure(figsize=(14, 12))

    # GridSpec with 2 rows, 2 cols — but bottom spans both cols
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    ax_sub   = fig.add_subplot(gs[0, 0])  # top-left: sub-demands
    ax_delta = fig.add_subplot(gs[0, 1])  # top-right: aggregate ΔP
    ax_price = fig.add_subplot(gs[1, :])  # bottom row, spanning both columns

    #ax_price, ax_delta, ax_sub = axes
    ax_sub, ax_delta, ax_price = axes   # <— new order!
    ax_price.set_title("Upper panel: Prices vs $L_C$")
    ax_delta.set_title("Middle panel: Aggregate $\\Delta P$ vs $L_C$")
    ax_sub.set_title("Lower panel: Sub-demand decomposition (direct demand)")

    # small tolerance for “near wall”
    dLC = np.diff(np.unique(L_grid)); dLC = dLC[dLC > 0]
    tol = (dLC.min()/2.0) if dLC.size else 0.0

    eq_points: Dict[str, Optional[Tuple[float, float]]] = {}

    # ΔP grid for decomposition
    dp_min = min(sub1_base.A - sub1_base.m*sub1_base.cap,
                 sub2_base.A - sub2_base.m*sub2_base.cap) - 1.0
    dp_max = max(sub1_base.A, sub2_base.A) + 1.0
    dp_grid = np.linspace(dp_min, dp_max, 400)

    for scen in scenarios:
        # supply (can be overridden, but for single demand shock we’ll leave it at base)
        S = replace(base_S, **scen.supply_override)

        # apply per-scenario sub-demand overrides
        sub1 = replace(sub1_base, **scen.sub1_override)
        sub2 = replace(sub2_base, **scen.sub2_override)

        # --- Top panel (supply curves) ---
        P_C_vals, P_R_vals = compute_top_curves(L_grid, S)
        if scen.show_supply:
            ax_price.plot(L_grid, P_C_vals, color=scen.color,
                          label=f"{scen.name}  $P_C$", **({"ls": "-"} | scen.supply_style))
            ax_price.plot(L_grid, P_R_vals, color=scen.color,
                          label=f"{scen.name}  $P_R$", **({"ls": "--"} | scen.supply_style))

        # --- Middle panel: aggregate ΔP from the two sub-demands + supply ---
        dP_agg = aggregate_inverse_demand_two_types(L_grid, sub1, sub2)
        s_vals = deltaP_supply(
            L_grid, S.L_total, S.sc, S.ic, S.Q_C, S.sr, S.ir, S.Q_R,
            EPS=S.DELTA, HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE,
            LOW_SUPPLY_VALUE=S.LOW_SUPPLY_VALUE
        )
        if scen.show_demand:
            ax_delta.plot(L_grid, dP_agg, color=scen.color,
                          label=f"{scen.name} demand (agg)", **({"ls": "-"} | scen.demand_style))
        if scen.show_supply:
            ax_delta.plot(L_grid, s_vals, color=scen.color,
                          label=f"{scen.name} supply", **({"ls": "-"} | scen.supply_style))

        # equilibrium (aggregate)
        eq = find_equilibrium(L_grid, dP_agg, s_vals)
        eq_points[scen.name] = eq

        # plot EQ markers
        if eq is not None:
            LC_EQ, DP_EQ = eq
            ax_delta.scatter([LC_EQ], [DP_EQ], color=scen.color, s=60, zorder=3, label=f"{scen.name} EQ")

            # top-panel EQ prices from bottom ΔP (anchor one side, enforce ΔP)
            P_R_EQ = inverse_rural_price_from_LC(
                LC_EQ, S.L_total, S.sr, S.ir, S.Q_R,
                DELTA=S.DELTA, HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE
            )
            if np.isfinite(P_R_EQ):
                P_C_EQ = P_R_EQ + DP_EQ
            else:
                P_C_EQ = inverse_city_price(
                    LC_EQ, S.sc, S.ic, S.Q_C, DELTA=S.DELTA, HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE
                )
                P_R_EQ = P_C_EQ - DP_EQ

            ax_price.scatter([LC_EQ, LC_EQ], [P_C_EQ, P_R_EQ],
                             color=scen.color, s=55, zorder=4, label=f"{scen.name} EQ (prices)")

        # --- Bottom panel: sub-demand decomposition (direct demand stack) ---
        Lcum1, Lcum2 = subdemand_cumulative_curves(dp_grid, sub1, sub2)
        ax_sub.plot(Lcum1, dp_grid, color=scen.color, alpha=0.65, ls="--",
                    label=f"{scen.name}: type 1")
        ax_sub.plot(Lcum2, dp_grid, color=scen.color, ls="-",
                    label=f"{scen.name}: type 1 + 2")

    # cosmetics
    ax_price.set_ylabel("Price"); ax_price.set_xlabel("$L_C$"); ax_price.legend()
    ax_delta.set_ylabel("$\\Delta P$ (city − rural)"); ax_delta.set_xlabel("$L_C$"); ax_delta.legend()
    ax_sub.set_xlabel("$L_C$ (cumulative)"); ax_sub.set_ylabel("$\\Delta P$"); ax_sub.legend()
    fig.tight_layout()
    return fig, (ax_price, ax_delta, ax_sub), eq_points
# Base sub-demands
sub1_base = SubDemand(A=28.0, m=0.20, cap=55.0)   # type 1
sub2_base = SubDemand(A=29.0, m=0.20, cap=45.0)   # type 2

# Benefit for type 2 (pure demand shock)
benefit = 3.0

scenarios = [
    Scenario(name="Baseline", color="blue", show_demand=True, show_supply=True),
    Scenario(name="Demand shock", color="green", show_demand=True, show_supply=True,
             sub2_override={"A": sub2_base.A + benefit}),   # supply unchanged
]
fig, axes, eq_pts = plot_scenarios_with_subdemand(
    L_grid, base_D, base_S, scenarios,
    sub1_base=sub1_base, sub2_base=sub2_base,
    figsize=(14, 16)
)
print(eq_pts)
