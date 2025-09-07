# %%
#
# from dataclasses import dataclass


@dataclass
class SubDemand:
    # Inverse demand for type i: ΔP_i(L_i) = A_i - m_i * L_i,  with 0 ≤ L_i ≤ cap_i
    A: float  # intercept (city benefit shift goes here)
    m: float  # slope (>0); equals (e_high - e_low)_i in your notation
    cap: float  # population / max city adoption for type i


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

    denom = 1.0 / m1 + 1.0 / m2
    DP_both = (A1 / m1 + A2 / m2 - LC) / denom

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
    demand_override: dict = field(default_factory=dict)  # (kept for compatibility)
    supply_override: dict = field(default_factory=dict)  # SupplyParams overrides
    # NEW: per-scenario sub-demand overrides
    sub1_override: dict = field(
        default_factory=dict
    )  # fields of SubDemand, e.g. {"A": 30.0}
    sub2_override: dict = field(default_factory=dict)
    demand_style: dict = field(default_factory=dict)
    supply_style: dict = field(default_factory=dict)


def plot_scenarios_with_subdemand(
    L_grid: np.ndarray,
    base_D: DemandParams,
    base_S: SupplyParams,
    scenarios: List[Scenario],
    *,
    sub1_base: SubDemand,  # base type-1 sub-demand
    sub2_base: SubDemand,  # base type-2 sub-demand
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

    ax_sub = fig.add_subplot(gs[0, 0])  # top-left: sub-demands
    ax_delta = fig.add_subplot(gs[0, 1])  # top-right: aggregate ΔP
    ax_price = fig.add_subplot(gs[1, :])  # bottom row, spanning both columns

    # ax_price, ax_delta, ax_sub = axes
    ax_sub, ax_delta, ax_price = axes  # <— new order!
    ax_price.set_title("Upper panel: Prices vs $L_C$")
    ax_delta.set_title("Middle panel: Aggregate $\\Delta P$ vs $L_C$")
    ax_sub.set_title("Lower panel: Sub-demand decomposition (direct demand)")

    # small tolerance for “near wall”
    dLC = np.diff(np.unique(L_grid))
    dLC = dLC[dLC > 0]
    tol = (dLC.min() / 2.0) if dLC.size else 0.0

    eq_points: Dict[str, Optional[Tuple[float, float]]] = {}

    # ΔP grid for decomposition
    dp_min = (
        min(
            sub1_base.A - sub1_base.m * sub1_base.cap,
            sub2_base.A - sub2_base.m * sub2_base.cap,
        )
        - 1.0
    )
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
            ax_price.plot(
                L_grid,
                P_C_vals,
                color=scen.color,
                label=f"{scen.name}  $P_C$",
                **({"ls": "-"} | scen.supply_style),
            )
            ax_price.plot(
                L_grid,
                P_R_vals,
                color=scen.color,
                label=f"{scen.name}  $P_R$",
                **({"ls": "--"} | scen.supply_style),
            )

        # --- Middle panel: aggregate ΔP from the two sub-demands + supply ---
        dP_agg = aggregate_inverse_demand_two_types(L_grid, sub1, sub2)
        s_vals = deltaP_supply(
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
        if scen.show_demand:
            ax_delta.plot(
                L_grid,
                dP_agg,
                color=scen.color,
                label=f"{scen.name} demand (agg)",
                **({"ls": "-"} | scen.demand_style),
            )
        if scen.show_supply:
            ax_delta.plot(
                L_grid,
                s_vals,
                color=scen.color,
                label=f"{scen.name} supply",
                **({"ls": "-"} | scen.supply_style),
            )

        # equilibrium (aggregate)
        eq = find_equilibrium(L_grid, dP_agg, s_vals)
        eq_points[scen.name] = eq

        # plot EQ markers
        if eq is not None:
            LC_EQ, DP_EQ = eq
            ax_delta.scatter(
                [LC_EQ],
                [DP_EQ],
                color=scen.color,
                s=60,
                zorder=3,
                label=f"{scen.name} EQ",
            )

            # top-panel EQ prices from bottom ΔP (anchor one side, enforce ΔP)
            P_R_EQ = inverse_rural_price_from_LC(
                LC_EQ,
                S.L_total,
                S.sr,
                S.ir,
                S.Q_R,
                DELTA=S.DELTA,
                HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE,
            )
            if np.isfinite(P_R_EQ):
                P_C_EQ = P_R_EQ + DP_EQ
            else:
                P_C_EQ = inverse_city_price(
                    LC_EQ,
                    S.sc,
                    S.ic,
                    S.Q_C,
                    DELTA=S.DELTA,
                    HIGH_SUPPLY_VALUE=S.HIGH_SUPPLY_VALUE,
                )
                P_R_EQ = P_C_EQ - DP_EQ

            ax_price.scatter(
                [LC_EQ, LC_EQ],
                [P_C_EQ, P_R_EQ],
                color=scen.color,
                s=55,
                zorder=4,
                label=f"{scen.name} EQ (prices)",
            )

        # --- Bottom panel: sub-demand decomposition (direct demand stack) ---
        Lcum1, Lcum2 = subdemand_cumulative_curves(dp_grid, sub1, sub2)
        ax_sub.plot(
            Lcum1,
            dp_grid,
            color=scen.color,
            alpha=0.65,
            ls="--",
            label=f"{scen.name}: type 1",
        )
        ax_sub.plot(
            Lcum2, dp_grid, color=scen.color, ls="-", label=f"{scen.name}: type 1 + 2"
        )

    # cosmetics
    ax_price.set_ylabel("Price")
    ax_price.set_xlabel("$L_C$")
    ax_price.legend()
    ax_delta.set_ylabel("$\\Delta P$ (city − rural)")
    ax_delta.set_xlabel("$L_C$")
    ax_delta.legend()
    ax_sub.set_xlabel("$L_C$ (cumulative)")
    ax_sub.set_ylabel("$\\Delta P$")
    ax_sub.legend()
    fig.tight_layout()
    return fig, (ax_price, ax_delta, ax_sub), eq_points


# Base sub-demands
sub1_base = SubDemand(A=28.0, m=0.20, cap=55.0)  # type 1
sub2_base = SubDemand(A=29.0, m=0.20, cap=45.0)  # type 2

# Benefit for type 2 (pure demand shock)
benefit = 3.0

scenarios = [
    Scenario(name="Baseline", color="blue", show_demand=True, show_supply=True),
    Scenario(
        name="Demand shock",
        color="green",
        show_demand=True,
        show_supply=True,
        sub2_override={"A": sub2_base.A + benefit},
    ),  # supply unchanged
]
fig, axes, eq_pts = plot_scenarios_with_subdemand(
    L_grid,
    base_D,
    base_S,
    scenarios,
    sub1_base=sub1_base,
    sub2_base=sub2_base,
    figsize=(14, 16),
)
print(eq_pts)
