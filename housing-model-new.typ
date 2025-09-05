= A stylized model of housing and agglomoration

== Introduction

This model aim to present a stylized model that allows users to easily reason about housing policy. Especially, in a time where YIMBY/NIMBY debates have become ever more present.

== The Static Model

We consider a world where there exist to types of dwellings; city and rural. They differ by the fact that city have agglomoration effects, which can be important to consider in when considering housing policies. We denote city and rural, $C$ and $R$ respectively. We additionally, consider that there already is a stock of housing $Q_R, Q_C$ which is fixed (in the short run at least). There are $L$ households in the economy, that can either live in the city or in rural areas, denoted by $L_R, L_C$, and we easily conclude that $L = L_R + L_C$, and trivially we assume the stock oh housing exceeds the number of individuals $L < Q_R + Q_C$.

We can consider the utility functions of the individual households (which we for simplicity lets correspond to their income). The income for family I depends on whether they live in the city or in a rural area:

$
  Y_(C i) &= underbrace(K_C + tilde(L)_C^(alpha) + e_i, W_C)- P_C \
  Y_(R i) &= W_R - P_R
$

I.e. the utility is the wage $(W)$ minus the price housing. $K_C$ is a constant, $tilde(L)_C$ is the agglomoration effect and $e_i$ is just some small wage perturbance, so the aggregated demand slopes down. In other words, in this formulation, since we are mostly interested in housing in the cities, capture the relevant dynamics. $P_R, P_C$ is the price of housing in rural areas and cities respectively. We are interested in how this clears the economy. Finally a note about $tilde(L)_C$ this can be thought of households in the city last period (myopic agents), or the expected number of households in the city (shared by all agents). Just note, that $L_C$ and $tilde(L)_C$ does not represent the same. This model yields $P_C, P_R, L_C, L_R$ given $tilde(L)_C$ (and the other parameters).

We consider the agents to be utility maximizing and they chose the city if $Y_C > Y_R$ and vice versa. Using this we can derive the aggregate demand (for simplicty we let $e tilde "Uniform"(underline(e), overline(e))$, such that the demand curve is linear. We can derive an expression for the aggregate demand for city housing as function of relative prices *maybe better expressed as $Y_C > Y_R$*:

$
  Y_C - Y_R = \ K_C + tilde(L)_C^alpha + e_i - P_C - (W_R -P_R) &>= 0 \
  => e_i &>= Delta P + B( tilde(L)_C)
$

Where $Delta P equiv P_C - P_R$, and $B$ is just a generic function of $tilde(L)_C$, i.e. $K_C + tilde(L)_C^alpha $. We use that expression of the density for uniform distributions and write:

$ L_C = D_C ( Delta P) =( overline(e) + B(tilde(L)) - Delta P ) / ( overline(e) - underline(e) ) $

With the inverse demand:

$ Delta P = overline(e) + B(tilde(L)) - L_C (overline(e) - underline(e)) $

We also need to consider the supply side of the houseing. Since capital in the short run is fixed, we could think of the supply to be mirrored "L" shaped, however, we model it a bit more flexible as:

$
  S_C = min(a_C + b_C L_C, Q_C) \
  S_R = min(a_R + b_R L_R, Q_R)
$

This is to say the supply is upward sloping (with slope $b$), and vertical when reaching the maximum $Q$.

We can plot these separately (see first plot), but it makes reasoning easier when we plot them again eachother (next plot). If we consider that $L = L_C + L_R => L_C = L - L_R$. In other words we can plot the two supply functions on the same axis but opposite directions $L_C$ as we normally would while $L_R$ is mirrored. We now the total length of the x-axis is $L$ since that is the total number of households in the economy. The flipped supply $underline(S)_R = max(L - a_r - b_r (L - L_C), L - Q_R)) $.

Underneath this plot we draw a second plot, which corresponds to $Z (L_C) equiv S_C - underline(S)_R$ and $D_C (L_c)$, with $Delta P$ on the $Y$-axis and $L_C$ on the $X$-axis.

These two plot allow us to reason about demand and supply and how supply shocks translate into other prices and even longer run adjustments through agglomoration.

#figure(
  image("assets/supply-demand-fig1.png", width: 100%),
  caption: [
    Supply and Demand example
  ],
)

How to read the figure

What’s on each axis

- Horizontal (both panels): the number of households living in the city. Moving right means more people in the city and fewer in rural areas.
- Top panel (Prices): the city housing price curve and the rural housing price curve, both drawn against city population so you can compare them on one axis.
- Bottom panel (Relative price): the price gap between city and rural (city price minus rural price). Two curves live here: a demand curve and a supply-implied curve. **Demand slopes down**, supply slopes up.

The example above shows how a very often considered thought experiment of building more housing in the city. Before going into this specific example, let us summarize the reasoning:

1. We begin in EQ 1 where we have a price on houses in both rural and city areas.
2. Policy makers increase the supply of housing in the cities, which starts an adjustment of the supply curve (from blue to red), which is both reflected in top and bottom plot. Basically we see (in this instance), the vertical right most line moves right and the upward sloped full (not dashed) line moves down. A classic example of increased supply. In the lower panel this is also reflected in $Z$-curve that captures the supply as expressed only by the number of households that end in the city $L_C$. We are now in EQ 2. which imply more people are now living in the city, fewer are living in rural areas, and the price of housing in both city and rural areas have decreased.
3. Due to more people living ijn the city, agglomoration effects begin to kick in, and demand adjusts (long term adjustment). This is reflected in the green demand curve moving up/right in the bottom figure. Which implies even more people are now living in the city, which is reflected in prices in the city now increasing, while the prices in rural areas are now even more decreased.

Summarizing effect of $Q_R arrow.t$

- in the city: $P_C arrow.b => L_C arrow.t => "agglomoration" arrow.t => L_C arrow.t => P_C arrow.t$
- in rural areas: $L_R arrow.b => P_R arrow.b$ Then agglomoration happens such that more people want to live in city. substitution implies: $L_R arrow.b => P_R arrow.b $

Importantly, even though we know for sure that the effect will be lower prices in rural areas and more people living in the city, the price effect is more ambigous. Yes, we can know for sure that the prices will be lower in rural areas, however, we cannot know for sure that the agglomoration effect is dominated by the supply shock. This depends entirely on the models parameters, or rather on the underlying preferences of people in the real world.

== case 2: Demand shock with multiple types

Before demand was defined the following way:

$ Y_(C i) = K_C + tilde(L)_C^alpha + e_i - P_C $

and

$ Y_(R i) = K_R - P_R $

We now consider two types $A$ and $B$ where $B$ now gets a benefit when they live in the city. We can consider some government benefit to parents. So we can think of $K_(C A)$ and $K_(C B) = K_(C A) + rho$ assume there is $kappa$ and $1 - kappa$ of each type. We can now derive the demand, using the same logic as previously: Agents are utility maximizing and they chose the city if $Y_C > Y_R$ and vice versa. Using this we can derive the aggregate demand (for simplicty we let $e tilde "Uniform"(underline(e), overline(e))$, such that the sub-demand curve is linear. We can derive an expression for the aggregate demand for city housing as function of relative prices *maybe better expressed as $Y_C > Y_R$*:

$
  Y_C - Y_R = \ K_C + tilde(L)_C^alpha + e_i - P_C - (W_R -P_R) &>= 0 \
  => e_i &>= Delta P + B( tilde(L)_C)
$
o
= Two-Type City Demand with a City-Specific Benefit

Assume two agent types, A and B, with population shares $kappa$ and $1-kappa$.

Preferences:
$
  Y_(C i) = K_C + tilde(L)_C^alpha + e_i - P_C, quad
  Y_(R i) = K_R - P_R.
$

Type B receives an additional city benefit $rho$ (so $K_(C B) = K_(C A) + rho$).
Idiosyncratic taste $e_i ~ "Uniform"([e_min, e_max])$.

Define relative price and baseline term:
$
  Delta P = P_C - P_R, quad
  B(tilde(L)_C) = -(K_C - K_R) - tilde(L)_C^alpha.
$

---

== Individual choice

An individual chooses the city iff $Y_C >= Y_R$, which is equivalent to a threshold:

- Type A: $e_i >= tau_A = Delta P + B(tilde(L)_C)$
- Type B: $e_i >= tau_B = Delta P + B(tilde(L)_C) - rho$

---

== Sub-demand shares

With $e_i ~ "Uniform"([e_min, e_max])$, the share living in the city is

$
  s_t = "clip"( (e_max - tau_t) / E , 0 , 1 ), quad
  E = e_max - e_min.
$

So
$ s_A = (e_max - Delta P - B(tilde(L)_C)) / E, $
$ s_B = (e_max - Delta P - B(tilde(L)_C) + rho) / E, $
in the interior case ($e_min < tau_t < e_max$).

---

== Aggregate demand

Total city demand:
$ L_C(Delta P, tilde(L)_C) = kappa s_A + (1-kappa) s_B. $

= Inverse demand (interior)

$ L_C = frac( e_max - Delta P - B(tilde(L)_C) + (1-kappa) rho , E ) $

Invert:
$ Delta P = e_max - B(tilde(L)_C) + (1-kappa) rho - E L_C. $

---

== Sub-demands by type

$ Delta P = e_max - B(tilde(L)_C) - frac(E, kappa) L_C^A $
$ Delta P = e_max - B(tilde(L)_C) + rho - frac(E, 1-kappa) L_C^B $

---

== Corner cases

If $tau_t <= e_min$ then $s_t=1$.
If $tau_t >= e_max$ then $s_t=0$.
Always: $L_C in [0,1]$.

== Observations:

Now we realize, for a given clearing price delta $Delta P$, we have two prices, one for rural $P_R$ and one for the city $P_C$. Holding say $P_R$ constant (as is the case in a given equilibrium, we can plot the sub demands for type $A$ and type $B$ as a function of $L_C$, i.e. for given $Delta P$ we can have a plot of $P_C$ vs $L_(C A)$ and $L_(C B)$. In this way we can compare before an after a benefit of size $rho$ is introduced.

Concretely:

We start by $rho=0$, and find the equilibrium in the three relevant plots. Then we change $rho$. This corresponds to a demand shock (in bottom right corner plot). This yields new prices (upper right plot), which then allow us to plot the new $L_(C A)$ and $L_(C B)$ in the upper left corner plot.

