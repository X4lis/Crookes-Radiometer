"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CROOKES RADIOMETER — MULTIPHYSICS SIMULATION                       ║
║                                                                              ║
║  Physics modelled:                                                           ║
║   1. Radiative heat transfer  → vane temperature field (black/white sides)  ║
║   2. Thermal conduction       → steady-state vane temperature profile        ║
║   3. Kinetic gas theory       → Maxwell-Boltzmann momentum transfer          ║
║   4. Thermal creep (LGE)      → dominant radiometric drive force             ║
║   5. Photon radiation pressure→ tiny but physically present                  ║
║   6. Rigid-body rotation      → moment of inertia, angular velocity         ║
║   7. Viscous drag             → low-pressure gas damping                     ║
║   8. Convective bulk airflow  → buoyancy-driven torque at high pressure      ║
║   9. Electrostatic charging   → Coulomb torque on metallised vanes           ║
╚══════════════════════════════════════════════════════════════════════════════╝

The Crookes radiometer consists of 4 vanes on a spindle inside a partial-vacuum
glass bulb.  Each vane is black on one face and shiny/white on the other.
Light heats the black face more than the white face, creating a temperature
gradient.  In the low-pressure regime (Knudsen number Kn ~ 1) the dominant
torque comes from *thermal creep* (also called the LGE / thermophoretic edge
effect): gas molecules near the hot edge of the vane gain extra momentum,
producing a net reaction force that spins the wheel so the black faces retreat
from the light.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe; switch to "TkAgg" for live window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
k_B   = 1.380649e-23    # Boltzmann constant          [J K⁻¹]
sigma = 5.670374419e-8  # Stefan-Boltzmann constant   [W m⁻² K⁻⁴]
c     = 2.99792458e8    # speed of light              [m s⁻¹]
N_A   = 6.02214076e23   # Avogadro number             [mol⁻¹]
R_gas = 8.314462618     # universal gas constant      [J mol⁻¹ K⁻¹]

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS  (all SI unless noted)
# ─────────────────────────────────────────────────────────────────────────────
class Params:
    # ── Geometry ─────────────────────────────────────────────────────────────
    n_vanes        = 4           # number of vanes
    vane_width     = 0.015       # m   (15 mm)
    vane_height    = 0.015       # m   (15 mm)
    vane_thickness = 5e-4        # m   (0.5 mm)
    arm_length     = 0.012       # m   pivot → vane centre  (12 mm)
    vane_mass      = 3e-4        # kg  each vane + arm share (~0.3 g)
    spindle_mass   = 1e-4        # kg

    # ── Optical / thermal properties ─────────────────────────────────────────
    absorptivity_black = 0.95    # black (painted) face
    absorptivity_white = 0.15    # white / reflective face
    emissivity_black   = 0.95
    emissivity_white   = 0.10
    thermal_conductivity_vane = 15.0   # W m⁻¹ K⁻¹  (aluminium-ish)

    # ── Radiation source ─────────────────────────────────────────────────────
    irradiance     = 800.0       # W m⁻²  (roughly 1 sun, normal incidence)

    # ── Gas ──────────────────────────────────────────────────────────────────
    gas            = "air"
    molar_mass_air = 28.97e-3    # kg mol⁻¹
    pressure_Pa    = 10.0        # Pa   (partial vacuum, ~0.1 mbar)
    T_gas_ambient  = 293.15      # K  (20 °C = 68 °F)
    # Knudsen-number regime selector (auto-computed)
    vane_gap       = 0.003       # m    mean free space near vane edges

    # ── Numerical ────────────────────────────────────────────────────────────
    dt             = 1e-3        # s    time step
    t_end          = 60.0        # s    total simulation time
    n_vane_nodes   = 20          # 1-D finite-difference nodes across vane width

    # ── Moment of inertia (4 point masses on arms) ───────────────────────────
    @property
    def I_total(self):
        I_vanes  = self.n_vanes * self.vane_mass * self.arm_length**2
        I_spindle = 0.5 * self.spindle_mass * (1e-3)**2   # thin rod ≈ 0
        return I_vanes + I_spindle

    # ── Gas mean free path ────────────────────────────────────────────────────
    @property
    def mean_free_path(self):
        # λ = k_B T / (√2 π d² P),   d_air ≈ 3.7 Å
        d_air = 3.7e-10
        return k_B * self.T_gas_ambient / (math.sqrt(2) * math.pi * d_air**2 * self.pressure_Pa)

    @property
    def knudsen_number(self):
        return self.mean_free_path / self.vane_gap

P = Params()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  THERMAL MODEL  — 1-D finite-difference across vane width
#     x = 0 : black face    x = L : white face
#     Steady-state: d²T/dx² = 0  → linear profile in thin vane
#     We solve the *surface* temperatures from energy balance.
# ─────────────────────────────────────────────────────────────────────────────

def solve_vane_temperatures(irradiance, T_gas, alpha_b, alpha_w,
                             eps_b, eps_w, k, thickness, n_nodes=P.n_vane_nodes):
    """
    Returns (T_black, T_white, T_profile[n_nodes]) via Newton iteration on
    steady-state surface energy balance for each face plus 1-D conduction.

    Energy balance (per unit area):
      Black face:   α_b · G - ε_b · σ · T_b⁴ - k/L·(T_b - T_w) = 0
      White face:   α_w · G - ε_w · σ · T_w⁴ + k/L·(T_b - T_w) = 0
    We also add gas convective exchange q_conv = h(T_face - T_gas) with
    h estimated from free-molecular heat transfer.
    """
    L = thickness
    # Free-molecular heat-transfer coefficient (Knudsen > 1)
    gamma = 1.4       # air
    Pr    = 0.71
    M_air = P.molar_mass_air
    # accommodation coefficient
    alpha_acc = 0.9
    rho_gas   = P.pressure_Pa * M_air / (R_gas * T_gas)
    v_mean    = math.sqrt(8 * k_B * T_gas / (math.pi * M_air / N_A))
    # free-molecular h  [W m⁻² K⁻¹]
    h_fm = alpha_acc * (gamma + 1) / (gamma - 1) * k_B / (4 * M_air / N_A) \
           * rho_gas * v_mean

    def equations(T):
        Tb, Tw = T
        cond = k / L * (Tb - Tw)
        conv_b = h_fm * (Tb - T_gas)
        conv_w = h_fm * (Tw - T_gas)
        f1 = alpha_b * irradiance - eps_b * sigma * Tb**4 - cond - conv_b
        f2 = alpha_w * irradiance - eps_w * sigma * Tw**4 + cond - conv_w
        return np.array([f1, f2])

    # Newton-Raphson
    T_vec = np.array([T_gas + 20.0, T_gas + 5.0])
    for _ in range(200):
        F = equations(T_vec)
        if np.max(np.abs(F)) < 1e-6:
            break
        # Jacobian (analytical)
        Tb, Tw = T_vec
        cond_coef = P.thermal_conductivity_vane / L
        J = np.array([
            [-4 * eps_b * sigma * Tb**3 - cond_coef - h_fm,  cond_coef],
            [ cond_coef, -4 * eps_w * sigma * Tw**3 - cond_coef - h_fm],
        ])
        dT = np.linalg.solve(J, -F)
        T_vec = T_vec + dT
        T_vec = np.clip(T_vec, T_gas * 0.5, T_gas * 10)

    T_black, T_white = T_vec
    # Linear conduction profile
    x_nodes  = np.linspace(0, 1, n_nodes)
    T_profile = T_black + (T_white - T_black) * x_nodes

    return float(T_black), float(T_white), T_profile


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RADIOMETRIC (THERMAL CREEP) FORCE  — Knudsen-regime torque
#
#     In the transition / free-molecular regime the dominant mechanism is
#     thermal creep along the vane edge.  Reynolds (1879) and later Knudsen
#     showed the tangential force on a flat plate with a temperature gradient
#     ∇T along its surface scales as:
#
#         F_creep ≈ C_tc · (μ / T) · (∇T / L) · A_edge
#
#     where C_tc is a numerical coefficient ~1.14 from kinetic theory (Sone),
#     μ is dynamic viscosity, L is a characteristic length.
#
#     In our geometry ∇T = (T_black - T_white) / vane_width,
#     acting along the vane height (edge area = h × thickness).
#     The net force per vane acts perpendicular to the arm → pure torque.
# ─────────────────────────────────────────────────────────────────────────────

def thermal_creep_force(T_black, T_white, T_gas, pressure):
    """
    Returns force [N] on one vane due to thermal creep along vane surface.
    Positive → rotates black face away from light (observed direction).
    """
    delta_T   = T_black - T_white          # K
    T_mean    = (T_black + T_white) / 2.0

    if abs(delta_T) < 1e-6:
        return 0.0

    # Dynamic viscosity of air (Sutherland's law)
    mu_ref, T_ref, S = 1.716e-5, 273.15, 110.4
    mu = mu_ref * (T_mean / T_ref)**1.5 * (T_ref + S) / (T_mean + S)

    # Thermal creep coefficient (Sone's result, free-molecular limit)
    C_tc = 1.147

    # ∇T along vane face (from edge to edge ≈ across vane width)
    grad_T = delta_T / P.vane_width       # K m⁻¹

    # Edge area (two long edges of the vane)
    A_edge = 2 * P.vane_height * P.vane_thickness   # m²

    # Creep force magnitude
    F_creep = C_tc * (mu / T_mean) * grad_T * A_edge * (R_gas * T_mean / (P.molar_mass_air))

    # Scale by Knudsen-number correction  (Loyalka interpolation)
    Kn = P.knudsen_number
    # Interpolation factor: 1 at Kn>>1, 0 at Kn<<1
    f_Kn = Kn / (1.0 + Kn)

    return F_creep * f_Kn


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DIRECT MOLECULAR MOMENTUM-TRANSFER FORCE
#
#     In the free-molecular regime molecules hitting the hot face bounce off
#     with higher thermal speed than those hitting the cold face.
#     Net force per unit area:
#
#         ΔP = n · m · (v̄_hot² - v̄_cold²) / 4   [Pa]
#         v̄ = √(8 k_B T / π m)
#
#     We compute the differential normal pressure on both faces.
#     This gives a smaller contribution than creep in the Kn~1 regime.
# ─────────────────────────────────────────────────────────────────────────────

def molecular_pressure_force(T_black, T_white, pressure):
    """
    Net force on one vane face due to differential molecular rebound pressure.
    This mechanism is only active in the free-molecular regime (Kn >> 1).
    In the continuum regime viscosity homogenises momentum: force → 0.
    Suppressed by the same Loyalka f_Kn = Kn/(1+Kn) factor used for creep.
    """
    m_mol = P.molar_mass_air / N_A
    n_density = pressure / (k_B * ((T_black + T_white) / 2))

    v_black = math.sqrt(8 * k_B * T_black / (math.pi * m_mol))
    v_white = math.sqrt(8 * k_B * T_white / (math.pi * m_mol))

    P_black_face = n_density * m_mol * (v_black**2) / 4.0
    P_white_face = n_density * m_mol * (v_white**2) / 4.0

    delta_P = P_black_face - P_white_face
    A_vane  = P.vane_width * P.vane_height

    # Kn suppression: free-molecular mechanism vanishes in the continuum limit
    Kn   = P.knudsen_number
    f_Kn = Kn / (1.0 + Kn)

    return delta_P * A_vane * f_Kn


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RADIATION PRESSURE FORCE
#
#     F_rad = (α · G · A) / c   (absorbed photons transfer p = E/c)
#     Differential force between black and white faces.
# ─────────────────────────────────────────────────────────────────────────────

def radiation_pressure_force(irradiance):
    """Net radiation pressure force on one vane [N]."""
    A_vane = P.vane_width * P.vane_height
    # Black absorbs α_b·G, white reflects (1-α_w)·G back → gives 2× momentum
    F_black = P.absorptivity_black * irradiance * A_vane / c       # absorbed
    F_white = (2.0 * (1.0 - P.absorptivity_white) * irradiance * A_vane / c
               + P.absorptivity_white * irradiance * A_vane / c)    # reflected + absorbed
    # Net: white side pushes MORE (more reflection) → vane pushed from white
    # But here we return force that drives rotation (black retreats from light)
    # So net torque-driving force = F_white_push - F_black_push
    return F_white - F_black


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VISCOUS DRAG TORQUE
#
#     In the partial-vacuum regime, drag comes from gas viscosity.
#     We use a simple model:  τ_drag = -b · ω
#     with b estimated from Stokes drag on a rotating disc.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 8.  CONVECTIVE BULK-AIRFLOW TORQUE
#
#     At atmospheric pressure the gas is a continuous viscous fluid.  A nearby
#     warm body (e.g. a human hand) heats the air next to the bulb, which rises
#     by buoyancy (Rayleigh-Bénard / natural convection).  This bulk flow wraps
#     around the vanes asymmetrically and exerts an aerodynamic drag torque.
#
#     Model (thin-plate aerodynamic drag in crossflow):
#       Each vane sees a local flow velocity  U_i  that depends on the angle
#       between the vane arm and the dominant convective plume direction.
#       Force on one vane face:  F = ½ ρ U² C_D A  (flat-plate C_D ≈ 1.17)
#       Net tangential component → torque contribution.
#
#     The convective velocity magnitude is estimated from the plume model:
#       U_plume ≈ √(2 g β ΔT_plume H)  (buoyant updraft scaling)
#     where β = 1/T_gas, ΔT_plume is the excess temperature at the heat source,
#     and H is the characteristic height (~bulb diameter).
#
#     This torque is proportional to ρ ∝ pressure, so it naturally vanishes at
#     low pressure and dominates at atmosphere.
# ─────────────────────────────────────────────────────────────────────────────

_conv_rng = np.random.default_rng(42)   # private RNG for turbulence noise

# ── Glass boundary-layer attenuation ─────────────────────────────────────────
# The glass bulb presents two sequential resistances to the exterior convective
# plume before it can exert a force on the interior vanes:
#
#   1. EXTERIOR BOUNDARY LAYER  (Blasius flat-plate approximation)
#      The exterior flow decelerates inside a viscous boundary layer over the
#      sphere.  The effective velocity seen at the glass surface is:
#        U_surface ≈ U_∞ · (1 − 1/√Re_ext)
#      For typical plume speeds 0.05–0.15 m/s and R_bulb = 30 mm,
#      Re_ext ≈ 100–300  →  f_BL ≈ 0.88–0.94.
#
#   2. GLASS SHELL THERMAL-MOMENTUM TRANSMISSION
#      The 2 mm borosilicate wall does not transmit bulk momentum.  Instead,
#      the exterior plume couples to the interior gas via conduction:
#        Q'' = k_glass · ΔT_ext / t_glass          [W m⁻²]
#        ΔT_int = Q'' / h_int                       [K]
#        U_int  = √(g β ΔT_int H_bulb)             [m s⁻¹]
#      where h_int combines the free-molecular and natural-convection
#      coefficients in parallel (each dominant in its pressure regime).
#      Typical values (k_glass=1.2, t=2 mm, ΔT_ext~1–5 K):
#        ΔT_int ≈ 0.06–0.3 K  →  U_int ≈ 0.01–0.03 m/s  →  η ≈ 0.10–0.25
#
#   3. PRESSURE SIGMOID
#      At Kn ≫ 1 the interior gas is too rarefied for bulk flow regardless of
#      surface temperature, so η → 0.  Modelled as η_p = P/(P + P_half).
#
# Combined:  f_glass = f_BL · (U_int/U_ext) · η_p   clamped to [0, 1].
# Both the exterior U_ext and attenuated U_int are returned so the visualiser
# can draw them with distinct colours and opacity.

_GLASS_K      = 1.2      # W m⁻¹ K⁻¹  borosilicate thermal conductivity
_GLASS_T      = 2e-3     # m           glass wall thickness
_GLASS_R_BULB = 0.030    # m           outer sphere radius
_GLASS_P_HALF = 500.0    # Pa          pressure at which η_p = 0.5
_MU_AIR_20C   = 1.81e-5  # Pa·s        air viscosity at 20 °C
_RHO_AIR_ATM  = 1.204    # kg m⁻³      air density at 1 atm, 20 °C


def _glass_attenuation(U_ext, pressure, T_gas, dT_ext):
    """
    Compute glass boundary-layer attenuation.

    Returns
    -------
    f_total  : float in [0, 1]
        Multiply U_ext by this to get the effective velocity acting on vanes.
    U_int    : float [m/s]
        Interior buoyant velocity driven by conduction through the glass wall.
        Used by the visualiser to shade the interior streamlines distinctly.
    """
    if U_ext < 1e-9:
        return 0.0, 0.0

    # 1. Exterior boundary-layer factor
    Re_ext = _RHO_AIR_ATM * U_ext * _GLASS_R_BULB / _MU_AIR_20C
    Re_ext = max(Re_ext, 1.0)
    f_BL   = 1.0 - 1.0 / math.sqrt(Re_ext)   # ~0.90 at Re=100

    # 2. Interior heat-transfer coefficient: free-molecular + natural convection
    m_mol    = P.molar_mass_air / N_A
    rho_int  = pressure * P.molar_mass_air / (R_gas * T_gas)
    v_mean   = math.sqrt(8 * k_B * T_gas / (math.pi * m_mol))
    # Free-molecular h (accommodation α=0.9, γ_air=1.4):
    h_fm_int = 0.9 * (2.4 / 0.4) * (k_B / (4.0 * m_mol)) * rho_int * v_mean

    # Natural-convection h via Churchill-Chu Nu ≈ 0.59 Ra^0.25 (10⁴ < Ra < 10⁹)
    k_air   = 0.026
    g_acc   = 9.81
    beta    = 1.0 / T_gas
    # Thermal diffusivity of interior gas; guard against near-zero rho_int
    alpha_d = k_air / (max(rho_int, 1e-6) * 1006.0)
    nu_int  = _MU_AIR_20C / max(rho_int, 1e-9)   # kinematic viscosity
    L_char  = 2 * _GLASS_R_BULB
    Ra = g_acc * beta * max(dT_ext, 0.01) * L_char**3 / (nu_int * alpha_d)
    Ra = max(Ra, 0.0)
    Nu_nc = 0.59 * Ra**0.25 if Ra > 1.0 else 1.0
    h_nc  = Nu_nc * k_air / L_char

    h_int = h_fm_int + h_nc   # parallel combination

    # 3. Interior surface ΔT and resulting buoyant velocity
    Q_flux = _GLASS_K * dT_ext / _GLASS_T        # W m⁻²
    dT_int = Q_flux / max(h_int, 0.1)
    dT_int = min(dT_int, dT_ext)                  # cannot exceed exterior ΔT
    U_int  = math.sqrt(max(0.0, g_acc * beta * dT_int * L_char))

    # 4. Pressure sigmoid: bulk flow cannot develop at very low interior pressure
    eta_p = pressure / (pressure + _GLASS_P_HALF)

    # 5. Combined factor
    f_conduct = (U_int / U_ext) * eta_p
    f_total   = max(0.0, min(1.0, f_BL * f_conduct))

    return f_total, U_int


def convective_torque(heat_spots_phys, T_gas, pressure, theta):
    """
    Returns (tau [N·m], U_dominant_ext [m/s]) with glass boundary-layer
    attenuation applied to the exterior plume before computing vane forces.

    heat_spots_phys : list of dicts with keys
        'angle_rad'  – azimuthal direction of the heat source around the bulb
        'dist_m'     – distance from bulb centre to source [m]
        'power_W'    – radiated power of source [W]
    theta           : current rotor angle [rad]
    pressure        : gas pressure [Pa]
    T_gas           : ambient gas temperature [K]
    """
    if not heat_spots_phys:
        return 0.0, 0.0

    # Air properties at interior conditions
    mu_ref, T_ref, S_suth = 1.716e-5, 273.15, 110.4
    mu  = mu_ref * (T_gas / T_ref)**1.5 * (T_ref + S_suth) / (T_gas + S_suth)
    rho = pressure * P.molar_mass_air / (R_gas * T_gas)
    g   = 9.81
    beta = 1.0 / T_gas

    H_bulb = 0.060   # m  characteristic height (bulb diameter)
    A_v    = P.vane_width * P.vane_height
    r      = P.arm_length

    tau_total  = 0.0
    U_dominant = 0.0   # largest exterior plume speed across all sources

    for hs in heat_spots_phys:
        P_w    = hs['power_W']
        ang_hs = hs['angle_rad']
        dist   = max(hs['dist_m'], 0.010)

        # Exterior plume temperature excess and speed
        k_air    = 0.026
        dT_plume = min(P_w / (4 * math.pi * k_air * dist), 80.0)
        U_vert   = math.sqrt(max(0.0, g * beta * dT_plume * H_bulb))
        U_horiz_mean = 0.45 * U_vert
        noise    = float(_conv_rng.normal(0.0, 0.20 * max(U_horiz_mean, 1e-4)))
        U_ext    = max(0.0, U_horiz_mean + noise)
        U_dominant = max(U_dominant, U_ext)

        # Attenuate through glass boundary layer + conduction + pressure sigmoid
        f_glass, _U_int = _glass_attenuation(U_ext, pressure, T_gas, dT_plume)
        U_flow = U_ext * f_glass    # effective velocity acting on the vanes

        # Tangential bulk flow past the bulb in the azimuthal direction of the source
        flow_dir = ang_hs + math.pi / 2
        Ux = U_flow * math.cos(flow_dir)
        Uy = U_flow * math.sin(flow_dir)

        # Differential drag on black (C_D=1.30, rough) vs white (C_D=0.80, smooth)
        # gives a net torque that does not cancel across opposing vanes.
        C_D_black = 1.30
        C_D_white = 0.80
        for k in range(4):
            vane_ang  = theta + k * math.pi / 2
            nx_b =  math.cos(vane_ang)
            ny_b =  math.sin(vane_ang)
            nx_w = -nx_b;  ny_w = -ny_b

            U_n_black = Ux * nx_b + Uy * ny_b
            U_n_white = Ux * nx_w + Uy * ny_w

            if U_n_black > 0:
                F_black = 0.5 * rho * U_n_black**2 * C_D_black * A_v
            else:
                F_black = 0.5 * rho * U_n_black**2 * C_D_black * A_v * 0.05
            if U_n_white > 0:
                F_white = 0.5 * rho * U_n_white**2 * C_D_white * A_v
            else:
                F_white = 0.5 * rho * U_n_white**2 * C_D_white * A_v * 0.05

            cross          = Ux * ny_b - Uy * nx_b
            sign_rotation  = 1.0 if cross >= 0 else -1.0
            tau_total     += (F_black - F_white) * sign_rotation * r

    return tau_total, U_dominant


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ELECTROSTATIC TORQUE
#
#     Rubbing or handling the glass bulb deposits triboelectric charge
#     Q [C] in a small number of discrete patches on the surface
#     (Schein 2007, J Electrostatics).  Each patch is well-approximated
#     as a point charge at the glass surface.  The polished-aluminium vane
#     faces (σ ~ 10⁷ S/m) respond immediately to the near-field Coulomb
#     force via induced image charges.  The black-paint faces (σ ~ 10⁻⁶ S/m)
#     do not: they cannot redistribute charge on the millisecond timescale
#     of a patch passing by a rotating vane.  The net torque is therefore
#     the image-charge attraction on the Al face only.
#
#     Model:
#       Approximate the total charge Q as N_patch ~ 8 equal point charges
#       q = Q / N_patch located at random positions on the sphere surface.
#       For each patch at position (R_b, φ_patch), the field at vane k
#       at (r, θ + k·π/2) is computed exactly (Coulomb), and the induced
#       charge on the Al face is q_ind = ε₀ · E_patch · A_vane.
#       Force on that face: F = q_ind · E_patch (Coulomb, same direction).
#       The tangential component contributes torque about the spin axis.
#       The black face contributes nothing (no induced charge).
#
#     The torque scales as Q² (since both E_patch ∝ Q and q_ind ∝ E_patch).
#     At Q = 1 nC this gives ~5 nN·m; at Q = 5 nC, ~130 nN·m.
#
#     Per-step Gaussian noise (σ = 25 % of |τ|) models the non-uniform
#     and time-varying distribution of charge patches on the glass.
# ─────────────────────────────────────────────────────────────────────────────

_elec_rng    = np.random.default_rng(99)
# A single rubbing event deposits charge in a spot ~1 cm² on the glass.
# We model it as one effective point charge at a fixed azimuthal angle
# (arbitrary; set once per program run).  Slow thermal drift shifts the
# spot, modelled by advancing _patch_phi by a small random walk each call.
_patch_phi   = 0.61   # initial patch angle [rad] (arbitrary)


def electrostatic_torque(charge_C, theta):
    """
    Returns net electrostatic torque [N·m] on the vane rotor.

    charge_C : net charge on the glass bulb [Coulombs]  (signed; + or -)
    theta    : current rotor angle [rad]

    The torque comes from image-charge attraction on the conducting
    (Al) vane faces only; the black-paint faces do not respond on the
    relevant timescale.  See module-level comment for derivation.
    """
    global _patch_phi
    if abs(charge_C) < 1e-14:
        return 0.0

    eps0   = 8.854187817e-12   # F m⁻¹
    R_bulb = 0.030             # glass sphere radius [m]
    r      = P.arm_length
    A_vane = P.vane_width * P.vane_height

    # Slow random walk of the charge patch (models charge redistribution
    # as the glass warms slightly; step σ ~ 3° per call at 60 fps)
    _patch_phi += float(_elec_rng.normal(0.0, math.radians(3.0)))

    xp = R_bulb * math.cos(_patch_phi)
    yp = R_bulb * math.sin(_patch_phi)

    tau_total = 0.0
    for k in range(4):
        vane_ang = theta + k * math.pi / 2
        xv = r * math.cos(vane_ang)
        yv = r * math.sin(vane_ang)

        dx = xv - xp;  dy = yv - yp
        d  = math.hypot(dx, dy)
        if d < 1e-6:
            continue

        # Coulomb field from the charge patch at the vane position
        E_patch = abs(charge_C) / (4.0 * math.pi * eps0 * d * d)

        # Induced charge on the conducting (Al) vane face (ε₀ E A)
        q_ind = eps0 * E_patch * A_vane

        # Attractive Coulomb force on the induced charge
        sign = 1.0 if charge_C > 0 else -1.0
        Fx = sign * q_ind * E_patch * (dx / d)
        Fy = sign * q_ind * E_patch * (dy / d)

        # Tangential component → torque about z-axis
        tau_total += (Fy * math.cos(vane_ang) - Fx * math.sin(vane_ang)) * r

    # Gaussian noise: patch is not a perfect point charge; shape fluctuations
    # add ~25 % amplitude uncertainty per time step.
    sigma_noise = 0.25 * abs(tau_total)
    noise       = float(_elec_rng.normal(0.0, max(sigma_noise, 1e-14)))
    return tau_total + noise


def compute_drag_coefficient(pressure, T_gas):
    """
    Returns rotational drag coefficient b [N·m·s/rad].

    Two additive contributions, each correct in its own regime:

    1. FREE-MOLECULAR drag (Kn >> 1):
       Molecules bounce off the vane face and carry away angular momentum.
       Raw formula: b_fm ∝ n · m · v_th · A · r²  (∝ pressure).
       This must be suppressed in the continuum regime with the same
       Loyalka factor f_Kn = Kn/(1+Kn) used for the drive torques.
       Without this suppression b_fm at 1 atm (Kn ≈ 2×10⁻⁵) is ~45 000×
       too large, making the vane appear frozen when driven by atmospheric
       convection (τ_conv ~ 0.25 nN·m vs b_fm·ω ~ 6.5×10⁻⁵ N·m·s/rad).

    2. CONTINUUM (viscous) drag (Kn << 1):
       Stokes shear of the vane face against the surrounding gas.
       The effective gap is the clearance between the vane tip and the
       bulb wall (R_bulb - arm_length ~ 12 mm).
       b_cont ∝ μ (independent of pressure in the viscous regime).

    The two are added; each naturally dominates in the appropriate Kn regime.
    At Kn >> 1: b_fm·f_Kn ≈ b_fm  (f_Kn → 1),  b_cont is negligible.
    At Kn << 1: b_fm·f_Kn ≈ b_fm·Kn → 0,         b_cont dominates.
    """
    mu_ref, T_ref, S = 1.716e-5, 273.15, 110.4
    mu = mu_ref * (T_gas / T_ref)**1.5 * (T_ref + S) / (T_gas + S)

    Kn   = P.knudsen_number
    f_Kn = Kn / (1.0 + Kn)   # Loyalka interpolation, same as drive torques
    A_v  = P.vane_width * P.vane_height
    r    = P.arm_length

    # ── Free-molecular drag (with Kn suppression) ─────────────────────────────
    # Each molecule hitting the vane face transfers momentum ∝ m·v_th.
    # Flux Γ = n·v_th/4, so raw b_fm ∝ n·m·v_th·A·r² ∝ pressure.
    # Multiplied by f_Kn so it vanishes in the continuum limit, consistent
    # with how thermal_creep_force and molecular_pressure_force are treated.
    alpha_acc = 0.9
    m_mol     = P.molar_mass_air / N_A
    v_th      = math.sqrt(8 * k_B * T_gas / (math.pi * m_mol))
    b_fm_one  = alpha_acc * (pressure / (k_B * T_gas)) * m_mol * v_th * A_v * r**2
    b_fm      = P.n_vanes * b_fm_one * f_Kn

    # ── Continuum (viscous) drag ──────────────────────────────────────────────
    # Stokes shear: τ = μ · A · v_tip / clearance = μ · A · r² · ω / clearance
    R_bulb     = 0.030          # glass sphere radius [m]
    clearance  = R_bulb - (r + P.vane_width / 2)   # ~12 mm
    clearance  = max(clearance, 1e-3)              # safety floor
    b_cont_one = mu * A_v * r**2 / clearance
    b_cont     = P.n_vanes * b_cont_one

    # Bearing / pivot friction (realistic PTFE needle bearing)
    b_bearing  = 5e-10   # N·m·s

    return b_fm + b_cont + b_bearing


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TORQUE ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_torque(omega, irradiance, T_gas, pressure, T_black, T_white,
                         heat_spots_phys=None, charge_C=0.0, theta=0.0):
    """
    Returns (tau_net, tau_creep, tau_molecular, tau_rad, tau_drag,
             tau_conv, tau_elec) [N·m]
    """
    r = P.arm_length

    F_creep = thermal_creep_force(T_black, T_white, T_gas, pressure)
    F_mol   = molecular_pressure_force(T_black, T_white, pressure)
    F_rad   = radiation_pressure_force(irradiance)

    tau_creep     =  F_creep * r * P.n_vanes
    tau_molecular =  F_mol   * r * P.n_vanes
    tau_rad       =  F_rad   * r * P.n_vanes

    tau_conv, _   = convective_torque(
        heat_spots_phys if heat_spots_phys is not None else [],
        T_gas, pressure, theta)
    tau_elec      = electrostatic_torque(charge_C, theta)

    b         = compute_drag_coefficient(pressure, T_gas)
    tau_drag  = -b * omega

    tau_net = tau_creep + tau_molecular + tau_rad + tau_drag + tau_conv + tau_elec

    return tau_net, tau_creep, tau_molecular, tau_rad, tau_drag, tau_conv, tau_elec


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TIME INTEGRATION  (simple Euler + optional RK4)
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation():
    print("=" * 70)
    print("  CROOKES RADIOMETER  —  MULTIPHYSICS SIMULATION")
    print("=" * 70)
    print(f"\n  Gas pressure    : {P.pressure_Pa:.1f} Pa  ({P.pressure_Pa/133.322:.4f} Torr)")
    print(f"  Mean free path  : {P.mean_free_path*1e6:.1f} μm")
    print(f"  Knudsen number  : {P.knudsen_number:.3f}  (", end="")
    Kn = P.knudsen_number
    if Kn < 0.1:
        print("continuum regime — radiometer barely spins)")
    elif Kn < 10:
        print("transition regime — optimal for radiometer)")
    else:
        print("free-molecular regime)")
    print(f"  Moment of inertia: {P.I_total:.3e} kg·m²")

    # Solve steady-state temperatures once (they don't depend on ω)
    T_black, T_white, T_profile = solve_vane_temperatures(
        P.irradiance, P.T_gas_ambient,
        P.absorptivity_black, P.absorptivity_white,
        P.emissivity_black, P.emissivity_white,
        P.thermal_conductivity_vane, P.vane_thickness
    )
    delta_T = T_black - T_white
    print(f"\n  Vane temperatures:")
    print(f"    Black face  : {T_black - 273.15:.2f} °C  ({T_black:.2f} K)")
    print(f"    White face  : {T_white - 273.15:.2f} °C  ({T_white:.2f} K)")
    print(f"    ΔT          : {delta_T:.3f} K")

    # Initial conditions
    omega  = 0.0          # rad s⁻¹
    theta  = 0.0          # rad
    t      = 0.0
    dt     = P.dt

    # Storage
    n_steps = int(P.t_end / dt) + 1
    times        = np.zeros(n_steps)
    omegas       = np.zeros(n_steps)
    thetas       = np.zeros(n_steps)
    tau_nets     = np.zeros(n_steps)
    tau_creeps   = np.zeros(n_steps)
    tau_mols     = np.zeros(n_steps)
    tau_rads     = np.zeros(n_steps)
    tau_drags    = np.zeros(n_steps)
    rpm_arr      = np.zeros(n_steps)

    I = P.I_total

    print(f"\n  Running {n_steps:,} time steps (dt={dt*1000:.1f} ms, T_end={P.t_end:.0f}s)...\n")

    for i in range(n_steps):
        times[i]  = t
        omegas[i] = omega
        thetas[i] = theta
        rpm_arr[i] = omega * 60 / (2 * math.pi)

        tau_net, tau_c, tau_m, tau_r, tau_d, tau_cv, tau_el = compute_total_torque(
            omega, P.irradiance, P.T_gas_ambient, P.pressure_Pa, T_black, T_white
        )
        tau_nets[i]   = tau_net
        tau_creeps[i] = tau_c
        tau_mols[i]   = tau_m
        tau_rads[i]   = tau_r
        tau_drags[i]  = tau_d

        # RK4 integration for omega
        def domega_dt(w):
            tn, *_ = compute_total_torque(
                w, P.irradiance, P.T_gas_ambient,
                P.pressure_Pa, T_black, T_white
            )
            return tn / I

        k1 = domega_dt(omega)
        k2 = domega_dt(omega + 0.5 * dt * k1)
        k3 = domega_dt(omega + 0.5 * dt * k2)
        k4 = domega_dt(omega + dt * k3)
        alpha = (k1 + 2*k2 + 2*k3 + k4) / 6.0

        omega += alpha * dt
        theta += omega * dt
        t     += dt

    # Terminal (steady-state) values
    omega_ss  = omegas[-1]
    rpm_ss    = rpm_arr[-1]
    period_ss = abs(2 * math.pi / omega_ss) if abs(omega_ss) > 1e-6 else float('inf')

    print(f"  ── Results ────────────────────────────────────────────")
    print(f"  Terminal angular velocity : {omega_ss:.4f} rad/s")
    print(f"  Terminal RPM              : {rpm_ss:.3f} rpm")
    print(f"  Period per revolution     : {period_ss:.2f} s")
    print(f"\n  Torque breakdown at steady state:")
    print(f"    Thermal creep  : {tau_creeps[-1]:.4e} N·m  ({100*tau_creeps[-1]/max(abs(tau_nets[-1]),1e-20):.1f}%)")
    print(f"    Molecular rebound: {tau_mols[-1]:.4e} N·m  ({100*tau_mols[-1]/max(abs(tau_nets[-1]),1e-20):.1f}%)")
    print(f"    Radiation press: {tau_rads[-1]:.4e} N·m  ({100*tau_rads[-1]/max(abs(tau_nets[-1]),1e-20):.1f}%)")
    print(f"    Viscous drag   : {tau_drags[-1]:.4e} N·m")
    print(f"    Net torque     : {tau_nets[-1]:.4e} N·m")
    print()

    return {
        "times": times, "omegas": omegas, "thetas": thetas, "rpm": rpm_arr,
        "tau_net": tau_nets, "tau_creep": tau_creeps, "tau_mol": tau_mols,
        "tau_rad": tau_rads, "tau_drag": tau_drags,
        "T_black": T_black, "T_white": T_white, "T_profile": T_profile,
        "omega_ss": omega_ss, "rpm_ss": rpm_ss,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PRESSURE SWEEP  — RPM vs gas pressure
# ─────────────────────────────────────────────────────────────────────────────

def pressure_sweep():
    """Compute terminal RPM across a range of pressures (0.01 – 10000 Pa)."""
    pressures = np.logspace(-2, 4, 60)   # Pa
    rpm_values = []
    omega_ss_values = []

    for press in pressures:
        P.pressure_Pa = press
        Tb, Tw, _ = solve_vane_temperatures(
            P.irradiance, P.T_gas_ambient,
            P.absorptivity_black, P.absorptivity_white,
            P.emissivity_black, P.emissivity_white,
            P.thermal_conductivity_vane, P.vane_thickness
        )
        # Terminal omega: tau_drive = tau_drag  → omega_ss = tau_drive / b
        b = compute_drag_coefficient(press, P.T_gas_ambient)
        r = P.arm_length
        F_c = thermal_creep_force(Tb, Tw, P.T_gas_ambient, press)
        F_m = molecular_pressure_force(Tb, Tw, press)
        F_r = radiation_pressure_force(P.irradiance)
        tau_drive = (F_c + F_m + F_r) * r * P.n_vanes
        omega_t   = max(tau_drive / b, 0.0)
        rpm_t     = omega_t * 60 / (2 * math.pi)
        rpm_values.append(rpm_t)
        omega_ss_values.append(omega_t)

    P.pressure_Pa = 10.0   # restore default
    return pressures, np.array(rpm_values)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG  = "#0d1117"
ACCENT1  = "#58a6ff"    # blue   – thermal creep
ACCENT2  = "#f78166"    # red    – molecular
ACCENT3  = "#ffa657"    # orange – radiation
ACCENT4  = "#3fb950"    # green  – drag
WHITE_FG = "#e6edf3"
GRID_COL = "#21262d"
VANE_BLACK_COL = "#2c2c2c"
VANE_WHITE_COL = "#f0f0f0"
BULB_COL = "#1f6feb"


def draw_radiometer_frame(ax, theta, T_black, T_white, omega):
    """Draw a top-down schematic of the 4-vane radiometer at angle theta."""
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-0.030, 0.030)
    ax.set_ylim(-0.030, 0.030)
    ax.set_aspect('equal')
    ax.axis('off')

    # Glass bulb
    bulb = plt.Circle((0, 0), 0.026, color=BULB_COL, alpha=0.12, zorder=1)
    ax.add_patch(bulb)
    bulb_ring = plt.Circle((0, 0), 0.026, color=BULB_COL, alpha=0.5,
                             fill=False, linewidth=1.5, zorder=2)
    ax.add_patch(bulb_ring)

    # Spindle
    spindle = plt.Circle((0, 0), 0.0008, color="#8b949e", zorder=5)
    ax.add_patch(spindle)

    # 4 vanes at 90° intervals
    for k in range(4):
        angle = theta + k * math.pi / 2

        # Arm from spindle to vane centre
        cx = P.arm_length * math.cos(angle)
        cy = P.arm_length * math.sin(angle)
        ax.plot([0, cx], [0, cy], color="#8b949e", linewidth=1.2, zorder=3)

        # Vane rectangle – perpendicular to arm
        perp = angle + math.pi / 2
        hw   = P.vane_width / 2
        hh   = P.vane_height / 2

        # Four corners of the vane
        dx, dy = math.cos(perp), math.sin(perp)
        ex, ey = math.cos(angle),  math.sin(angle)

        corners_black = np.array([
            [cx + hw * dx - 0.0001 * ex,  cy + hw * dy - 0.0001 * ey],
            [cx - hw * dx - 0.0001 * ex,  cy - hw * dy - 0.0001 * ey],
            [cx - hw * dx + hh * ex,       cy - hw * dy + hh * ey],
            [cx + hw * dx + hh * ex,       cy + hw * dy + hh * ey],
        ])
        corners_white = np.array([
            [cx + hw * dx + 0.0001 * ex,  cy + hw * dy + 0.0001 * ey],
            [cx - hw * dx + 0.0001 * ex,  cy - hw * dy + 0.0001 * ey],
            [cx - hw * dx - hh * ex,       cy - hw * dy - hh * ey],
            [cx + hw * dx - hh * ex,       cy + hw * dy - hh * ey],
        ])

        t_norm = (T_black - 273.15 - 20) / 30.0   # 0–1 scale
        t_norm = max(0, min(1, t_norm))
        black_color = plt.cm.hot(0.4 + 0.4 * t_norm)

        poly_b = plt.Polygon(corners_black, closed=True,
                              facecolor=black_color, edgecolor='none', zorder=4)
        poly_w = plt.Polygon(corners_white, closed=True,
                              facecolor=VANE_WHITE_COL, edgecolor='none', zorder=4)
        ax.add_patch(poly_b)
        ax.add_patch(poly_w)

    # Info text
    rpm = omega * 60 / (2 * math.pi)
    ax.text(0, -0.029, f"{rpm:.2f} RPM", ha='center', va='bottom',
            color=WHITE_FG, fontsize=8, zorder=6)


def plot_results(sim, pressures, rpm_sweep):
    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    fig.suptitle("CROOKES RADIOMETER  —  MULTIPHYSICS SIMULATION",
                 color=WHITE_FG, fontsize=15, fontweight='bold', y=0.98)

    gs = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.38,
                  left=0.06, right=0.97, top=0.92, bottom=0.07)

    # ── Panel helpers ─────────────────────────────────────────────────────────
    def style_ax(ax, title):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=WHITE_FG, labelsize=8)
        ax.xaxis.label.set_color(WHITE_FG)
        ax.yaxis.label.set_color(WHITE_FG)
        ax.title.set_color(WHITE_FG)
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(color=GRID_COL, linewidth=0.5, linestyle='--', alpha=0.7)

    t  = sim["times"]
    om = sim["omegas"]
    rp = sim["rpm"]

    # 1. Angular velocity vs time
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t, om, color=ACCENT1, linewidth=1.8, label='ω (rad/s)')
    ax1_r = ax1.twinx()
    ax1_r.plot(t, rp, color=ACCENT3, linewidth=1.2, linestyle='--', label='RPM')
    ax1_r.tick_params(colors=WHITE_FG, labelsize=8)
    ax1_r.yaxis.label.set_color(ACCENT3)
    ax1_r.set_ylabel("RPM", fontsize=8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angular velocity (rad/s)")
    style_ax(ax1, "Angular Velocity & RPM vs Time")
    ax1.legend(fontsize=7, facecolor=DARK_BG, labelcolor=WHITE_FG,
               loc='upper left')
    ax1_r.legend(fontsize=7, facecolor=DARK_BG, labelcolor=WHITE_FG,
                 loc='lower right')

    # 2. Torque breakdown vs time
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.plot(t, sim["tau_creep"] * 1e9, color=ACCENT1,  lw=1.5, label='Thermal creep')
    ax2.plot(t, sim["tau_mol"]   * 1e9, color=ACCENT2,  lw=1.5, label='Molecular rebound')
    ax2.plot(t, sim["tau_rad"]   * 1e9, color=ACCENT3,  lw=1.5, label='Radiation pressure')
    ax2.plot(t, sim["tau_drag"]  * 1e9, color=ACCENT4,  lw=1.5, label='Viscous drag')
    ax2.plot(t, sim["tau_net"]   * 1e9, color=WHITE_FG, lw=2.2, label='Net torque', alpha=0.9)
    ax2.axhline(0, color=GRID_COL, linewidth=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Torque (nN·m)")
    style_ax(ax2, "Torque Breakdown vs Time")
    ax2.legend(fontsize=7, facecolor=DARK_BG, labelcolor=WHITE_FG,
               ncol=2, loc='upper right')

    # 3. Temperature profile across vane
    ax3 = fig.add_subplot(gs[1, :2])
    x_nodes = np.linspace(0, P.vane_thickness * 1e3, P.n_vane_nodes)
    T_prof  = sim["T_profile"]
    cmap    = matplotlib.cm.hot
    for i, (x_i, T_i) in enumerate(zip(x_nodes, T_prof)):
        ax3.bar(x_i, T_i - 273.15, width=x_nodes[1]-x_nodes[0] if len(x_nodes)>1 else 0.02,
                color=cmap(0.2 + 0.6 * i / len(x_nodes)), alpha=0.85)
    ax3.plot(x_nodes, T_prof - 273.15, color=WHITE_FG, lw=1.5)
    ax3.set_xlabel("Position across vane thickness (mm)")
    ax3.set_ylabel("Temperature (°C)")
    # Annotate faces
    ax3.axvline(x_nodes[0],  color=VANE_BLACK_COL, lw=2, label='Black face')
    ax3.axvline(x_nodes[-1], color=VANE_WHITE_COL, lw=2, label='White face')
    ax3.text(x_nodes[0],  sim["T_black"]-273.15+0.1, f'  {sim["T_black"]-273.15:.2f}°C',
             color=WHITE_FG, fontsize=8, va='bottom')
    ax3.text(x_nodes[-1], sim["T_white"]-273.15+0.1, f'  {sim["T_white"]-273.15:.2f}°C',
             color=WHITE_FG, fontsize=8, va='bottom')
    style_ax(ax3, "Vane Temperature Profile (1-D Conduction)")
    ax3.legend(fontsize=7, facecolor=DARK_BG, labelcolor=WHITE_FG)

    # 4. RPM vs Pressure sweep
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.semilogx(pressures, rpm_sweep, color=ACCENT1, lw=2.0)
    ax4.axvline(10.0, color=ACCENT3, lw=1.2, linestyle='--', label='Sim pressure (10 Pa)')
    ax4.fill_between(pressures, rpm_sweep, alpha=0.15, color=ACCENT1)
    ax4.set_xlabel("Gas pressure (Pa)")
    ax4.set_ylabel("Terminal RPM")
    style_ax(ax4, "Terminal RPM vs Gas Pressure")
    ax4.legend(fontsize=7, facecolor=DARK_BG, labelcolor=WHITE_FG)
    # Regime labels
    ymax = max(rpm_sweep) if max(rpm_sweep) > 0 else 1
    ax4.text(0.02, ymax * 0.85, "Free-molecular\n(Kn ≫ 1)",
             color=WHITE_FG, fontsize=7, alpha=0.7)
    ax4.text(50, ymax * 0.85, "Transition\n(Kn ~ 1)\n← optimal",
             color=ACCENT3, fontsize=7, alpha=0.9)
    ax4.text(3000, ymax * 0.3, "Continuum\n(Kn ≪ 1)",
             color=WHITE_FG, fontsize=7, alpha=0.7)

    # 5. Torque pie chart at steady state
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor(DARK_BG)
    labels = ['Thermal\nCreep', 'Molecular\nRebound', 'Radiation\nPressure']
    tau_ss = [abs(sim["tau_creep"][-1]),
              abs(sim["tau_mol"][-1]),
              abs(sim["tau_rad"][-1])]
    total = sum(tau_ss)
    if total > 0:
        pcts  = [v / total * 100 for v in tau_ss]
        colors = [ACCENT1, ACCENT2, ACCENT3]
        wedges, texts, autotexts = ax5.pie(
            tau_ss, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90,
            textprops={'color': WHITE_FG, 'fontsize': 7},
        )
        for at in autotexts:
            at.set_color(DARK_BG)
            at.set_fontsize(7)
    ax5.set_title("Drive Torque Breakdown\n(steady state)",
                  color=WHITE_FG, fontsize=9, fontweight='bold')

    # 6. Phase portrait (omega vs theta)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(sim["thetas"] % (2 * math.pi), sim["omegas"],
             color=ACCENT1, lw=0.8, alpha=0.6)
    ax6.set_xlabel("θ mod 2π (rad)")
    ax6.set_ylabel("ω (rad/s)")
    style_ax(ax6, "Phase Portrait (ω vs θ)")

    # 7. Kinetic gas properties summary
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(DARK_BG)
    ax7.axis('off')
    props = [
        ("Gas pressure",         f"{P.pressure_Pa:.1f} Pa"),
        ("Mean free path λ",     f"{P.mean_free_path*1e6:.1f} μm"),
        ("Knudsen number Kn",    f"{P.knudsen_number:.3f}"),
        ("Irradiance G",         f"{P.irradiance:.0f} W m⁻²"),
        ("T_black",              f"{sim['T_black']-273.15:.2f} °C"),
        ("T_white",              f"{sim['T_white']-273.15:.2f} °C"),
        ("ΔT",                   f"{sim['T_black']-sim['T_white']:.3f} K"),
        ("Terminal ω",           f"{sim['omega_ss']:.4f} rad/s"),
        ("Terminal RPM",         f"{sim['rpm_ss']:.3f} rpm"),
        ("Moment of inertia I",  f"{P.I_total:.3e} kg·m²"),
        ("Vane arm length",      f"{P.arm_length*100:.1f} cm"),
        ("n_vanes",              f"{P.n_vanes}"),
    ]
    y_start = 0.97
    ax7.text(0.5, 1.02, "Simulation Parameters & Results",
             transform=ax7.transAxes, ha='center', va='top',
             color=WHITE_FG, fontsize=9, fontweight='bold')
    for i, (k, v) in enumerate(props):
        ax7.text(0.02, y_start - i * 0.08, k + ":",
                 transform=ax7.transAxes, ha='left', va='top',
                 color="#8b949e", fontsize=7.5)
        ax7.text(0.98, y_start - i * 0.08, v,
                 transform=ax7.transAxes, ha='right', va='top',
                 color=ACCENT1, fontsize=7.5, fontweight='bold')

    # 8. Schematic top-down view
    ax8 = fig.add_subplot(gs[2, 3])
    draw_radiometer_frame(ax8, sim["thetas"][-1],
                          sim["T_black"], sim["T_white"], sim["omega_ss"])
    ax8.set_title("Top-Down View (final state)",
                  color=WHITE_FG, fontsize=9, fontweight='bold',
                  pad=6)
    # Legend patches
    bp = mpatches.Patch(color=plt.cm.hot(0.7),  label='Black face (hot)')
    wp = mpatches.Patch(color=VANE_WHITE_COL, label='White face (cool)')
    ax8.legend(handles=[bp, wp], fontsize=7, facecolor=DARK_BG,
               labelcolor=WHITE_FG, loc='lower center')

    plt.savefig("crookes_radiometer_sim.png", dpi=150, facecolor=DARK_BG,
                bbox_inches='tight')
    print("  Static plot saved → crookes_radiometer_sim.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  ANIMATION  (top-down spinning radiometer)
# ─────────────────────────────────────────────────────────────────────────────

def make_animation(sim):
    print("  Generating animation (this may take ~30 s)…")
    fig_a, ax_a = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)

    # Sample every N-th frame to make a ~5 s gif at 24 fps
    n_frames = 120
    total    = len(sim["times"])
    indices  = np.linspace(0, total - 1, n_frames, dtype=int)

    def update(frame_idx):
        ax_a.cla()
        i = indices[frame_idx]
        draw_radiometer_frame(ax_a, sim["thetas"][i],
                               sim["T_black"], sim["T_white"], sim["omegas"][i])
        t_s = sim["times"][i]
        rpm = sim["omegas"][i] * 60 / (2 * math.pi)
        fig_a.suptitle(f"Crookes Radiometer  |  t = {t_s:.1f} s  |  {rpm:.2f} RPM",
                        color=WHITE_FG, fontsize=10, y=0.97)

    ani = FuncAnimation(fig_a, update, frames=n_frames, interval=42)
    ani.save("crookes_radiometer_anim.gif",
             writer=PillowWriter(fps=24),
             savefig_kwargs={'facecolor': DARK_BG})
    plt.close(fig_a)
    print("  Animation saved   → crookes_radiometer_anim.gif")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Main simulation
    sim_data = run_simulation()

    # Pressure sweep
    print("  Running pressure sweep (60 points)…")
    pressures, rpm_sweep = pressure_sweep()
    print("  Pressure sweep complete.\n")

    # Plots
    plot_results(sim_data, pressures, rpm_sweep)

    # Animation
    make_animation(sim_data)

    print("\n  ✓  Simulation complete.")
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  Output files:                                        │")
    print("  │   • crookes_radiometer_sim.png   (static dashboard)  │")
    print("  │   • crookes_radiometer_anim.gif  (spinning animation) │")
    print("  └──────────────────────────────────────────────────────┘")
