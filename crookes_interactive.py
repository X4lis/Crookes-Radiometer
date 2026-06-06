"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   CROOKES RADIOMETER — INTERACTIVE MULTIPHYSICS SIMULATION                  ║
║                                                                              ║
║   Controls:                                                                  ║
║   • Drag the ☀  SUN icon   — moves the collimated light beam                ║
║   • Drag the 🔥 HEAT spots — independent point heat sources                 ║
║   • Scroll wheel on bulb   — change gas pressure (Kn regime)                ║
║   • R key                  — reset to default                                ║
║   • Space                  — pause / resume                                  ║
║   • L key                  — toggle light on / off                          ║
║   • +/- keys               — irradiance up / down                           ║
║   • E key                  — cycle electrostatic charge (none/low/high/off) ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math, sys, os, time, collections
os.environ.setdefault("SDL_VIDEODRIVER", "x11")

import pygame
import pygame.gfxdraw
import numpy as np

from crookes_radiometer import (
    P, k_B, sigma, R_gas, N_A,
    thermal_creep_force, molecular_pressure_force,
    radiation_pressure_force, compute_drag_coefficient,
    convective_torque, electrostatic_torque,
    _glass_attenuation,
)

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
W, H       = 1400, 860
FPS_TARGET = 60

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":         (  8, 12, 16),
    "panel":      ( 13, 17, 23),
    "border":     ( 30, 38, 50),
    "fg":         (230, 237, 243),
    "dim":        (139, 148, 158),
    "blue":       ( 88, 166, 255),
    "red":        (247, 129, 102),
    "orange":     (255, 166,  87),
    "green":      ( 63, 185,  80),
    "purple":     (188, 140, 255),
    "teal":       ( 57, 211, 187),
    "gold":       (227, 179,  65),
    "glass":      ( 31, 111, 235),
    "glass_dim":  ( 15,  55, 120),
    "black_vane": ( 30,  30,  35),
    "white_vane": (235, 240, 248),
    "sun":        (255, 220,  50),
    "heat":       (255,  80,  20),
    "conv":       ( 80, 200, 255),   # convection streamlines
    "elec":       (210, 130, 255),   # electrostatic field lines
}

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT  (pixel regions)
# ─────────────────────────────────────────────────────────────────────────────
BULB_CX, BULB_CY = 490, 420        # 3-D view centre
BULB_R           = 210              # glass sphere radius (pixels)
SCALE            = BULB_R / P.arm_length / 17.5   # pixels per metre

DASH_X = 990                        # dashboard left edge
DASH_W = W - DASH_X - 12

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS STATE
# ─────────────────────────────────────────────────────────────────────────────
class State:
    # Rotor
    omega       = 0.0        # rad/s
    theta       = 0.0        # rad
    # Light source  (pixel coords of the handle)
    sun_px      = BULB_CX - 260
    sun_py      = BULB_CY - 240
    irradiance  = 800.0      # W/m²
    light_on    = True       # toggleable
    # Heat spots (up to 4 draggable point sources — default power = skin temperature 91°F)
    # power field × 15 W = actual radiated watts; 0.1018 ≈ palm at 91°F in 68°F room
    # Default positions: one spot below (90°) and one to the left (180°), both at
    # ~22–25 cm from the bulb centre, so they sit outside the drawn glass sphere
    # on screen.  These two positions produce reinforcing convective flows that
    # add constructively (τ ≈ 0.6 nN·m, ~2.7 RPM at atmospheric pressure).
    # Symmetric placement (e.g. opposite sides) causes the torques to cancel.
    SKIN_POWER  = 0.1018
    heat_spots  = [
        {"px": BULB_CX,        "py": BULB_CY + 250, "power": SKIN_POWER, "active": True},  # below
        {"px": BULB_CX - 220,  "py": BULB_CY,       "power": SKIN_POWER, "active": True},  # left
    ]
    # Gas
    pressure    = 10.0       # Pa
    T_gas       = 293.15     # K  (68°F ambient)
    # Sim time
    t           = 0.0
    paused      = False
    sim_speed   = 60.0      # simulated seconds per real second (1× = real-time)
    # History for plots  (ring buffers)
    HIST        = 400
    hist_t      = collections.deque(maxlen=400)
    hist_omega  = collections.deque(maxlen=400)
    hist_rpm    = collections.deque(maxlen=400)
    hist_tc     = collections.deque(maxlen=400)   # tau creep
    hist_tm     = collections.deque(maxlen=400)   # tau molecular
    hist_tr     = collections.deque(maxlen=400)   # tau radiation
    hist_td     = collections.deque(maxlen=400)   # tau drag
    hist_tcv    = collections.deque(maxlen=400)   # tau convection
    hist_te     = collections.deque(maxlen=400)   # tau electrostatic
    # Particle cloud
    N_PART      = 180
    # Vane temps (computed each tick)
    T_black     = 296.0
    T_white     = 293.2
    # Interaction
    dragging    = None      # "sun" | "heat:0" | "heat:1" | "heat:2"
    # Cached torques for display
    tau_creep   = 0.0
    tau_mol     = 0.0
    tau_rad     = 0.0
    tau_drag    = 0.0
    tau_conv    = 0.0
    tau_elec    = 0.0
    tau_net     = 0.0
    # Convection visualisation state
    conv_U      = 0.0        # dominant exterior plume speed [m/s] for display
    # Per-heat-spot glass attenuation data cached each physics tick for the
    # visualiser: list of dicts with keys 'f_glass', 'U_ext', 'U_int'
    conv_glass_data = []
    # Electrostatic charge state
    # Cycle: 0 = none, 1 = low (+1 nC), 2 = high (+5 nC), 3 = low (-1 nC)
    _CHARGE_LEVELS = [0.0, 1e-9, 5e-9, -1e-9]
    _charge_idx    = 0
    @property
    def charge_C(self):
        return self._CHARGE_LEVELS[self._charge_idx]
    def cycle_charge(self):
        self._charge_idx = (self._charge_idx + 1) % len(self._CHARGE_LEVELS)

S = State()

# ─────────────────────────────────────────────────────────────────────────────
# PARTICLE CLOUD
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)
# Positions in pixel space relative to bulb centre
r_p   = BULB_R * 0.88 * rng.random(S.N_PART) ** (1/3)
th_p  = rng.uniform(0, 2*math.pi, S.N_PART)
px_p  = (r_p * np.cos(th_p)).astype(float)
py_p  = (r_p * np.sin(th_p)).astype(float)
vx_p  = rng.uniform(-0.9, 0.9, S.N_PART)
vy_p  = rng.uniform(-0.9, 0.9, S.N_PART)

def step_particles(dt_px=1.0):
    global px_p, py_p, vx_p, vy_p
    # Thermal boost near hot vane faces
    T_norm = min(1.0, (S.T_black - S.T_gas) / 10.0)
    speed  = 0.55 + 1.8 * T_norm
    vx_p  += rng.uniform(-0.06, 0.06, S.N_PART) * speed
    vy_p  += rng.uniform(-0.06, 0.06, S.N_PART) * speed
    # Clamp speed
    spd    = np.hypot(vx_p, vy_p)
    cap    = 1.6 + 1.2 * T_norm
    mask   = spd > cap
    vx_p[mask] *= cap / spd[mask]
    vy_p[mask] *= cap / spd[mask]
    px_p  += vx_p
    py_p  += vy_p
    # Bounce off sphere
    r2    = px_p**2 + py_p**2
    rlim  = (BULB_R * 0.91)**2
    hit   = r2 > rlim
    if hit.any():
        r_hit   = np.sqrt(r2[hit])
        nx      = px_p[hit] / r_hit
        ny      = py_p[hit] / r_hit
        vn      = vx_p[hit] * nx + vy_p[hit] * ny
        vx_p[hit] -= 2 * vn * nx
        vy_p[hit] -= 2 * vn * ny
        px_p[hit]  = nx * BULB_R * 0.88
        py_p[hit]  = ny * BULB_R * 0.88

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS TICK
# ─────────────────────────────────────────────────────────────────────────────
def sun_direction():
    """Unit vector from sun handle toward bulb centre."""
    dx = BULB_CX - S.sun_px
    dy = BULB_CY - S.sun_py
    d  = math.hypot(dx, dy) + 1e-9
    return dx / d, dy / d


def effective_irradiance_on_vane(vane_angle):
    """
    Irradiance on one vane's black face from the sun + heat spots.
    vane_angle: arm angle in world frame.
    Black face normal = outward along arm.
    """
    sdx, sdy = sun_direction()
    # Black face normal (outward from pivot)
    nx = math.cos(vane_angle)
    ny = math.sin(vane_angle)
    cos_sun = max(0.0, sdx * nx + sdy * ny)
    G_sun   = S.irradiance * cos_sun if S.light_on else 0.0

    # Heat spots: inverse-square from spot to vane centre
    G_heat = 0.0
    vcx    = BULB_CX + (P.arm_length * SCALE) * math.cos(vane_angle)
    vcy    = BULB_CY + (P.arm_length * SCALE) * math.sin(vane_angle)
    for hs in S.heat_spots:
        if not hs["active"]:
            continue
        dpx = vcx - hs["px"]
        dpy = vcy - hs["py"]
        dist_px = math.hypot(dpx, dpy) + 1e-6
        dist_m  = dist_px / SCALE
        # Irradiance from point source: P / (2π r²)  (one-sided hemisphere)
        P_watt  = hs["power"] * 15.0   # power in watts
        G_heat += P_watt / (2 * math.pi * dist_m**2)

    return G_sun + G_heat


def solve_temperatures_fast(G_black, G_white):
    """
    Fast Newton solve for (T_black, T_white) given per-face irradiance.
    Same physics as the full solver but inlined for speed.
    """
    L     = P.vane_thickness
    k     = P.thermal_conductivity_vane
    eps_b = P.emissivity_black
    eps_w = P.emissivity_white
    Tg    = S.T_gas
    M_air = P.molar_mass_air
    # Free-molecular h
    rho   = S.pressure * M_air / (R_gas * Tg)
    v_m   = math.sqrt(8 * k_B * Tg / (math.pi * M_air / N_A))
    h_fm  = 0.9 * (1.4 + 1) / (1.4 - 1) * k_B / (4 * M_air / N_A) * rho * v_m
    c_k   = k / L
    # Warm-start from ambient — never assume heat is present
    Tb, Tw = Tg, Tg
    # If both faces receive identical (or zero) irradiance, temperatures must
    # be equal — short-circuit to avoid Newton drift creating phantom ΔT.
    if abs(G_black - G_white) < 1e-9 and G_black < 1e-9:
        return Tg, Tg
    for _ in range(30):
        f1 = G_black - eps_b * sigma * Tb**4 - c_k*(Tb-Tw) - h_fm*(Tb-Tg)
        f2 = G_white - eps_w * sigma * Tw**4 + c_k*(Tb-Tw) - h_fm*(Tw-Tg)
        j11 = -4*eps_b*sigma*Tb**3 - c_k - h_fm
        j12 =  c_k
        j21 =  c_k
        j22 = -4*eps_w*sigma*Tw**3 - c_k - h_fm
        det = j11*j22 - j12*j21
        if abs(det) < 1e-30: break
        dTb = -(f1*j22 - f2*j12) / det
        dTw = -(j11*f2 - j21*f1) / det
        Tb += dTb;  Tw += dTw
        Tb = max(Tg*0.5, min(Tg*8, Tb))
        Tw = max(Tg*0.5, min(Tg*8, Tw))
        if abs(dTb) + abs(dTw) < 1e-5: break
    return float(Tb), float(Tw)


def physics_tick(dt_real):
    """Advance simulation by dt_real seconds of wall time."""
    if S.paused or dt_real <= 0:
        return
    # Scale wall-clock dt by sim_speed, then sub-step so each physics
    # step stays ≤ 20 ms of simulated time (keeps RK4 stable at high speeds)
    sim_dt_total = min(dt_real, 0.05) * S.sim_speed
    max_substep  = 0.020   # s of sim time per substep
    n_sub        = max(1, int(math.ceil(sim_dt_total / max_substep)))
    dt           = sim_dt_total / n_sub
    for _ in range(n_sub):
        _physics_substep(dt)

def _physics_substep(dt):

    # Aggregate irradiance on black/white faces across all 4 vanes
    G_black_avg, G_white_avg = 0.0, 0.0
    for k in range(4):
        ang      = S.theta + k * math.pi / 2
        G_b      = effective_irradiance_on_vane(ang)
        # White face normal is opposite
        G_w      = effective_irradiance_on_vane(ang + math.pi)
        G_black_avg += G_b
        G_white_avg += G_w
    G_black_avg /= 4;  G_white_avg /= 4

    # Vane temperatures
    S.T_black, S.T_white = solve_temperatures_fast(G_black_avg, G_white_avg)

    # Build heat-spot physics list (convert pixel positions to physical coords)
    heat_spots_phys = []
    for hs in S.heat_spots:
        if not hs["active"]:
            continue
        dpx = hs["px"] - BULB_CX
        dpy = hs["py"] - BULB_CY
        dist_px = math.hypot(dpx, dpy) + 1e-6
        heat_spots_phys.append({
            "angle_rad": math.atan2(dpy, dpx),
            "dist_m":    dist_px / SCALE,
            "power_W":   hs["power"] * 15.0,
        })

    # Torques
    P.pressure_Pa = S.pressure
    F_creep  = thermal_creep_force(S.T_black, S.T_white, S.T_gas, S.pressure)
    F_mol    = molecular_pressure_force(S.T_black, S.T_white, S.pressure)
    F_rad_v  = 0.0
    # Per-vane radiation force using true geometry
    for k in range(4):
        ang      = S.theta + k * math.pi / 2
        G_b      = effective_irradiance_on_vane(ang)
        G_w      = effective_irradiance_on_vane(ang + math.pi)
        A        = P.vane_width * P.vane_height
        from crookes_radiometer import c
        F_rad_v += ((2*(1-P.absorptivity_white)*G_w + P.absorptivity_white*G_w)
                    - P.absorptivity_black * G_b) * A / c

    r         = P.arm_length
    tau_c     = F_creep * r * P.n_vanes
    tau_m     = F_mol   * r * P.n_vanes
    tau_r     = F_rad_v * r
    b         = compute_drag_coefficient(S.pressure, S.T_gas)
    tau_d     = -b * S.omega

    # ── New: convective and electrostatic torques ─────────────────────────────
    # Cache per-spot glass attenuation data for the visualiser.
    # We replicate the exterior plume speed calculation here (cheap) so the
    # visualiser has U_ext, U_int, and f_glass without re-running physics.
    _k_air_vis = 0.026
    _g_vis     = 9.81
    _beta_vis  = 1.0 / S.T_gas
    _H_vis     = 0.060
    _glass_cache = []
    for _hs in heat_spots_phys:
        _dist   = max(_hs["dist_m"], 0.010)
        _dT_ext = min(_hs["power_W"] / (4 * math.pi * _k_air_vis * _dist), 80.0)
        _U_vert = math.sqrt(max(0.0, _g_vis * _beta_vis * _dT_ext * _H_vis))
        _U_ext  = max(0.0, 0.45 * _U_vert)   # mean horiz component (no noise for vis)
        _f, _Ui = _glass_attenuation(_U_ext, S.pressure, S.T_gas, _dT_ext)
        _glass_cache.append({
            "f_glass": _f,
            "U_ext":   _U_ext,
            "U_int":   _Ui,
            "dT_ext":  _dT_ext,
        })
    S.conv_glass_data = _glass_cache

    tau_cv, S.conv_U = convective_torque(
        heat_spots_phys, S.T_gas, S.pressure, S.theta)
    tau_el = electrostatic_torque(S.charge_C, S.theta)

    tau_drive = tau_c + tau_m + tau_r + tau_cv + tau_el
    tau_net   = tau_drive + tau_d

    I = P.I_total
    # Exact analytical integrator for linear drag system:
    #   dω/dt = (tau_drive - b*ω) / I
    # Solution: ω(t+dt) = ω⋅exp(-b⋅dt/I) + (tau_drive/b)⋅(1 - exp(-b⋅dt/I))
    # Unconditionally stable at any timestep, unlike RK4 which needs dt < 2.79⋅I/b.
    decay     = math.exp(-b * dt / I)
    S.omega   = S.omega * decay + (tau_drive / b) * (1.0 - decay)
    S.theta  += S.omega * dt
    S.t      += dt

    # Cache
    S.tau_creep = tau_c;  S.tau_mol = tau_m
    S.tau_rad   = tau_r;  S.tau_drag = -b * S.omega
    S.tau_conv  = tau_cv; S.tau_elec = tau_el
    S.tau_net   = tau_drive + S.tau_drag

    # History
    S.hist_t.append(S.t)
    S.hist_omega.append(S.omega)
    S.hist_rpm.append(S.omega * 60 / (2*math.pi))
    S.hist_tc.append(tau_c  * 1e9)
    S.hist_tm.append(tau_m  * 1e9)
    S.hist_tr.append(tau_r  * 1e9)
    S.hist_td.append(tau_d  * 1e9)
    S.hist_tcv.append(tau_cv * 1e9)
    S.hist_te.append(tau_el  * 1e9)


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def lerp_color(c1, c2, t):
    t = max(0, min(1, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def heat_color(T, T_min, T_max):
    frac = (T - T_min) / max(T_max - T_min, 0.001)
    frac = max(0.0, min(1.0, frac))
    if frac < 0.5:
        return lerp_color((30, 30, 80), (200, 80, 20), frac * 2)
    else:
        return lerp_color((200, 80, 20), (255, 240, 60), (frac - 0.5) * 2)

def alpha_surface(surf, alpha):
    surf.set_alpha(alpha)
    return surf

def draw_aa_circle(surf, color, cx, cy, r, width=1):
    if r <= 0: return
    pygame.gfxdraw.aacircle(surf, int(cx), int(cy), int(r), color)
    if width > 1:
        for i in range(1, width):
            pygame.gfxdraw.aacircle(surf, int(cx), int(cy), int(r - i), color)

def draw_filled_circle(surf, color, cx, cy, r):
    if r <= 0: return
    pygame.gfxdraw.aacircle(surf, int(cx), int(cy), int(r), color)
    pygame.gfxdraw.filled_circle(surf, int(cx), int(cy), int(r), color)

def draw_glow(surf, color, cx, cy, r, layers=5):
    """Soft glow by drawing concentric transparent circles."""
    for i in range(layers, 0, -1):
        alpha = int(55 / layers * i)
        rad   = r + (layers - i + 1) * 3
        gsurf = pygame.Surface((rad*2+2, rad*2+2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(gsurf, rad, rad, rad,
                                     (*color[:3], alpha))
        surf.blit(gsurf, (cx - rad, cy - rad))

def draw_line_aa(surf, color, p1, p2, width=1):
    if width == 1:
        pygame.draw.aaline(surf, color, p1, p2)
    else:
        pygame.draw.line(surf, color, p1, p2, width)

def polygon_aa(surf, color, points):
    if len(points) < 3: return
    pygame.gfxdraw.aapolygon(surf, [(int(x), int(y)) for x, y in points], color)
    pygame.gfxdraw.filled_polygon(surf, [(int(x), int(y)) for x, y in points], color)

def draw_arrow(surf, color, x0, y0, dx, dy, head=8, shaft=2):
    x1, y1 = x0 + dx, y0 + dy
    draw_line_aa(surf, color, (x0, y0), (x1, y1), shaft)
    if abs(dx) + abs(dy) < 3: return
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip    = (x1, y1)
    b1     = (x1 - ux*head + px*(head*0.4),
               y1 - uy*head + py*(head*0.4))
    b2     = (x1 - ux*head - px*(head*0.4),
               y1 - uy*head - py*(head*0.4))
    polygon_aa(surf, color, [tip, b1, b2])

def draw_dashed_circle(surf, color, cx, cy, r, n_dashes=32, dash_frac=0.55):
    for i in range(n_dashes):
        a0 = 2*math.pi * i / n_dashes
        a1 = 2*math.pi * (i + dash_frac) / n_dashes
        pts = []
        for t in np.linspace(a0, a1, 6):
            pts.append((cx + r*math.cos(t), cy + r*math.sin(t)))
        for j in range(len(pts)-1):
            pygame.draw.aaline(surf, color, pts[j], pts[j+1])

# ─────────────────────────────────────────────────────────────────────────────
# RADIOMETER DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def vane_corners(theta, k, offset_frac=0.0):
    """
    Returns (black_quad, white_quad) as lists of (px,py) tuples.
    offset_frac: slight perspective tilt (0 = flat top-down).
    """
    ang  = theta + k * math.pi / 2
    cx   = BULB_CX + P.arm_length * SCALE * math.cos(ang)
    cy   = BULB_CY + P.arm_length * SCALE * math.sin(ang)
    perp = ang + math.pi/2
    hw   = (P.vane_width / 2)  * SCALE
    hh   = (P.vane_height / 2) * SCALE * 0.55   # foreshorten for perspective
    eps  = 1.5   # pixel gap between faces

    dx, dy   = math.cos(ang),  math.sin(ang)
    px_, py_ = math.cos(perp), math.sin(perp)

    def c(sw, sh, face_off):
        x = cx + sw*hw*px_ + sh*hh*dx + face_off*dx
        y = cy + sw*hw*py_ + sh*hh*dy + face_off*dy
        return (x, y)

    black = [c(-1,-1,+eps), c(+1,-1,+eps), c(+1,+1,+eps), c(-1,+1,+eps)]
    white = [c(-1,-1,-eps), c(+1,-1,-eps), c(+1,+1,-eps), c(-1,+1,-eps)]
    return black, white, (cx, cy), ang


def draw_photon_rays(surf, frame):
    """Animated ray bundle from sun position toward bulb."""
    # Always draw the sun icon so it stays draggable when off
    sun_icon_col = C["sun"] if S.light_on else (80, 75, 40)
    spike_col    = C["gold"] if S.light_on else (60, 55, 30)

    if S.light_on:
        sdx, sdy = sun_direction()
        dist = math.hypot(BULB_CX - S.sun_px, BULB_CY - S.sun_py)
        perp_x, perp_y = -sdy, sdx
        n_rays  = 10
        spread  = 60
        phase   = (frame * 0.18) % 1.0
        for i in range(n_rays):
            t       = (i / (n_rays - 1) - 0.5) * spread
            phase_i = (phase + i * 0.1) % 1.0
            sx = S.sun_px + t * perp_x + phase_i * sdx * 18
            sy = S.sun_py + t * perp_y + phase_i * sdy * 18
            ex = sx + sdx * dist * 1.1
            ey = sy + sdy * dist * 1.1
            alpha = int(180 * (1 - abs(t) / spread) * (0.5 + 0.5 * math.sin(phase_i * math.pi)))
            ray_surf = pygame.Surface((W, H), pygame.SRCALPHA)
            pygame.draw.line(ray_surf, (*C["gold"], alpha),
                             (int(sx), int(sy)), (int(ex), int(ey)), 2)
            surf.blit(ray_surf, (0, 0))

    # Sun icon (always visible)
    draw_glow(surf, sun_icon_col, int(S.sun_px), int(S.sun_py), 14, layers=6)
    draw_filled_circle(surf, sun_icon_col, int(S.sun_px), int(S.sun_py), 14)
    draw_aa_circle(surf, (255,255,255) if S.light_on else (100,100,80),
                   int(S.sun_px), int(S.sun_py), 14, 1)
    # Crossed-out line when off
    if not S.light_on:
        pygame.draw.line(surf, (200, 60, 60),
                         (int(S.sun_px)-12, int(S.sun_py)-12),
                         (int(S.sun_px)+12, int(S.sun_py)+12), 2)
        pygame.draw.line(surf, (200, 60, 60),
                         (int(S.sun_px)+12, int(S.sun_py)-12),
                         (int(S.sun_px)-12, int(S.sun_py)+12), 2)
    # Spike rays
    for i in range(8):
        a   = i * math.pi / 4
        rx  = S.sun_px + 19 * math.cos(a)
        ry  = S.sun_py + 19 * math.sin(a)
        rx2 = S.sun_px + 25 * math.cos(a)
        ry2 = S.sun_py + 25 * math.sin(a)
        pygame.draw.aaline(surf, spike_col, (int(rx), int(ry)), (int(rx2), int(ry2)))


def draw_heat_spots(surf, frame):
    for idx, hs in enumerate(S.heat_spots):
        if not hs["active"]: continue
        px, py = int(hs["px"]), int(hs["py"])
        # Pulsing glow
        pulse  = 0.7 + 0.3 * math.sin(frame * 0.12 + idx * 1.3)
        r_glow = int(18 * hs["power"] / S.SKIN_POWER * pulse)
        draw_glow(surf, C["heat"], px, py, r_glow, layers=5)
        draw_filled_circle(surf, C["heat"], px, py, 10)
        draw_aa_circle(surf, (255,200,100), px, py, 10, 1)
        # Heat waves (concentric fading rings)
        for ring in range(3):
            phase = (frame * 0.07 + ring * 0.33) % 1.0
            r_ring = int(14 + phase * 28)
            alpha  = int(110 * (1 - phase))
            draw_aa_circle(surf, (*C["heat"], alpha), px, py, r_ring)


# Colour for interior (attenuated) convection streamlines — slightly warmer
# than the exterior "conv" blue so the two regions are visually distinct.
C_CONV_INT = (130, 220, 180)   # muted cyan-green: interior, attenuated


def draw_convection_flows(surf, frame):
    """
    Draw physically-derived convection streamlines for each active heat spot,
    showing:
      • EXTERIOR  – the buoyant plume rising from the source toward / around
                    the glass bulb in the direction of gravity (screen-down =
                    +y, so plumes rise in the -y direction).  Colour: cyan.
      • BOUNDARY  – a visual 'damping' flash at the glass surface to indicate
                    the boundary-layer resistance.  Colour: white flash.
      • INTERIOR  – the weak re-driven buoyant recirculation inside the bulb,
                    driven by heat conducted through the glass wall.
                    Colour: muted cyan-green, opacity ∝ f_glass.
    Opacity of exterior streamlines ∝ U_ext (pressure-independent: the plume
    exists at any pressure outside the sealed bulb).
    Opacity of interior streamlines ∝ f_glass = f_BL · (U_int/U_ext) · η_p,
    which naturally vanishes at low interior pressure.
    """
    active_spots = [(i, hs) for i, hs in enumerate(S.heat_spots) if hs["active"]]
    if not active_spots:
        return

    # One-time shared alpha surface for line drawing
    line_surf = pygame.Surface((W, H), pygame.SRCALPHA)

    for spot_i, (idx, hs) in enumerate(active_spots):
        hx, hy = hs["px"], hs["py"]

        # Retrieve per-spot glass attenuation data cached by the physics tick.
        glass = (S.conv_glass_data[spot_i]
                 if spot_i < len(S.conv_glass_data)
                 else {"f_glass": 0.0, "U_ext": 0.0, "U_int": 0.0, "dT_ext": 0.0})
        U_ext  = glass["U_ext"]
        U_int  = glass["U_int"]
        f_glass = glass["f_glass"]

        # ── Exterior streamlines ──────────────────────────────────────────────
        # Plumes always exist outside regardless of interior pressure.
        # Maximum opacity caps at U_ext = 0.25 m/s (typical warm-hand plume).
        ext_norm = min(1.0, U_ext / 0.25)
        if ext_norm > 0.01:
            # The plume rises vertically (-y in screen space) from the source,
            # then curves around the glass bulb.  We parameterise the path in
            # two segments:
            #   Segment A  (0 ≤ s < 0.55): straight vertical rise toward bulb.
            #   Segment B  (0.55 ≤ s ≤ 1): wraps tangentially around the bulb
            #              surface and fades at the antipodal point.
            n_streams = 4
            for s_i in range(n_streams):
                x_off = (s_i - (n_streams - 1) / 2.0) * 9
                phase = (frame * 0.038 + s_i * 0.22 + idx * 0.6) % 1.0
                n_pts = 22
                pts = []
                for j in range(n_pts):
                    s = (j / (n_pts - 1) + phase) % 1.0

                    if s < 0.55:
                        # Vertical buoyant rise from source toward bulb
                        # Gravity is screen-down (+y), so rise is -y.
                        rise_frac = s / 0.55   # 0→1 as plume travels to bulb
                        # Distance from source to bulb rim along the connecting line
                        dx_to_bulb = BULB_CX - hx
                        dy_to_bulb = BULB_CY - hy
                        dist_to_bulb = math.hypot(dx_to_bulb, dy_to_bulb) + 1e-6
                        # Plume starts rising (−y) and curves toward bulb centre
                        # as buoyancy entrains flow toward the obstacle.
                        # Blend: 0% at source → 100% pointing straight at bulb.
                        mix = rise_frac ** 1.4
                        # Pure vertical component (−y = upward on screen)
                        vx_rise = 0.0
                        vy_rise = -1.0
                        # Component directed at bulb centre
                        vx_bulb = dx_to_bulb / dist_to_bulb
                        vy_bulb = dy_to_bulb / dist_to_bulb
                        # Blended direction
                        bx = (1 - mix) * vx_rise + mix * vx_bulb
                        by = (1 - mix) * vy_rise + mix * vy_bulb
                        bl = math.hypot(bx, by) + 1e-9
                        bx /= bl;  by /= bl
                        # Travel distance: full gap from source to bulb rim
                        travel = rise_frac * (dist_to_bulb - BULB_R * 0.95)
                        px_s = hx + x_off + bx * travel
                        py_s = hy          + by * travel
                    else:
                        # Wrap around the bulb surface tangentially.
                        # Entry angle: direction from bulb centre to source.
                        wrap_frac = (s - 0.55) / 0.45   # 0→1 around the rim
                        ang_source = math.atan2(hy - BULB_CY, hx - BULB_CX)
                        # Sweep ±70° around the rim from the entry point
                        sweep = math.pi * 0.39 * wrap_frac
                        ang_rim = ang_source + math.pi + sweep   # antipodal side
                        r_rim = BULB_R * 1.06    # just outside the glass
                        px_s = BULB_CX + r_rim * math.cos(ang_rim)
                        py_s = BULB_CY + r_rim * math.sin(ang_rim)
                        x_off_local = 0   # lateral offset absorbed into sweep

                    pts.append((int(px_s), int(py_s)))

                for j in range(len(pts) - 1):
                    seg_alpha = int(ext_norm * 155 * (1.0 - j / n_pts))
                    if seg_alpha < 4:
                        continue
                    pygame.draw.line(line_surf, (*C["conv"], seg_alpha),
                                     pts[j], pts[j + 1], 2)

            # Upward-direction arrow at source
            pygame.draw.line(line_surf, (*C["conv"], int(ext_norm * 180)),
                             (int(hx), int(hy) - 6), (int(hx), int(hy) - 24), 2)
            tip = (int(hx), int(hy) - 24)
            b1  = (int(hx) - 4, int(hy) - 17)
            b2  = (int(hx) + 4, int(hy) - 17)
            pygame.gfxdraw.filled_trigon(line_surf,
                                          tip[0], tip[1], b1[0], b1[1], b2[0], b2[1],
                                          (*C["conv"], int(ext_norm * 180)))

        # ── Glass boundary ring flash ─────────────────────────────────────────
        # A thin arc on the bulb rim near the heat source shows where the
        # boundary layer resistance acts.  Brightness ∝ U_ext.
        if ext_norm > 0.02:
            ang_source = math.atan2(hy - BULB_CY, hx - BULB_CX)
            arc_half   = 0.55   # radians (≈ ±31°) arc half-width
            n_arc      = 14
            pulse      = 0.65 + 0.35 * math.sin(frame * 0.09 + idx)
            for ai in range(n_arc):
                a_frac = ai / (n_arc - 1)   # 0→1
                a_ang  = ang_source - arc_half + a_frac * 2 * arc_half
                # Fade toward arc edges
                edge_fade = 1.0 - abs(2 * a_frac - 1)
                arc_alpha = int(ext_norm * pulse * edge_fade * 120)
                rx = int(BULB_CX + BULB_R * math.cos(a_ang))
                ry = int(BULB_CY + BULB_R * math.sin(a_ang))
                if arc_alpha > 4:
                    pygame.gfxdraw.filled_circle(line_surf, rx, ry, 2,
                                                  (220, 240, 255, arc_alpha))

        # ── Interior streamlines ──────────────────────────────────────────────
        # Only visible at pressures where bulk interior flow can develop.
        # f_glass encodes the full attenuation (BL + conduction + pressure).
        int_norm = min(1.0, f_glass * 3.0)   # scale up so it's visible at f≈0.15
        if int_norm > 0.01:
            # Interior re-circulation: warm glass wall on the source-facing side
            # drives a gentle toroidal roll.  We draw it as a small circular
            # arc inside the bulb, centred on the bulb centre, rotating in the
            # direction the plume sweeps (clockwise or anti, depending on side).
            ang_source = math.atan2(hy - BULB_CY, hx - BULB_CX)
            n_int = 3
            for s_i in range(n_int):
                phase_i = (frame * 0.028 + s_i * 0.33 + idx * 0.5) % 1.0
                n_ipts  = 16
                i_pts   = []
                r_int   = BULB_R * (0.50 + 0.18 * s_i)   # nested arcs at 50/68/86 % R
                arc_span = math.pi * 0.7   # 126° arc
                for j in range(n_ipts):
                    t_frac = (j / (n_ipts - 1) + phase_i) % 1.0
                    a_ang  = ang_source + math.pi + t_frac * arc_span - arc_span / 2
                    ix = int(BULB_CX + r_int * math.cos(a_ang))
                    iy = int(BULB_CY + r_int * math.sin(a_ang))
                    i_pts.append((ix, iy))
                for j in range(len(i_pts) - 1):
                    seg_a = int(int_norm * 110 * (1.0 - j / n_ipts))
                    if seg_a < 4:
                        continue
                    pygame.draw.line(line_surf, (*C_CONV_INT, seg_a),
                                     i_pts[j], i_pts[j + 1], 1)

            # Small circulating arrow at the bulb centre side
            ang_arrow = ang_source + math.pi   # antipodal = inside near source
            ax_ = int(BULB_CX + BULB_R * 0.55 * math.cos(ang_arrow))
            ay_ = int(BULB_CY + BULB_R * 0.55 * math.sin(ang_arrow))
            draw_arrow(surf, (*C_CONV_INT, int(int_norm * 140)),
                       ax_, ay_,
                       int(-14 * math.sin(ang_arrow)),
                       int( 14 * math.cos(ang_arrow)),
                       head=5, shaft=1)

    surf.blit(line_surf, (0, 0))


def draw_electrostatic_field(surf, frame):
    """
    Draw faint radial field lines emanating from the glass bulb when charged.
    Uses a pulsing glow on the bulb rim and N equally-spaced dashed field lines.
    """
    Q = S.charge_C
    if abs(Q) < 5e-11:   # below display threshold
        return

    # Intensity proportional to |Q|
    q_norm  = min(1.0, abs(Q) / 5e-9)
    alpha_r = int(q_norm * 140)
    pulse   = 0.75 + 0.25 * math.sin(frame * 0.08)

    # Colour: positive charge → purple, negative → cool blue
    rim_col = C["elec"] if Q > 0 else C["blue"]

    # Glowing halo on the bulb rim
    draw_glow(surf, rim_col, BULB_CX, BULB_CY,
              int(BULB_R * pulse), layers=4)
    draw_aa_circle(surf, (*rim_col, alpha_r),
                   BULB_CX, BULB_CY, BULB_R + 3, 2)

    # Radial field lines
    n_lines = 16
    line_len = int(50 * q_norm)   # pixels beyond bulb rim
    for i in range(n_lines):
        angle = 2 * math.pi * i / n_lines + frame * 0.004
        x0 = BULB_CX + BULB_R * math.cos(angle)
        y0 = BULB_CY + BULB_R * math.sin(angle)
        x1 = BULB_CX + (BULB_R + line_len) * math.cos(angle)
        y1 = BULB_CY + (BULB_R + line_len) * math.sin(angle)
        seg_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.line(seg_surf, (*rim_col, alpha_r), (int(x0), int(y0)),
                         (int(x1), int(y1)), 1)
        surf.blit(seg_surf, (0, 0))
        # Small arrowhead at tip (direction: outward for +Q, inward for -Q)
        if Q > 0:
            draw_arrow(surf, (*rim_col, alpha_r),
                       int(x0 + (x1-x0)*0.6), int(y0 + (y1-y0)*0.6),
                       int((x1-x0)*0.4), int((y1-y0)*0.4), head=5, shaft=1)
        else:
            draw_arrow(surf, (*rim_col, alpha_r),
                       int(x1), int(y1),
                       int(-(x1-x0)*0.4), int(-(y1-y0)*0.4), head=5, shaft=1)

    # Charge label
    q_nC   = Q * 1e9
    sign_s = "+" if Q > 0 else ""
    lbl    = f"Q = {sign_s}{q_nC:.1f} nC"
    pygame.font.init()
    _fnt   = pygame.font.SysFont("monospace", 11)
    lbl_s  = _fnt.render(lbl, True, rim_col)
    surf.blit(lbl_s, (BULB_CX - lbl_s.get_width() // 2,
                      BULB_CY - BULB_R - 28))


def draw_radiometer(surf, frame):
    cx, cy = BULB_CX, BULB_CY
    T_b = S.T_black;  T_w = S.T_white

    # ── Bulb shadow / base ────────────────────────────────────────────────────
    shadow = pygame.Surface((BULB_R*2+40, 30), pygame.SRCALPHA)
    for i in range(15):
        alpha = int(80 * (1 - i/15))
        ellipse_rect = pygame.Rect(i, i//2, (BULB_R*2+40)-i*2, 30 - i)
        pygame.gfxdraw.filled_ellipse(shadow,
                                       (BULB_R+20), 15,
                                       BULB_R + 20 - i, 14 - i//2,
                                       (0, 0, 0, alpha))
    surf.blit(shadow, (cx - BULB_R - 20, cy + BULB_R - 10))

    # ── Glass bulb (multiple concentric circles for glass effect) ─────────────
    # Outer glow
    draw_glow(surf, C["glass_dim"], cx, cy, BULB_R, layers=4)
    # Fill with very dark tint
    glass_fill = pygame.Surface((BULB_R*2, BULB_R*2), pygame.SRCALPHA)
    pygame.gfxdraw.filled_circle(glass_fill, BULB_R, BULB_R, BULB_R,
                                  (20, 60, 120, 30))
    surf.blit(glass_fill, (cx - BULB_R, cy - BULB_R))
    # Main rim
    draw_aa_circle(surf, (*C["glass"], 200), cx, cy, BULB_R, 2)
    # Inner rim highlight
    draw_aa_circle(surf, (*C["glass"], 80), cx, cy, BULB_R - 4, 1)
    # Glare specular spot (top-left)
    glare_surf = pygame.Surface((50, 30), pygame.SRCALPHA)
    pygame.gfxdraw.filled_ellipse(glare_surf, 25, 15, 22, 12,
                                   (200, 230, 255, 45))
    surf.blit(glare_surf, (cx - BULB_R//3 - 25, cy - BULB_R//2 - 15))

    # ── Particles ─────────────────────────────────────────────────────────────
    T_norm = min(1.0, (T_b - S.T_gas) / 15.0)
    for i in range(S.N_PART):
        rx = cx + int(px_p[i])
        ry = cy + int(py_p[i])
        # Colour: hot near vane arms
        arm_dist = min(
            math.hypot(px_p[i] - P.arm_length * SCALE * math.cos(S.theta + k*math.pi/2),
                       py_p[i] - P.arm_length * SCALE * math.sin(S.theta + k*math.pi/2))
            for k in range(4)
        )
        warmth = max(0, 1 - arm_dist / (P.vane_width * SCALE * 1.5))
        col = lerp_color((60, 90, 180), (255, 140, 30), warmth * T_norm + warmth * 0.2)
        alpha = int(100 + 80 * warmth)
        p_surf = pygame.Surface((5, 5), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(p_surf, 2, 2, 2, (*col, alpha))
        surf.blit(p_surf, (rx - 2, ry - 2))

    # ── Spindle ───────────────────────────────────────────────────────────────
    pygame.draw.line(surf, C["dim"], (cx, cy - BULB_R + 20), (cx, cy + BULB_R - 20), 2)
    draw_filled_circle(surf, (100, 110, 120), cx, cy, 5)
    draw_aa_circle(surf, C["fg"], cx, cy, 5, 1)

    # ── Vane arms + vanes ─────────────────────────────────────────────────────
    for k in range(4):
        ang = S.theta + k * math.pi / 2
        ax_ = cx + P.arm_length * SCALE * math.cos(ang)
        ay_ = cy + P.arm_length * SCALE * math.sin(ang)
        pygame.draw.aaline(surf, C["dim"], (cx, cy), (int(ax_), int(ay_)))

        black_q, white_q, (vcx, vcy), vang = vane_corners(S.theta, k)

        # Irradiance on this vane → brightness boost
        G_b = effective_irradiance_on_vane(ang)
        heat_t = min(1.0, G_b / (S.irradiance * 1.5 + 1))
        vane_hot  = heat_color(T_b, S.T_gas - 1, S.T_gas + 20)
        vane_warm = lerp_color(C["black_vane"], vane_hot, heat_t)

        polygon_aa(surf, vane_warm, black_q)
        polygon_aa(surf, C["white_vane"], white_q)
        # Thin edge
        edge = [(int(black_q[i][0]), int(black_q[i][1])) for i in range(4)]
        pygame.gfxdraw.aapolygon(surf, edge, (*C["dim"], 120))

    # ── Thermal creep force arrows on vane edges ──────────────────────────────
    if abs(S.tau_creep) > 1e-13:
        scale_arrow = min(abs(S.tau_creep) / 1e-10 * 32, 55)
        sign = 1 if S.tau_creep >= 0 else -1
        for k in range(4):
            ang   = S.theta + k * math.pi / 2
            perp  = ang + math.pi/2
            # Position: at the far edge of the vane
            ex = BULB_CX + (P.arm_length + P.vane_width*0.55) * SCALE * math.cos(ang)
            ey = BULB_CY + (P.arm_length + P.vane_width*0.55) * SCALE * math.sin(ang)
            # Arrow along tangential direction
            tx = -math.sin(ang) * sign * scale_arrow
            ty =  math.cos(ang) * sign * scale_arrow
            draw_arrow(surf, C["teal"], int(ex), int(ey), int(tx), int(ty), head=7)

    # ── Rotation arc ──────────────────────────────────────────────────────────
    if abs(S.omega) > 0.01:
        arc_r   = BULB_R * 0.38
        arc_span = min(abs(S.omega) * 1.8, math.pi * 0.9)
        sign_om  = 1 if S.omega >= 0 else -1
        arc_pts  = []
        for t in np.linspace(S.theta, S.theta + sign_om * arc_span, 35):
            arc_pts.append((cx + arc_r * math.cos(t), cy + arc_r * math.sin(t)))
        for i in range(len(arc_pts) - 1):
            alpha = int(160 * i / len(arc_pts))
            pygame.draw.aaline(surf, (*C["teal"], alpha), arc_pts[i], arc_pts[i+1])
        # Arrow tip
        if len(arc_pts) >= 2:
            ep = arc_pts[-1];  ep2 = arc_pts[-2]
            tx = ep[0] - ep2[0];  ty = ep[1] - ep2[1]
            draw_arrow(surf, C["teal"], int(ep2[0]), int(ep2[1]),
                       int(tx*3), int(ty*3), head=7)

    # ── Bottom glass base (decorative) ───────────────────────────────────────
    base_rect = pygame.Rect(cx - 22, cy + BULB_R - 5, 44, 32)
    pygame.draw.rect(surf, (50, 60, 75), base_rect, border_radius=3)
    pygame.draw.rect(surf, C["border"], base_rect, 1, border_radius=3)


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def draw_plot_bg(surf, rect, title, font_sm):
    pygame.draw.rect(surf, C["panel"], rect, border_radius=6)
    pygame.draw.rect(surf, C["border"], rect, 1, border_radius=6)
    t_surf = font_sm.render(title, True, C["fg"])
    surf.blit(t_surf, (rect.x + 8, rect.y + 6))


def plot_line(surf, rect, data, color, y_min, y_max, x_pad=6, y_pad=24):
    if len(data) < 2: return
    rx, ry, rw, rh = rect.x, rect.y, rect.width, rect.height
    data_a = list(data)
    n = len(data_a)
    pts = []
    for i, v in enumerate(data_a):
        px_ = rx + x_pad + i * (rw - 2*x_pad) / max(n-1, 1)
        py_ = ry + rh - y_pad - (v - y_min) / max(y_max - y_min, 1e-9) * (rh - y_pad - 4)
        py_ = max(ry + 2, min(ry + rh - 2, py_))
        pts.append((int(px_), int(py_)))
    if len(pts) >= 2:
        pygame.draw.lines(surf, color, False, pts, 2)
    # Filled area under curve
    if len(pts) >= 2:
        fill_pts = pts + [(pts[-1][0], ry+rh-2), (pts[0][0], ry+rh-2)]
        fill_surf = pygame.Surface((rw, rh), pygame.SRCALPHA)
        local_pts = [(p[0]-rx, p[1]-ry) for p in fill_pts]
        if len(local_pts) >= 3:
            pygame.gfxdraw.filled_polygon(fill_surf, local_pts, (*color[:3], 35))
        surf.blit(fill_surf, (rx, ry))


def draw_gauge(surf, cx, cy, r, value, v_min, v_max, color, font, label):
    """Semicircular speedometer gauge."""
    # Background arc
    for i in range(120):
        a = math.pi + i * math.pi / 120
        frac = i / 119
        col  = lerp_color(C["blue"], C["red"], frac)
        px_  = cx + int((r-6) * math.cos(a))
        py_  = cy + int((r-6) * math.sin(a))
        pygame.gfxdraw.filled_circle(surf, px_, py_, 3, (*col, 90))

    # Needle
    frac = (value - v_min) / max(v_max - v_min, 1e-9)
    frac = max(0, min(1, frac))
    needle_a = math.pi + frac * math.pi
    nx = cx + int(r * 0.82 * math.cos(needle_a))
    ny = cy + int(r * 0.82 * math.sin(needle_a))
    n_col = lerp_color(C["blue"], C["red"], frac)
    draw_line_aa(surf, n_col, (cx, cy), (nx, ny), 3)
    draw_filled_circle(surf, C["fg"], cx, cy, 5)

    # Value text — placed ABOVE the pivot so it stays within the gauge box.
    val_surf = font.render(f"{value:.2f}", True, color)
    lbl_surf = font.render(label, True, C["dim"])
    # Stack: label on top, value below it, both centred, sitting above cy.
    total_h = lbl_surf.get_height() + 2 + val_surf.get_height()
    label_y = cy - total_h - 4
    surf.blit(lbl_surf, (cx - lbl_surf.get_width()//2, label_y))
    surf.blit(val_surf, (cx - val_surf.get_width()//2,
                         label_y + lbl_surf.get_height() + 2))

    # Tick marks — kept inside the arc radius (no outward labels to save space)
    for tick in range(6):
        ta = math.pi + tick * math.pi / 5
        t1x = cx + int((r-10) * math.cos(ta))
        t1y = cy + int((r-10) * math.sin(ta))
        t2x = cx + int(r * math.cos(ta))
        t2y = cy + int(r * math.sin(ta))
        pygame.draw.aaline(surf, C["dim"], (t1x, t1y), (t2x, t2y))


def draw_torque_bars(surf, rect, font_sm):
    labels  = ["Creep", "Mol.", "Rad.", "Drag", "Conv", "Elec"]
    values  = [S.tau_creep*1e9, S.tau_mol*1e9, S.tau_rad*1e9,
               S.tau_drag*1e9,  S.tau_conv*1e9, S.tau_elec*1e9]
    colors  = [C["blue"], C["red"], C["orange"], C["green"], C["conv"], C["elec"]]
    v_max   = max(abs(v) for v in values) * 1.3 if any(v for v in values) else 1.0

    bw = (rect.width - 20) // 6 - 3
    for i, (lbl, val, col) in enumerate(zip(labels, values, colors)):
        bx  = rect.x + 10 + i * (bw + 3)
        mid = rect.y + rect.height // 2
        bar_h = int(abs(val) / max(v_max, 1e-15) * (rect.height//2 - 28))
        bar_h = max(bar_h, 2)
        if val >= 0:
            bar_r = pygame.Rect(bx, mid - bar_h, bw, bar_h)
        else:
            bar_r = pygame.Rect(bx, mid, bw, bar_h)
        pygame.draw.rect(surf, col, bar_r, border_radius=3)
        pygame.draw.rect(surf, (*col, 120), bar_r.inflate(2,2), 1, border_radius=3)
        # Zero line
        pygame.draw.line(surf, C["border"],
                          (rect.x+5, mid), (rect.x+rect.width-5, mid), 1)
        # Label — always at the very bottom of the bar region
        l_s = font_sm.render(lbl, True, col)
        surf.blit(l_s, (bx + bw//2 - l_s.get_width()//2, rect.y + rect.height - 18))
        # Value — for positive bars draw below the bar top (not above the title);
        # for negative bars draw just below the bar bottom.
        v_s = font_sm.render(f"{val:.2f}", True, C["dim"])
        if val >= 0:
            # Just below the bar's top edge, clear of the title row
            v_y = mid - bar_h + 2
        else:
            v_y = mid + bar_h + 2
        surf.blit(v_s, (bx + bw//2 - v_s.get_width()//2, v_y))


def draw_temp_gradient(surf, rect, font_sm):
    """Horizontal gradient bar showing T_black → T_white."""
    T_b, T_w = S.T_black, S.T_white
    gx, gy   = rect.x + 6, rect.y + 28
    gw, gh   = rect.width - 12, 30
    # Draw gradient
    for i in range(gw):
        frac = i / max(gw-1, 1)
        T    = T_b + (T_w - T_b) * frac
        col  = heat_color(T, min(T_b, T_w) - 0.5, max(T_b, T_w) + 0.5)
        pygame.draw.line(surf, col, (gx+i, gy), (gx+i, gy+gh))
    pygame.draw.rect(surf, C["border"], (gx, gy, gw, gh), 1)

    # Labels
    Tb_s = font_sm.render(f"{T_b-273.15:.3f}°C", True, C["orange"])
    Tw_s = font_sm.render(f"{T_w-273.15:.3f}°C", True, C["blue"])
    dT_s = font_sm.render(f"ΔT = {T_b-T_w:.4f} K", True, C["fg"])
    surf.blit(Tb_s, (gx, gy + gh + 4))
    surf.blit(Tw_s, (gx + gw - Tw_s.get_width(), gy + gh + 4))
    surf.blit(dT_s, (gx + gw//2 - dT_s.get_width()//2, gy + gh + 4))

    # Black/white face markers — placed below the temperature labels (below the bar)
    b_s = font_sm.render("■ Black", True, C["dim"])
    w_s = font_sm.render("White ■", True, C["dim"])
    marker_y = gy + gh + 4 + font_sm.get_height() + 2
    surf.blit(b_s, (gx, marker_y))
    surf.blit(w_s, (gx + gw - w_s.get_width(), marker_y))


def draw_dashboard(surf, fonts, frame):
    font_lg, font_md, font_sm, font_xs = fonts
    x0, y0 = DASH_X, 10
    W_D, H_D = DASH_W, H - 20

    # Panel background
    pygame.draw.rect(surf, C["panel"],
                     pygame.Rect(x0, y0, W_D, H_D), border_radius=8)
    pygame.draw.rect(surf, C["border"],
                     pygame.Rect(x0, y0, W_D, H_D), 1, border_radius=8)

    # Title
    title = font_md.render("CROOKES RADIOMETER", True, C["fg"])
    surf.blit(title, (x0 + W_D//2 - title.get_width()//2, y0 + 8))
    sub = font_sm.render("Multiphysics · Interactive", True, C["dim"])
    surf.blit(sub, (x0 + W_D//2 - sub.get_width()//2, y0 + 30))

    pad = 8
    y   = y0 + 54

    # ── GAUGE ────────────────────────────────────────────────────────────────
    g_h  = 130
    gauge_rect = pygame.Rect(x0 + pad, y, W_D - 2*pad, g_h)
    pygame.draw.rect(surf, C["bg"], gauge_rect, border_radius=5)
    pygame.draw.rect(surf, C["border"], gauge_rect, 1, border_radius=5)
    rpm  = S.omega * 60 / (2*math.pi)
    rpm_max = max(10.0, abs(rpm) * 1.5, 5.0)
    # Place gauge centre so arc (radius 68) and value text (cy+r//4+20) stay inside the box.
    # cy = y + 80 → arc top = y+12, arc bottom = y+148 (fits in g_h=130 with minor clip).
    # Use r=58 so full arc bottom = y+80+58 = y+138 < y+130 is tight; use cy = y+68, r=56.
    _gauge_r  = 56
    _gauge_cy = y + _gauge_r + 14          # arc top at y+14, arc bottom at y+14+2*56=y+126
    draw_gauge(surf, x0 + W_D//2, _gauge_cy, _gauge_r,
               abs(rpm), 0, rpm_max, C["blue"], font_sm, "RPM")
    y += g_h + pad

    # ── ω TRACE ───────────────────────────────────────────────────────────────
    trace_h = 90
    tr = pygame.Rect(x0 + pad, y, W_D - 2*pad, trace_h)
    draw_plot_bg(surf, tr, "ω  angular velocity (rad/s)", font_sm)
    if S.hist_omega:
        mn = min(S.hist_omega)
        mx = max(S.hist_omega) if max(S.hist_omega) > 0 else 0.5
        plot_line(surf, tr, S.hist_omega, C["blue"], mn * 1.1, mx * 1.15)
    # Current value
    v_s = font_md.render(f"{S.omega:.4f} rad/s", True, C["blue"])
    surf.blit(v_s, (tr.right - v_s.get_width() - 6, tr.y + 4))
    y += trace_h + pad

    # ── TORQUE BARS ───────────────────────────────────────────────────────────
    tb_h = 108
    tbr  = pygame.Rect(x0 + pad, y, W_D - 2*pad, tb_h)
    draw_plot_bg(surf, tbr, "Torque components (nN·m)", font_sm)
    draw_torque_bars(surf, tbr, font_sm)
    # Net torque value — placed at the bottom-right to avoid the title row
    nt_s = font_sm.render(f"net: {S.tau_net*1e9:.3f} nN·m", True, C["fg"])
    surf.blit(nt_s, (tbr.right - nt_s.get_width() - 6,
                     tbr.y + tbr.height - font_sm.get_height() - 4))
    y += tb_h + pad

    # ── TEMPERATURE GRADIENT ──────────────────────────────────────────────────
    tg_h = 100
    tgr  = pygame.Rect(x0 + pad, y, W_D - 2*pad, tg_h)
    draw_plot_bg(surf, tgr, "Vane temperature gradient", font_sm)
    draw_temp_gradient(surf, tgr, font_sm)
    y += tg_h + pad

    # ── PARAMETERS BOX ───────────────────────────────────────────────────────
    pb_h = H_D - (y - y0) - pad - 4
    pbr  = pygame.Rect(x0 + pad, y, W_D - 2*pad, pb_h)
    pygame.draw.rect(surf, C["bg"], pbr, border_radius=5)
    pygame.draw.rect(surf, C["border"], pbr, 1, border_radius=5)

    kn   = P.knudsen_number
    mfp  = P.mean_free_path * 1e6
    def k2f(k): return (k - 273.15) * 9/5 + 32   # Kelvin → °F
    charge_labels = ["none", "low +", "high +", "low -"]
    charge_lbl    = charge_labels[S._charge_idx]
    params = [
        ("Sim speed",  f"{S.sim_speed:.1f}x"),
        ("Ambient",    f"{k2f(S.T_gas):.1f}°F  ({S.T_gas-273.15:.1f}°C)"),
        ("Pressure",   f"{S.pressure:.2f} Pa"),
        ("Kn",         f"{kn:.3f}"),
        ("λ (MFP)",    f"{mfp:.0f} μm"),
        ("Irradiance", f"{S.irradiance:.0f} W/m²"),
        ("T_black",    f"{k2f(S.T_black):.2f}°F  ({S.T_black-273.15:.3f}°C)"),
        ("T_white",    f"{k2f(S.T_white):.2f}°F  ({S.T_white-273.15:.3f}°C)"),
        ("ΔT vane",    f"{S.T_black-S.T_white:.4f} K"),
        ("Heat src",   f"{sum(hs['power']*15 for hs in S.heat_spots if hs['active']):.2f} W  (~{k2f(305.93):.0f}°F skin)"),
        ("ω",          f"{S.omega:.4f} rad/s"),
        ("RPM",        f"{rpm:.3f}"),
        ("τ creep",    f"{S.tau_creep*1e9:.3f} nN·m"),
        ("τ drag",     f"{S.tau_drag*1e9:.3f} nN·m"),
        ("τ conv",     f"{S.tau_conv*1e9:.3f} nN·m  U={S.conv_U*100:.1f} cm/s"),
        ("τ elec",     f"{S.tau_elec*1e9:.3f} nN·m  Q={charge_lbl}"),
        ("Charge",     f"{S.charge_C*1e9:.2f} nC  [E]=cycle"),
    ]
    line_h = min(18, (pb_h - 10) // max(len(params), 1))
    for i, (k_, v_) in enumerate(params):
        ky = pbr.y + 6 + i * line_h
        k_s = font_xs.render(k_ + ":", True, C["dim"])
        v_s = font_xs.render(v_,       True, C["blue"])
        surf.blit(k_s, (pbr.x + 6, ky))
        surf.blit(v_s, (pbr.right - v_s.get_width() - 6, ky))


# ─────────────────────────────────────────────────────────────────────────────
# CONTROLS / HELP OVERLAY
# ─────────────────────────────────────────────────────────────────────────────


def draw_controls(surf, font_sm, font_xs):
    controls = [
        ("[ / ]",            "Sim speed ÷2 / ×2"),
        ("☀  Sun",          "Drag to move light source"),
        ("L",               "Toggle light on/off"),
        ("🔥 Heat spots",   "Drag to reposition"),
        ("Scroll (bulb)",   "Change gas pressure"),
        ("+/-",             "Irradiance ±50 W/m²"),
        ("R",               "Reset all"),
        ("Space",           "Pause / Resume"),
        ("A",               "Add heat spot (max 3)"),
        ("D",               "Delete last heat spot"),
        ("E",               "Cycle charge: none→+1nC→+5nC→-1nC"),
    ]
    bx, by = 10, H - 10 - len(controls) * 18 - 14
    bw, bh = 310, len(controls) * 18 + 14
    bg_s = pygame.Surface((bw, bh), pygame.SRCALPHA)
    bg_s.fill((13, 17, 23, 200))
    surf.blit(bg_s, (bx, by))
    pygame.draw.rect(surf, C["border"], (bx, by, bw, bh), 1, border_radius=5)
    hdr = font_sm.render("Controls", True, C["fg"])
    surf.blit(hdr, (bx + bw//2 - hdr.get_width()//2, by + 3))
    for i, (key, desc) in enumerate(controls):
        ky = by + 18 + i * 18
        k_s = font_xs.render(key,  True, C["gold"])
        d_s = font_xs.render(desc, True, C["dim"])
        surf.blit(k_s, (bx + 6,  ky))
        surf.blit(d_s, (bx + 115, ky))


def draw_status_bar(surf, font_xs, paused):
    def k2f(k): return (k - 273.15) * 9/5 + 32
    light_str = "☀ ON " if S.light_on else "☀ OFF"
    status = "  ⏸  PAUSED  —  press SPACE to resume" if paused else \
             f"  {light_str}  |  t = {S.t:.2f}s   |   ω = {S.omega:.4f} rad/s   |   RPM = {S.omega*60/(2*math.pi):.3f}   |   P = {S.pressure:.1f} Pa   |   amb = {k2f(S.T_gas):.0f}°F   |   speed = {S.sim_speed:.1f}x"
    col = C["orange"] if paused else C["dim"]
    s   = font_xs.render(status, True, col)
    surf.blit(s, (6, H - 14))


# ─────────────────────────────────────────────────────────────────────────────
# HIT-TESTING
# ─────────────────────────────────────────────────────────────────────────────

def hit_sun(mx, my):
    return math.hypot(mx - S.sun_px, my - S.sun_py) < 22

def hit_heat(mx, my):
    for idx, hs in enumerate(S.heat_spots):
        if hs["active"] and math.hypot(mx - hs["px"], my - hs["py"]) < 20:
            return f"heat:{idx}"
    return None

def in_bulb(mx, my):
    return math.hypot(mx - BULB_CX, my - BULB_CY) < BULB_R


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("Crookes Radiometer — Interactive Multiphysics")

    clock  = pygame.time.Clock()

    # Fonts
    pygame.font.init()
    def make_font(size, mono=False):
        name = "monospace" if mono else "sans-serif"
        try:
            return pygame.font.SysFont(name, size)
        except:
            return pygame.font.Font(None, size)

    font_lg = make_font(22, mono=True)
    font_md = make_font(16, mono=True)
    font_sm = make_font(13, mono=True)
    font_xs = make_font(11, mono=True)
    fonts   = (font_lg, font_md, font_sm, font_xs)

    frame = 0
    last_time = time.perf_counter()

    running = True
    while running:
        now      = time.perf_counter()
        dt_real  = now - last_time
        last_time = now

        # ── EVENTS ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    S.paused = not S.paused
                elif event.key == pygame.K_l:
                    S.light_on = not S.light_on
                elif event.key == pygame.K_r:
                    S.omega      = 0.0
                    S.theta      = 0.0
                    S.t          = 0.0
                    S.sun_px     = BULB_CX - 260
                    S.sun_py     = BULB_CY - 240
                    S.irradiance = 800.0
                    S.light_on   = True
                    S.pressure   = 10.0
                    S.sim_speed  = 60.0
                    S.heat_spots = [
                        {"px": BULB_CX,       "py": BULB_CY + 250, "power": S.SKIN_POWER, "active": True},
                        {"px": BULB_CX - 220, "py": BULB_CY,       "power": S.SKIN_POWER, "active": True},
                    ]
                    S.hist_t.clear(); S.hist_omega.clear(); S.hist_rpm.clear()
                    S.hist_tc.clear(); S.hist_tm.clear()
                    S.hist_tr.clear(); S.hist_td.clear()
                    S.hist_tcv.clear(); S.hist_te.clear()
                    S._charge_idx = 0
                    S.tau_conv = 0.0; S.tau_elec = 0.0; S.conv_U = 0.0
                    S.conv_glass_data = []
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                   pygame.K_KP_PLUS):
                    S.irradiance = min(S.irradiance + 50, 3000)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    S.irradiance = max(S.irradiance - 50, 0)
                elif event.key == pygame.K_RIGHTBRACKET:
                    S.sim_speed = min(S.sim_speed * 2.0, 512.0)
                elif event.key == pygame.K_LEFTBRACKET:
                    S.sim_speed = max(S.sim_speed / 2.0, 0.25)
                elif event.key == pygame.K_a:
                    if len(S.heat_spots) < 4:
                        S.heat_spots.append({
                            "px": BULB_CX + rng.integers(-150, 150),
                            "py": BULB_CY + rng.integers(-150, 150),
                            "power": S.SKIN_POWER, "active": True,
                        })
                elif event.key == pygame.K_d:
                    if S.heat_spots:
                        S.heat_spots.pop()
                elif event.key == pygame.K_e:
                    S.cycle_charge()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button == 1:
                    if hit_sun(mx, my):
                        S.dragging = "sun"
                    else:
                        h = hit_heat(mx, my)
                        if h:
                            S.dragging = h

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    S.dragging = None

            elif event.type == pygame.MOUSEMOTION:
                if S.dragging == "sun":
                    S.sun_px, S.sun_py = event.pos
                elif S.dragging and S.dragging.startswith("heat:"):
                    idx = int(S.dragging.split(":")[1])
                    if idx < len(S.heat_spots):
                        S.heat_spots[idx]["px"] = event.pos[0]
                        S.heat_spots[idx]["py"] = event.pos[1]

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if in_bulb(mx, my):
                    S.pressure = max(0.1, S.pressure * (1.15 if event.y > 0 else 0.87))
                    P.pressure_Pa = S.pressure

        # ── PHYSICS ──────────────────────────────────────────────────────────
        physics_tick(dt_real)
        step_particles()

        # ── DRAW ─────────────────────────────────────────────────────────────
        screen.fill(C["bg"])

        # Left panel background
        pygame.draw.rect(screen, C["panel"],
                          pygame.Rect(0, 0, DASH_X - 2, H), border_radius=0)
        pygame.draw.line(screen, C["border"], (DASH_X-2, 0), (DASH_X-2, H), 1)

        # Title text in view area
        title_s = font_lg.render("Crookes Radiometer — Multiphysics", True, C["fg"])
        screen.blit(title_s, (10, 8))
        sub_s = font_xs.render(
            "Drag ☀ light · drag 🔥 heat sources · L = light · scroll = pressure · E = charge",
            True, C["dim"])
        screen.blit(sub_s, (10, 32))
        # Sim-speed badge
        spd_col  = C["teal"] if S.sim_speed >= 1.0 else C["orange"]
        spd_text = font_sm.render(f"⏩ {S.sim_speed:.1f}x  [ slower  faster ]", True, spd_col)
        # Place above the controls legend box (which starts at H - 10 - 11*18 - 14 = 638)
        screen.blit(spd_text, (10, H - 10 - 11*18 - 14 - font_sm.get_height() - 6))

        # Photon rays (behind bulb)
        draw_photon_rays(screen, frame)

        # Convection streamlines (behind bulb, in front of rays)
        draw_convection_flows(screen, frame)

        # Electrostatic field lines (on the bulb rim)
        draw_electrostatic_field(screen, frame)

        # Heat spots (behind bulb)
        draw_heat_spots(screen, frame)

        # Radiometer
        draw_radiometer(screen, frame)

        # Dashboard
        draw_dashboard(screen, fonts, frame)

        # Controls legend
        draw_controls(screen, font_sm, font_xs)

        # Status bar
        draw_status_bar(screen, font_xs, S.paused)

        pygame.display.flip()
        clock.tick(FPS_TARGET)
        frame += 1

    pygame.quit()


if __name__ == "__main__":
    main()
