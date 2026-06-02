import aerosandbox as asb
import aerosandbox.numpy as np

# =============================================================================
# 1. DEFINE CONTROL SURFACES
# =============================================================================
# For an aileron, we set symmetric=False (anti-symmetric deflection for roll).
# For flaps or elevators, you would set symmetric=True.
aileron = asb.ControlSurface(
    name="aileron",
    symmetric=False,    # Anti-symmetric deflection (one goes up, one goes down)
    deflection=10.0,     # Deflection angle in degrees (downwards-positive)
    hinge_point=0.75,   # Hinged at 75% of the local chord
    trailing_edge=True
)

# =============================================================================
# 2. DEFINE WING CROSS SECTIONS (WingXSec)
# =============================================================================
# We will build a wing with 3 sections to isolate the aileron to the outer panel.
root_xsec = asb.WingXSec(
    xyz_le=[0.0, 0.0, 0.0],
    chord=1.0,
    airfoil=asb.Airfoil("naca0012"),
    twist=0.0,
    control_surfaces=[] # No control surfaces on the inboard section
)

mid_xsec = asb.WingXSec(
    xyz_le=[0.1, 0.6, 0.0],
    chord=0.8,
    airfoil=asb.Airfoil("naca0012"),
    twist=0.0,
    control_surfaces=[aileron] # Aileron starts here and lofts to the tip
)

tip_xsec = asb.WingXSec(
    xyz_le=[0.25, 1.5, 0.0],
    chord=0.5,
    airfoil=asb.Airfoil("naca0012"),
    twist=0.0,
    control_surfaces=[] # Terminates at the tip
)

# =============================================================================
# 3. ASSEMBLE WING AND AIRPLANE
# =============================================================================
main_wing = asb.Wing(
    name="Main Wing",
    xsecs=[root_xsec, mid_xsec, tip_xsec],
    symmetric=True # Automatically mirrors the right wing to the left side
)

airplane = asb.Airplane(
    name="Simple AeroBuildup Airplane",
    xyz_ref=[0.25, 0.0, 0.0], # Reference point for moment calculations (typically CG)
    wings=[main_wing],
    fuselages=[]              # Can add asb.Fuselage objects here if needed
)

# =============================================================================
# 4. DEFINE FLIGHT ENVIRONMENT & OPERATING POINT
# =============================================================================
op_point = asb.OperatingPoint(
    atmosphere=asb.Atmosphere(altitude=0), # Sea level standard atmosphere
    velocity=30.0,                         # Freestream velocity (m/s)
    alpha=4.0,                             # Angle of attack (degrees)
    beta=0.0                               # Sideslip angle (degrees)
)

# =============================================================================
# 5. RUN AEROBUILDUP ANALYSIS
# =============================================================================
# AeroBuildup is an analytical/empirical method ideal for rapid conceptual design
analysis = asb.AeroBuildup(
    airplane=airplane,
    op_point=op_point
)

# Execute the analysis to get a dictionary of coefficients
aero_results = analysis.run()

# =============================================================================
# 6. DISPLAY RESULTS
# =============================================================================
print(f"--- Aerodynamic Coefficients for '{airplane.name}' ---")
print(f"Lift Coefficient (CL):     {aero_results['CL'][0]:.4f}")
print(f"Drag Coefficient (CD):     {aero_results['CD'][0]:.4f}")
print(f"Pitching Moment (Cm):      {aero_results['Cm'][0]:.4f}")
print(f"Rolling Moment (Cl):       {aero_results['Cl'][0]:.4f}")