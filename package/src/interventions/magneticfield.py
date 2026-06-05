"""
Functions in this file will verify the differential equations in task assigned.
We want the calculations in the document to be correct.
"""

#Compute Loop Dipole Moment
#Checking whether a current-carrying wire around Mars creates a dipole moment.
def compute_loop_dipole_moment(
	loop_current: float,
	radius: float,
) ->float:


	"""
	Equation: M = I * pi * R^2

	This computes the magnetic dipole moment of a superconducting loop around Mars

	Important parameters:
	loop_current : float
		current through loop (A)

	radius:float
		radius of loop (m)

	Returns
	float
		dipole moment (A*m^2)
	"""

	loop_area = math.pi * radius**2 #computing are enclosed by loop
	dipole_moment = loop_current * loop_area #applying the dipole moment equation
	return dipole_moment

#Compute Required Magnetopause Field
#Checking how much magnetic field is needed to balance solar wind pressure.
def compute_required_magnetopause_field(
	solar_wind_pressure: float,
	mu_0: float,
) -> float:

	"""
	Equation: B_mp = sqrt(2 * mu_0 * P_ram)

	This computes the magnetic field required to balance the incoming solar wind pressure.

	Important parameters:
	solar_wind_pressure: float
		solar wind pressure (Pa)

	mu_0 : float
		permeability of free space

	Returns
	float
		required magnetic field (Tesla)
	"""

	required_field = math.sqrt( 2.0 * mu_0 * solar_wind_pressure )
	return required_field

#Compute Dipole Field Strength
#Checking how strong the magnetic field is at a given distance.
def compute_dipole_field(
	dipole_moment: float,
	distance: float,
	mu_0: float,
) ->float:

	"""
	Equation: B(r) = mu_0 * M / (4 * pi * r^3)

	This computes the magnetic field strength at a specified distance from Mars.

	Important parameters:
	dipole_moment : float
		dipole moment (A*m^2)

	distance : float
		distance from Mars center (m)

	mu_0 : float
		permeability of free space

	Returns
	float
		magnetic field strength (Tesla)
	"""

	numerator = mu_0 * dipole_moment
	denominator = ( 4.0 * math.pi * distance**3 )
	magnetic_field = numerator / denominator
	return magnetic_field

#Compute Magnetopause Standoff Distance
#Checking how far the magnetic shield extends from Mars.
def compute_magnetopause_standoff(
	dipole_moment: float,
	solar_wind_pressure: float,
	mu_0: float,
) ->float:

	"""
	Equation: r_mp = (mu_0 * M / (4 * pi * B_mp))^(1/3)

	This computes the distance where the magnetic field balances the solar wind pressure.

	Important parameters:
	dipole_moment : float
		dipole moment (A*m^2)

	solar_wind_pressure : float
		solar wind pressure (Pa)

	mu_0 : float
		permeability of free space

	Returns
	float
		magnetopause standoff distance (m)
	"""

	boundary_field = compute_required_magnetopause_field(
		solar_wind_pressure,
		mu_0,
	)
	standoff_distance = (
		(mu_0 * dipole_moment)
		/ (4.0 * math.pi * boundary_field)
	) ** (1.0 / 3.0)
	return standoff_distance

#Compute Shield Effectiveness
#Checking how effective the magnetic shield is.
def compute_shield_effectiveness(
	standoff_distance: float,
	mars_radius: float,
) -> float:
	"""
	Assumption:
	
	r_mp <= 1.0 R_M -> no shield
	r_mp >= 1.5 R_M -> full shield

	This converts magnetopause distance into a shielding effectiveness value.

	Important parameters:
	standoff_distance : float
		magnetopause distance (m)

	mars_radius : float
		Mard radius (m)

	Returns
	float
		shield effectiveness from 0 to 1
	"""

	effectiveness = (
		standoff_distance - mars_radius
	) / (0.5 * mars_radius)

	effectiveness = max(
		0.0,
		min(1.0, effectiveness)
	)
	return effectiveness

#Compute Effective Escape Rate
#Checking how much atmospheric escape remains after shielding.
def compute_effective_escape_rate(
	base_escape_rate: float,
	shield_fraction: float,
) -> float:
	
	"""
	Equation : E_eff = E_base * (1 - f_shield)

	This computes the atmospheric escape rate after magnetic shielding has been applied.

	Important parameters:
	base_escape_rate : float
		original atmospheric escape rate (kg/s)

	shield_fraction : float
		shield effectiveness from0 to 1

	Returns
	float
		effective atmospheric escape rate (kg/s)
	"""

	effective_escape = (
		base_escape_rate * (1.0 - shield_fraction))
	return effective_escape

#Compute Pressure Lost Rate
#Checking how atmospheric escape affects surface pressure.
def compute_pressure_loss_rate(
	escape_rate: float,
	gravity: float,
	planet_radius: float,
) -> float:

	"""
	Equation: dP/dt = -(escape_rate * g) / (4 * pi * R^2)

	This converts atmospheric mass loss into pressure loss over time.

	Important parameters:
	escape_rate : float
		atmospheric escape rate (kg/s)
	
	gravity : float
		Mars gravity (m/s^2)
	
	planet_radius : float
		Mars radius (m)

	Returns
	float
		pressure loss rate (Pa/s)
	"""
	surface_area = ( 4.0 * math.pi * planet_radius**2)

	pressure_loss_rate = ( -escape_rate * gravity / surface_area)
	return pressure_loss_rate

#Verify Magnetic Intervetion
#Runs the full chain of calculations from the paper.
def verify_magnetic_intervention(
	loop_current: float,
	mars_radius: float,
	solar_wind_pressure: float,
	base_escape_rate: float,
	gravity: float,
	mu_0: float,
):

	"""
	This verifies the complete magnetic intervention.

	Calculation chain:
	
	loop current
		->
	dipole moment
		->
	magnetopause distance
		->
	shield effectiveness
		->
	escape rate
		->
	pressure lost
	
	Returns
	dict
		all calculated values
	"""
	dipole_moment = compute_loop_dipole_moment(
		loop_current,
		mars_radius,
	)

	standoff_distance = compute_magnetopause_standoff(
		dipole_moment,
		solar_wind_pressure
		mu_0,
	)

	shield_fraction = compute_shield_effectiveness(
		standoff_distance,
		mars_radius,
	)

	escape_rate = compute_effective_escape_rate(
		base_escape_rate,
		shield_fraction,
	)

	return{
		"dipole_moment": dipole_moment,
		"standoff_distance": standoff_distance,
		"shield_fraction": shield_fraction,
		"escape_rate": escape_rate
		"pressure_loss_rate": pressure_loss_rate,
	}
