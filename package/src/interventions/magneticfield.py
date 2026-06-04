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

