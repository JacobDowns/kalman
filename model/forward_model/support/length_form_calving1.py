import numpy as np
from dolfin import *

class LengthForm(object):
    """
    Set up the variational form for length, or more specically, the H(x=1)=0
    boundary condition used to determine length.
    """

    def __init__(self, model):

        # DG thickness
        H = model.H
        H_c = model.H_c
        # Rate of change of H
        dHdt = model.dHdt
        # Ice sheet length
        L = model.L
        # Rate of change of L
        dLdt = model.dLdt
        # Real test function
        chi = model.chi
        # Boundary measure
        ds1 = dolfin.ds(subdomain_data = model.boundaries)
        # SMB expression
        adot_prime = model.adot_prime
        # Ice stream width
        width = model.width
        # Spatial coordinate
        x_spatial = SpatialCoordinate(model.mesh)
        # Time partial of width
        dWdt = dLdt*x_spatial[0]/L*width.dx(0)
        # Calving velocity
        H_calving = Constant(10.0)
        # Velocity
        ubar = model.ubar

        # HELPER FUNCTIONS
        def softplus(y1,y2,alpha=1):
            # The softplus function is a differentiable approximation
            # to the ramp function.  Its derivative is the logistic function.
            # Larger alpha makes a sharper transition.
            return dolfin.Max(y1,y2) + (1./alpha)*dolfin.ln(1.+dolfin.exp(alpha*(dolfin.Min(y1,y2)-dolfin.Max(y1,y2))))

        U_calving = Constant(0.0)#Constant(10.0)*softplus(H_calving - H, 0, alpha=0.5)

        # GLOBAL - derived from integrating the mass conservation equation across
        # the model domain then using the boundary condition u_c = u_t - dL/dt.
        # Very stable, works good, not clear how to generalize to multidimensional case
        #R_len = (dLdt*width*H + L*width*(dHdt - adot_prime))*chi*dx + U_calving*width*H*chi*ds1(1)

        ### Length residual
        ### ====================================================================

        #R_length = (dHdt - 1./L*H_calving.dx(0)*dLdt)*chi*df.ds(1)
        #R_length = (dLdt*width*H + L*dWdt*H + L*width*(dHdt - adot_prime))*chi*dx + U_calving*width*H*chi*ds1(1)
        #R_length = dHdt*chi*ds1(1)

        R_length = (H_c - Constant(15.0))*chi*ds1(1)

        self.R_length = R_length
