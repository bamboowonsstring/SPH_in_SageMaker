#!/usr/bin/python
# -*- coding: utf-8 -*-

#from __future__ import print_function
from pysph.solver.application import Application

import pickle
import os

# Sagemaker Mount Path
prefix='/opt/ml/'
input_path=prefix+'input/data'

output_path = os.path.join(prefix, 'output/')                       # Error
model_path = os.path.join(prefix, 'model/')                         # Result
param_path = os.path.join(prefix, 
     'input/config/hyperparameters.json')     # Parameter

channel_name='training'
training_path = os.path.join(input_path, channel_name)

#ここからコピペ----------------------------------------------------------------------------------------------
#SageMaker実行時、実行前にカーネルをリスタートすること
from pysph.solver.application import Application

from numpy import ones_like, mgrid, sqrt

from pysph.base.utils import get_particle_array
from pysph.base.utils import get_particle_array_wcsph
from pysph.base.kernels import CubicSpline,Gaussian

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator,EulerIntegrator
from pysph.sph.integrator_step import WCSPHStep,EulerStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import XSPHCorrection, ContinuityEquation
from pysph.sph.wc.basic import TaitEOS, MomentumEquation
#define Step--------------------------------------------------------------------------------------------------
from pysph.sph.integrator_step import IntegratorStep
class WCSPHStep2(IntegratorStep):
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]
        
        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                   d_aw, d_ax, d_ay, d_az, d_arho, dt):
        dtb2 = 0.5*dt
        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_az[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]


    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                   d_aw, d_ax, d_ay, d_az, d_arho, dt):

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_az[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]
#define Application--------------------------------------------------------------------------------------------
class Test(Application):
    def initialize(self):
        import json
        with open(param_path, mode='r') as f:
            par = json.load(f)        
        self.co = float(par['co'])
        self.ro = float(par['ro'])
        self.hdx = float(par['hdx'])
        self.dx = float(par['dx'])
        self.alpha = float(par['alpha'])
        self.dt = float(par['dt'])
        self.tf = float(par['tf'])
        
    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(fluid=WCSPHStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        dt=self.dt, tf=self.tf,pfreq=10)

        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    TaitEOS(
                        dest='fluid', sources=None, 
                        rho0=self.ro,c0=self.co, gamma=7.0
                    ),
                ],
                real=False
            ),
            Group(equations=[
                ContinuityEquation(dest='fluid',  sources=['fluid']),
                MomentumEquation(
                    dest='fluid', sources=['fluid'],
                    alpha=self.alpha, beta=0.0, c0=self.co
                ),
                XSPHCorrection(
                    dest='fluid', sources=['fluid'], eps=0.0)
            ]),
        ]
        return equations    

    def create_particles(self):
        """Create the circular patch of fluid."""
        name = 'fluid'
        dx=self.dx
        hdx=self.hdx
        ro=self.ro

        x, y = mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
        x = x.ravel()
        y = y.ravel()

        m = ones_like(x)*dx*dx*ro
        h = ones_like(x)*hdx*dx
        rho = ones_like(x) *ro
        u = -100*x
        v = 100*y

        # remove particles outside the circle
        indices = []
        for i in range(len(x)):
            if sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
                indices.append(i)

        pa = get_particle_array_wcsph(x=x, y=y, m=m, rho=rho, h=h, u=u, v=v,
                                name=name)
        pa.remove_particles(indices)

        print("Elliptical drop :: %d particles"
              % (pa.get_number_of_particles()))

        pa.set_output_arrays(['x','y','u', 'v', 'rho', 'h', 'p', 'pid', 'tag', 'gid'])
        return [pa]
    
if __name__ == '__main__':
    app = Test(fname=model_path+"Data")
    app.run()    