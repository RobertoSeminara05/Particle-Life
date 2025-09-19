#%% IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as anim
import time

#%% SYSTEM

class System():
    
    def __init__(self, population, ncols=5, dist="grid", extent=50,
                 symm=False):
        
        self.force_matrix = 40*np.random.rand(ncols,ncols) - 20
        #self.force_matrix = -np.random.lognormal(2.4, .6, (ncols,ncols))+20
        self.bound_matrix = 9*np.random.rand(ncols,ncols,3) + 1
        self.bound_matrix.sort(axis=-1)
        
        if symm:
            self.force_matrix = (self.force_matrix+self.force_matrix.T)/2
            self.bound_matrix = (self.bound_matrix+
                                 np.transpose(self.bound_matrix, (1,0,2)))/2
        
        #np.fill_diagonal(self.force_matrix, -8)
        
        cols = np.random.rand(ncols, 3)
        
        self.extent = extent
        
        self.fig, self.ax = plt.subplots(figsize=(16,9))
        self.ax.set_facecolor([.2,.2,.25])
        self.ax.set_xlim(0,self.extent)
        self.ax.set_ylim(0,self.extent)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.pop = population
        self.poss = np.empty((self.pop,2))
        self.vels = np.zeros((self.pop,2))
        self.newposs = np.empty((self.pop,2))
        self.newvels = np.empty((self.pop,2))
        self.cols = np.empty((self.pop,3))
        self.types = np.empty(self.pop, dtype=int)
        
        if dist == "grid":
            grid = int(np.sqrt(self.pop))
            if grid%1 != 0:
                raise TypeError("Population must be a square number")
                
            coords = np.linspace(1,self.extent-1,grid)
            for i in range(self.pop):
                pos = np.array([coords[i//grid], coords[i%grid]])
                typ = np.random.randint(0,ncols)
                self.poss[i] = pos
                self.cols[i] = cols[typ]
                self.types[i] = typ
                
        self.scat = self.ax.scatter(self.poss[:,0], self.poss[:,1],
                                    c=self.cols, s=10)

        
        
    def force(self, vector, coeff):
        
        distance = np.sqrt(vector.dot(vector))
        direction = vector/distance
        if distance <=1: return (distance-1)*direction
        if distance <=8: return ((coeff/7)*(distance-1))*direction
        if distance <=15: return (-(coeff/7)*(distance-15))*direction
        if distance >15: return 0*direction



def step2(f, sys):
    """
    For fixed ranges but varying force strengths

    Parameters
    ----------
    f : int
        unused animation thing.
    sys : object
        the system.

    Returns
    -------
    ??
        for animation.

    """
    
    ts = .4
    
    rel_vecs = sys.poss[:, np.newaxis, :] - sys.poss[np.newaxis, :, :]
    rel_vecs = rel_vecs - sys.extent * np.round(rel_vecs / sys.extent)
    distances = np.linalg.norm(rel_vecs, axis=-1)
    directions = rel_vecs / distances[..., np.newaxis]
    coeffs = sys.force_matrix[sys.types[:, None], sys.types[None, :]]
    
    low = 1
    mid = 5
    hih = 9
    rep = 5
            
            
    reg1 = distances <= low
    reg2 = (distances > low) & (distances <= mid)
    reg3 = (distances > mid) & (distances <= hih)
    reg4 = distances > hih
    
    forces = np.zeros_like(rel_vecs)
    
    forces[reg1] = (-rep/low)*(distances[reg1]-low)[..., np.newaxis]*directions[reg1]
    forces[reg2] = ((coeffs[reg2]/(mid-low))*(distances[reg2]-low))[..., np.newaxis]*directions[reg2]
    forces[reg3] = ((-coeffs[reg3]/(hih-mid))*(distances[reg3]-hih))[..., np.newaxis]*directions[reg3]
    forces[reg4] = 0*directions[reg4]
    
    np.fill_diagonal(forces[:, :, 0], 0)
    np.fill_diagonal(forces[:, :, 1], 0)
    
    total_force = forces.sum(axis=1)
    
    sys.newvels = sys.vels + total_force * ts
    sys.newposs = (sys.poss + sys.newvels * ts)%sys.extent

    sys.scat.set_offsets(sys.newposs)
    sys.poss = sys.newposs
    
    return sys.scat,
    

def step3(f, sys):
    """
    variable forces and ranges

    Parameters
    ----------
    f : int
        unused animation thing.
    sys : object
        the system.

    Returns
    -------
    ??
        for animation.

    """
    
    ts = .25
    
    rel_vecs = sys.poss[:, np.newaxis, :] - sys.poss[np.newaxis, :, :]
    rel_vecs = rel_vecs - sys.extent * np.round(rel_vecs / sys.extent)
    distances = np.linalg.norm(rel_vecs, axis=-1)
    directions = rel_vecs / distances[..., np.newaxis]
    coeffs = sys.force_matrix[sys.types[:, None], sys.types[None, :]]
    bounds = sys.bound_matrix[sys.types[:, None], sys.types[None, :]]
            
    
    rep = 10
    low = 10
    
    reg1 = distances <= low    
    reg2 = (distances > low) & (distances <= bounds[:,:,0])
    reg3 = (distances > bounds[:,:,0]) & (distances <= bounds[:,:,1])
    reg4 = (distances > bounds[:,:,1]) & (distances <= bounds[:,:,2])
    reg5 = distances > bounds[:,:,2]
    
    lefts = bounds[...,0]
    peaks = bounds[...,1]
    right = bounds[...,2]
    
    forces = np.zeros_like(rel_vecs)
    
    forces[reg1] = (-rep/low)*(distances[reg1]-low)[..., np.newaxis]*directions[reg1]
    forces[reg2] = 0*directions[reg2]
    forces[reg3] = ((coeffs[reg3]/(peaks[reg3]-lefts[reg3]))*(distances[reg3]-lefts[reg3]))[..., np.newaxis]*directions[reg3]
    forces[reg4] = ((-coeffs[reg4]/(right[reg4]-peaks[reg4]))*(distances[reg4]-right[reg4]))[..., np.newaxis]*directions[reg4]
    forces[reg5] = 0*directions[reg5]
    
    np.fill_diagonal(forces[:, :, 0], 0)
    np.fill_diagonal(forces[:, :, 1], 0)
    
    total_force = forces.sum(axis=1)
    
    sys.newvels = sys.vels + total_force * ts
    sys.newposs = (sys.poss + sys.newvels * ts)%sys.extent

    sys.scat.set_offsets(sys.newposs)
    sys.poss = sys.newposs
    
    return sys.scat,


def step4(f, sys):
    """
    damped, variable forces and ranges

    Parameters
    ----------
    f : int
        unused animation thing.
    sys : object
        the system.

    Returns
    -------
    ??
        for animation.

    """
    
    ts = .1
    damping = .1
    
    rel_vecs = sys.poss[:, np.newaxis, :] - sys.poss[np.newaxis, :, :]
    rel_vecs = rel_vecs - sys.extent * np.round(rel_vecs / sys.extent)
    distances = np.linalg.norm(rel_vecs, axis=-1)
    directions = rel_vecs / distances[..., np.newaxis]
    coeffs = sys.force_matrix[sys.types[:, None], sys.types[None, :]]
    bounds = sys.bound_matrix[sys.types[:, None], sys.types[None, :]]
            
    rep = 2
    low = 1
    
    reg1 = distances <= low    
    reg2 = (distances > low) & (distances <= bounds[:,:,0])
    reg3 = (distances > bounds[:,:,0]) & (distances <= bounds[:,:,1])
    reg4 = (distances > bounds[:,:,1]) & (distances <= bounds[:,:,2])
    reg5 = distances > bounds[:,:,2]
    
    lefts = bounds[...,0]
    peaks = bounds[...,1]
    right = bounds[...,2]
    
    forces = np.zeros_like(rel_vecs)
    
    forces[reg1] = (-rep/low)*(distances[reg1]-low)[..., np.newaxis]*directions[reg1]
    forces[reg2] = 0*directions[reg2]
    forces[reg3] = ((coeffs[reg3]/(peaks[reg3]-lefts[reg3]))*(distances[reg3]-lefts[reg3]))[..., np.newaxis]*directions[reg3]
    forces[reg4] = ((-coeffs[reg4]/(right[reg4]-peaks[reg4]))*(distances[reg4]-right[reg4]))[..., np.newaxis]*directions[reg4]
    forces[reg5] = 0*directions[reg5]
    
    np.fill_diagonal(forces[:, :, 0], 0)
    np.fill_diagonal(forces[:, :, 1], 0)
    
    total_force = forces.sum(axis=1) - damping * sys.vels
    
    sys.newvels = sys.vels + total_force * ts
    sys.newposs = (sys.poss + sys.newvels * ts)%sys.extent

    sys.scat.set_offsets(sys.newposs)
    sys.poss = sys.newposs
    sys.vels = sys.newvels
    
    return sys.scat,


def step5(f, sys):
    """
    maxspeed, variable forces and ranges

    Parameters
    ----------
    f : int
        unused animation thing.
    sys : object
        the system.

    Returns
    -------
    ??
        for animation.

    """
    
    t1 = time.time()
    ts = .025
    damping = .1
    
    rel_vecs = sys.poss[:, np.newaxis, :] - sys.poss[np.newaxis, :, :]
    rel_vecs = rel_vecs - sys.extent * np.round(rel_vecs / sys.extent)
    distances = np.linalg.norm(rel_vecs, axis=-1)
    directions = rel_vecs / distances[..., np.newaxis]
    coeffs = sys.force_matrix[sys.types[:, None], sys.types[None, :]]
    bounds = sys.bound_matrix[sys.types[:, None], sys.types[None, :]]
            
    rep = 100
    low = 1
    
    reg1 = distances <= low    
    reg2 = (distances > low) & (distances <= bounds[:,:,0])
    reg3 = (distances > bounds[:,:,0]) & (distances <= bounds[:,:,1])
    reg4 = (distances > bounds[:,:,1]) & (distances <= bounds[:,:,2])
    reg5 = distances > bounds[:,:,2]
    
    lefts = bounds[...,0]
    peaks = bounds[...,1]
    right = bounds[...,2]
    
    forces = np.zeros_like(rel_vecs)
    
    forces[reg1] = rep*directions[reg1]
    forces[reg2] = 0*directions[reg2]
    forces[reg3] = ((coeffs[reg3]/(peaks[reg3]-lefts[reg3]))*(distances[reg3]-lefts[reg3]))[..., np.newaxis]*directions[reg3]
    forces[reg4] = ((-coeffs[reg4]/(right[reg4]-peaks[reg4]))*(distances[reg4]-right[reg4]))[..., np.newaxis]*directions[reg4]
    forces[reg5] = 0*directions[reg5]
    
    np.fill_diagonal(forces[:, :, 0], 0)
    np.fill_diagonal(forces[:, :, 1], 0)
    
    total_force = forces.sum(axis=1) - damping * sys.vels
    
    max_speed = 20
    sys.newvels = sys.vels + total_force * ts
    sys.newspeeds = np.linalg.norm(sys.newvels, axis=-1)
    factors = np.ones_like(sys.newspeeds)
    factors[sys.newspeeds>max_speed] = sys.newspeeds[sys.newspeeds>max_speed]/max_speed
    factors = np.repeat(factors[:, np.newaxis], 2, axis=1)
    sys.newvels /= factors
    
    sys.newposs = (sys.poss + sys.newvels * ts)%sys.extent
    
    t2 = time.time()

    sys.scat.set_offsets(sys.newposs)
    t3 = time.time()
    sys.poss = sys.newposs
    sys.vels = sys.newvels
    print(f"calc: {t2-t1}\nplot: {t3-t2}\n")
    
    return sys.scat,


def step6(f, sys):
    """
    walls!, maxspeed, variable forces and ranges

    Parameters
    ----------
    f : int
        unused animation thing.
    sys : object
        the system.

    Returns
    -------
    ??
        for animation.

    """
    
    def wall_rebound(pos,vel,size):
        
        righ = pos[:,0] > size
        pos[righ,0] = 2*size - pos[righ,0]
        vel[righ,0] *= -3
        
        left = pos[:,0] < 0
        pos[left,0] *= -1
        vel[left,0] *= -3
        
        uppp = pos[:,1] > size
        pos[uppp,1] = 2*size - pos[uppp,1]
        vel[uppp,1] *= -3
        
        down = pos[:,1] < 0
        pos[down,1] *= -1
        vel[down,1] *= -3
        
        return pos,vel
    
    ts = .1
    damping = 0
    
    rel_vecs = sys.poss[:, np.newaxis, :] - sys.poss[np.newaxis, :, :]
    rel_vecs = rel_vecs - sys.extent * np.round(rel_vecs / sys.extent)
    distances = np.linalg.norm(rel_vecs, axis=-1)
    directions = rel_vecs / distances[..., np.newaxis]
    coeffs = sys.force_matrix[sys.types[:, None], sys.types[None, :]]
    bounds = sys.bound_matrix[sys.types[:, None], sys.types[None, :]]
            
    rep = 2
    low = 1
    
    reg1 = distances <= low    
    reg2 = (distances > low) & (distances <= bounds[:,:,0])
    reg3 = (distances > bounds[:,:,0]) & (distances <= bounds[:,:,1])
    reg4 = (distances > bounds[:,:,1]) & (distances <= bounds[:,:,2])
    reg5 = distances > bounds[:,:,2]
    
    lefts = bounds[...,0]
    peaks = bounds[...,1]
    right = bounds[...,2]
    
    forces = np.zeros_like(rel_vecs)
    
    forces[reg1] = (-rep/low)*(distances[reg1]-low)[..., np.newaxis]*directions[reg1]
    forces[reg2] = 0*directions[reg2]
    forces[reg3] = ((coeffs[reg3]/(peaks[reg3]-lefts[reg3]))*(distances[reg3]-lefts[reg3]))[..., np.newaxis]*directions[reg3]
    forces[reg4] = ((-coeffs[reg4]/(right[reg4]-peaks[reg4]))*(distances[reg4]-right[reg4]))[..., np.newaxis]*directions[reg4]
    forces[reg5] = 0*directions[reg5]
    
    np.fill_diagonal(forces[:, :, 0], 0)
    np.fill_diagonal(forces[:, :, 1], 0)
    
    total_force = forces.sum(axis=1) - damping * sys.vels
    
    max_speed = 10
    sys.newvels = sys.vels + total_force * ts
    sys.newspeeds = np.linalg.norm(sys.newvels, axis=-1)
    factors = np.ones_like(sys.newspeeds)
    factors[sys.newspeeds>max_speed] = sys.newspeeds[sys.newspeeds>max_speed]/max_speed
    factors = np.repeat(factors[:, np.newaxis], 2, axis=1)
    sys.newvels /= factors
    
    sys.newposs = (sys.poss + sys.newvels * ts)
    sys.newposs, sys.newvels = wall_rebound(sys.newposs,
                                            sys.newvels,
                                            sys.extent)
    

    sys.scat.set_offsets(sys.newposs)
    sys.poss = sys.newposs
    sys.vels = sys.newvels
    
    return sys.scat,

s=System(625, extent=100, ncols=6, symm=True)
ani = anim(s.fig, step5, fargs=[s], frames=200, interval=1, blit=True)
#ani.save("particle life.mp4", writer="ffmpeg", fps=30)
plt.show() 
        
        
        
        
        
        
        
        

        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        