#%% IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as anim
import time

#%% SYSTEM

class System():
    
    def __init__(self, population, ncols=5, dist="grid", extent=50,
                 symm=False):
        
        # Force matrix that governs particle interaction, -20 to 20
        self.force_matrix = 40*np.random.rand(ncols,ncols) - 20
        # Governs where the force starts/peaks/stops for each type
        self.bound_matrix = 9*np.random.rand(ncols,ncols,3) + 1
        self.bound_matrix.sort(axis=-1)
        
        # Make the interaction symmetric, if specified
        if symm:
            self.force_matrix = (self.force_matrix+self.force_matrix.T)/2
            self.bound_matrix = (self.bound_matrix+
                                 np.transpose(self.bound_matrix, (1,0,2)))/2
        
        # Particle colours
        cols = np.random.rand(ncols, 3)
        
        # Universe size
        self.extent = extent
        
        # Set up figure and axis
        self.fig, self.ax = plt.subplots(figsize=(16,9))
        self.ax.set_facecolor([.2,.2,.25])
        self.ax.set_xlim(0,self.extent)
        self.ax.set_ylim(0,self.extent)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Generate particles assign positions/velocities(random)/colours/types
        self.pop = int((np.sqrt(population)//1)**2)
        self.poss = np.empty((self.pop,2))
        self.vels = np.zeros((self.pop,2))
        self.vels = 16*np.random.rand(self.pop,2)-8
        self.newposs = np.empty((self.pop,2))
        self.newvels = np.empty((self.pop,2))
        self.cols = np.empty((self.pop,3))
        self.types = np.empty(self.pop, dtype=int)
        
        # Generate particles in a grid (can expand for different generation
        # methods)
        if dist == "grid":
            grid = int(np.sqrt(self.pop))
                
            coords = np.linspace(1,self.extent-1,grid)
            for i in range(self.pop):
                pos = np.array([coords[i//grid], coords[i%grid]])
                typ = np.random.randint(0,ncols)
                self.poss[i] = pos
                self.cols[i] = cols[typ]
                self.types[i] = typ
                
        # Scatter plot object with each particle scattered
        self.scat = self.ax.scatter(self.poss[:,0], self.poss[:,1],
                                    c=self.cols, s=10)



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
    
    # Time step
    ts = .05
    
    # Relative displacement vectors between particles (with periodic boundary)
    rel_vecs = sys.poss[:, np.newaxis, :] - sys.poss[np.newaxis, :, :]
    rel_vecs = rel_vecs - sys.extent * np.round(rel_vecs / sys.extent)
    # Distances and unit direction vectors
    distances = np.linalg.norm(rel_vecs, axis=-1)
    directions = rel_vecs / distances[..., np.newaxis]
    # Force strengths based on particle types
    coeffs = sys.force_matrix[sys.types[:, None], sys.types[None, :]]
    
    # Fixed distance thresholds (ignores bound_matrix)
    low = .5
    mid = 5
    hih = 9
    # Repulsion strength
    rep = 3
            
    # Boolean masks for regions of interaction
    reg1 = distances <= low
    reg2 = (distances > low) & (distances <= mid)
    reg3 = (distances > mid) & (distances <= hih)
    reg4 = distances > hih
    
    # Initialise force array
    forces = np.zeros_like(rel_vecs)
    
    # Calculate the forces
    forces[reg1] = (-rep/low)*(distances[reg1]-low)[..., np.newaxis]*directions[reg1]
    forces[reg2] = ((coeffs[reg2]/(mid-low))*(distances[reg2]-low))[..., np.newaxis]*directions[reg2]
    forces[reg3] = ((-coeffs[reg3]/(hih-mid))*(distances[reg3]-hih))[..., np.newaxis]*directions[reg3]
    forces[reg4] = 0*directions[reg4]
    
    # Remove self-interaction (diagonal entries)
    np.fill_diagonal(forces[:, :, 0], 0)
    np.fill_diagonal(forces[:, :, 1], 0)
    
    # Net force
    total_force = forces.sum(axis=1)
    
    # Update velocities and positions (with periodic boundaries)
    sys.newvels = sys.vels + total_force * ts
    sys.newposs = (sys.poss + sys.newvels * ts)%sys.extent

    # Update scatter plot with new positions
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
    
    # Time step
    ts = .05
    
    # Relative displacement vectors between particles (with periodic boundary)
    rel_vecs = sys.poss[:, np.newaxis, :] - sys.poss[np.newaxis, :, :]
    rel_vecs = rel_vecs - sys.extent * np.round(rel_vecs / sys.extent)
    # Distances and unit direction vectors
    distances = np.linalg.norm(rel_vecs, axis=-1)
    directions = rel_vecs / distances[..., np.newaxis]
    # Force strengths based on particle types
    coeffs = sys.force_matrix[sys.types[:, None], sys.types[None, :]]
    # Interaction distance thresholds from bound_matrix
    bounds = sys.bound_matrix[sys.types[:, None], sys.types[None, :]]
            
    # Repulsion parameters for very short distances
    rep = 10
    low = 10
    
    # Boolean masks for interaction regions
    reg1 = distances <= low    
    reg2 = (distances > low) & (distances <= bounds[:,:,0])
    reg3 = (distances > bounds[:,:,0]) & (distances <= bounds[:,:,1])
    reg4 = (distances > bounds[:,:,1]) & (distances <= bounds[:,:,2])
    reg5 = distances > bounds[:,:,2]
    
    # Extract left, peak, and right thresholds for easier indexing
    lefts = bounds[...,0]
    peaks = bounds[...,1]
    right = bounds[...,2]
    
    # Initialise force array
    forces = np.zeros_like(rel_vecs)
    
    # Calculate forces
    forces[reg1] = (-rep/low)*(distances[reg1]-low)[..., np.newaxis]*directions[reg1]
    forces[reg2] = 0*directions[reg2]
    forces[reg3] = ((coeffs[reg3]/(peaks[reg3]-lefts[reg3]))*(distances[reg3]-lefts[reg3]))[..., np.newaxis]*directions[reg3]
    forces[reg4] = ((-coeffs[reg4]/(right[reg4]-peaks[reg4]))*(distances[reg4]-right[reg4]))[..., np.newaxis]*directions[reg4]
    forces[reg5] = 0*directions[reg5]
    
    # Remove self-interaction (diagonal entries)
    np.fill_diagonal(forces[:, :, 0], 0)
    np.fill_diagonal(forces[:, :, 1], 0)
    
    # Net force on each particle
    total_force = forces.sum(axis=1)
    
    # Update velocities and positions (with periodic boundaries)
    sys.newvels = sys.vels + total_force * ts
    sys.newposs = (sys.poss + sys.newvels * ts)%sys.extent

    # Update scatter plot with new positions
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
    
    # Time step and damping coefficient
    ts = .02
    damping = .1
    
    # Relative displacement vectors between particles (with periodic boundary)
    rel_vecs = sys.poss[:, np.newaxis, :] - sys.poss[np.newaxis, :, :]
    rel_vecs = rel_vecs - sys.extent * np.round(rel_vecs / sys.extent)
    # Distances and unit direction vectors
    distances = np.linalg.norm(rel_vecs, axis=-1)
    directions = rel_vecs / distances[..., np.newaxis]
    # Force strengths based on particle types
    coeffs = sys.force_matrix[sys.types[:, None], sys.types[None, :]]
    # Interaction distance thresholds from bound_matrix
    bounds = sys.bound_matrix[sys.types[:, None], sys.types[None, :]]
           
    # Repulsion parameters for very short distances
    rep = 1
    low = .5
    
    # Boolean masks for interaction regions
    reg1 = distances <= low    
    reg2 = (distances > low) & (distances <= bounds[:,:,0])
    reg3 = (distances > bounds[:,:,0]) & (distances <= bounds[:,:,1])
    reg4 = (distances > bounds[:,:,1]) & (distances <= bounds[:,:,2])
    reg5 = distances > bounds[:,:,2]
    
    # Extract left, peak, and right thresholds for easier indexing
    lefts = bounds[...,0]
    peaks = bounds[...,1]
    right = bounds[...,2]
    
    # Initialise force array
    forces = np.zeros_like(rel_vecs)
    
    # Calculate forces
    forces[reg1] = (-rep/low)*(distances[reg1]-low)[..., np.newaxis]*directions[reg1]
    forces[reg2] = 0*directions[reg2]
    forces[reg3] = ((coeffs[reg3]/(peaks[reg3]-lefts[reg3]))*(distances[reg3]-lefts[reg3]))[..., np.newaxis]*directions[reg3]
    forces[reg4] = ((-coeffs[reg4]/(right[reg4]-peaks[reg4]))*(distances[reg4]-right[reg4]))[..., np.newaxis]*directions[reg4]
    forces[reg5] = 0*directions[reg5]
    
    # Remove self-interaction (diagonal entries)
    np.fill_diagonal(forces[:, :, 0], 0)
    np.fill_diagonal(forces[:, :, 1], 0)
    
    # Net force on each particle, including damping proportional to velocity
    total_force = forces.sum(axis=1) - damping * sys.vels
    
    # Update velocities and positions (with periodic boundaries)
    sys.newvels = sys.vels + total_force * ts
    sys.newposs = (sys.poss + sys.newvels * ts)%sys.extent

    # Update scatter plot with new positions
    sys.scat.set_offsets(sys.newposs)
    # Save updated positions and velocities
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
    
    # Timing for profiling
    t1 = time.time()
    # Time step and damping coefficient
    ts = .025
    damping = .1
    
    # Relative displacement vectors between particles (with periodic boundary)
    rel_vecs = sys.poss[:, np.newaxis, :] - sys.poss[np.newaxis, :, :]
    rel_vecs = rel_vecs - sys.extent * np.round(rel_vecs / sys.extent)
    # Distances and unit direction vectors
    distances = np.linalg.norm(rel_vecs, axis=-1)
    directions = rel_vecs / distances[..., np.newaxis]
    # Force strengths based on particle types
    coeffs = sys.force_matrix[sys.types[:, None], sys.types[None, :]]
    # Interaction distance thresholds from bound_matrix
    bounds = sys.bound_matrix[sys.types[:, None], sys.types[None, :]]
            
    # Repulsion parameters for very short distances
    rep = 100
    low = 1
    
    # Boolean masks for interaction regions
    reg1 = distances <= low    
    reg2 = (distances > low) & (distances <= bounds[:,:,0])
    reg3 = (distances > bounds[:,:,0]) & (distances <= bounds[:,:,1])
    reg4 = (distances > bounds[:,:,1]) & (distances <= bounds[:,:,2])
    reg5 = distances > bounds[:,:,2]
    
    # Extract left, peak, and right thresholds for easier indexing
    lefts = bounds[...,0]
    peaks = bounds[...,1]
    right = bounds[...,2]
    
    # Initialise force array
    forces = np.zeros_like(rel_vecs)
    
    # Calculate forces
    forces[reg1] = rep*directions[reg1]
    forces[reg2] = 0*directions[reg2]
    forces[reg3] = ((coeffs[reg3]/(peaks[reg3]-lefts[reg3]))*(distances[reg3]-lefts[reg3]))[..., np.newaxis]*directions[reg3]
    forces[reg4] = ((-coeffs[reg4]/(right[reg4]-peaks[reg4]))*(distances[reg4]-right[reg4]))[..., np.newaxis]*directions[reg4]
    forces[reg5] = 0*directions[reg5]
    
    # Remove self-interaction (diagonal entries)
    np.fill_diagonal(forces[:, :, 0], 0)
    np.fill_diagonal(forces[:, :, 1], 0)
    
    # Net force on each particle, including damping proportional to velocity
    total_force = forces.sum(axis=1) - damping * sys.vels
    
    # Limit maximum particle speed
    max_speed = 20
    sys.newvels = sys.vels + total_force * ts
    sys.newspeeds = np.linalg.norm(sys.newvels, axis=-1)
    factors = np.ones_like(sys.newspeeds)
    factors[sys.newspeeds>max_speed] = sys.newspeeds[sys.newspeeds>max_speed]/max_speed
    factors = np.repeat(factors[:, np.newaxis], 2, axis=1)
    sys.newvels /= factors
    
    # Update positions (with periodic boundaries)
    sys.newposs = (sys.poss + sys.newvels * ts)%sys.extent
    
    # Timing for profiling
    t2 = time.time()

     # Update scatter plot with new positions
    sys.scat.set_offsets(sys.newposs)
    # Timing for profiling
    t3 = time.time()
    # Save updated positions and velocities
    sys.poss = sys.newposs
    sys.vels = sys.newvels
    # Print computation/plot times
    #####print(f"calc: {t2-t1}\nplot: {t3-t2}\n")
    
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
    
    # Helper function to implement wall collisions
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
    
    # Time step and damping coefficient
    ts = .04
    damping = 0
    
    # Relative displacement vectors between particles (with periodic boundary)
    rel_vecs = sys.poss[:, np.newaxis, :] - sys.poss[np.newaxis, :, :]
    #rel_vecs = rel_vecs - sys.extent * np.round(rel_vecs / sys.extent)
    # Distances and unit direction vectors 
    distances = np.linalg.norm(rel_vecs, axis=-1)
    directions = rel_vecs / distances[..., np.newaxis]
    # Force strengths based on particle types
    coeffs = sys.force_matrix[sys.types[:, None], sys.types[None, :]]
    # Interaction distance thresholds from bound_matrix
    bounds = sys.bound_matrix[sys.types[:, None], sys.types[None, :]]
            
    # Repulsion parameters for very short distances
    rep = 2
    low = 1
    
    # Boolean masks for interaction regions
    reg1 = distances <= low    
    reg2 = (distances > low) & (distances <= bounds[:,:,0])
    reg3 = (distances > bounds[:,:,0]) & (distances <= bounds[:,:,1])
    reg4 = (distances > bounds[:,:,1]) & (distances <= bounds[:,:,2])
    reg5 = distances > bounds[:,:,2]
    
    # Extract left, peak, and right thresholds for easier indexing
    lefts = bounds[...,0]
    peaks = bounds[...,1]
    right = bounds[...,2]
    
    # Initialise force array
    forces = np.zeros_like(rel_vecs)
    
    # Calculate forces
    forces[reg1] = (-rep/low)*(distances[reg1]-low)[..., np.newaxis]*directions[reg1]
    forces[reg2] = 0*directions[reg2]
    forces[reg3] = ((coeffs[reg3]/(peaks[reg3]-lefts[reg3]))*(distances[reg3]-lefts[reg3]))[..., np.newaxis]*directions[reg3]
    forces[reg4] = ((-coeffs[reg4]/(right[reg4]-peaks[reg4]))*(distances[reg4]-right[reg4]))[..., np.newaxis]*directions[reg4]
    forces[reg5] = 0*directions[reg5]
    
    # Remove self-interaction (diagonal entries)
    np.fill_diagonal(forces[:, :, 0], 0)
    np.fill_diagonal(forces[:, :, 1], 0)
    
    # Net force on each particle, including damping proportional to velocity
    total_force = forces.sum(axis=1) - damping * sys.vels
    
    # Limit maximum particle speed
    max_speed = 15
    sys.newvels = sys.vels + total_force * ts
    sys.newspeeds = np.linalg.norm(sys.newvels, axis=-1)
    factors = np.ones_like(sys.newspeeds)
    factors[sys.newspeeds>max_speed] = sys.newspeeds[sys.newspeeds>max_speed]/max_speed
    factors = np.repeat(factors[:, np.newaxis], 2, axis=1)
    sys.newvels /= factors
    
    # Update positions (walls instead of periodic boundaries)
    sys.newposs = (sys.poss + sys.newvels * ts)
    sys.newposs, sys.newvels = wall_rebound(sys.newposs,
                                            sys.newvels,
                                            sys.extent)
    

    # Update scatter plot with new positions
    sys.scat.set_offsets(sys.newposs)
    # Save updated positions and velocities
    sys.poss = sys.newposs
    sys.vels = sys.newvels
    
    return sys.scat,

# # Initialise a particle system with 625 particles, 6 types, symmetric interactions, and universe size 100
# s=System(625, extent=100, ncols=6, symm=True)
# # Create an animation using step5 (max speed, variable forces)
# # - fargs=[s] passes the system object to the step function
# # - frames=200 runs 200 steps
# # - interval=1 sets the delay between frames in ms
# # - blit=True improves performance by only redrawing changed elements
# ani = anim(s.fig, step5, fargs=[s], frames=200, interval=1, blit=True)
# # Display the animation
# plt.show() 

# Menu for selecting the simulation step
print("Particle Simulation Steps Menu:")
print("1: Fixed distance ranges, varying force strengths")
print("2: Variable distance ranges and forces (prone to pulsating)")
print("3: Damped motion, variable forces and ranges")
print("4: Max speed limiter, variable forces and ranges")
print("5: 2Adds wall collisions, max speed, variable forces and ranges")

choice = input("Enter the number of the simulation step to run (1-5): ")

if choice == "1":
    selected_step = step2
elif choice == "2":
    selected_step = step3
elif choice == "3":
    selected_step = step4
elif choice == "4":
    selected_step = step5
elif choice == "5":
    selected_step = step6
else:
    print("Invalid choice. Defaulting to step2.")
    selected_step = step2

print(f"Running simulation using: {selected_step.__name__}")

particles = int(input("\nEnter number of particles: "))
if particles % 1 != 0 or particles > 1000:
    print("Must be an integer <= 1000, defaulting to 400")
    particles = 400

ncols = int(input("\nEnter number of particle types: "))
if ncols % 1 != 0:
    print("Must be an integer, defaulting to 5")
    ncols = 5
if ncols >= 20:
    print("Too many, defaulting to 5")
    ncols = 5
        
# Initialise the particle system
s = System(particles, ncols, extent=50, symm=False)

# Run the animation with the selected step function
ani = anim(s.fig, selected_step, fargs=[s], frames=200, interval=1, blit=True)
plt.show()
        
        
        
        
        
        

        
        
        

        
        
    
