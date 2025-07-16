import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from brainio.assemblies import BehavioralAssembly
from brainio.stimuli import StimulusSet


data_dir = '/home/ytang/workspace/tmp/data-FEF/data'
video_dir = '/home/ytang/workspace/data/FEF/videos'
COHERENCE = [0.032, 0.064, 0.128, 0.256, 0.512]
# DIRECTION = [0, 45, 90, 135, 180, 225, 270, 315]
DIRECTION = [0, 180]

# - square with side of length 4 degrees (for monkey C, 3.5 for monkey F)
# - velocity 6 degrees per second

# Function to generate random dots
def generate_vel(speed, direction, dt):
    # Compute displacements based on speed and directions
    displacements = speed * dt
    dx = displacements * np.cos(np.radians(direction))
    dy = displacements * np.sin(np.radians(direction))

    return np.array((dx, dy))

"""
Because the dots stimulus is often used to understand decisions that involve integrating evidence over space and time, 
a lot of the weirdness of drawing the dots is to ensure that people can't use other strategies 
- which usually means avoiding having individual dots carry too much information. 
In other words, we want to avoid drawing it in a way that the subject can just track a single dot and figure out the answer.
The “thisFrame” variable is part of one of the main ways we do that. 
Basically instead of drawing a single set of dots, some fraction of which moves from frame-to-frame, 
we usually divide them up into three subsets. The “motion” is the displacement of dots from each subset every third frame.
"""

def make_dots(
        id, 
        density = 150, # 16.7,
        coherence = 0.03,  # Set coherence level (0 to 1)
        speed = 6.0,  # 6.0,  # Set motion speed
        direction = 90,  # Set motion direction (in degrees)
        fps = 60,
        screen_radius = 5.0,  # Radius of the circular display area
        screen_size = 8.0,
        dot_size = 10,
        duration = 2, # seconds
        replacement_frames = 3,
    ):

    dt = 1 / fps  # Time step (inverse of FPS)
    num_dots = int(density * screen_radius**2 / fps)
    
    dot_positions = np.random.uniform(-screen_radius, screen_radius, (2, num_dots))
    life_times = np.zeros(num_dots, dtype=int)

    # Create the animation
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    # Function to update dot positions in the animation
    def update(frame):
        ax.clear()
        this_frame = np.zeros(num_dots, dtype=bool)
        this_frame[frame % replacement_frames::replacement_frames] = True

        # prioritize short life times
        num_consistent_dots = int(coherence * num_dots)
        consistent_dots_ = np.argsort(life_times)[:num_consistent_dots]
        consistent_dots = np.zeros(num_dots, dtype=bool)
        consistent_dots[consistent_dots_] = True
        non_consistent_dots = ~consistent_dots
        consistent_dots[~this_frame] = False
        non_consistent_dots[~this_frame] = False

        life_times[consistent_dots] += 1

        dot_positions[:, consistent_dots] += generate_vel(speed, direction, dt)[:,None] * replacement_frames
        dot_positions[:, non_consistent_dots] = np.random.uniform(-screen_radius, screen_radius, (2, non_consistent_dots.sum()))
        # dot_positions[:, non_consistent_dots] += generate_vel(speed, np.random.uniform(0, 360, non_consistent_dots.sum()), dt) * replacement_frames

        # wrap around
        dot_positions[0, :] = np.mod(dot_positions[0, :] + screen_radius, 2*screen_radius) - screen_radius
        dot_positions[1, :] = np.mod(dot_positions[1, :] + screen_radius, 2*screen_radius) - screen_radius

        # Only display dots within the circular area
        mask = np.sqrt(dot_positions[0, :]**2 + dot_positions[1, :]**2) < screen_radius
        ax.set_xlim(-screen_size, screen_size)
        ax.set_ylim(-screen_size, screen_size)  
        # axis off
        ax.axis('off')
        # Plot the dots
        ax.scatter(dot_positions[0, mask], dot_positions[1, mask], s=dot_size, c='white')

    ani = FuncAnimation(fig, update, frames=fps * duration, interval=1000 / fps)
    # save mp4
    ani.save(f'{video_dir}/{id}.mp4', writer='ffmpeg', fps=fps)

def make_stimuli(sample_per_condition):
    from joblib import Parallel, delayed
    # parallize
    Parallel(n_jobs=-1)(delayed(make_dots)(f"{coh*100}_{dir}_{i}", coherence=coh, direction=dir) 
                        for i in range(sample_per_condition) for coh in COHERENCE for dir in DIRECTION)


def load_dataset(identifier='ding2012', sample_per_condition=100):
    # make stimulus_set

    make_stimuli(sample_per_condition)

    stimulus_ids = []
    stimulus_paths = []
    stimulus_directions = []
    cohs = []

    # make num_samples samples for each case
    for i in range(sample_per_condition):
        for coh in COHERENCE:
            for dir in DIRECTION:
                sp = f"{coh*100}_{dir}_{i}"
                stimulus_ids.append(sp)
                stimulus_paths.append(os.path.join(video_dir, f"{sp}.mp4"))
                stimulus_directions.append(dir)
                cohs.append(coh)

    stimulus_paths = [os.path.join(video_dir, f"{stimulus_id}.mp4") for stimulus_id in stimulus_ids]

    stimulus_set = {}
    stimulus_set["stimulus_id"] = stimulus_ids
    stimulus_set["direction"] = stimulus_directions
    stimulus_set['truth'] = stimulus_directions
    stimulus_set['coh'] = cohs
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}
    stimulus_set.identifier = identifier

    assembly = BehavioralAssembly(
        np.array(stimulus_directions)[:, None],
        dims=('presentation', 'label'),
        coords={
            'stimulus_id': ('presentation', stimulus_ids),
            'coherence': ('presentation', cohs),
            'direction': ('presentation', stimulus_directions),
        }
    )

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly