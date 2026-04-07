# UAV Trajectory Planning with A* and Convex Optimization

This project implements a hybrid path planning algorithm for Unmanned Aerial Vehicles (UAVs) operating in obstacle-dense environments. 
It combines the **A* search algorithm** for global pathfinding with **Convex Optimization (CVXPY)** for trajectory smoothing and obstacle avoidance.

## Features
* **A* Algorithm:** Efficient grid-based global path search.
* **Convex Optimization:** Real-time trajectory generation using Sequential Convex Programming (SCP) concepts.
* **Visualization:** High-quality plots using Matplotlib, including custom UAV markers (icons).
* **Image Processing:** Obstacle map generation from images using PIL/Pillow.

## Prerequisites
Ensure you have Python installed. You can install all necessary libraries using the following command:

```bash
pip install -r requirements.txt
