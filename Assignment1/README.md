***Task 1: Torus Simulator and Braitenberg Vehicles***

**Sub-tasks:**

* TorusSimulator class (wrap-around 2D world) – handles position wrapping and motion updates.

* LightField class – defines a toroidal light intensity field around a source.

* BraitenbergRobot class – two light sensors steering by cross/direct coupling.

* Simulation function – steps the robot, records trajectories.

* Static plots:

    * Light intensity field (task1_light_field.png).

    * Trajectories of the “aggressor” (cross-coupled) and “fear” (direct-coupled) vehicles (task1_traj.png).

* Animation:

    Generates task1_anim.gif showing vehicles moving on the torus.

***Task 2: Proximity Sensors and Rule-Based Navigation***

**Sub-tasks:**

* WallEnvironment class – defines arena boundaries and interior walls, ray intersection.

* ProximityRobot class – three forward-facing proximity sensors (left/front/right).

* Reactive controller in step() – obstacle avoidance rules to turn and move.

* Static plot of trajectory through the maze-like walls (task2_traj.png).

* Animation of the robot exploring (task2_anim.gif).    

***How to run the codes in Google Colab***
* Open a new Colab notebook at https://colab.research.google.com

* Copy the Task1  code block into a cell and run:

    * It generates and displays the light field, trajectories, and task1_anim.gif files.
* Copy the Task2 code block into a cell and run:
    * It produces task2_traj.png and task2_anim.gif files.
* Download the generated PNG/GIF files via Colab’s file browser (left sidebar > Files).

* I have also included the downloaded png and gif files in each task folder.