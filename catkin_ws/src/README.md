# Assets for the exercises
This catkin workspace contains the assets needed to run the exercises. The `evolutionary_robotics` package contains all those assets. The directory consists of 3 sub-directories:

- `launch` contains the launch files for the exercise. The launch file from an exercise is linked to a file here that runs the exercise.

- `models` contains the models used in the exercise. Most of the models used come from the [JdeRobot/assets](https://github.com/JdeRobot/assets) repository. The modified and new models are present in this directory.

- `worlds` contains the world files for the exercise. The launch files run the simulation from the world files.

## Summary of changes
The following headings describe the summary of changes and additions to the `JdeRobot/assets`:

### Launch

- `obstacle_avoidance.launch` runs `empty_world.launch` and includes the world file `obstacle_avoidance.world`

### Worlds

- `obstacle_avoidance.world` contains the world representation of the obstacle avoidance exercise. The file includes `sun`, `ground_plane_transparent`, `roombaIR` and `simpleWorld` models.

### Models

- `roombaIR` contains the Roomba robot with 8 IR sensors fitted accross it. The file is an edited version of `roombaROS` where the Hokuyo Laser sensor is removed and Sonar sensors are added.
