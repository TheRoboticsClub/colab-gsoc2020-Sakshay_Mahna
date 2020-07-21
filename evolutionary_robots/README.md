# Evolutionary Robots

## Description
Exercises on Evolutionary Robotics for JdeRobot Robotics Academy

## Prerequisites
The libraries and exercises are developed and tested in **Python 2.7.17, Pip 20.0.2, ROS Melodic, Ubuntu 18.04**.

### Git
The instructions to install Git(command terminal) for Ubuntu 18.04 are:

- Update the Default Packages

```bash
sudo apt update
```

- Install Git

```bash
sudo apt install git
```

- Check if the following command does not give a missing error

```bash
git
```

### Python and Pip
The instructions to install Python2.7 for Ubuntu 18.04 are:

- Update and Upgrade the Default Packages

```bash
sudo apt update
sudo apt upgrade
```

- Install Python2.7

```bash
sudo apt install python2.7
```

- To check correct installation, the following command should open a Python interpreter

```bash
python2
```

- Install Pip for Python2

```bash
sudo apt install python-pip
```

- Check if the following command does not give a missing error

```bash
pip
```

### Generic Infrastructure of Robotics Academy
Follow the Installation instructions as given on the [Robotics Academy webpage](http://jderobot.github.io/RoboticsAcademy/installation/#generic-infrastructure).

The installation is done correctly if we can successfully run the following commands:

- Source the environment variables

```bash
source ~/.bashrc
```

- Start the ROS Master server. This would keep running in the terminal without giving any errors

```bash
roscore
```

- The Gazebo Model variable should contain paths to jderobot directories

```bash
echo $GAZEBO_MODEL_PATH
```

## Dependencies
The project uses the following python modules

```
numpy==1.16.5
graphviz==0.14
tensorflow==2.1.0
matplotlib==2.1.0
```

## Libraries
The libraries developed for the project are available in [libraries](./libraries). These libraries are useful for solving the exercises. The API reference and examples are also provided.

## Installation
Before running the installation, make sure that all the prerequisites are already installed on the system which are **Git, Python, Pip and Generic Infrastructure of Robotics Academy.**

- Open a new terminal and navigate to the directory where the exercises should be downloaded.

- Clone the Github Repository.

```bash
git clone https://github.com/TheRoboticsClub/colab-gsoc2020-Sakshay_Mahna
```

- Navigate to the working directory inside the cloned repository.

```bash
cd colab-gsoc2020-Sakshay_Mahna/evolutionary_robotics
```

- Update Pip to the latest version. Some dependencies need the latest version to install correctly.

```bash
pip install --upgrade pip
```

- Install the dependencies. All the dependencies would be installed without giving any errors.

```bash
pip install -r requirements.txt
```

- Source the ROS environment variables.

```bash
source /opt/ros/melodic/setup.bash
```

- Run the installation script to configure the Gazebo Assets. This will build the workspace and make new directories `devel` and `build` inside `colab-gsoc2020-Sakshay_Mahna/catkin_ws` directory.

```bash
. installation.bash
```

- Run the source script to source the Gazebo Assets. This command would add new paths to `GAZEBO_MODEL_PATH` environment variable.

```bash
. source.bash
```

- Specific instructions to run each of the exercises are given in their respective directories.

### Obstacle Avoidance
The code for this exercise is present in [obstacle_avoidance](./obstacle_avoidance)




