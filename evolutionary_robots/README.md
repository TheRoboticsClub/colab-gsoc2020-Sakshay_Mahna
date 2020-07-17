# Evolutionary Robots

## Description
Exercises on Evolutionary Robotics for JdeRobot Robotics Academy

## Prerequisites
The libraries and exercises are developed and tested in **Python 2.7.17, ROS Melodic, Ubuntu 18.04**. These are the system requirements that JdeRobot currently specifies for users to run their Robotics-Academy software.

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

### Python and Pip
The instructions to install Python2.7 for Ubuntu 18.04 are:

- Update and Upgrade the Default Packages

```bash
sudo apt update
sudo apt upgrade
```

- Install Python2.7 and Pip for Python2

```bash
sudo apt install python2.7
sudo apt install python-pip
```

### Generic Infrastructure of Robotics Academy
Follow the Installation instructions as given on the [Robotics Academy webpage](http://jderobot.github.io/RoboticsAcademy/installation/#generic-infrastructure).

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
Before running the installation, make sure that all the prerequisites are already installed on the system.

- Clone the Github Repository

```bash
git clone https://github.com/TheRoboticsClub/colab-gsoc2020-Sakshay_Mahna
```

- Install the dependencies

```bash
pip install -r requirements.txt
```

- Run the installation script to configure the Gazebo Assets

```bash
. installation.bash
```

- Specific instructions to run each of the exercises are given in their respective directories.

### Obstacle Avoidance
The code for this exercise is present in [obstacle_avoidance](./obstacle_avoidance)




