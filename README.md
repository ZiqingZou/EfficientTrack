# EfficientTrack

This is the official implement od our paper: "High-precision and high-efficiency trajectory tracking for excavators based on closed-loop dynamics."

EfficientTrack is a learning-enhanced trajectory tracking framework designed to tackle the complex nonlinear dynamics of hydraulic excavators, including time delays and control coupling. Traditional control methods often struggle with these challenges, while standard learning-based approaches require extensive environment interaction.

EfficientTrack combines model-based learning with closed-loop dynamics to significantly improve learning efficiency and minimize tracking errors. It achieves superior precision and smoothness with fewer interactions.

<img src="images/control block.png" alt="Control Block" width="1000"/>

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ZiqingZou/EfficientTrack.git
cd EfficientTrack
pip install -r requirements.txt
```

## Overview

EfficientTrack follows a structured pipeline to achieve high-precision and efficient trajectory tracking for hydraulic excavators.

<img src="images/data collection.png" alt="Data Collection" width="400"/>


### Data Collection
First, we collect expert loading trajectories (dataset/zc_interp_train) and introduce Gaussian noise during closed-loop tracking to generate a robust training dataset for the dynamics model.

To run the data collection process:

```bash
python -m simulation.collect_data
```

### Closed-Loop Dynamics Model
Second, we train a closed-loop dynamics model (network_model/predictor.py) using multi-step forward and backward propagation to predict excavator observations from reference trajectories, minimizing prediction error with regularization.

<img src="images/model learning.png" alt="Model Learning" width="500"/>

### Trajectory Adjustment Policy
Next, we optimize a trajectory adjustment policy (network_model/controller.py) via multi-step backpropagation to refine reference positions for accurate tracking, incorporating regularization to ensure smooth and stable adjustments.

<img src="images/policy learning.png" alt="Model Learning" width="550"/>

### Train
To train the closed-loop dynamics model and the trajectory adjustment policy, run:

```bash
python -m offline_train.train
```

## Outcome
To verify the tracking performance of the learned policy in simulation, run:

```bash
python -m test.test
python -m test.analyze_test
```

To visualize the tracking performance on a real-world excavator, run the following script to generate the comparison plot:

```bash
python -m test.analyze_pid_vs_test
```


<img src="images/field outcome.png" alt="Real-World Tracking" width="1000"/>
