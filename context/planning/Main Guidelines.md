


## PES UNIVERSITY
Department of Computer Science and Engineering
RR Campus
## UE24CS352A: MACHINE LEARNING
## PATHSNAP: DISTRIBUTED MAP-MATCHING &
## ROUTE RECONSTRUCTION
## Hackathon Guide
## 1. Challenge Overview
GPS tracking systems often produce inaccurate location measurements due to urban buildings, signal
interference, multipath effects, GPS dropouts and low sampling frequencies.
Consequently, a vehicle travelling along a road may appear to move onto a parallel street, jump across
buildings or deviate from its actual route.
Your task is to develop PathSnap, a distributed machine learning system that reconstructs realistic
vehicle trajectories by matching noisy GPS observations to a road network.
The challenge involves developing an end-to-end solution, from understanding and processing raw GPS
data to training machine learning models, reconstructing routes and deploying the final system.

This is a folium map visualization of a trip from the provided dataset where the blue dots represent the
raw gps points and the green line is the actual vehicle path (ground truth).


## 2. Important: Read Before Beginning
Students are strongly encouraged to thoroughly read all the supporting documents provided for
this hackathon before beginning implementation or using AI tools.
These documents contain important information about the problem context, dataset, expected outcomes,
technical requirements, evaluation criteria and submission guidelines. Understanding these details is
essential for making informed technical decisions and successfully completing the challenge. Relying
solely on AI-generated suggestions without first understanding the problem and the provided material
may lead to incorrect implementation choices and incomplete solutions.
- Two-Laptop Deployment Architecture
## Laptop Component Responsibilities
## Laptop 1
Model and
## API
Hosts the trained classifier, HMM/Viterbi implementation, road-network
information and server-side evaluation process
## Laptop 2 User Interface
Displays the ten evaluation-trip options, sends the selected trip to the API
and displays the returned route and results

## 4. Expected Work Process
The overall challenge should be approached in the following stages:
- Understand the problem.
- Explore and understand the raw dataset.
- Develop an independent data-processing and feature-engineering approach.
- Train and compare multiple machine learning models.
- Select the best-performing model based on experimental evidence.
- Perform Bayesian hyperparameter optimisation.
- Implement HMM and Viterbi-based route reconstruction.
- Export the trained model and required artifacts.
- Develop the FastAPI backend.
- Develop the evaluator-facing UI.
- Integrate and test the complete system.
- Prepare the final technical report.
