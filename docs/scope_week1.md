# Bachelor Thesis Scope (Week 1)

## Title
VLMap-Based Semantic Object Search with YOLOE

## Problem Statement

Robots operating in indoor environments must often search for objects specified by a human using natural language. 
A naive strategy would simply explore the environment randomly or move toward the nearest detected object. 
However, such strategies are inefficient when multiple candidate locations exist.

This thesis investigates **intelligent object search** in indoor environments by combining:

- open-vocabulary perception
- semantic spatial mapping
- navigation-aware goal selection

The system receives an object query (for example "mug" or "chair") and must search the environment efficiently by deciding **where to look first and which candidate goal to visit**.

---

## Hypothesis

Combining open-vocabulary object detection with **YOLOE** and semantic spatial priors from **VLMaps** can improve object search efficiency compared to simple baselines such as random exploration or nearest-candidate navigation.

---

## Compared Methods

The perception module of the system will be based on **YOLOE**, which provides open-vocabulary object detection from text prompts.

The thesis will compare several **object search strategies** built on top of this perception system:

1. Random Exploration  
   The robot explores the environment randomly while using YOLOE to detect potential target objects.

2. Nearest Candidate Selection  
   When YOLOE detects multiple candidate objects, the robot selects the closest one based on navigation distance.

3. Semantic Region Search using VLMaps  
   The robot uses the semantic information stored in the VLMap to prioritize regions where the target object is more likely to appear.

4. Hybrid Semantic Goal Selection (VLMap + YOLOE)  
   Candidate objects detected by YOLOE are ranked using both semantic context from the VLMap and navigation cost.

---

## Core Scope (Mandatory)

The core of the thesis focuses on **simulation experiments**.

The system will:

- use the official **VLMaps repository** as the semantic mapping backbone
- integrate **YOLOE** as an open-vocabulary perception module
- implement several object search strategies
- compare these strategies experimentally in simulation

The minimum expected outcome is a **working simulation pipeline with quantitative comparison between methods**.

---

## Extension (Optional)

As an extension, the pipeline may be transferred to a **real robot** and integrated with **ROS 2**.

This would demonstrate that the same semantic search approach can be applied outside simulation.

---

## Risks

Potential risks include:

- dependency conflicts when installing VLMaps
- dependency conflicts when installing YOLOE
- integration complexity between perception and semantic mapping
- navigation complexity in simulation environments
- limited time for real robot integration

---

## Contingency Plan

If the real robot extension cannot be completed, the thesis will still be considered successful if:

- the full simulation pipeline works
- several search strategies are implemented
- the experimental comparison is completed
