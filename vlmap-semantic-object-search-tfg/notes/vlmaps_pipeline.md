VLMaps takes RGB-D observations and poses, builds a semantic 3D map, allows language-based indexing through heatmaps or masks, and provides a navigation framework for object-goal tasks. In my thesis, I will use VLMaps mainly as semantic spatial memory, while YOLOE will provide open-vocabulary detections and my search strategy will combine both.

# VLMaps Pipeline

## What VLMaps is
VLMaps is a semantic spatial map representation for robot navigation.
It builds a 3D map from RGB-D observations and stores visual-language semantic information in the map.

## Role in this thesis
In this thesis, VLMaps will be used as the semantic spatial memory of the environment.
It will provide semantic priors about where target objects are more likely to appear.
YOLOE will be used as the open-vocabulary detector.

## Input data
VLMaps requires:
- RGB images
- depth images
- poses

Typical dataset structure:
- rgb/
- depth/
- poses.txt

## Stage 1: Dataset generation

Script:
`dataset/generate_dataset.py`

Purpose:
Generate RGB-D observations from Habitat simulator scenes using a sequence of poses.

Input:
- Habitat scene (.glb)
- poses.txt
- sensor configuration

Output:
- RGB images
- depth images
- optional semantic observations
- scene folder with poses

Important details:
- uses Habitat Simulator
- uses Hydra config
- configuration file: `config/generate_dataset.yaml`

## Stage 2: Map creation

Script:
`application/create_map.py`

Purpose:
Build the VLMap from a generated dataset.

Input:
- scene folder with RGB, depth and poses
- map creation config

Output:
- semantic 3D map with LSeg embeddings stored per voxel

Important details:
- uses Hydra config
- configuration file: `config/map_creation_cfg.yaml`

## Stage 3: Map indexing

Script:
`application/index_map.py`

Purpose:
Load a previously created VLMap and query it using natural language.

Input:
- created VLMap
- text category or query

Output:
- semantic mask
- 2D or 3D heatmap
- visualization of relevant regions

Important details:
- uses Hydra config
- configuration file: `config/map_indexing_cfg.yaml`
- can work with predefined Matterport3D categories or free-text queries
- initializes CLIP before indexing

## Stage 4: Object-goal navigation evaluation

Script:
`application/evaluation/evaluate_object_goal_navigation.py`

Purpose:
Evaluate navigation to object categories using the VLMap system.

Process:
- setup robot and task in Habitat
- load object-goal tasks
- parse instruction
- navigate to target object categories
- replay actions and evaluate task performance
- save metrics per task

Related metrics script:
`application/evaluation/compute_object_goal_navigation_metrics.py`

## Important configuration files
- `config/generate_dataset.yaml`
- `config/map_creation_cfg.yaml`
- `config/map_indexing_cfg.yaml`
- `config/object_goal_navigation_cfg.json`
- `config/data_paths/default.yaml`
- `config/vlmaps.yaml`
- `config/params/default.yaml`

## How VLMaps fits my thesis
VLMaps will provide:
- semantic map creation
- language-based spatial priors
- relevant regions for object search
- a reference navigation evaluation setup

YOLOE will provide:
- open-vocabulary object detections
- visual candidates
- confidence scores

My contribution will be the search / goal selection layer that combines:
- YOLOE detections
- VLMap semantic priors
- navigation cost

## What I do not need to master yet
- every configuration parameter
- full spatial-goal navigation pipeline
- customized real-robot dataset integration
- internal implementation details of all VLMap classes

## Questions / doubts
- Should I start from an already generated scene before generating my own dataset?
- Which output of `index_map.py` is the easiest to connect to candidate ranking?
- How much of the original VLMaps navigation stack should I reuse?
