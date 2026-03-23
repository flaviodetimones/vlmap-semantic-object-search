#!/bin/bash
export PYTHONPATH=/workspace/third_party/vlmaps:$PYTHONPATH
export OPENAI_KEY="${OPENAI_KEY:-$OPENAI_API_KEY}"
APP=/workspace/third_party/vlmaps/application
DATASET=/workspace/third_party/vlmaps/dataset

while true; do
    echo ""
    echo "┌─────────────────────────────────────────────────────┐"
    echo "│                 Available commands                  │"
    echo "├─────────────────────────────────────────────────────┤"
    echo "│  1) Check GPU / CUDA                                │"
    echo "│  2) Show workspace structure                        │"
    echo "│  3) VLMaps pipeline                                 │"
    echo "│  4) Start Jupyter Notebook (port 8888)              │"
    echo "│  5) Install / update Python dependencies            │"
    echo "│  6) Open interactive Python shell (conda tfg)       │"
    echo "│  q) Quit                                            │"
    echo "└─────────────────────────────────────────────────────┘"
    read -rp "  Select an option: " opcion

    case "$opcion" in
        1)
            echo ""
            echo "► GPU / CUDA status:"
            echo "─────────────────────────────────────────"
            nvidia-smi 2>/dev/null || echo "  nvidia-smi not available"
            echo ""
            python -c "
import torch
print(f'  PyTorch   : {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU       : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
            ;;
        2)
            echo ""
            echo "► Workspace structure:"
            echo "─────────────────────────────────────────"
            find /workspace -maxdepth 2 -not -path '*/\.*' \
                -not -path '*/data/*' -not -path '*/results/*' \
                | sort | sed 's|/workspace/||; s|^|  |'
            ;;
        3)
            while true; do
                echo ""
                echo "  ┌─────────────────────────────────────────────────┐"
                echo "  │              VLMaps Pipeline                    │"
                echo "  ├─────────────────────────────────────────────────┤"
                echo "  │  r) Scripts reference (full pipeline overview)  │"
                echo "  │  s) List available scenes                       │"
                echo "  │  c) Collect dataset (smart frame filtering)     │"
                echo "  │  m) Create VLMap          (scene_id required)   │"
                echo "  │  i) Index map             (scene_id required)   │"
                echo "  │  1) Baseline M1: Random   (scene_id required)   │"
                echo "  │  2) Baseline M2: Nearest  (scene_id required)   │"
                echo "  │  y) YOLOE detection       (scene_id required)   │"
                echo "  │  p) Project detections 3D (scene_id required)   │"
                echo "  │  h) Heatmap region analysis(scene_id required)   │"
                echo "  │  l) LabelMe room labeling (export + convert)     │"
                echo "  │  n) Interactive LLM navigation                  │"
                echo "  │  b) Back                                        │"
                echo "  └─────────────────────────────────────────────────┘"
                read -rp "  Select: " sub

                case "$sub" in
                    r|R)
                        echo ""
                        echo "════════════════════════════════════════════════════════════════"
                        echo "  VLMaps Pipeline — Scripts Reference"
                        echo "════════════════════════════════════════════════════════════════"
                        echo ""
                        echo "  All scripts use Hydra. Run them from inside the container."
                        echo "  data_paths=docker always required to use /workspace/data paths."
                        echo "  Config files: third_party/vlmaps/config/"
                        echo ""
                        echo "  ── Step 0 · Collect dataset ─────────────────────────────────"
                        echo "  Collect a Habitat-Sim dataset (RGB-D + poses) for a scene."
                        echo ""
                        echo "    python $DATASET/collect_custom_dataset.py data_paths=docker \\"
                        echo "        scene_names=[\"SceneName\"]"
                        echo ""
                        echo "  ── Step 1 · Create VLMap ────────────────────────────────────"
                        echo "  Build the language-grounded 3-D voxel map from the dataset."
                        echo ""
                        echo "    python $APP/create_map.py data_paths=docker scene_id=0"
                        echo ""
                        echo "  ── Step 2 · Index map (semantic query) ──────────────────────"
                        echo "  Load a saved VLMap and query it with a text category."
                        echo ""
                        echo "    python $APP/index_map.py data_paths=docker init_categories=true scene_id=0"
                        echo ""
                        echo "  ── Step 3 · Interactive LLM navigation (GPT-4o-mini) ────────"
                        echo "  Type a natural-language instruction, the robot navigates live."
                        echo "  Requires: OPENAI_API_KEY env var set."
                        echo ""
                        echo "    python $APP/interactive_object_nav.py data_paths=docker scene_id=0"
                        echo ""
                        echo "  ── Utilities ────────────────────────────────────────────────"
                        echo "    python $APP/generate_object_nav_tasks.py data_paths=docker"
                        echo "    python $APP/generate_obstacle_map.py data_paths=docker scene_id=0"
                        echo "    python $APP/evaluation/evaluate_object_goal_navigation.py data_paths=docker"
                        echo ""
                        echo "════════════════════════════════════════════════════════════════"
                        ;;
                    s|S)
                        echo ""
                        echo "  Available scenes:"
                        echo "  ─────────────────────────────────────────────────"
                        SCENES_DIR=/workspace/data/vlmaps_dataset
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                echo "    scene_id=$i  →  $(basename "$dir")"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                            if [ "$i" -eq 0 ]; then
                                echo "    (no scene folders found)"
                            fi
                        else
                            echo "    Directory not found: $SCENES_DIR"
                            echo "    Make sure the data volume is mounted."
                        fi
                        ;;
                    c|C)
                        echo ""
                        echo "  NOTE: requires X11 display. If you get display errors,"
                        echo "  run on your HOST terminal first:  xhost +local:docker"
                        echo ""
                        HABITAT_DIR=/workspace/data/mp3d
                        echo "  Available MP3D scenes:"
                        echo "  ────────────────────────────────────────────────────"
                        if [ -d "$HABITAT_DIR" ]; then
                            for scene_path in $(find "$HABITAT_DIR" -mindepth 1 -maxdepth 1 -type d | sort); do
                                echo "    $(basename "$scene_path")"
                            done
                        else
                            echo "    (no scenes found at $HABITAT_DIR)"
                        fi
                        echo ""
                        echo "  Output: /workspace/data/vlmaps_dataset/<scene>_<id>/"
                        echo ""
                        read -rp "  Enter scene name (or b=back): " chosen_scene
                        if [ -z "$chosen_scene" ] || [ "$chosen_scene" = "b" ] || [ "$chosen_scene" = "B" ]; then
                            echo "  Cancelled."
                        elif [ -d "$HABITAT_DIR/$chosen_scene" ]; then
                            echo ""
                            echo "► python dataset/collect_custom_dataset.py data_paths=docker scene_names=[\"$chosen_scene\"]"
                            echo ""
                            cd /workspace/third_party/vlmaps
                            python "$DATASET/collect_custom_dataset.py" \
                                data_paths=docker "scene_names=[\"$chosen_scene\"]"
                        else
                            echo "  Scene '$chosen_scene' not found in $HABITAT_DIR."
                        fi
                        ;;
                    m|M)
                        echo ""
                        echo "  Available scenes:"
                        echo "  ─────────────────────────────────────────────────"
                        SCENES_DIR=/workspace/data/vlmaps_dataset
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                echo "    scene_id=$i  →  $(basename "$dir")"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                        else
                            echo "    (data directory not found)"
                        fi
                        echo ""
                        read -rp "  scene_id to build (default 0): " scene
                        scene=${scene:-0}
                        echo ""
                        echo "► Building VLMap for scene_id=$scene..."
                        cd /workspace/third_party/vlmaps
                        python "$APP/create_map.py" data_paths=docker scene_id="$scene"
                        ;;
                    i|I)
                        echo ""
                        echo "  Available scenes:"
                        echo "  ─────────────────────────────────────────────────"
                        SCENES_DIR=/workspace/data/vlmaps_dataset
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                echo "    scene_id=$i  →  $(basename "$dir")"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                        else
                            echo "    (data directory not found)"
                        fi
                        echo ""
                        read -rp "  scene_id to index (default 0): " scene
                        scene=${scene:-0}
                        echo ""
                        echo "► Indexing VLMap for scene_id=$scene..."
                        cd /workspace/third_party/vlmaps
                        python "$APP/index_map.py" data_paths=docker init_categories=true scene_id="$scene"
                        ;;
                    1)
                        echo ""
                        echo "  Available scenes:"
                        echo "  ─────────────────────────────────────────────────"
                        SCENES_DIR=/workspace/data/vlmaps_dataset
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                echo "    scene_id=$i  →  $(basename "$dir")"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                        else
                            echo "    (data directory not found)"
                        fi
                        echo ""
                        read -rp "  scene_id (default 0): " scene
                        scene=${scene:-0}
                        read -rp "  query object (default chair): " query
                        query=${query:-chair}
                        read -rp "  episodes (default 5): " eps
                        eps=${eps:-5}
                        echo ""
                        echo "► Running M1 Random baseline: query=$query, scene=$scene, episodes=$eps..."
                        cd /workspace
                        python src/run_baselines.py data_paths=docker \
                            scene_id="$scene" nav.vis=True +query="$query" +method=random +episodes="$eps"
                        ;;
                    2)
                        echo ""
                        echo "  Available scenes:"
                        echo "  ─────────────────────────────────────────────────"
                        SCENES_DIR=/workspace/data/vlmaps_dataset
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                echo "    scene_id=$i  →  $(basename "$dir")"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                        else
                            echo "    (data directory not found)"
                        fi
                        echo ""
                        read -rp "  scene_id (default 0): " scene
                        scene=${scene:-0}
                        read -rp "  query object (default chair): " query
                        query=${query:-chair}
                        read -rp "  episodes (default 5): " eps
                        eps=${eps:-5}
                        echo ""
                        echo "► Running M2 Nearest baseline: query=$query, scene=$scene, episodes=$eps..."
                        cd /workspace
                        python src/run_baselines.py data_paths=docker \
                            scene_id="$scene" nav.vis=True +query="$query" +method=nearest +episodes="$eps"
                        ;;
                    y|Y)
                        echo ""
                        echo "  Available scenes:"
                        echo "  ─────────────────────────────────────────────────"
                        SCENES_DIR=/workspace/data/vlmaps_dataset
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                echo "    scene_id=$i  →  $(basename "$dir")"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                        else
                            echo "    (data directory not found)"
                        fi
                        echo ""
                        read -rp "  scene_id (default 0): " scene
                        scene=${scene:-0}
                        read -rp "  queries (default chair,table,sofa,bed,toilet): " queries
                        queries=${queries:-chair,table,sofa,bed,toilet}
                        IFS=',' read -ra q_arr <<< "$queries"
                        q_hydra="[$(printf '"%s",' "${q_arr[@]}" | sed 's/,$//' )]"
                        read -rp "  sample_rate (default 10): " sr
                        sr=${sr:-10}
                        echo ""
                        echo "► Running YOLOE detection: scene=$scene, queries=$q_hydra, sample_rate=$sr..."
                        cd /workspace
                        python src/run_yoloe_detect.py data_paths=docker \
                            scene_id="$scene" "+queries=$q_hydra" "+sample_rate=$sr"
                        ;;
                    p|P)
                        echo ""
                        echo "  Available scenes:"
                        echo "  ─────────────────────────────────────────────────"
                        SCENES_DIR=/workspace/data/vlmaps_dataset
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                echo "    scene_id=$i  →  $(basename "$dir")"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                        else
                            echo "    (data directory not found)"
                        fi
                        echo ""
                        read -rp "  scene_id (default 0): " scene
                        scene=${scene:-0}
                        echo ""
                        echo "► Projecting 2D detections to 3D: scene=$scene..."
                        cd /workspace
                        python src/project_detections_3d.py data_paths=docker scene_id="$scene"
                        ;;
                    h|H)
                        echo ""
                        echo "  Available scenes:"
                        echo "  ─────────────────────────────────────────────────"
                        SCENES_DIR=/workspace/data/vlmaps_dataset
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                echo "    scene_id=$i  →  $(basename "$dir")"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                        else
                            echo "    (data directory not found)"
                        fi
                        echo ""
                        read -rp "  scene_id (default 0): " scene
                        scene=${scene:-0}
                        read -rp "  query object (default chair): " query
                        query=${query:-chair}
                        echo ""
                        echo "► Heatmap region analysis: scene=$scene, query=$query..."
                        cd /workspace
                        python src/analyze_vlmap_heatmap.py data_paths=docker \
                            scene_id="$scene" "+query=$query"
                        ;;
                    l|L)
                        echo ""
                        echo "  Available scenes:"
                        echo "  ─────────────────────────────────────────────────"
                        SCENES_DIR=/workspace/data/vlmaps_dataset
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                echo "    scene_id=$i  →  $(basename "$dir")"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                        else
                            echo "    (data directory not found)"
                        fi
                        echo ""
                        read -rp "  scene_id (default 0): " scene
                        scene=${scene:-0}
                        # Resolve scene dir for the PNG path
                        SCENES_DIR=/workspace/data/vlmaps_dataset
                        scene_name=$(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort | sed -n "$((scene+1))p")
                        png_path="$scene_name/topdown_rgb.png"

                        echo ""
                        echo "► Step 1: Exporting top-down RGB map..."
                        cd /workspace
                        python src/export_topdown_map.py data_paths=docker scene_id="$scene"

                        echo ""
                        echo "► Step 2: Opening LabelMe..."
                        echo "  Draw polygons for each room (kitchen, bedroom, hallway, etc.)"
                        echo "  Save (Ctrl+S) and close LabelMe when done."
                        echo ""
                        pip install -q labelme 2>/dev/null
                        LD_LIBRARY_PATH="/opt/conda/envs/tfg/lib:$LD_LIBRARY_PATH" \
                            python -c "
import os, sys, numpy as np
np.bool = bool; np.int = int; np.float = float
# Force PyQt5 plugins, block cv2 Qt
import PyQt5
qt_dir = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins')
os.environ['QT_PLUGIN_PATH'] = qt_dir
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(qt_dir, 'platforms')
# Block cv2 from overriding Qt
sys.modules['cv2'] = type(sys)('cv2')
sys.argv = ['labelme'] + sys.argv[1:]
from labelme.__main__ import main; main()
" "$png_path" --output "$scene_name/topdown_rgb.json"

                        lm_json="$scene_name/topdown_rgb.json"
                        if [ -f "$lm_json" ]; then
                            echo ""
                            echo "► Step 3: Converting LabelMe annotations to room_map..."
                            python src/convert_labelme_rooms.py data_paths=docker \
                                scene_id="$scene" "+labelme_json=$lm_json"
                        else
                            echo "  No LabelMe JSON found — skipping conversion."
                        fi
                        ;;
                    n|N)
                        echo ""
                        if [ -z "$OPENAI_API_KEY" ]; then
                            echo "  WARNING: OPENAI_API_KEY is not set. The script will fail."
                            echo "  Set it with: export OPENAI_API_KEY=sk-..."
                        fi
                        echo "  Available scenes:"
                        echo "  ─────────────────────────────────────────────────"
                        SCENES_DIR=/workspace/data/vlmaps_dataset
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                echo "    scene_id=$i  →  $(basename "$dir")"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                        else
                            echo "    (data directory not found)"
                        fi
                        echo ""
                        read -rp "  scene_id to use (default 0): " scene
                        scene=${scene:-0}
                        echo ""
                        echo "► Launching interactive LLM navigation (scene $scene)..."
                        echo "  Type instructions at the prompt. Press any key in the OpenCV window to advance."
                        echo "  Type 'quit' or 'exit' to stop."
                        echo ""
                        python "$APP/interactive_object_nav.py" data_paths=docker scene_id="$scene"
                        ;;
                    b|B)
                        break
                        ;;
                    *)
                        echo "  Invalid option."
                        ;;
                esac
            done
            ;;
        4)
            echo ""
            echo "► Launching Jupyter Notebook at http://localhost:8888 ..."
            jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
                --NotebookApp.token='' --NotebookApp.password=''
            ;;
        5)
            echo ""
            echo "► Installing/updating dependencies from requirements.txt..."
            if [ -f /workspace/docker/requirements.txt ]; then
                pip install -r /workspace/docker/requirements.txt
            else
                echo "  /workspace/docker/requirements.txt not found."
            fi
            ;;
        6)
            echo ""
            echo "► Opening interactive Python (conda env: tfg)..."
            python
            ;;
        q|Q)
            echo ""
            echo "  Bye!"
            echo ""
            break
            ;;
        *)
            echo "  Invalid option. Choose 1-6 or 'q'."
            ;;
    esac
done
