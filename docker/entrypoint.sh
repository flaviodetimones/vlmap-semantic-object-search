#!/bin/bash
set -e

# Activate the tfg conda environment for interactive shells
source /opt/conda/etc/profile.d/conda.sh
conda activate tfg

# ── Install 'menu' as a standalone executable (runs once per container) ───────
if [ ! -f /usr/local/bin/menu ]; then
    cat > /usr/local/bin/menu << 'MENU_SCRIPT'
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
    echo -n "  Select an option: "
    read -r opcion

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
                echo "  │  c) Collect custom dataset (enter path)         │"
                echo "  │  l) Interactive LLM navigation                  │"
                echo "  │  b) Back                                        │"
                echo "  └─────────────────────────────────────────────────┘"
                echo -n "  Select: "
                read -r sub

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
                        echo "    python $DATASET/generate_dataset.py data_paths=docker"
                        echo "    python $DATASET/collect_minimal_dataset.py data_paths=docker"
                        echo "    python $DATASET/collect_hm3d_headless.py data_paths=docker"
                        echo "    python $DATASET/collect_custom_dataset.py data_paths=docker \\"
                        echo "        data_paths.vlmaps_data_dir=/your/path"
                        echo ""
                        echo "  ── Step 1 · Create VLMap ────────────────────────────────────"
                        echo "  Build the language-grounded 3-D voxel map from the dataset."
                        echo ""
                        echo "    python $APP/create_map.py data_paths=docker scene_id=0"
                        echo "    python $APP/create_map.py data_paths=docker scene_id=1"
                        echo ""
                        echo "  ── Step 2 · Index map (semantic query) ──────────────────────"
                        echo "  Load a saved VLMap and query it with a text category."
                        echo "  Asks interactively: 'What is your interested category?'"
                        echo ""
                        echo "    python $APP/index_map.py data_paths=docker scene_id=0"
                        echo "    python $APP/index_map.py data_paths=docker init_categories=true scene_id=0"
                        echo ""
                        echo "  ── Step 3a · Object-goal navigation ─────────────────────────"
                        echo "  Navigate the robot to an object specified by language."
                        echo ""
                        echo "    python $APP/object_goal_navigation.py data_paths=docker scene_id=0"
                        echo ""
                        echo "  ── Step 3b · Point navigation on map ────────────────────────"
                        echo "  Navigate to a goal expressed as a map coordinate."
                        echo ""
                        echo "    python $APP/point_navigation_on_map.py data_paths=docker scene_id=0"
                        echo ""
                        echo "  ── Step 3c · Interactive LLM navigation (GPT-4o-mini) ───────"
                        echo "  Type a natural-language instruction, the robot navigates live."
                        echo "  Requires: OPENAI_API_KEY env var set."
                        echo "  Controls: OpenCV window — press any key to advance each step."
                        echo "            type 'quit' or 'exit' at the prompt to stop."
                        echo ""
                        echo "    python $APP/interactive_object_nav.py data_paths=docker scene_id=0"
                        echo ""
                        echo "  ── Step 3d · Generate object nav tasks ──────────────────────"
                        echo "  Pre-generate object_navigation_tasks.json for each scene."
                        echo ""
                        echo "    python $APP/generate_object_nav_tasks.py data_paths=docker"
                        echo ""
                        echo "  ── Utilities ────────────────────────────────────────────────"
                        echo "  Generate obstacle map from occupancy grid:"
                        echo "    python $APP/generate_obstacle_map.py data_paths=docker scene_id=0"
                        echo ""
                        echo "  Test rotation / transform primitives (no Habitat needed):"
                        echo "    python $APP/test_primitives.py"
                        echo ""
                        echo "  Evaluation:"
                        echo "    python $APP/evaluation/evaluate_object_goal_navigation.py data_paths=docker"
                        echo "    python $APP/evaluation/evaluate_spatial_goal_navigation.py data_paths=docker"
                        echo "    python $APP/evaluation/compute_object_goal_navigation_metrics.py data_paths=docker"
                        echo "    python $APP/evaluation/compute_spatial_goal_navigation_metrics.py data_paths=docker"
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
                        echo -n "  Enter dataset path (e.g. /workspace/data/my_scene): "
                        read -r datapath
                        if [ -z "$datapath" ]; then
                            echo "  No path provided, cancelled."
                        else
                            echo ""
                            echo "► Running collect_custom_dataset.py with path: $datapath"
                            python "$DATASET/collect_custom_dataset.py" \
                                data_paths.vlmaps_data_dir="$datapath"
                        fi
                        ;;
                    l|L)
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
                        echo -n "  scene_id to use (default 0): "
                        read -r scene
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
MENU_SCRIPT
    chmod +x /usr/local/bin/menu
fi

# ── Ensure vlmaps is always importable ───────────────────────────────────────
grep -qxF 'export PYTHONPATH=/workspace/third_party/vlmaps:$PYTHONPATH' /root/.bashrc \
    || echo 'export PYTHONPATH=/workspace/third_party/vlmaps:$PYTHONPATH' >> /root/.bashrc

# ── Alias OPENAI_KEY for scripts that expect it instead of OPENAI_API_KEY ────
grep -qxF 'export OPENAI_KEY=$OPENAI_API_KEY' /root/.bashrc \
    || echo 'export OPENAI_KEY=$OPENAI_API_KEY' >> /root/.bashrc
export OPENAI_KEY=$OPENAI_API_KEY

# ── Welcome banner ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         TFG — VLMaps Semantic Object Search                     ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Python  : $(python --version 2>&1)"
echo "║  CUDA    : $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "║  GPU     : $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'No GPU detected')"
echo "║  Workdir : $(pwd)"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Type  'menu'  to see available commands                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

exec "$@"
