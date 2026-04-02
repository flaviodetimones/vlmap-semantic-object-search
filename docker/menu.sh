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
            # ── Dataset selection ─────────────────────────────────────────
            echo ""
            echo "  ┌─────────────────────────────────────────────────┐"
            echo "  │              Select dataset                     │"
            echo "  ├─────────────────────────────────────────────────┤"
            echo "  │  1) MP3D  (Matterport3D — escenas reales)       │"
            echo "  │  2) HSSD  (Habitat Static Scene Dataset)        │"
            echo "  └─────────────────────────────────────────────────┘"
            echo -n "  Dataset (1/2, default 1): "
            read -r ds_choice
            ds_choice=${ds_choice:-1}

            if [ "$ds_choice" = "2" ]; then
                DATASET_TYPE="hssd"
                DATA_PATHS="hssd"
                SCENES_DIR=/workspace/data/vlmaps_dataset_hssd
                HSSD_CFG=/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json
                NAV_EXTRA="dataset_type=hssd scene_dataset_config_file=$HSSD_CFG"
                DS_LABEL="HSSD"
            else
                DATASET_TYPE="mp3d"
                DATA_PATHS="docker"
                SCENES_DIR=/workspace/data/vlmaps_dataset
                NAV_EXTRA=""
                DS_LABEL="MP3D"
            fi

            while true; do
                echo ""
                echo "  ┌─────────────────────────────────────────────────┐"
                echo "  │       VLMaps Pipeline  [$DS_LABEL]$([ "$DS_LABEL" = "MP3D" ] && echo "                  " || echo "                 ")│"
                echo "  ├─────────────────────────────────────────────────┤"
                echo "  │  r) Scripts reference (full pipeline overview)  │"
                echo "  │  s) List available scenes                       │"
                echo "  │  c) Collect dataset                             │"
                echo "  │  m) Create VLMap          (scene_id required)   │"
                echo "  │  i) Index map             (scene_id required)   │"
                echo "  │  l) Interactive LLM navigation                  │"
                echo "  │  g) Generate obstacle map image                 │"
                if [ "$DATASET_TYPE" = "mp3d" ]; then
                echo "  │  n) Label rooms (LabelMe → room_map)            │"
                fi
                echo "  │  b) Back                                        │"
                echo "  └─────────────────────────────────────────────────┘"
                echo -n "  Select: "
                read -r sub

                case "$sub" in
                    r|R)
                        echo ""
                        echo "════════════════════════════════════════════════════════════════"
                        echo "  VLMaps Pipeline — Scripts Reference  [$DS_LABEL]"
                        echo "════════════════════════════════════════════════════════════════"
                        echo ""
                        if [ "$DATASET_TYPE" = "hssd" ]; then
                        echo "  ── Step 0 · Collect dataset (navegación manual) ─────────────"
                        echo ""
                        echo "    python $DATASET/collect_hssd_dataset.py \\"
                        echo "        --scene_dataset_config $HSSD_CFG \\"
                        echo "        --scene_id 102344280"
                        echo ""
                        echo "  ── Step 1 · Create VLMap ────────────────────────────────────"
                        echo ""
                        echo "    python $APP/create_map.py data_paths=hssd scene_id=0"
                        echo ""
                        echo "  ── Step 2 · Index map ───────────────────────────────────────"
                        echo ""
                        echo "    python $APP/index_map.py data_paths=hssd init_categories=true scene_id=0"
                        echo ""
                        echo "  ── Step 3 · Interactive LLM navigation ──────────────────────"
                        echo ""
                        echo "    python $APP/interactive_object_nav.py data_paths=hssd scene_id=0 \\"
                        echo "        dataset_type=hssd \\"
                        echo "        scene_dataset_config_file=$HSSD_CFG"
                        else
                        echo "  All scripts use Hydra. Run them from inside the container."
                        echo "  data_paths=docker uses /workspace/data paths."
                        echo ""
                        echo "  ── Step 0 · Collect dataset ─────────────────────────────────"
                        echo ""
                        echo "    python $DATASET/collect_custom_dataset.py data_paths=docker \\"
                        echo "        scene_names=[\"SceneName\"]"
                        echo ""
                        echo "  ── Step 1 · Create VLMap ────────────────────────────────────"
                        echo ""
                        echo "    python $APP/create_map.py data_paths=docker scene_id=0"
                        echo ""
                        echo "  ── Step 2 · Index map ───────────────────────────────────────"
                        echo ""
                        echo "    python $APP/index_map.py data_paths=docker init_categories=true scene_id=0"
                        echo ""
                        echo "  ── Step 3 · Interactive LLM navigation ──────────────────────"
                        echo ""
                        echo "    python $APP/interactive_object_nav.py data_paths=docker scene_id=0"
                        fi
                        echo ""
                        echo "════════════════════════════════════════════════════════════════"
                        ;;
                    s|S)
                        echo ""
                        echo "  Available scenes [$DS_LABEL]:"
                        echo "  ─────────────────────────────────────────────────"
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                echo "    scene_id=$i  →  $(basename "$dir")"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                            if [ "$i" -eq 0 ]; then
                                echo "    (no scene folders found in $SCENES_DIR)"
                            fi
                        else
                            echo "    Directory not found: $SCENES_DIR"
                        fi
                        ;;
                    c|C)
                        echo ""
                        echo "  NOTE: requires X11 display (xhost +local:docker on host if needed)"
                        echo ""
                        if [ "$DATASET_TYPE" = "hssd" ]; then
                            if [ ! -f "$HSSD_CFG" ]; then
                                echo "  ERROR: HSSD dataset config not found: $HSSD_CFG"
                                echo "  Run:  apt-get install -y git-lfs && git lfs install"
                                echo "        cd /workspace/data/versioned_data/hssd-hab && git lfs pull"
                            else
                                echo "  Scene: 102344280  (15 habitaciones, ~447m²)"
                                echo "  Output: $SCENES_DIR/102344280_1/"
                                echo ""
                                echo "  Controls: w=forward  a=left  d=right  q=quit"
                                echo "  Each movement auto-saves RGB + depth + pose."
                                echo "  Aim for 500+ frames covering all rooms."
                                echo ""
                                echo -n "  scene_id to collect (default 102344280): "
                                read -r hssd_scene
                                hssd_scene=${hssd_scene:-102344280}
                                echo ""
                                echo "► python dataset/collect_hssd_dataset.py --scene_id $hssd_scene"
                                echo ""
                                cd /workspace/third_party/vlmaps
                                python "$DATASET/collect_hssd_dataset.py" \
                                    --scene_dataset_config "$HSSD_CFG" \
                                    --scene_id "$hssd_scene"
                            fi
                        else
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
                            echo -n "  Enter scene name (or b=back): "
                            read -r chosen_scene
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
                        fi
                        ;;
                    m|M)
                        echo ""
                        echo "  Available scenes [$DS_LABEL]:"
                        echo "  ─────────────────────────────────────────────────"
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
                        echo -n "  scene_id to build (default 0): "
                        read -r scene
                        scene=${scene:-0}
                        echo ""
                        echo "► Building VLMap for scene_id=$scene  [$DS_LABEL]..."
                        cd /workspace/third_party/vlmaps
                        python "$APP/create_map.py" data_paths="$DATA_PATHS" scene_id="$scene"
                        ;;
                    i|I)
                        echo ""
                        echo "  Available scenes [$DS_LABEL]:"
                        echo "  ─────────────────────────────────────────────────"
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
                        echo -n "  scene_id to index (default 0): "
                        read -r scene
                        scene=${scene:-0}
                        echo ""
                        echo "► Indexing VLMap for scene_id=$scene  [$DS_LABEL]..."
                        cd /workspace/third_party/vlmaps
                        python "$APP/index_map.py" data_paths="$DATA_PATHS" init_categories=true scene_id="$scene"
                        ;;
                    l|L)
                        echo ""
                        if [ -z "$OPENAI_API_KEY" ]; then
                            echo "  WARNING: OPENAI_API_KEY is not set. The script will fail."
                            echo "  Set it with: export OPENAI_API_KEY=sk-..."
                        fi
                        echo "  Available scenes [$DS_LABEL]:"
                        echo "  ─────────────────────────────────────────────────"
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
                        echo "► Launching interactive LLM navigation (scene $scene)  [$DS_LABEL]..."
                        echo "  Type instructions at the prompt. Type 'quit' to stop."
                        echo ""
                        cd /workspace/third_party/vlmaps
                        python "$APP/interactive_object_nav.py" \
                            data_paths="$DATA_PATHS" scene_id="$scene" $NAV_EXTRA
                        ;;
                    g|G)
                        echo ""
                        echo "  Available scenes [$DS_LABEL]:"
                        echo "  ─────────────────────────────────────────────────"
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                HAS_MAP=""
                                [ -f "$dir/obstacle_map.png" ] && HAS_MAP=" [map ready]"
                                echo "    scene_id=$i  →  $(basename "$dir")$HAS_MAP"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                        fi
                        echo ""
                        echo -n "  scene_id (default 0): "
                        read -r scene
                        scene=${scene:-0}
                        echo ""
                        echo "► Generating obstacle map images for scene $scene  [$DS_LABEL]..."
                        cd /workspace/third_party/vlmaps
                        python "$APP/generate_obstacle_map_png.py" \
                            data_paths="$DATA_PATHS" scene_id="$scene" $NAV_EXTRA
                        ;;
                    n|N)
                        if [ "$DATASET_TYPE" = "hssd" ]; then
                            echo "  Room labeling not needed for HSSD — annotations are automatic"
                            echo "  (read from semantics/scenes/<id>.semantic_config.json)."
                        else
                        echo ""
                        echo "  ╔═══════════════════════════════════════════════════╗"
                        echo "  ║           Room Labeling Workflow                  ║"
                        echo "  ╠═══════════════════════════════════════════════════╣"
                        echo "  ║  Step 1: Generate obstacle map (option g)         ║"
                        echo "  ║  Step 2: LabelMe opens → draw room polygons       ║"
                        echo "  ║  Step 3: Save in LabelMe (Ctrl+S), then close     ║"
                        echo "  ║  Step 4: Automatic conversion to room_map         ║"
                        echo "  ╠═══════════════════════════════════════════════════╣"
                        echo "  ║  TIPS:                                            ║"
                        echo "  ║  • Use Polygon tool (not Rectangle)               ║"
                        echo "  ║  • Draw LARGE polygons covering the whole room    ║"
                        echo "  ║  • Labels: living_room  bedroom  kitchen          ║"
                        echo "  ║    bathroom  office  hallway  dining_room         ║"
                        echo "  ║  • Multiple rooms same type: bedroom_1 bedroom_2  ║"
                        echo "  ╚═══════════════════════════════════════════════════╝"
                        echo ""
                        echo "  Available scenes:"
                        echo "  ─────────────────────────────────────────────────"
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r dir; do
                                HAS_MAP=""
                                ( [ -f "$dir/topdown_labeled.png" ] || [ -f "$dir/obstacle_map.png" ] ) && HAS_MAP=" [map ready]"
                                HAS_ROOMS=""
                                [ -f "$dir/room_map/room_map.npy" ] && HAS_ROOMS=" [rooms labeled]"
                                echo "    scene_id=$i  →  $(basename "$dir")$HAS_MAP$HAS_ROOMS"
                                i=$((i+1))
                            done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
                        fi
                        echo ""
                        echo -n "  scene_id (default 0): "
                        read -r scene
                        scene=${scene:-0}
                        SCENE_DIR=$(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort | sed -n "$((scene+1))p")
                        if [ -z "$SCENE_DIR" ]; then
                            echo "  Scene not found."
                        else
                            MAP_IMG="$SCENE_DIR/topdown_labeled.png"
                            [ ! -f "$MAP_IMG" ] && MAP_IMG="$SCENE_DIR/obstacle_map.png"
                            if [ ! -f "$MAP_IMG" ]; then
                                echo "  Map image not found. Run option 'g' first."
                            else
                                LABEL_JSON="$SCENE_DIR/room_map/room_labels.json"
                                mkdir -p "$SCENE_DIR/room_map"
                                echo ""
                                echo "► Opening LabelMe... (draw polygons, Ctrl+S to save, then close)"
                                echo "  Image: $MAP_IMG"
                                echo "  Output JSON: $LABEL_JSON"
                                echo ""
                                LD_PRELOAD=/opt/conda/envs/tfg/lib/libstdc++.so.6 \
                                    labelme "$MAP_IMG" \
                                    --output "$LABEL_JSON" \
                                    --autosave \
                                    --nodata
                                if [ -f "$LABEL_JSON" ]; then
                                    echo ""
                                    echo "► Converting LabelMe JSON → room_map..."
                                    cd /workspace/third_party/vlmaps
                                    python application/labelme_to_room_map.py \
                                        --json "$LABEL_JSON" \
                                        --scene "$SCENE_DIR"
                                else
                                    echo "  No JSON saved (LabelMe closed without saving)."
                                fi
                            fi
                        fi
                        fi
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
                # Patch labelme 5.10.1: np.bool removed in NumPy >=1.24
                LABELME_FILE=/opt/conda/envs/tfg/lib/python3.9/site-packages/labelme/_label_file.py
                if [ -f "$LABELME_FILE" ]; then
                    sed -i 's/NDArray\[np\.bool\]/NDArray[np.bool_]/g' "$LABELME_FILE"
                    echo "  labelme np.bool patch applied."
                fi
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
