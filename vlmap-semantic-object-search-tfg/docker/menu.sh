#!/bin/bash
export PYTHONPATH=/workspace/third_party/vlmaps:$PYTHONPATH
export OPENAI_KEY="${OPENAI_KEY:-$OPENAI_API_KEY}"
APP=/workspace/third_party/vlmaps/application
DATASET=/workspace/third_party/vlmaps/dataset

run_other_menu() {
    while true; do
        echo ""
        echo "  ┌─────────────────────────────────────────────────────┐"
        echo "  │                    Other tools                      │"
        echo "  ├─────────────────────────────────────────────────────┤"
        echo "  │  1) Check GPU / CUDA                                │"
        echo "  │  2) Show workspace structure                        │"
        echo "  │  3) Start Jupyter Notebook (port 8888)              │"
        echo "  │  4) Install / update Python dependencies            │"
        echo "  │  5) Open interactive Python shell (conda tfg)       │"
        echo "  │  b) Back                                            │"
        echo "  └─────────────────────────────────────────────────────┘"
        echo -n "  Select an option: "
        read -r other_opt

        case "$other_opt" in
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
                echo ""
                echo "► Launching Jupyter Notebook at http://localhost:8888 ..."
                jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
                    --NotebookApp.token='' --NotebookApp.password=''
                ;;
            4)
                echo ""
                echo "► Installing/updating dependencies from requirements.txt..."
                if [ -f /workspace/docker/requirements.txt ]; then
                    pip install -r /workspace/docker/requirements.txt
                    LABELME_FILE=/opt/conda/envs/tfg/lib/python3.9/site-packages/labelme/_label_file.py
                    if [ -f "$LABELME_FILE" ]; then
                        sed -i 's/NDArray\[np\.bool\]/NDArray[np.bool_]/g' "$LABELME_FILE"
                        echo "  labelme np.bool patch applied."
                    fi
                else
                    echo "  /workspace/docker/requirements.txt not found."
                fi
                ;;
            5)
                echo ""
                echo "► Opening interactive Python (conda env: tfg)..."
                python
                ;;
            b|B)
                break
                ;;
            *)
                echo "  Invalid option."
                ;;
        esac
    done
}

print_scene_list() {
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
}

scene_name_from_id() {
    local scene_id="$1"
    if [ ! -d "$SCENES_DIR" ]; then
        return 1
    fi
    find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort | sed -n "$((scene_id + 1))p" | xargs -r basename
}

scene_count() {
    if [ ! -d "$SCENES_DIR" ]; then
        echo 0
        return
    fi
    find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort | wc -l
}

prompt_valid_scene_id() {
    local prompt_text="$1"
    local default_scene="$2"
    local count
    local max_scene
    local scene

    count=$(scene_count)
    if [ "$count" -le 0 ]; then
        echo "  No scenes available in $SCENES_DIR."
        return 1
    fi

    max_scene=$((count - 1))
    if [ "$default_scene" -gt "$max_scene" ]; then
        default_scene=0
    fi

    while true; do
        echo -n "  $prompt_text (default $default_scene): "
        read -r scene
        scene=${scene:-$default_scene}
        if [[ "$scene" =~ ^[0-9]+$ ]] && [ "$scene" -ge 0 ] && [ "$scene" -lt "$count" ]; then
            SELECTED_SCENE_ID="$scene"
            return 0
        fi
        echo "  Invalid scene_id '$scene'. Valid range: 0-$max_scene."
    done
}

run_testing_menu() {
    while true; do
        echo ""
        echo "  ┌─────────────────────────────────────────────────┐"
        echo "  │         Testing / Evaluation  [$DS_LABEL]$([ "$DS_LABEL" = "MP3D" ] && echo "           " || echo "          ")│"
        echo "  ├─────────────────────────────────────────────────┤"
        echo "  │  1) Generate test set                           │"
        echo "  │  2) Compare full 2x2 pipeline                   │"
        echo "  │  3) Heatmap-only offline analysis               │"
        echo "  │  b) Back                                        │"
        echo "  └─────────────────────────────────────────────────┘"
        echo -n "  Select: "
        read -r test_opt

        case "$test_opt" in
            1)
                echo ""
                if [ "$DATASET_TYPE" != "hssd" ]; then
                    echo "  Test set generation is currently HSSD-only."
                    echo "  Switch to HSSD from the dataset menu."
                else
                    echo "  Available scenes [$DS_LABEL]:"
                    echo "  ─────────────────────────────────────────────────"
                    print_scene_list
                    echo ""
                    echo -n "  Scene ids comma-separated (default 0,1): "
                    read -r eval_scene_ids
                    eval_scene_ids=${eval_scene_ids:-0,1}
                    echo -n "  Queries per scene (default 50): "
                    read -r eval_qps
                    eval_qps=${eval_qps:-50}
                    echo -n "  Min navigable room ratio (default 0.25): "
                    read -r eval_min_nav
                    eval_min_nav=${eval_min_nav:-0.25}
                    echo -n "  Seed (default 21042026): "
                    read -r eval_seed
                    eval_seed=${eval_seed:-21042026}
                    echo ""
                    echo "► Generating normalized evaluation query JSONL..."
                    echo "  Output: /workspace/tools/eval_queries/{scene_name}.jsonl"
                    cd /workspace
                    python tools/build_eval_queries.py \
                        --scene-ids "$eval_scene_ids" \
                        --queries-per-scene "$eval_qps" \
                        --dataset-type "$DATASET_TYPE" \
                        --data-paths "$DATA_PATHS" \
                        --scene-dataset-config-file "$HSSD_CFG" \
                        --min-room-navigable "$eval_min_nav" \
                        --seed "$eval_seed"
                fi
                ;;
            2)
                echo ""
                if [ "$DATASET_TYPE" != "hssd" ]; then
                    echo "  Full 2x2 comparison is currently HSSD-only."
                    echo "  Switch to HSSD from the dataset menu."
                elif [ -z "$OPENAI_API_KEY" ]; then
                    echo "  WARNING: OPENAI_API_KEY is not set. Set it before running the 2x2 comparison."
                else
                    echo "  Available scenes [$DS_LABEL]:"
                    echo "  ─────────────────────────────────────────────────"
                    print_scene_list
                    echo ""
                    echo -n "  Scene ids comma-separated (default 0): "
                    read -r scene_ids
                    scene_ids=${scene_ids:-0}
                    echo -n "  Queries path or directory (blank = default eval_queries): "
                    read -r eval_queries
                    echo -n "  Executor policy mode [heuristic|hybrid|llm] (default hybrid): "
                    read -r policy_mode
                    policy_mode=${policy_mode:-hybrid}
                    STAMP=$(date +%Y%m%d_%H%M%S)
                    OUT_DIR="/workspace/results/eval_runs/${STAMP}"
                    echo ""
                    echo "► Running full 2x2 pipeline evaluation..."
                    echo "  Output root: $OUT_DIR"
                    cd /workspace
                    if [ -n "$eval_queries" ]; then
                        python tools/run_full_2x2_eval.py \
                            --scene-ids "$scene_ids" \
                            --queries "$eval_queries" \
                            --dataset-type "$DATASET_TYPE" \
                            --data-paths "$DATA_PATHS" \
                            --scene-dataset-config-file "$HSSD_CFG" \
                            --policy-mode "$policy_mode" \
                            --out "$OUT_DIR"
                    else
                        python tools/run_full_2x2_eval.py \
                            --scene-ids "$scene_ids" \
                            --dataset-type "$DATASET_TYPE" \
                            --data-paths "$DATA_PATHS" \
                            --scene-dataset-config-file "$HSSD_CFG" \
                            --policy-mode "$policy_mode" \
                            --out "$OUT_DIR"
                    fi
                    echo ""
                    echo "  Results root:  $OUT_DIR"
                fi
                ;;
            3)
                echo ""
                if [ "$DATASET_TYPE" != "hssd" ]; then
                    echo "  Heatmap-only offline analysis is currently HSSD-only."
                    echo "  Switch to HSSD from the dataset menu."
                else
                    echo "  Available scenes [$DS_LABEL]:"
                    echo "  ─────────────────────────────────────────────────"
                    print_scene_list
                    echo ""
                    echo -n "  Scene ids comma-separated (default 0): "
                    read -r scene_ids
                    scene_ids=${scene_ids:-0}
                    echo -n "  Queries path or directory (blank = default eval_queries): "
                    read -r heat_queries
                    echo -n "  Save overlay images? [y/N]: "
                    read -r save_imgs
                    STAMP=$(date +%Y%m%d_%H%M%S)
                    OUT_DIR="/workspace/results/eval_runs/${STAMP}"
                    echo ""
                    echo "► Running heatmap-only offline analysis..."
                    echo "  Output root: $OUT_DIR"
                    cd /workspace
                    if [[ "$save_imgs" =~ ^[Yy]$ ]]; then
                        if [ -n "$heat_queries" ]; then
                            python tools/run_heatmap_offline_eval.py \
                                --scene-ids "$scene_ids" \
                                --queries "$heat_queries" \
                                --dataset-type "$DATASET_TYPE" \
                                --data-paths "$DATA_PATHS" \
                                --scene-dataset-config-file "$HSSD_CFG" \
                                --save-images \
                                --out "$OUT_DIR"
                        else
                            python tools/run_heatmap_offline_eval.py \
                                --scene-ids "$scene_ids" \
                                --dataset-type "$DATASET_TYPE" \
                                --data-paths "$DATA_PATHS" \
                                --scene-dataset-config-file "$HSSD_CFG" \
                                --save-images \
                                --out "$OUT_DIR"
                        fi
                    else
                        if [ -n "$heat_queries" ]; then
                            python tools/run_heatmap_offline_eval.py \
                                --scene-ids "$scene_ids" \
                                --queries "$heat_queries" \
                                --dataset-type "$DATASET_TYPE" \
                                --data-paths "$DATA_PATHS" \
                                --scene-dataset-config-file "$HSSD_CFG" \
                                --out "$OUT_DIR"
                        else
                            python tools/run_heatmap_offline_eval.py \
                                --scene-ids "$scene_ids" \
                                --dataset-type "$DATASET_TYPE" \
                                --data-paths "$DATA_PATHS" \
                                --scene-dataset-config-file "$HSSD_CFG" \
                                --out "$OUT_DIR"
                        fi
                    fi
                    echo ""
                    echo "  Results root:  $OUT_DIR"
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
}

while true; do
    echo ""
    echo "┌─────────────────────────────────────────────────────┐"
    echo "│                 Available commands                  │"
    echo "├─────────────────────────────────────────────────────┤"
    echo "│  1) VLMaps pipeline                  [default]      │"
    echo "│  2) Other tools (GPU, Jupyter, deps, shell, ...)    │"
    echo "│  q) Quit                                            │"
    echo "└─────────────────────────────────────────────────────┘"
    echo -n "  Select an option (default 1): "
    read -r opcion
    opcion=${opcion:-1}

    case "$opcion" in
        2)
            run_other_menu
            ;;
        q|Q)
            echo ""
            echo "  Bye!"
            echo ""
            break
            ;;
        1)
            # ── Dataset selection ─────────────────────────────────────────
            echo ""
            echo "  ┌─────────────────────────────────────────────────┐"
            echo "  │              Select dataset                     │"
            echo "  ├─────────────────────────────────────────────────┤"
            echo "  │  1) MP3D  (Matterport3D — escenas reales)       │"
            echo "  │  2) HSSD  (Habitat Static Scene Dataset)        │"
            echo "  └─────────────────────────────────────────────────┘"
            echo -n "  Dataset (1/2, default 2): "
            read -r ds_choice
            ds_choice=${ds_choice:-2}

            if [ "$ds_choice" = "1" ]; then
                DATASET_TYPE="mp3d"
                DATA_PATHS="docker"
                SCENES_DIR=/workspace/data/vlmaps_dataset
                NAV_EXTRA=""
                DS_LABEL="MP3D"
            else
                DATASET_TYPE="hssd"
                DATA_PATHS="hssd"
                SCENES_DIR=/workspace/data/vlmaps_dataset_hssd
                HSSD_CFG=/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json
                NAV_EXTRA="dataset_type=hssd scene_dataset_config_file=$HSSD_CFG"
                DS_LABEL="HSSD"
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
                echo "  │  l) Interactive LLM navigation       [default]  │"
                echo "  │  e) Interactive executor navigation             │"
                echo "  │  t) Testing / evaluation submenu                │"
                echo "  │  g) Generate obstacle map image                 │"
                if [ "$DATASET_TYPE" = "mp3d" ]; then
                echo "  │  n) Label rooms (LabelMe → room_map)            │"
                fi
                echo "  │  b) Back                                        │"
                echo "  └─────────────────────────────────────────────────┘"
                echo -n "  Select (default l): "
                read -r sub
                sub=${sub:-l}

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
                        echo ""
                        echo "  ── Step 4 · Interactive executor navigation ────────────────"
                        echo ""
                        echo "    python $APP/interactive_object_nav_executor.py data_paths=hssd scene_id=0 \\"
                        echo "        dataset_type=hssd \\"
                        echo "        scene_dataset_config_file=$HSSD_CFG"
                        echo "    # Optional policy mode:"
                        echo "    VLMAPS_POLICY_MODE=hybrid   # or heuristic / llm"
                        echo ""
                        echo "  ── Step 5 · Testing / evaluation workflows ─────────────────"
                        echo ""
                        echo "    # Preferred path: use menu option 't' and choose one of:"
                        echo "    #   1) Generate test set"
                        echo "    #   2) Compare full 2x2 pipeline"
                        echo "    #   3) Heatmap-only offline analysis"
                        echo ""
                        echo "    # Direct runners from inside the container:"
                        echo "    python /workspace/tools/run_full_2x2_eval.py --scene-ids 0 --out /workspace/results/eval_runs/demo"
                        echo "    python /workspace/tools/run_heatmap_offline_eval.py --scene-ids 0 --out /workspace/results/eval_runs/demo"
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
                        echo ""
                        echo "  ── Step 4 · Interactive executor navigation ────────────────"
                        echo ""
                        echo "    python $APP/interactive_object_nav_executor.py data_paths=docker scene_id=0"
                        echo "    # Optional policy mode:"
                        echo "    VLMAPS_POLICY_MODE=hybrid   # or heuristic / llm"
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
                        if ! prompt_valid_scene_id "scene_id to build" 0; then
                            continue
                        fi
                        scene="$SELECTED_SCENE_ID"
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
                        if ! prompt_valid_scene_id "scene_id to index" 0; then
                            continue
                        fi
                        scene="$SELECTED_SCENE_ID"
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
                        if ! prompt_valid_scene_id "scene_id to use" 0; then
                            continue
                        fi
                        scene="$SELECTED_SCENE_ID"
                        echo ""
                        echo "► Launching interactive LLM navigation (scene $scene)  [$DS_LABEL]..."
                        echo "  Type instructions at the prompt. Type 'quit' to stop."
                        echo ""
                        cd /workspace/third_party/vlmaps
                        python "$APP/interactive_object_nav.py" \
                            data_paths="$DATA_PATHS" scene_id="$scene" $NAV_EXTRA
                        ;;
                    e|E)
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
                        if ! prompt_valid_scene_id "scene_id to use" 0; then
                            continue
                        fi
                        scene="$SELECTED_SCENE_ID"
                        echo -n "  policy mode [heuristic|hybrid|llm] (default hybrid): "
                        read -r policy_mode
                        policy_mode=${policy_mode:-hybrid}
                        echo ""
                        echo "► Launching interactive executor navigation (scene $scene, policy=$policy_mode)  [$DS_LABEL]..."
                        echo "  Type instructions at the prompt. Type 'quit' to stop."
                        echo ""
                        cd /workspace/third_party/vlmaps
                        VLMAPS_POLICY_MODE="$policy_mode" python "$APP/interactive_object_nav_executor.py" \
                            data_paths="$DATA_PATHS" scene_id="$scene" $NAV_EXTRA
                        ;;
                    t|T)
                        run_testing_menu
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
                        if ! prompt_valid_scene_id "scene_id" 0; then
                            continue
                        fi
                        scene="$SELECTED_SCENE_ID"
                        echo ""
                        echo "► Generating obstacle map images for scene $scene  [$DS_LABEL]..."
                        cd /workspace/third_party/vlmaps
                        python "$APP/generate_obstacle_map_png.py" \
                            data_paths="$DATA_PATHS" scene_id="$scene"
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
        *)
            echo "  Invalid option. Choose 1, 2 or 'q'."
            ;;
    esac
done
