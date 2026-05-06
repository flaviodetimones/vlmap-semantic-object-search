#!/bin/bash
export PYTHONPATH=/workspace/third_party/vlmaps:$PYTHONPATH
export OPENAI_KEY="${OPENAI_KEY:-$OPENAI_API_KEY}"
APP=/workspace/third_party/vlmaps/application
DATASET=/workspace/third_party/vlmaps/dataset

is_scene_excluded() {
    local scene_name="$1"
    case "$DATASET_TYPE:$scene_name" in
        hssd:108736884_177263634_4)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

iter_scene_names() {
    if [ ! -d "$SCENES_DIR" ]; then
        return
    fi

    while IFS= read -r dir; do
        local scene_name
        scene_name=$(basename "$dir")
        if ! is_scene_excluded "$scene_name"; then
            echo "$scene_name"
        fi
    done < <(find "$SCENES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
}

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
        while IFS= read -r scene_name; do
            echo "    scene_id=$i  →  $scene_name"
            i=$((i+1))
        done < <(iter_scene_names)
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
    iter_scene_names | sed -n "$((scene_id + 1))p"
}

scene_count() {
    if [ ! -d "$SCENES_DIR" ]; then
        echo 0
        return
    fi
    iter_scene_names | wc -l
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
        echo "  │  4) YOLOE conf-thresh sweep                     │"
        echo "  │  5) Place benchmark small objects               │"
        echo "  │  6) Build benchmark small-object queries        │"
        echo "  │  7) Run benchmark small-object eval             │"
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
                    echo "  Full pipeline comparison is currently HSSD-only."
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
                    echo -n "  YOLOE conf threshold [0.30|0.35|0.40] (default 0.30): "
                    read -r yoloe_conf
                    yoloe_conf=${yoloe_conf:-0.30}
                    STAMP=$(date +%Y%m%d_%H%M%S)
                    OUT_DIR="/workspace/results/eval_runs/${STAMP}"
                    echo ""
                    echo "► Running full 2x2 pipeline evaluation..."
                    echo "  Output root: $OUT_DIR"
                    cd /workspace
                    if [ -n "$eval_queries" ]; then
                        python tools/run_full_eval.py \
                            --scene-ids "$scene_ids" \
                            --queries "$eval_queries" \
                            --dataset-type "$DATASET_TYPE" \
                            --data-paths "$DATA_PATHS" \
                            --scene-dataset-config-file "$HSSD_CFG" \
                            --policy-mode "$policy_mode" \
                            --yoloe-conf-thresh "$yoloe_conf" \
                            --out "$OUT_DIR"
                    else
                        python tools/run_full_eval.py \
                            --scene-ids "$scene_ids" \
                            --dataset-type "$DATASET_TYPE" \
                            --data-paths "$DATA_PATHS" \
                            --scene-dataset-config-file "$HSSD_CFG" \
                            --policy-mode "$policy_mode" \
                            --yoloe-conf-thresh "$yoloe_conf" \
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
            4)
                echo ""
                if [ "$DATASET_TYPE" != "hssd" ]; then
                    echo "  YOLOE conf-thresh sweep is currently HSSD-only."
                    echo "  Switch to HSSD from the dataset menu."
                elif [ -z "$OPENAI_API_KEY" ]; then
                    echo "  WARNING: OPENAI_API_KEY is not set. Set it before running the sweep."
                else
                    echo "  Available scenes [$DS_LABEL]:"
                    echo "  ─────────────────────────────────────────────────"
                    print_scene_list
                    echo ""
                    echo -n "  Scene id (default 0): "
                    read -r sweep_scene_id
                    sweep_scene_id=${sweep_scene_id:-0}
                    echo -n "  Queries path (blank = default eval_queries/{scene_name}.jsonl): "
                    read -r sweep_queries
                    echo -n "  Thresholds CSV (default 0.30,0.40,0.50,0.60): "
                    read -r sweep_thresholds
                    sweep_thresholds=${sweep_thresholds:-0.30,0.40,0.50,0.60}
                    echo -n "  Method key [Ob_Hb|Ob_Hp|Oe_Hb|Oe_Hp] (default Oe_Hp): "
                    read -r sweep_method
                    sweep_method=${sweep_method:-Oe_Hp}
                    echo -n "  Executor policy mode [heuristic|hybrid|llm] (default hybrid): "
                    read -r sweep_policy_mode
                    sweep_policy_mode=${sweep_policy_mode:-hybrid}
                    STAMP=$(date +%Y%m%d_%H%M%S)
                    OUT_DIR="/workspace/results/yoloe_sweep/${STAMP}"
                    echo ""
                    echo "► Running YOLOE conf-thresh sweep..."
                    echo "  Output root: $OUT_DIR"
                    cd /workspace
                    if [ -n "$sweep_queries" ]; then
                        python tools/run_yoloe_thresh_sweep.py \
                            --scene-id "$sweep_scene_id" \
                            --queries "$sweep_queries" \
                            --thresholds "$sweep_thresholds" \
                            --method-key "$sweep_method" \
                            --dataset-type "$DATASET_TYPE" \
                            --data-paths "$DATA_PATHS" \
                            --scene-dataset-config-file "$HSSD_CFG" \
                            --policy-mode "$sweep_policy_mode" \
                            --out "$OUT_DIR"
                    else
                        scene_name=$(scene_name_from_id "$sweep_scene_id")
                        if [ -z "$scene_name" ]; then
                            echo "  Could not resolve scene_name for scene_id=$sweep_scene_id"
                        else
                            python tools/run_yoloe_thresh_sweep.py \
                                --scene-id "$sweep_scene_id" \
                                --queries "/workspace/tools/eval_queries/${scene_name}.jsonl" \
                                --thresholds "$sweep_thresholds" \
                                --method-key "$sweep_method" \
                                --dataset-type "$DATASET_TYPE" \
                                --data-paths "$DATA_PATHS" \
                                --scene-dataset-config-file "$HSSD_CFG" \
                                --policy-mode "$sweep_policy_mode" \
                                --out "$OUT_DIR"
                        fi
                    fi
                    echo ""
                    echo "  Results root:  $OUT_DIR"
                fi
                ;;
            5)
                echo ""
                if [ "$DATASET_TYPE" != "hssd" ]; then
                    echo "  Benchmark small-object placement is currently HSSD-only."
                    echo "  Switch to HSSD from the dataset menu."
                else
                    echo "  Available scenes [$DS_LABEL]:"
                    echo "  ─────────────────────────────────────────────────"
                    print_scene_list
                    echo ""
                    echo -n "  Scene ids comma-separated (default 0,1,2): "
                    read -r ov_place_scene_ids
                    ov_place_scene_ids=${ov_place_scene_ids:-0,1,2}
                    echo ""
                    echo "► Placing benchmark small objects into HSSD scenes..."
                    echo "  Query JSONL output: /workspace/tools/eval_queries/small_objects_placed/{scene_name}.jsonl"
                    cd /workspace
                    IFS=',' read -r -a _ov_place_ids <<< "$ov_place_scene_ids"
                    for _sid in "${_ov_place_ids[@]}"; do
                        _sid=$(echo "$_sid" | xargs)
                        [ -z "$_sid" ] && continue
                        _scene_name=$(scene_name_from_id "$_sid")
                        if [ -z "$_scene_name" ]; then
                            echo "  [skip] could not resolve scene_name for scene_id=$_sid"
                            continue
                        fi
                        _base_scene=$(echo "$_scene_name" | sed -E 's/_[0-9]+$//')
                        _scene_path="/workspace/data/versioned_data/hssd-hab/scenes/${_base_scene}.scene_instance.json"
                        _spec_path="/workspace/tools/eval_queries/specs/small_objects_${_base_scene}.json"
                        _semantic_path="/workspace/data/versioned_data/hssd-hab/semantics/scenes/${_base_scene}.semantic_config.json"
                        _out_jsonl="/workspace/tools/eval_queries/small_objects_placed/${_scene_name}.jsonl"
                        if [ ! -f "$_spec_path" ]; then
                            echo "  [skip] missing spec: $_spec_path"
                            continue
                        fi
                        python /workspace/tools/place_small_objects.py \
                            --scene "$_scene_path" \
                            --spec "$_spec_path" \
                            --out-jsonl "$_out_jsonl" \
                            --metadata-csv /workspace/data/versioned_data/hssd-hab/metadata/object_categories_filtered.csv \
                            --furniture-csv /workspace/data/versioned_data/hssd-hab/metadata/fpmodels-with-decomposed.csv \
                            --semantic-config "$_semantic_path"
                    done
                fi
                ;;
            6)
                echo ""
                if [ "$DATASET_TYPE" != "hssd" ]; then
                    echo "  Benchmark small-object query generation is currently HSSD-only."
                    echo "  Switch to HSSD from the dataset menu."
                else
                    echo "  Available scenes [$DS_LABEL]:"
                    echo "  ─────────────────────────────────────────────────"
                    print_scene_list
                    echo ""
                    echo -n "  Scene names CSV (blank = all current eval_queries scenes): "
                    read -r ov_scene_names
                    echo -n "  Targets CSV (blank = bottle,laptop,mug,kettle,toaster,coffee maker,trash bin,teapot): "
                    read -r ov_targets
                    ov_targets=${ov_targets:-bottle,laptop,mug,kettle,toaster,coffee maker,trash bin,teapot}
                    echo -n "  Include room_object variants? [y/N]: "
                    read -r ov_room_object
                    echo ""
                    echo "► Building benchmark small-object query battery..."
                    echo "  Output: /workspace/tools/eval_queries/small_objects/{scene_name}.jsonl"
                    cd /workspace
                    if [[ "$ov_room_object" =~ ^[Yy]$ ]]; then
                        if [ -n "$ov_scene_names" ]; then
                            python tools/build_open_vocab_eval_queries.py \
                                --source-dir /workspace/tools/eval_queries \
                                --out-dir /workspace/tools/eval_queries/small_objects \
                                --dataset-root /workspace/data/versioned_data/hssd-hab \
                                --scene-names "$ov_scene_names" \
                                --targets "$ov_targets" \
                                --include-room-object
                        else
                            python tools/build_open_vocab_eval_queries.py \
                                --source-dir /workspace/tools/eval_queries \
                                --out-dir /workspace/tools/eval_queries/small_objects \
                                --dataset-root /workspace/data/versioned_data/hssd-hab \
                                --targets "$ov_targets" \
                                --include-room-object
                        fi
                    else
                        if [ -n "$ov_scene_names" ]; then
                            python tools/build_open_vocab_eval_queries.py \
                                --source-dir /workspace/tools/eval_queries \
                                --out-dir /workspace/tools/eval_queries/small_objects \
                                --dataset-root /workspace/data/versioned_data/hssd-hab \
                                --scene-names "$ov_scene_names" \
                                --targets "$ov_targets"
                        else
                            python tools/build_open_vocab_eval_queries.py \
                                --source-dir /workspace/tools/eval_queries \
                                --out-dir /workspace/tools/eval_queries/small_objects \
                                --dataset-root /workspace/data/versioned_data/hssd-hab \
                                --targets "$ov_targets"
                        fi
                    fi
                fi
                ;;
            7)
                echo ""
                if [ "$DATASET_TYPE" != "hssd" ]; then
                    echo "  Benchmark small-object evaluation is currently HSSD-only."
                    echo "  Switch to HSSD from the dataset menu."
                else
                    if [ -z "$OPENAI_API_KEY" ]; then
                        echo "  NOTE: OPENAI_API_KEY is not set."
                        echo "  The run will still work, but open-vocab resolution may fall back more often."
                    fi
                    echo "  Available scenes [$DS_LABEL]:"
                    echo "  ─────────────────────────────────────────────────"
                    print_scene_list
                    echo ""
                    echo -n "  Scene ids comma-separated (default 0,1,2): "
                    read -r ov_scene_ids
                    ov_scene_ids=${ov_scene_ids:-0,1,2}
                    echo -n "  Queries dir or JSONL (blank = /workspace/tools/eval_queries/small_objects): "
                    read -r ov_queries
                    ov_queries=${ov_queries:-/workspace/tools/eval_queries/small_objects}
                    echo -n "  Policy mode [heuristic|hybrid|llm] (default hybrid): "
                    read -r ov_policy
                    ov_policy=${ov_policy:-hybrid}
                    echo -n "  YOLOE conf threshold (default 0.30): "
                    read -r ov_yoloe_conf
                    ov_yoloe_conf=${ov_yoloe_conf:-0.30}
                    echo -n "  Methods CSV (default Ob_Hp): "
                    read -r ov_methods
                    ov_methods=${ov_methods:-Ob_Hp}
                    STAMP=$(date +%Y%m%d_%H%M%S)
                    OUT_DIR="/workspace/results/small_objects_eval/${STAMP}"
                    echo ""
                    echo "► Running benchmark small-object evaluation..."
                    echo "  Output root: $OUT_DIR"
                    cd /workspace
                    python tools/run_open_vocab_eval.py \
                        --scene-ids "$ov_scene_ids" \
                        --queries "$ov_queries" \
                        --dataset-type "$DATASET_TYPE" \
                        --data-paths "$DATA_PATHS" \
                        --scene-dataset-config-file "$HSSD_CFG" \
                        --policy-mode "$ov_policy" \
                        --yoloe-conf-thresh "$ov_yoloe_conf" \
                        --methods "$ov_methods" \
                        --out "$OUT_DIR"
                    echo ""
                    echo "  Results root:  $OUT_DIR"
                    echo "  Summary:       $OUT_DIR/open_vocab_analysis/open_vocab_by_method.md"
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
                echo "  │  n) Label rooms (LabelMe → room_map)            │"
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
                        echo "    #   4) YOLOE conf-thresh sweep"
                        echo "    #   5) Place benchmark small objects"
                        echo "    #   6) Build benchmark small-object queries"
                        echo "    #   7) Run benchmark small-object eval"
                        echo ""
                        echo "    # Direct runners from inside the container:"
                        echo "    python /workspace/tools/run_full_eval.py --scene-ids 0 --yoloe-conf-thresh 0.30 --out /workspace/results/eval_runs/demo"
                        echo "    python /workspace/tools/run_yoloe_thresh_sweep.py --scene-id 0 --queries /workspace/tools/eval_queries/SCENE.jsonl --thresholds 0.30,0.35,0.40 --out /workspace/results/yoloe_sweep/demo"
                        echo "    python /workspace/tools/run_heatmap_offline_eval.py --scene-ids 0 --out /workspace/results/eval_runs/demo"
                        echo "    python /workspace/tools/place_small_objects.py --scene /workspace/data/versioned_data/hssd-hab/scenes/SCENE.scene_instance.json --spec /workspace/tools/eval_queries/specs/small_objects_SCENE.json --out-jsonl /workspace/tools/eval_queries/small_objects_placed/SCENE.jsonl"
                        echo "    python /workspace/tools/build_open_vocab_eval_queries.py --targets 'bottle,laptop,mug,kettle,toaster,coffee maker,trash bin,teapot' --include-room-object"
                        echo "    python /workspace/tools/run_open_vocab_eval.py --scene-ids 0,1,2 --queries /workspace/tools/eval_queries/small_objects --out /workspace/results/small_objects_eval/demo"
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
                            while IFS= read -r scene_name; do
                                echo "    scene_id=$i  →  $scene_name"
                                i=$((i+1))
                            done < <(iter_scene_names)
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
                                # Fixed spawn points per scene (X Z).
                                # Avoids bad random spawns (e.g. on top of furniture).
                                # Add new scenes here as needed.
                                declare -A _SPAWN_POINTS=(
                                    ["102344193"]="0.0 -5.0"
                                    ["102344280"]="-3.0 -8.0"
                                    ["108736884_177263634"]="2.0 -6.0"
                                    ["107734479_176000442"]="3.49 -10.77"
                                )
                                _spawn_args=""
                                if [[ -n "${_SPAWN_POINTS[$hssd_scene]+_}" ]]; then
                                    read -r _sx _sz <<< "${_SPAWN_POINTS[$hssd_scene]}"
                                    _spawn_args="--start-pos $_sx $_sz"
                                    echo "  Spawn : fixed ($_sx, $_sz)  [hallway/entryway preset]"
                                else
                                    echo "  Spawn : random (no preset for '$hssd_scene')"
                                fi
                                echo "► python dataset/collect_hssd_dataset.py --scene_id $hssd_scene $_spawn_args"
                                echo ""
                                cd /workspace/third_party/vlmaps
                                # shellcheck disable=SC2086
                                python "$DATASET/collect_hssd_dataset.py" \
                                    --scene_dataset_config "$HSSD_CFG" \
                                    --scene_id "$hssd_scene" \
                                    $_spawn_args
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
                            while IFS= read -r scene_name; do
                                echo "    scene_id=$i  →  $scene_name"
                                i=$((i+1))
                            done < <(iter_scene_names)
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
                            while IFS= read -r scene_name; do
                                echo "    scene_id=$i  →  $scene_name"
                                i=$((i+1))
                            done < <(iter_scene_names)
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
                            while IFS= read -r scene_name; do
                                echo "    scene_id=$i  →  $scene_name"
                                i=$((i+1))
                            done < <(iter_scene_names)
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
                            while IFS= read -r scene_name; do
                                echo "    scene_id=$i  →  $scene_name"
                                i=$((i+1))
                            done < <(iter_scene_names)
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
                            while IFS= read -r scene_name; do
                                dir="$SCENES_DIR/$scene_name"
                                HAS_MAP=""
                                [ -f "$dir/obstacle_map.png" ] && HAS_MAP=" [map ready]"
                                echo "    scene_id=$i  →  $scene_name$HAS_MAP"
                                i=$((i+1))
                            done < <(iter_scene_names)
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
                            data_paths="$DATA_PATHS" scene_id="$scene" $NAV_EXTRA
                        ;;
                    n|N)
                        echo ""
                        echo "  ╔══════════════════════════════════════════════════════╗"
                        echo "  ║        Room labeling workflow (LabelMe)             ║"
                        echo "  ╠══════════════════════════════════════════════════════╣"
                        echo "  ║  Esta opción hace el flujo completo:                ║"
                        echo "  ║  preparar -> abrir LabelMe -> convertir a room_map  ║"
                        echo "  ║  Tú solo dibujas y guardas.                         ║"
                        echo "  ╚══════════════════════════════════════════════════════╝"
                        echo ""
                        echo "  Escenas disponibles [$DS_LABEL]:"
                        echo "  ─────────────────────────────────────────────────"
                        if [ -d "$SCENES_DIR" ]; then
                            i=0
                            while IFS= read -r scene_name; do
                                dir="$SCENES_DIR/$scene_name"
                                HAS_MAP=""
                                ( [ -f "$dir/topdown_labeled.png" ] || [ -f "$dir/obstacle_map.png" ] ) && HAS_MAP=" [mapa listo]"
                                HAS_ROOMS=""
                                [ -f "$dir/room_map/room_map.npy" ] && HAS_ROOMS=" [room_map manual]"
                                echo "    scene_id=$i  →  $scene_name$HAS_MAP$HAS_ROOMS"
                                i=$((i+1))
                            done < <(iter_scene_names)
                        fi
                        echo ""
                        if ! prompt_valid_scene_id "scene_id" 0; then
                            continue
                        fi
                        scene="$SELECTED_SCENE_ID"
                        SCENE_NAME=$(scene_name_from_id "$scene")
                        if [ -z "$SCENE_NAME" ]; then
                            echo "  Escena no encontrada."
                            continue
                        fi
                        SCENE_DIR="$SCENES_DIR/$SCENE_NAME"
                        ANNO_DIR="/workspace/annotations/room_labels/$DATASET_TYPE/$SCENE_NAME"
                        HOST_ANNO_DIR="/home/mario/tfg/vlmap-semantic-object-search-tfg/annotations/room_labels/$DATASET_TYPE/$SCENE_NAME"
                        MAP_IMG="$ANNO_DIR/topdown_labeled.png"
                        LABEL_JSON="$ANNO_DIR/room_labels.json"
                        ALT_LABEL_JSON="$ANNO_DIR/topdown_labeled.json"
                        echo ""
                        echo "  Escena       : $SCENE_NAME"
                        echo "  Anotaciones  : $ANNO_DIR"
                        echo "  JSON esperado: $LABEL_JSON"
                        echo "  JSON alterno : $ALT_LABEL_JSON"
                        echo ""
                        echo "► Preparando assets para LabelMe..."
                        if [ "$DATASET_TYPE" = "hssd" ]; then
                            python /workspace/tools/prepare_room_labelme.py \
                                --dataset-type "$DATASET_TYPE" \
                                --data-paths "$DATA_PATHS" \
                                --scene-id "$scene" \
                                --scene-dataset-config-file "$HSSD_CFG" \
                                --regenerate
                        else
                            python /workspace/tools/prepare_room_labelme.py \
                                --dataset-type "$DATASET_TYPE" \
                                --data-paths "$DATA_PATHS" \
                                --scene-id "$scene" \
                                --regenerate
                        fi
                        echo ""
                        echo "► Asegurando compatibilidad de LabelMe con NumPy..."
                        python /workspace/tools/ensure_labelme_compat.py
                        echo ""
                        if [ ! -f "$MAP_IMG" ]; then
                            echo "  No se ha podido preparar la imagen de anotación."
                            continue
                        fi
                        echo "► Abriendo LabelMe dentro del contenedor..."
                        echo "  Dibuja los polígonos, guarda y cierra LabelMe."
                        echo "  Imagen : $MAP_IMG"
                        echo "  Salida : $LABEL_JSON"
                        QT_PLUGIN_PATH=/opt/conda/envs/tfg/lib/python3.9/site-packages/PyQt5/Qt5/plugins \
                        QT_QPA_PLATFORM_PLUGIN_PATH=/opt/conda/envs/tfg/lib/python3.9/site-packages/PyQt5/Qt5/plugins/platforms \
                        QT_QPA_PLATFORM=xcb \
                        LD_PRELOAD=/opt/conda/envs/tfg/lib/libstdc++.so.6 \
                            labelme "$MAP_IMG" \
                            --output "$LABEL_JSON" \
                            --autosave \
                            --nodata
                        if [ -f "$LABEL_JSON" ] || [ -f "$ALT_LABEL_JSON" ]; then
                            echo ""
                            echo "► Convirtiendo anotación manual a room_map..."
                            python /workspace/tools/convert_room_labelme.py \
                                --dataset-type "$DATASET_TYPE" \
                                --scene-id "$scene"
                        else
                            echo ""
                            echo "  No se encontró JSON guardado; se omite la conversión."
                            echo "  Si la ventana no se abrió bien dentro del contenedor,"
                            echo "  usa este comando en el host y luego vuelve a pulsar 'n':"
                            echo ""
                            echo "  labelme \"$HOST_ANNO_DIR/topdown_labeled.png\" -O \"$HOST_ANNO_DIR/room_labels.json\" --autosave --nodata"
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
