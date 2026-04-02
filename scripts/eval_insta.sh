#!/bin/bash

dataset_path="/media/yuhao/bluessd/insta_frn/"

datasets=(
    # /2026-01-31_run1/
    # /2026-01-31_run2/
    /2026-01-31_run3/
    /2026-03-07_run1/
    /2026-03-07_run3/
)

no_calib=false
print_only=false
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-calib)
            no_calib=true
            ;;
        --print)
            print_only=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

if [ "$print_only" = false ]; then
    for dataset in ${datasets[@]}; do
        dataset_name="$dataset_path""$dataset"/
        if [ "$no_calib" = true ]; then
            python main.py --dataset $dataset_name --no-viz --save-as insta/no_calib/$dataset --config config/eval_no_calib.yaml
        else
            echo "=== Evaluate with calibration ==="
            python main.py --dataset $dataset_name --no-viz --save-as insta/calib/$dataset --config config/eval_calib.yaml
        fi
    done
fi
