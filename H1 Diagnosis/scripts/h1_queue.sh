set -euo pipefail
cd /root/autodl-tmp
py=/root/autodl-tmp/h1_env/bin/python
archive=/root/autodl-tmp/h1_small.tar
for i in $(seq 1 720); do
    if test -s "$archive" && test -s h1_work/models/roberta-base/model.safetensors && sha256sum -c --status h1_small.sha256 && test -s h1_runner.tar.gz && "$py" -c 'import datasets, transformers, peft, torch; assert torch.cuda.is_available()' 2>/dev/null; then break; fi
    sleep 5
done
test -s "$archive" && test -s h1_work/models/roberta-base/model.safetensors && sha256sum -c --status h1_small.sha256
tar -xf "$archive" -C h1_work
tar -xzf h1_runner.tar.gz -C h1_work
cd h1_work
test "$(find outputs/gw_eva/shared_study -name head_warmup.pt | wc -l)" -eq 12
echo '5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81  models/roberta-base/model.safetensors' | sha256sum -c
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
run() {
    "$py" -u run_h1.py --task "$1" --seed "$2" --checkpoint-root outputs/gw_eva/shared_study
}
run rte 0
test -s G5G6/results/h1/rte_seed0.json
echo 'PILOT PASSED'
for task in rte mrpc sst2 sst2_lowres; do
    for seed in 0 1 2; do
        if test "$task:$seed" != rte:0; then run "$task" "$seed"; fi
    done
done
"$py" -c 'import matplotlib' || "$py" -m pip install matplotlib
"$py" summarize_h1.py G5G6/results/h1
echo 'H1 COMPLETED'
