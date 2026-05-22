# Plotting

## Create JSON run files

Normally, a json log file should be created automatically after running `lauch.sh` (see the patch `patches/0002-log-metrics-to-json.patch`).

Otherwise, if you do not have the `.json` log file, use `logparse.py` and provide the input stdout log, a descriptive name for the run (e.g. *8B BF16*, this will be displayed in the plot legend), and the output JSON file name.

```bash
python logparse.py \
    ../example_logs/gipfel-fp8-throughput-1.5b-tp1pp1-bf16-200s-1n-2176055.log \
    "8B BF16" \
    --output 8B_bf16.json 
```

## Create a violin plot

Use `violinplot.py` to create a violin plot comparing multiple json run files. For example:

```bash
python violinplot.py -f runs/8b-bf16-1n.json runs/8b-bf16-8n.json
```

You can also provide a directory to compare all json files in it:

```bash
python violinplot.py -f runs/
```

# Create a scalability plot

To create a scalability plot, first make sure that different JSON files with the same model settings have the same name (e.g., 'fp8' or 'baseline'). Then, use `scalabilityplot.py` to create a scatter plot of throughput vs model size, grouped by model name. For example:

```bash
python scalabilityplot.py -f runs
```
