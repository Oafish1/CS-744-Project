# Efficient Training of Graph Classification Models
Housed in this repository are various tests and algorithms for the benchmarking of graph classification model speed, including our hybrid approach to weight synchronization, *DGLight*.

### How to Use
The main algorithm here is housed in `torch_classify.py`.  To run a benchmark, the following command can be used:
```bash
python3 torch_classify.py <.NEL DATA> --master-ip <MASTER-IP> \
  --master-port <MASTER-PORT (Default 6585)> \
  --num-nodes <NUM-NODES> \
  --rank <NODE-RANK>
```

There are several additional arguments that can be used for configuring the benchmark.

`--silent` silences updates every 10 epochs.

`--skip` enables weight synchronization skipping, and defaults to once every 10 updates (set in `torch_classify.py`).

`--gs`, `--rr`, `--hy` chooses the synchronization techniques gather-scatter, ring-reduce, and our hybrid approach.  If none are selected, torch DDP is used.
