# CRF++
## train
```shell script
crf_learn -f 2 -c 2 -t data/template.txt data/train_data.txt model/crf-seg.model
```

## test
```shell script
crf_test -m model/crf-seg.model data/train_data.txt
```

