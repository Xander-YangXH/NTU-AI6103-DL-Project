# G5/G6 server study

Completed formal jobs: 32/32. Incomplete entries are not predictions.

| Task | Group | Seeds | Accuracy mean | Sample std | MRPC F1 mean | F1 std |
|---|---|---|---|---|---|---|
| rte | G5 | [0, 1, 2, 3, 4] | 0.7357400722021661 | 0.02427109937035614 | None | None |
| rte | G6 | [0, 1, 2, 3, 4] | 0.7292418772563177 | 0.020895012459910163 | None | None |
| mrpc | G5 | [0, 1, 2, 3, 4] | 0.8759803921568627 | 0.010046030162705452 | 0.9113998455806862 | 0.00690923158497408 |
| mrpc | G6 | [0, 1, 2, 3, 4] | 0.8808823529411764 | 0.006845215707729895 | 0.9140883426294157 | 0.005171056509397226 |
| sst2 | G5 | [0, 1, 2] | 0.9403669724770642 | 0.003034118475991537 | None | None |
| sst2 | G6 | [0, 1, 2] | 0.9430428134556575 | 0.0023872316507639038 | None | None |
| sst2_lowres | G5 | [0, 1, 2] | 0.8845565749235473 | 0.0028860223376416957 | None | None |
| sst2_lowres | G6 | [0, 1, 2] | 0.8998470948012233 | 0.001751749118866919 | None | None |

RTE10 / MRPC10 / SST2 3 / SST2-1k 20 epochs; max length128; final official dev only. G5 uses 576 total ranks; G6 uses a parameter budget. Actual counts are in per-seed initialization verification.json.

[
  {
    "task": "rte",
    "seeds": [
      0,
      1,
      2,
      3,
      4
    ],
    "G6_minus_G5_accuracy": -0.006498194945848401,
    "paired_t_95_CI": [
      -0.06188978582075999,
      0.048893395929063194
    ],
    "complete": true,
    "interpretation": "exploratory; G5 and G6 do not necessarily have identical parameter counts"
  },
  {
    "task": "mrpc",
    "seeds": [
      0,
      1,
      2,
      3,
      4
    ],
    "G6_minus_G5_accuracy": 0.004901960784313731,
    "paired_t_95_CI": [
      -0.008363450710791377,
      0.018167372279418836
    ],
    "complete": true,
    "interpretation": "exploratory; G5 and G6 do not necessarily have identical parameter counts"
  },
  {
    "task": "sst2",
    "seeds": [
      0,
      1,
      2
    ],
    "G6_minus_G5_accuracy": 0.0026758409785933024,
    "paired_t_95_CI": [
      -0.006481711213316272,
      0.011833393170502877
    ],
    "complete": true,
    "interpretation": "exploratory; G5 and G6 do not necessarily have identical parameter counts"
  },
  {
    "task": "sst2_lowres",
    "seeds": [
      0,
      1,
      2
    ],
    "G6_minus_G5_accuracy": 0.01529051987767586,
    "paired_t_95_CI": [
      0.009360307707402786,
      0.021220732047948936
    ],
    "complete": true,
    "interpretation": "exploratory; G5 and G6 do not necessarily have identical parameter counts"
  }
]