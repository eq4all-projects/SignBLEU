# SignBLEU

SignBLEU is a metric for evaluating the results of multichannel sign language translation.

<details open>
  <summary><b>Table of Contents</b></summary>

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Scoring](#scoring)
4. [Documentation](#documentation)
5. [Credit](#credit)
6. [TODO](#todo)

</details>

# Installation


Install from source:

```shell
git clone https://github.com/eq4all-projects/SignBLEU.git
cd SignBLEU
pip install -e .
```

# Quick Start

## Run SignBLEU with sample data

### Command-line Interface
```shell
$ signbleu -c configs/ns21.yml

Hypothesis file: sample_data/preds.json
References file: sample_data/references.json
Signature: m:SignBLEU|c:2|c:0.8508838698648301|e:True|t:3|s:3|s:[0.7430507186393743, 0.8824969025845955]
Score: 0.8508838698648301
```

### From Source Code
```python
from signbleu.metric import SignBLEU
import json

with open("sample_data/hypotheses.json", "r") as f_preds:
    hyp = json.load(f_preds)
with open("sample_data/references.json", "r") as f_refs:
    ref = json.load(f_refs)

channel_combine_map = {
    "face1": "face",
    "face2": "face",
    "face3": "face",
    "shoulder1": "shoulder",
    "shoulder2": "shoulder",
}
two_hand_map = {
    "both": ["left", "right"],
}

block_params = {
    "channel_combine_map": channel_combine_map,
    "two_hand_map": two_hand_map
}

gram_params = {
    "time_n": 3,
    "channel_n": 2,
    "channels": ["face", "left", "right"]
}

signbleu_params = {}

SIGNBLEU = SignBLEU(**block_params, **gram_params, **signbleu_params)
score = SIGNBLEU.calculate(hyp, ref)

print(score)
```

<!-- # Data Preprocessing

## Blockification
Since the existing data is mostly based on ELAN format, we provide preprocessing method called "Blockification".
### Command-line Interface

<details>
  <summary><code>$ blockify --help  # for blockifying data from the command line</code></summary>

```shell
Usage: blockify [OPTIONS]

  This function transforms ELAN files into a blockified file.

  e.g., blockify -i sample_data/annotations -o sample_data/blockified.json
  -sl sentence

Options:
  -i, --input PATH            Path to the ELAN files.  [required]
  -o, --output PATH           Path to the output file.  [required]
  -sl, --sentence_layer TEXT  Sentence layer (tier, channel) name. If
                              provided, ELAN annotations are segmented into
                              sentences, with annotations within each segment
                              transformed into blocked information.

  -h, --help                  Show this message and exit.
```
</details>

<details>
  <summary><code>$ blockify -i sample_data/annotations -o sample_data/blockified.json</code></summary>

```shell
```
</details>

### Source Code
```python
```

## Linearization
It also provides a way to convert a Block to a linearized representation, representing it like a normal text sequence. 
### Command-line Interface
```shell
```

### Source Code
```python
```

## Re-blockification
When a translation model produced translation outputs as a linearized format, you need to convert them into blocks to score them.
### Command-line Interface
```shell
```

### Source Code
```python
``` -->

# Scoring

## Command-line Interface Usage

<details>
  <summary><b>How to use SignBLEU?</b><br>
  <code>$ signbleu --help  # for running signbleu from the command line</code></summary>

```shell
Usage: signbleu [OPTIONS]

  This function calculates the SignBLEU score for given predictions and
  references.

  e.g., signbleu -c configs/ns21.yml

Options:
  -c, --config_path PATH          Configuration file path.
  -hp, --hyp_path PATH            File path containing a list of hypothesis
                                  sentences.

  -rp, --ref_path PATH            File path containing a list of reference
                                  sentences.

  -hf, --hyp_format [elan|ns21|block|gram]
                                  Hypothesis file format.
  -rf, --ref_format [elan|ns21|block|gram]
                                  Reference file format.
  -sk, --start_key TEXT           Blockification start key.
  -ek, --end_key TEXT             Blockification end key.
  -gk, --gloss_key TEXT           Blockification gloss key.
  -ot, --offset_threshold FLOAT   Blockification offset threshold.
  -ck, --channel_keys TEXT        Blockification channel keys.
  -thm, --two_hand_map TEXT       Blockification two hand map.
  -ccm, --channel_combine_map TEXT
                                  Blockification channel combine map.
  -mk, --mask_key TEXT            Blockification mask key.
  -tn, --time_n INTEGER           Gram time dimension.
  -cn, --channel_n INTEGER        Gram channel dimension.
  -ch, --channels TEXT            Gram channels list.
  -m, --method TEXT               Gram method.
  -sm, --swap_map TEXT            Gram swap map.
  -hc, --hand_channels TEXT       Gram hand channels list.
  -s, --smoothing TEXT            Metric smoothing method.
  -eo, --effective_order          Metric effective order flag.
  -v, --verbose                   Metric verbosity flag.
  -h, --help                      Show this message and exit.
```
</details>

<details>
  <summary><b>Using configuration file</b><br>
  <code>$ signbleu -c configs/ns21.yml</code></summary>

```shell
Predictions file: sample_data/preds.json
References file: sample_data/references.json
Signature: m:SignBLEU|c:2|c:0.8508838698648301|e:True|t:3|s:3|s:[0.7430507186393743, 0.8824969025845955]
Score: 0.8508838698648301
```
</details>


<details>
  <summary><b>Using parameters by passing them directly</b><br>
  <code>$ signbleu -hp sample_data\hypotheses.json -hf ns21 -rp sample_data\references.json -rf ns21</code></summary>

```shell
Predictions file: sample_data/preds.json
References file: sample_data/references.json
Signature: m:SignBLEU|c:2|c:0.8508838698648301|e:True|t:3|s:3|s:[0.7430507186393743, 0.8824969025845955]
Score: 0.8508838698648301
```
</details>

## From Source Code
<details>
  <summary><b>Using SignBLEU class directly</b></summary>

```python
from signbleu.metric import SignBLEU
import json

with open("sample_data/hypotheses.json", "r") as f_preds:
    hyp = json.load(f_preds)
with open("sample_data/references.json", "r") as f_refs:
    ref = json.load(f_refs)

channel_combine_map = {
    "face1": "face",
    "face2": "face",
    "face3": "face",
    "shoulder1": "shoulder",
    "shoulder2": "shoulder",
}
two_hand_map = {
    "both": ["left", "right"],
}

block_params = {
    "channel_combine_map": channel_combine_map,
    "two_hand_map": two_hand_map
}

gram_params = {
    "time_n": 3,
    "channel_n": 2,
    "channels": ["face", "left", "right"]
}

signbleu_params = {}

SIGNBLEU = SignBLEU(**block_params, **gram_params, **signbleu_params)
score = SIGNBLEU.calculate(hyp, ref)

print(score)
```
</details>

# Documentation

The documentation can be built using sphinx. Make sure to install the
development version of SignBLEU:
```shell
git clone https://github.com/eq4all-projects/SignBLEU.git
cd SignBLEU
pip install -e .[dev]
```

Then simply build the documentation from the doc directory:
```shell
cd docs
make html
```

The docs can be viewed locally or through a server:
```shell
docker compose up  # default port is 8123
```

# Credit

If you use SignBLEU, please cite the following:

```bibtex
@inproceedings{kim2024signbleu,
  title = "{SignBLEU}: Automatic Evaluation of Multi-channel Sign Language Translation",
  author = "Kim, Jung-Ho and Huerta-Enochian, Mathew and Ko, Changyong and Lee, Du Hui",
  booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
  month = "may",
  year = "2024",
  address = "Turin, Italy",
}
```

# TODO

There is a lot of cleaning up todo.
- Improve documentation and quick start guide.
  - Bugfix NS21, PDC, and NCSLGR examples.
  - Fix/improve API.
- Refactor temporal and channel gram functions.
- Make sure special characters are not hardcoded.
- General refactoring to reduce nesting.
- Run formatting with Black.
- Add support for 2D grams.
- Add alternative weighting schemes.
