# SignBLEU

SignBLEU is a metric for evaluating the results of multichannel sign language translation.

<details open>
  <summary><b>Table of Contents</b></summary>

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Using SignBLEU](#using-signbleu)
4. [Documentation](#documentation)
5. [Credit](#credit)
6. [TODO](#todo)

</details>

## Installation


Install from source:

```shell
git clone https://github.com/eq4all-projects/SignBLEU.git
cd SignBLEU
pip install -e .
```

## Quick Start

### Command-line Interface
The CLI is a work in progress and will be released soon!

### From Python

The below example loads an instance of DGS data and uses it to calculate
SignBLEU against itself.

```python
from signbleu.data import EAFParser
from signbleu.metric import SignBLEU


# Data files as a list of paths.
# Ex: The DGS sample ELAN file in sample_data/annotations/:
data_paths = ["sample_data/annotations/pdc_sample.eaf"]

# Set tier names for sentence-based instance segmentation.
time_tiers = ["Deutsche_Übersetzung_A", "Deutsche_Übersetzung_B"]

# Set tier names for extracting glosses.
gloss_tiers = [
    "Lexem_Gebärde_r_A",
    "Lexem_Gebärde_l_A",
    "Lexem_Gebärde_r_B",
    "Lexem_Gebärde_l_B",
]

parser = EAFParser(
    time_tiers=time_tiers,
    gloss_tiers=gloss_tiers,
)

# Get data as a list of dicts. We denote this format as "neutral" data.
# (Each dict holds glosses from the sentence segments specified by time_tiers)
data = parser.parse_all(data_paths)

# references should be a list neutral data, allowing for multiple references
# for each hypothesis.
# Ex: references = [ [{...}, {...}], [{...}, None] ] holds references for two
# hypotheses with two references for the first hypothesis and one reference for
# the second hypothesis.
references = [data]

# set parameters for temporal and channel grams
gram_params = {
    "time_n": 3,  # order for temporal grams
    "channel_n": 2,  # order for channel grams
    "channels": gloss_tiers,  # the channels to calculate over
}

SIGNBLEU = SignBLEU(**gram_params)
score = SIGNBLEU.calculate(data, references)
print(f'Signature: {score.signature}')
print(f'SignBLEU:  {score.score}')
```

This prints:
```
Signature: off:na||t:3|c:2|dim:1||m:sbleu|sm:exp|eff:y||v:0.1.0
SignBLEU:  1.0
```

## Using SignBLEU

As mentioned in the quick start section, the CLI for SignBLEU is not ready
and SignBLEU should be used from Python directly.

SignBLEU expects data to be formatted as an array of JSON objects (we refer to
this format as *neutral* data).
Each neutral JSON object corresponds to one segment of signing data (usually at
the sentence level) and contains key-values each annotation tier.

For example, the following neutral data contains two instances, each instance
has two tiers containing glosses, and each gloss is represented as a JSON
object with three fields: "gloss", "start", and "end".

```json
[
  {
    "tier1": [
      {"gloss": "...", "start": 0.1, "end": 0.9},
      {"gloss": "...", "start": 1.3, "end": 2.0}
    ],
    "tier2": [
      {"gloss": "...", "start": 0.1, "end": 2.0}
    ]
  },
  {
    "tier1": [
      {"gloss": "...", "start": 0.2, "end": 1.4}
    ],
    "tier2": []
  }
]
```

Model generations and reference labels should both be converted to this neutral
format before calling SignBLEU.

SignBLEU does offer basic conversion tools for converting .eaf files to the
neutral representation.
As an example, please see the below example processing of a DGS .eaf file (an
almost identical example to that in the quick start).

```python
import json
from signbleu.data import EAFParser

# Data files as a list of paths.
# Ex: The DGS sample ELAN file in sample_data/annotations/:
data_paths = ["sample_data/annotations/pdc_sample.eaf"]

# Set tier names for sentence-based instance segmentation.
time_tiers = ["Deutsche_Übersetzung_A", "Deutsche_Übersetzung_B"]

# Set tier names for extracting glosses. Specify both text and gloss tiers so
# that both will be collected in the neutral representation.
gloss_tiers = [
    # German text tiers
    "Deutsche_Übersetzung_A",
    "Deutsche_Übersetzung_B",

    # DGS tiers
    "Lexem_Gebärde_r_A",
    "Lexem_Gebärde_l_A",
    "Lexem_Gebärde_r_B",
    "Lexem_Gebärde_l_B",
]

parser = EAFParser(
    time_tiers=time_tiers,
    gloss_tiers=gloss_tiers,
)

# Get data as a list of dicts (neutral data).
# (Each dict holds glosses from the sentence segments specified by time_tiers)
data = parser.parse_all(data_paths)

# Optionally, save the data to disk.
with open('neutral_data.json', 'w') as f:
    json.dump(data, f, indent=2)
```

SignBLEU will require a list of hypothesis translations and one or more lists
of reference translations. Hypothesis translations should be formatted as
above (as netural data), but references should have an extra level of nesting.
For example, if we have four hypotheses and up to three references for every
hypothesis, the reference dataset should be an array containing three
sub-arrays.
The first sub-array should contain a reference for *each* hypothesis. For the
second and third sub-arrays, if the i-th hypothesis does not have a second or
third reference, respectively, then the entry should just be `null`.

```json
[
  [{...}, {...}, {...}, {...}],
  [{...}, null, {...}, null],
  [{...}, null, null, null]
]
```

Finally, SignBLEU can be calculated.
There are a number of parameters used for SignBLEU.
We divide them into three categories.
- block_params: Parameters for converting data into the block format.
- gram_params: Parameters for calculating temporal and channel grams.
- signbleu_params: Parameters for scoring.

We cover the most important parameters below but a more comprehensive overview
is available in the documentation.
Improved parameter loading and paring functions will be provided when the WIP
CLI is released.
Note that this example uses the toy neutral example files at
`sample_data/hypotheses.json` and `sample_data/references.json`.

```python
import json
from signbleu.metric import SignBLEU

with open("sample_data/hypotheses.json", "r") as f:
    hyps = json.load(f)
with open("sample_data/references.json", "r") as f:
    refs = json.load(f)

block_params = {
    # The channel_keys parameter should be a tuple or list of tier names to use
    # when building block data. If None, all tiers will be used. Useful for
    # ignoring spoken language tiers.
    "channel_keys": ["face1", "face2", "left", "right", "both"],

    # Tiers for matching articulators (and never have overlapping annotations)
    # may be combined into the same channel. Specify this n:1 mapping with
    # this parameter.
    "channel_combine_map": {
        "face1": "face",
        "face2": "face",
        "face3": "face",
        "shoulder1": "shoulder",
        "shoulder2": "shoulder",
    },

    # Some data, like NS21, annotates two-handed signs into a separate
    # "two-hand" or "both" tier. These glosses should be copied into individual
    # hand tiers (if they exist). Specify this 1:2 mapping with this parameter.
    "two_hand_map": {
        "both": ["left", "right"],
    }
}

gram_params = {
    # gram order should be specified with "time_n" and "channel_n" for temporal
    # and channel grams, respectively.
    "time_n": 3,
    "channel_n": 2,
}

signbleu_params = {
    # Smoothing for sentence SignBLEU can be specified with the "smoothing"
    # parameter. Possible values are None or "none", "epsilon", "add_k",
    # "exponential", and "floor". See the signbleu.metric module
    # for more information.
    "smoothing": "exponential",
}

SIGNBLEU = SignBLEU(**block_params, **gram_params, **signbleu_params)
score = SIGNBLEU.calculate(hyps, refs)

# score.signature returns the calculation signature.
print(f'Signature:           {score.signature}')

# score.score returns only the corpus SignBLEU score as a float.
print(f'Corpus SignBLEU:     {score.score}')

# score.scores returns a dict containing additional scoring information.
print(f'All SignBLEU Scores: {score.scores}')
```

## Documentation

We plan on hosting documentation soon. In the meantime, documentation can be
built using sphinx. Make sure to install the development version of SignBLEU:
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

## Credit

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

## TODO

There is a lot of cleaning up todo.
- Fix/improve CLI.
  - Bugfix NS21, PDC, and NCSLGR examples.
- Improve documentation and quick start guide.
- Refactor temporal and channel gram functions.
- Make sure special characters are not hardcoded.
- General refactoring to reduce nesting.
- Run formatting with Black.
- Add support for 2D grams.
- Add alternative weighting schemes.
