# Copyright 2024 EQ4ALL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


r"""
User-facing API.

TODO
   * Improve input file parsing. There are currently several problems:
      * Need to improve how multiple references are handled (currently no real parsing).
      * Should we assume that json data can contain multiple instances? I think best to just support eaf with single or multiple instances and then json only for the edge case of NS21 and then add better support as needed.
"""


import json
import yaml
import click
from pathlib import Path

from signbleu.data import (
    EAFParser,
    JSONParser,
    RegisteredParsers,
)
from signbleu.metric import SignBLEU


DATA_FORMATS = ["elan", "ns21", "neutral", "pdc", "ncslgr", "block", "gram"]


@click.group()
def cli():
    pass


class ClickParseJson(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            if value is None:
                return None
            return json.loads(value)
        except:
            breakpoint()
            raise click.BadParameter(value)


def parse_data_path(file_path, data_format, ref, **parser_kwargs):
    extension_message = 'Unsupported input extension "{}"'

    if data_format in ["block", "gram"]:
        message = f'Input format "{data_format}" not yet implemented.'
        raise NotImplementedError(message)

    path = Path(file_path)
    ext = path.suffix
    assert ext in [".json", ".eaf"], extension_message.format(ext)
    assert data_format in DATA_FORMATS

    # already in correct input format
    if ext == ".json" and data_format == "neutral":
        with open(path, "r") as f:
            data = json.load(path)
        return data, "neutral"

    # NS21 (NIASL2021) data
    if ext == ".json" and data_format == "ns21":
        parser = RegisteredParsers()
        data = parser.parse_ns21(path)
        # call json parser
        if ref:
            data = [data]
        return data, "neutral"

    if ext == ".json":
        raise TypeError()

    # general ELAN data (requires specifying time and gloss tiers)
    if ext == ".eaf" and data_format == "elan":
        # load using specified tiers
        parser = EAFParser(**parser_kwargs)
        data = parser.parse_file(path)
        if ref:
            data = [data]
        return data, "neutral"

    # PDC (DGS) data
    if ext == ".eaf" and data_format == "pdc":
        parser = RegisteredParsers()
        data = parser.parse_pdc(path)
        if ref:
            data = [data]
        return data, "neutral"

    # NCSLGR data
    if ext == ".eaf" and data_format == "ncslgr":
        parser = RegisteredParsers()
        data = parser.parse_ncslgr(path)
        if ref:
            data = [data]
        return data, "neutral"

    raise TypeError()

BLOCK_OPTIONS = [
    'offset_threshold',
    'channel_keys',
    'start_key',
    'end_key',
    'gloss_key',
    'two_hand_map',
    'channel_combine_map',
    'mask_key',
]
GRAM_OPTIONS = [
    'time_n',
    'channel_n',
    'channels',
    'method',
    'swap_map',
    'hand_channels',
]
METRIC_OPTIONS = [
    'smoothing',
    'effective_order',
    'verbose',
]

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option('-c', '--config_path', type=click.Path(exists=True), help='Configuration file path.')
@click.option('-hp', '--hyp_path', type=click.Path(exists=True), help='File path containing a list of hypothesis sentences.')
@click.option('-rp', '--ref_path', type=click.Path(exists=True), help='File path containing a list of reference sentences.')
@click.option('-hf', '--hyp_format', type=click.Choice(DATA_FORMATS), help='Hypothesis file format.')
@click.option('-rf', '--ref_format', type=click.Choice(DATA_FORMATS), help='Reference file format.')
@click.option('-sk','--start_key', help='Blockification start key.')
@click.option('-ek','--end_key', help='Blockification end key.')
@click.option('-gk','--gloss_key', help='Blockification gloss key.')
@click.option('-ot','--offset_threshold', type=float, help='Blockification offset threshold.')
@click.option('-ck','--channel_keys', cls=ClickParseJson, type=str, default=None, help='Blockification channel keys.')
@click.option('-thm','--two_hand_map', cls=ClickParseJson, type=str, default=None, help='Blockification two hand map.')
@click.option('-ccm','--channel_combine_map', cls=ClickParseJson, type=str, default=None, help='Blockification channel combine map.')
@click.option('-mk','--mask_key', help='Blockification mask key.')
@click.option('-tn','--time_n', type=int, help='Gram time dimension.')
@click.option('-cn','--channel_n', type=int, help='Gram channel dimension.')
@click.option('-ch','--channels', help='Gram channels list.')
@click.option('-m','--method', help='Gram method.')
@click.option('-sm','--swap_map', cls=ClickParseJson, type=str, default=None, help='Gram swap map.')
@click.option('-hc','--hand_channels', cls=ClickParseJson, type=str, default=None, help='Gram hand channels list.')
@click.option('-s','--smoothing', help='Metric smoothing method.')
@click.option('-eo','--effective_order', is_flag=True, help='Metric effective order flag.')
@click.option('-v','--verbose', is_flag=True, help='Metric verbosity flag.')
@click.option('-tt', '--time_tiers', cls=ClickParseJson, type=str, default=None, help='ELAN tiers for instance segmentation.')
@click.option('-gt', '--gloss_tiers', cls=ClickParseJson, type=str, default=None, help='ELAN tiers with named glosses.')
@click.option('-bt', '--bool_tiers', cls=ClickParseJson, type=str, default=None, help='ELAN tiers with unnamed glosses.')
def signbleu(**options):
    """
    This function calculates the SignBLEU score for given predictions and references. \n
    e.g., signbleu -c configs/pdc.yml
    """

    config_path = options.get('config_path', None)
    if config_path:
        # Load the configuration file
        with open(config_path, "r") as f_config:
            config = yaml.safe_load(f_config)

        # Update options with config values, defaulting to options values if not provided in the config
        for key in ['hyp_path', 'hyp_format', 'ref_path', 'ref_format']:
            options['data_' + key] = config['data'][key.replace('_path', '')]['path'] \
            if key in ['hyp_path', 'ref_path'] \
            else config['data'][key.replace('_format', '')]['format']

        for section in ['blockification', 'gram', 'metric']:
            section_options = {k.replace(section + '_', ''): v for k, v in options.items() if k.startswith(section)}
            for key, value in config.get(section, {}).items():
                options[section + '_' + key] = section_options.get(key, value)

        hyp_path = options['data_hyp_path']
        ref_path = options['data_ref_path']
        hyp_format = options['data_hyp_format']
        ref_format = options['data_ref_format']
    else:
        required_keys = ['hyp_path', 'hyp_format', 'ref_path', 'ref_format']
        assert all(key in options for key in required_keys), \
            "Missing one or more required options: 'hyp_path', 'hyp_format', 'ref_path', 'ref_format'."

        hyp_path = options['hyp_path']
        ref_path = options['ref_path']
        hyp_format = options['hyp_format']
        ref_format = options['ref_format']

    time_tiers = options.get('time_tiers')
    gloss_tiers = options.get('gloss_tiers')
    bool_tiers = options.get('bool_tiers')

    print(f"\nHypothesis file: {hyp_path} ({hyp_format})")
    print(f"Reference file: {ref_path} ({ref_format})")

    hypotheses, hyp_format = parse_data_path(
        hyp_path,
        hyp_format,
        ref=False,
        time_tiers=time_tiers,
        gloss_tiers=gloss_tiers,
        bool_tiers=bool_tiers,
    )
    references, ref_format = parse_data_path(
        ref_path,
        ref_format,
        ref=True,
        time_tiers=time_tiers,
        gloss_tiers=gloss_tiers,
        bool_tiers=bool_tiers,
    )

    # rename and categorize parameters
    #blockification_params = {k.replace('blockification_', ''): v for k, v in options.items() if k.startswith('blockification')}
    #gram_params = {k.replace('gram_', ''): v for k, v in options.items() if k.startswith('gram')}
    #metric_params = {k.replace('metric_', ''): v for k, v in options.items() if k.startswith('metric')}
    block_params = {k: v for k, v in options.items() if v is not None and k in BLOCK_OPTIONS}
    gram_params = {k: v for k, v in options.items() if v is not None and k in GRAM_OPTIONS}
    metric_params = {k: v for k, v in options.items() if v is not None and k in METRIC_OPTIONS}

    # calculate SignBLEU
    SIGNBLEU = SignBLEU(**block_params, **gram_params, **metric_params)
    results = SIGNBLEU.calculate(
        hypotheses=hypotheses,
        references=references,
        hypothesis_format=hyp_format,
        reference_format=ref_format,
    )

    print(f"Signature: {results.signature}")
    print(f"Score: {results.score}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option('-i', '--input', required=True, type=click.Path(exists=True), help='Path to the ELAN files.')
@click.option('-o', '--output', required=True, type=click.Path(exists=False), help='Path to the output file.')
@click.option('-sl', '--sentence_layer', type=str, help='Sentence layer (tier, channel) name. If provided, ELAN annotations are segmented into sentences, with annotations within each segment transformed into blocked information.')
def blockify(input, output, sentence_layer):
    """
    This function transforms ELAN files into a blockified file. \n
    e.g., blockify -i sample_data/annotations -o sample_data/blockified.json -sl sentence
    """
    print(f"Input path: {input}")
    print(f"Output file name: {output}")
    print(f"Sentence layer name: {sentence_layer}")
    # Here, add your logic to transform the input file into blocks.

    raise NotImplementedError("Blockification preprocessing by itself is not implemented in this version.")

if __name__ == '__main__':
    cli()
