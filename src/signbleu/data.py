r"""
Module for preprocessing raw data.

All loading and processing classes are provided for convenience and aim to
provide support for converting data from common sign language corpus formats to
the SignBLEU input format (denoted in this library as the "neutral" data
format).
An example of the neutral data format is:

>>> {
>>>   "tier_1": [{"gloss": "g_1", "start": 0.0, "end": 0.5}],
>>>   "tier_2": []
>>> }

which contains two glossing tiers ("tier_1" and "tier_2") and exactly one
annotation expressed as a dict containing a str-valued "gloss" field, a
float-valued "start" field, and a float-valued "end" field.

It should be relatively straightforward for users to pre-process their data
without this module.


**ELAN** parsers convert ELAN .eaf files to a SignBLEU-ready JSON format.

The recommended parser for .eaf files is :class:`EAFParser`\.
:class:`EAFParser` is a general class for parsing .eaf files and requires a
list of tiers containing ELAN's ALIGNABLE_ANNOTATIONs used for segmenting
individual instances (usually at the near-sentence level) and a list of tiers
containing target glosses and signals.

:class:`RegisteredParsers` is a convenience wrapper for parsing NCSLGR and PDC
(DGS) .eaf files, based on the tiers used in experiments from the original
SignBLEU paper.


**JSON** parsers convert JSON files to a SignBLEU-ready format.

Currently the only supported JSON loader is :class:`JSONParser`
which assumes that annotations are objects that contain gloss names, start
times, and end times; that annotations are contained in one or more arrays; and
that each file contains one instance.
Other convenience classes may be added in the future.

"""


import logging


logger = logging.getLogger(__name__)


import os
import json
from typing import Union
from pathlib import Path
import xml.etree.ElementTree as ET
from collections.abc import Sequence


class EAFParser:
    r"""A general ELAN .eaf file parser."""
    def __init__(
        self,
        time_tiers=None,
        gloss_tiers=None,
        bool_tiers=None,
    ):
        r"""
        Construct :class:`EAFParser`\.

        Args:
            time_tiers (Sequence[str], optional): All tiers containing time
                annotations for instance segmentation. These tiers usually
                belong to spoken language tiers that have been aligned to
                signing and can be used to segment a document into many
                near-sentence-level instances.
                The document will not be split if `time_tiers` is None.
                Defaults to None.
            gloss_tiers (Sequence[str], optional): All gloss and signal tiers
                containing annotations with gloss names and should be included
                in the output.
                Defaults to None.
            bool_tiers (Sequence[str]): Gloss tiers that should use the tier
                name as their gloss and should be included in the output.
                Usually used for tiers that annotate only a single signal type
                and thus need no glossings.

        Note:
            Example `time_tiers` would be "Deutsche_Übersetzung_A" and
            "Deutsche_Übersetzung_B" for PDC (DGS) data.
        """
        self.time_tiers = time_tiers
        self.gloss_tiers = gloss_tiers
        self.bool_tiers = bool_tiers

    def _collect_segment_refs(self, root):
        if self.time_tiers is None or len(self.time_tiers) == 0:
            return None
        ref = dict()
        for tier in root:
            if 'TIER_ID' in tier.attrib and tier.attrib['TIER_ID'] in self.time_tiers:
               for annotation in tier:
                   ref[annotation[0].attrib['ANNOTATION_ID'].split('_')[0]] = {
                       'TIME_SLOT_REF1': int(
                           annotation[0].attrib['TIME_SLOT_REF1'].replace('ts', '')
                       ),
                       'TIME_SLOT_REF2': int(
                           annotation[0].attrib['TIME_SLOT_REF2'].replace('ts', '')
                       ),
                   }

        return ref

    def _collect_time_refs(self, root):
        all_times = dict()
        for tier in root:
            if 'TIER_ID' in tier.attrib:
                for annotation in tier:
                    try:
                        all_times[annotation[0].attrib['ANNOTATION_ID'].split('_')[0]] = {
                            'TIME_SLOT_REF1': int(
                                annotation[0].attrib['TIME_SLOT_REF1'].replace('ts', '')
                            ),
                            'TIME_SLOT_REF2': int(
                                annotation[0].attrib['TIME_SLOT_REF2'].replace('ts', '')
                            ),
                        }
                    except KeyError as e:
                        break
                    except Exception as e:
                        message = (
                            'Unnexpected exception parsing .eaf file:'
                            f'\n{repr(e)}'
                        )
                        logger.warn(message)
                        continue
        return all_times

    def _collect_named_annotations(self, root, time_ref):
        if self.gloss_tiers is None or len(self.gloss_tiers) == 0:
            return dict()
        output = {tier: list() for tier in self.gloss_tiers}
        for tier in root:
            if 'TIER_ID' in tier.attrib and tier.attrib['TIER_ID'] in self.gloss_tiers:
                for annotation in tier:
                    aid = annotation[0].attrib['ANNOTATION_ID'].split('_')[0]
                    if aid not in time_ref:
                        message = (
                            f'Skipping annotation {aid} from document {doc_id}: '
                            'missing time alignment.'
                        )
                        logger.info(message)
                        continue
                    output[tier.attrib['TIER_ID']].append({
                        'gloss': annotation[0][0].text.replace('_', '-'),
                        'start': time_ref[aid]['TIME_SLOT_REF1'],
                        'end': time_ref[aid]['TIME_SLOT_REF2'],
                    })
        return output

    def _collect_unnamed_annotations(self, root, time_ref):
        if self.bool_tiers is None or len(self.bool_tiers) == 0:
            return dict()
        output = {tier: list() for tier in self.bool_tiers}
        for tier in root:
            if 'TIER_ID' in tier.attrib and tier.attrib['TIER_ID'] in self.bool_tiers:
                for annotation in tier:
                    aid = annotation[0].attrib['ANNOTATION_ID'].split('_')[0]
                    if aid not in time_ref:
                        message = (
                            f'Skipping annotation {aid} from document {doc_id}: '
                            'missing time alignment.'
                        )
                        logger.info(message)
                        continue
                    output[tier.attrib['TIER_ID']].append({
                        'gloss': tier.attrib['TIER_ID'],
                        'start': time_ref[aid]['TIME_SLOT_REF1'],
                        'end': time_ref[aid]['TIME_SLOT_REF2'],
                    })
        return output

    def _segment_data(self, segment_ref, annotations):
        if segment_ref is None:
            data = {
                k: sorted(annots, key=lambda x: x['start'])
                for k, annots in annotations.items()
            }
            return [data]
        output = list()
        for times in segment_ref.values():
            item = dict()
            for tier in annotations:
                item[tier] = [
                    annot
                    for annot in annotations[tier]
                    if annot['start'] >= times['TIME_SLOT_REF1'] \
                        and annot['end'] <= times['TIME_SLOT_REF2']
                ]
                item[tier] = sorted(item[tier], key=lambda x: x['start'])
            output.append(item)
        return output

    def parse_file(self, path: Union[str, os.PathLike]):
        r"""
        Get all sentence-level instances from a single .eaf file.

        Args:
            path (str or os.PathLike): Target .eaf file path.
        """
        doc_id = Path(path).name
        tree = ET.parse(path)
        root = tree.getroot()

        segment_ref = self._collect_segment_refs(root)
        time_ref = self._collect_time_refs(root)
        annotations = dict()
        annotations.update(self._collect_named_annotations(root, time_ref))
        annotations.update(self._collect_unnamed_annotations(root, time_ref))
        output = self._segment_data(segment_ref, annotations)

        return output

    def parse_all(self, paths: Sequence[Union[str, os.PathLike]]):
        r"""
        Get all sentence-level instances from .eaf file(s).

        Args:
            paths (Sequence[str or os.PathLike]): Target .eaf file paths.
        """
        return [
            instance
            for path in paths
            for instance in self.parse_file(path)
        ]


class RegisteredParsers:
    r"""
    Parser wrapper for recreating original SignBLEU experiments.

    This class provides simple call points for parsing PDC (DGS), NCSLGR,
    and NS21 data using tiers that match those used in the original SignBLEU
    experiments.

    This class is included for improved reproducibility, but we recommend using
    :class:`EAFParser` or :class`JSONParser` directly unless you are want to
    recreate the SignBLEU experiments.

    Note that PDC and NCSLGR use the following tiers:

    * PDC (DGS):
       * Time tiers (for instance segmentation)
          * "Deutsche_Übersetzung_A"
          * "Deutsche_Übersetzung_B"
       * Gloss tiers
          * "Lexem_Gebärde_r_A"
          * "Lexem_Gebärde_l_A"
          * "Lexem_Gebärde_r_B"
          * "Lexem_Gebärde_l_B"
    * NCSLGR:
       * Time tiers (for instance segmentation)
          * English translation
       * Gloss tiers
          * "main gloss"
          * non-dominant hand gloss"
          * head mvmt: nod"
          * head mvmt: shake"
          * eye brows"
          * eye aperture"

    """

    _pdc_time_tiers = ('Deutsche_Übersetzung_A', 'Deutsche_Übersetzung_B')
    _pdc_gloss_tiers = (
        'Lexem_Gebärde_r_A',
        'Lexem_Gebärde_l_A',
        'Lexem_Gebärde_r_B',
        'Lexem_Gebärde_l_B',
    )
    _ncslgr_time_tiers = ('English translation',)
    _ncslgr_gloss_tiers = (
        'main gloss',
        'non-dominant hand gloss',
        'head mvmt: nod',
        'head mvmt: shake',
        'eye brows',
        'eye aperture',
    )

    _ns21_gloss_key = 'gloss_id'
    _ns21_start_key = 'start'
    _ns21_end_key = 'end'
    _ns21_json_paths = (
        # all manual annotations contain glosses
        ('both', None, ('sign_script', 'sign_gestures_both')),
        ('right', None, ('sign_script', 'sign_gestures_strong')),
        ('left', None, ('sign_script', 'sign_gestures_weak')),

        # except for mouthings (not used here), no NMS annotations contain
        # a separate gloss key.
        ('mouth', 'cheek_inflate', ('nms_script', 'Ci')),
        ('mouth', 'mouth_open', ('nms_script', 'Mo1')),
        ('mouth', 'tongue_chew', ('nms_script', 'Tbt')),
        ('mouth', 'smile', ('nms_script', 'Mctr')),
        ('head', 'nod', ('nms_script', 'Hno')),
        ('head', 'shake', ('nms_script', 'Hs')),
        ('eye', 'brow_furrow', ('nms_script', 'EBf')),
    )

    def __init__(self):
        pass

    def parse_pdc(self, *paths):
        r"""
        Get all sentence-level instances from PDC (DGS) .eaf files.

        Calls :meth:`EAFParser.parse_all` on the given paths.

        Args:
            *paths (str): Target .eaf files.
        """
        parser = EAFParser(
            time_tiers=self._pdc_time_tiers,
            gloss_tiers=self._pdc_gloss_tiers,
        )
        return parser.parse_all(paths)

    def parse_ncslgr(self, *paths):
        r"""
        Get all sentence-level instances from NCSLGR .eaf files.

        Calls :meth:`EAFParser.parse_all` on the given paths.

        Args:
            *paths (str): Target .eaf files.
        """
        parser = EAFParser(
            time_tiers=self._ncslgr_time_tiers,
            gloss_tiers=self._ncslgr_gloss_tiers,
        )
        return parser.parse_all(paths)

    def parse_ns21(self, *paths):
        r"""
        Get all sentence-level NIASL2021 JSON files.

        Calls :meth:`JSONParser.parse_all` on the given paths.

        Args:
            *paths (str): Target NIASL2021 .json files.
            **load_kwargs: All kwargs are passed to open() to load the JSON
        """
        parser = JSONParser(
            self._ns21_gloss_key,
            self._ns21_start_key,
            self._ns21_end_key,
            *self._ns21_json_paths,
        )
        return parser.parse_all(paths)


class JSONParser:
    r"""A simple JSON data loading class"""
    def __init__(
        self,
        gloss_key='gloss',
        start_key='start',
        end_key='end',
        *json_paths,
    ):
        r"""
        Initialize :class:`JSONParser`\.

        Args:
            gloss_key (str, optional): The key for the gloss field.
            start_key (str, optional): The key for the start time.
            end_key (str, optional): The key for the end time.
            *json_paths (Tuple[str, Tuple, str]): Tuples of
                (<new-tier-name>, <new-gloss-name>,
                <json-path-to-annotation-array>).
                where all annotations will be put in <new-tier-name>,
                <new-gloss-name> is used as the gloss if not None, and
                <path-to-annotation-array> is a tuple of object keys and
                integer array positions.
                Note that if annotations contain a gloss name,
                <new-gloss-name> should always be None.

        Ex:
            >>> from signbleu.data import JSONParser
            >>> parser = JSONParser(
            >>>     # if annotations are structued as:
            >>>     # {"gloss": <gloss>, "start": <start-time>, "end": <end-time>}
            >>>     gloss_key="gloss",
            >>>     start_key="start",
            >>>     end_key="end",
            >>>     *(
            >>>         # if data is structured as:
            >>>         # {"annotations": {"left_hand": [...], "right_hand": [...]}}
            >>>         ("left", None, ("annotations", "left_hand")),
            >>>         ("right", None, ("annotations", "right_hand")),
            >>>     )
            >>> )
        """
        self._gloss_key = gloss_key
        self._start_key = start_key
        self._end_key = end_key
        self._paths = json_paths

    def parse_file(self, path: Union[str, os.PathLike], **load_kwargs):
        r"""
        Load and process a single .json file.

        Args:
            path (str or os.PathLike): Target .json file path.
            **load_kwargs: All kwargs are passed to open() to load the JSON
                data.
        """
        output = dict()

        kwargs = {'mode': 'rb'}
        kwargs.update(load_kwargs)
        with open(path, **kwargs) as f:
            data = json.load(f)
        for tier_name, gloss_name, path in self._paths:
            annots = data
            for part in path:
                annots = annots[part]
            for annotation in annots:
                if gloss_name is None:
                    gloss = annotation[self._gloss_key]
                else:
                    gloss = tier_name
                if tier_name not in output:
                    output[tier_name] = list()
                output[tier_name].append({
                    "gloss": gloss,
                    "start": annotation[self._start_key],
                    "end": annotation[self._end_key],
                })
        return output

    def parse_all(self, paths: Sequence[Union[str, os.PathLike]], **load_kwargs):
        r"""
        Load and process .json file(s).

        Args:
            paths (Sequence[str or os.PathLike]): Target .json file paths.
            **load_kwargs: All kwargs are passed to open() to load the JSON
        """
        return [
            self.parse_file(path, **load_kwargs)
            for path in paths
        ]


if __name__ == '__main__':
    # example
    rp = RegisteredParsers()
    data = rp.parse_pdc(Path(__file__).parent / '../../sample_data/annotations/pdc_sample.eaf')
    print(f'Length of PDC data: {len(data)}')
    data = rp.parse_ncslgr(Path(__file__).parent / '../../sample_data/annotations/ncslgr_sample.eaf')
    print(f'Length of NCSLGR data: {len(data)}')
