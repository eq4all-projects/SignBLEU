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


import numpy as np
from typing import Dict, List, Tuple


def get_elements(grams: List[Tuple[str]]):
    return list(set([
        element
        for gram in grams
        for element in gram
    ]))


def without(gram, idx):
    return tuple([g for g_i, g in enumerate(gram) if g_i != idx])


#def get_reference(gram_counts):
#    output = dict()
#    for gram, count in gram_counts:
#        output[gram] = count
#        if len(gram) == 1:
#            continue
#        for i in range(len(gram)):
#            gram_ = without(gram, i)
#            if gram_ in gram_counts:
#                continue
#            output[gram_] = 
#    return output


def _count_grams(corpus):
    counts = dict()
    for gram_dict in corpus:
        for grams in gram_dict.values():
            for gram in grams:
                gram = gram.get_elements(strip=True)
                if gram not in counts:
                    counts[gram] = 0
                counts[gram] += 1
    return counts


def count_grams(corpus=None, hypotheses=None, references=None):
    if corpus is None:
        corpus = list()
    if hypotheses is not None:
        corpus = corpus + hypotheses
    if references is not None:
        for ref_set in references:
            corpus = corpus + [ref for ref in ref_set if ref is not None]
    return _count_grams(corpus)


def marginal_count(
        gram_counts: Dict[Tuple[str], int],
):
    r"""
    Apply to channel grams only? Since removal of i
    """
    #elements = get_elements(list(gram_counts.keys()))
    output = dict()
    for gram, count in gram_counts.items():
        if len(gram) == 1:
            output[gram] = 1
            continue
        without_counts = list()
        for i in range(len(gram)):
            try:
                without_counts.append(gram_counts[without(gram, i)])
            except Exception as e:
                #print(e.__repr__())
                #without_counts.append(count+1)
                output[gram] = 1
                break
        else:
            subset_count = min(without_counts)
            output[gram] = (subset_count - count) / subset_count
    return output


def marginal_weights(
        gram_counts: Dict[Tuple[str], int],
):
    scores = marginal_count(gram_counts)
    return {k: 2**v / 2 for k, v in scores.items()}


if __name__ == '__main__':
    data = {
        ('g1', 'g2'): 50,
        ('g1',): 100,
        ('g2',): 100,
        ('g3',): 8,
        ('g2', 'g3'): 7,
        ('g1', 'g2', 'g3'): 1,
    }
    results = marginal_count(data)
    print(results)
    print(marginal_weights(data))
    #print({k: 2**v / 2 for k, v in results.items()})
