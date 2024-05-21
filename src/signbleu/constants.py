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
Constants used for gram processing.

Note that all constants can be set using environment variables.

+-------------+--------------+----------------------+----------------------------------+
| Constant    | Default      | Environment Var.     | Definition                       |
+=============+==============+======================+==================================+
| MASK_KEY    | "###"        | "SIGNBLEU_MASK_KEY"  | If colons (":") are used in gloss|
|             |              |                      | names, this string may be pre- or|
|             |              |                      | post-pended to glosses. If the   |
|             |              |                      | default "###" string is used in  |
|             |              |                      | glosses, another unused string   |
|             |              |                      | should be specified.             |
+-------------+--------------+----------------------+----------------------------------+
| SEP_KEY     | "_"          | "SIGNBLEU_SEP_KEY"   | Used internally as a separator   |
|             |              |                      | channel and gloss. Should be any |
|             |              |                      | string not used in glosses.      |
|             |              |                      |                                  |
+-------------+--------------+----------------------+----------------------------------+
"""


import os


MASK_KEY = os.environ.get('SIGNBLEU_MASK_KEY', '###')
SEP_KEY = os.environ.get('SIGNBLEU_SEP_KEY', '_')
