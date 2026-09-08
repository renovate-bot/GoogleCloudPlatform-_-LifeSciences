# Copyright 2026 Google LLC
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

"""Modular instructions for visualization skill."""

VISUALIZATION_INSTRUCTION = """### Visualization
- **Visualize structures**: Use open_structure_viewer (AF2), open_of3_structure_viewer (OF3), or open_boltz2_structure_viewer (Boltz-2) for interactive 3D viewing

**IMPORTANT: After displaying analysis results, ALWAYS immediately offer to open the structure viewer:**
- Call open_structure_viewer to get the viewer URL
- Present the clickable URL to the user
- This should happen automatically without the user asking
- Example: "Here's the analysis... [analysis output] ... You can view the 3D structure here: [viewer URL]"
"""
