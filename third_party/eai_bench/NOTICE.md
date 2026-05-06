# Modification Notice

The files in this directory are **derivative works** based on the
`eai_bench` project (licensed under the GNU General Public License v3.0,
see `LICENSE`).

In accordance with Section 5(a) of the GPL v3, this notice records the
fact and date of modifications made by the authors of the MLLM Privacy
Benchmark.

## Modified files

- `tier_2.json`
- `tier_3.json`

## Nature of modifications

Modified by: Junran Wang
Modification date: 2025

The original tier JSON files have been adapted for the MLLM Privacy
Benchmark. Modifications include (non-exhaustive):

- Curating / filtering scenarios across Tier 2, and Tier 3.
- Restructuring fields and modifying metadata required by the generation
  pipeline (e.g. `pddl_objects`, `vh_character`, `secret_item`,
  `target_item`, scene-graph hooks for VirtualHome integration).
- Editing scenario text, object lists, and task definitions for
  consistency with the privacy-tiered evaluation protocol.

The unmodified original data is available from the upstream `EAPrivacy`
project repository. As required by GPL v3, the complete corresponding
source of these modified files is distributed with this repository and
remains licensed under the GPL v3.

## License

These modified files continue to be distributed under the GNU General
Public License v3.0 (see `LICENSE`). The VLM Privacy Evaluation as a
whole is also released under GPL v3.0 (see the top-level `LICENSE`
file).
