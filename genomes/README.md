# genomes/ -- every hand-built hypothesis, as tested

One JSON per idea, named idea_universe. These are part of the record: the
registry (SURVIVORS.md) says what happened; these say exactly what was asked.

    python3.10 scripts/validate_genome.py --manifest manifests/<m>.yaml \
        --genome genomes/<idea>.json

deepstate_frozen.json is the deployed spec (deep_state.py sha 709f4e8b25b2a1e5),
regenerated via make_genome from the frozen parameters. Do not edit it.
