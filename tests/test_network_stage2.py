import random
import unittest

from fbnnet_core import _utils

from fbnnet_core.network import _drop_subsumed_rules


def _original_drop_subsumed(ruleset):
    """Verbatim copy of the pre-optimisation pairwise loop in mine_fbn_network_stage2."""
    dropped = []
    for j, rule in enumerate(ruleset):
        for k, rule2 in enumerate(ruleset):
            if k == j:
                continue
            if (int(rule['numOfInput']) < int(rule2['numOfInput']) and
                    int(rule['type']) == int(rule2['type']) and
                    int(rule['timestep']) == int(rule2['timestep']) and
                    all(gene in _utils.splitExpression(rule2['input'], 2, False)
                        for gene in _utils.splitExpression(rule['input'], 2, False))):
                dropped.append(rule2)
    return [rule for rule in ruleset if rule not in dropped]


def _random_ruleset(rng, n_rules, genes):
    rules = []
    for _ in range(n_rules):
        chosen = rng.sample(genes, rng.randint(1, min(4, len(genes))))
        rules.append({
            'input': ",".join(chosen),
            'numOfInput': len(chosen),
            'type': rng.choice([0, 1]),
            'timestep': rng.choice([1, 2]),
            'error': 0,
            'support': round(rng.random(), 3),
        })
    if rules and rng.random() < 0.5:
        rules.append(dict(rng.choice(rules)))
    return rules


class TestDropSubsumedRules(unittest.TestCase):
    def test_matches_original_pairwise_loop(self):
        rng = random.Random(2024)
        for trial in range(300):
            ruleset = _random_ruleset(rng, rng.randint(0, 25), list("ABCDEFGH")[:rng.randint(2, 8)])
            self.assertEqual(_drop_subsumed_rules(ruleset), _original_drop_subsumed(ruleset), msg=f"trial {trial}")

    def test_subset_rule_removes_superset_only_within_same_type_and_timestep(self):
        base = {'error': 0, 'support': 0.5}
        small = dict(base, input="A", numOfInput=1, type=1, timestep=1)
        big_same = dict(base, input="A,B", numOfInput=2, type=1, timestep=1)
        big_other_type = dict(base, input="A,B", numOfInput=2, type=0, timestep=1)
        big_other_step = dict(base, input="A,B", numOfInput=2, type=1, timestep=2)
        kept = _drop_subsumed_rules([small, big_same, big_other_type, big_other_step])
        self.assertEqual(kept, [small, big_other_type, big_other_step])


if __name__ == "__main__":
    unittest.main()
