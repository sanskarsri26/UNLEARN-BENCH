import copy
import unittest

import torch

from unlearn_bench.methods.baselines import sham_update, supervised_train
from unlearn_bench.methods.pcgu import compute_pcgu_selection, pcgu_step
from unlearn_bench.models import TinyAssociationLM, build_vocabulary
from unlearn_bench.utils.reproducibility import set_seed


class MethodTests(unittest.TestCase):
    def test_pcgu_selects_lowest_similarity_vector(self):
        parameter = torch.nn.Parameter(torch.zeros(2, 2))
        selection = compute_pcgu_selection(
            [("weight", parameter)],
            (torch.tensor([[1.0, 0.0], [1.0, 0.0]]),),
            (torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),),
            selected_fraction=0.5,
        )
        self.assertEqual(selection.selected, 1)
        self.assertFalse(selection.masks["weight"][0].any())
        self.assertTrue(selection.masks["weight"][1].all())

    def test_pcgu_changes_only_selected_entries(self):
        set_seed(3)
        records = [
            {"prompt": "entity alder", "completion": "red", "replacement": "blue"},
            {"prompt": "entity birch", "completion": "green", "replacement": "yellow"},
        ]
        vocabulary = build_vocabulary(records)
        model = TinyAssociationLM(len(vocabulary.tokens), 5)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        selection = pcgu_step(model, vocabulary, records, learning_rate=0.1, selected_fraction=0.2)
        for name, parameter in model.named_parameters():
            changed = parameter.detach() != before[name]
            self.assertFalse(torch.logical_and(changed, ~selection.masks[name]).any())
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters()))

    def test_frozen_parameter_remains_frozen(self):
        records = [{"prompt": "entity alder", "completion": "red", "replacement": None}]
        vocabulary = build_vocabulary(records)
        model = TinyAssociationLM(len(vocabulary.tokens), 4)
        model.embedding.weight.requires_grad_(False)
        before = model.embedding.weight.detach().clone()
        supervised_train(model, vocabulary, records, steps=2, learning_rate=0.1)
        self.assertTrue(torch.equal(before, model.embedding.weight))

    def test_sham_is_seeded_and_nonzero(self):
        first = TinyAssociationLM(5, 3)
        second = copy.deepcopy(first)
        initial = copy.deepcopy(first)
        sham_update(first, scale=0.1, seed=7)
        sham_update(second, scale=0.1, seed=7)
        parameters = zip(first.parameters(), second.parameters(), initial.parameters(), strict=True)
        squared_distance = 0.0
        for left, right, old in parameters:
            self.assertTrue(torch.equal(left, right))
            self.assertFalse(torch.equal(left, old))
            squared_distance += torch.sum((left - old) ** 2).item()
        self.assertAlmostEqual(squared_distance**0.5, 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
