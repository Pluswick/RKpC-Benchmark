import unittest

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from rat_kp_fusion.config import load_config
from rat_kp_fusion.models import (
    InjectionAttentiveFP,
    MolecularGCN,
    MolecularGINE,
    build_d_mpnn,
    trainable_parameter_count,
)
from rat_kp_fusion.statistics import exact_sign_flip_pvalue, percentile_seed_ci
from rat_kp_fusion.training import _pyg_graph
from rat_kp_core.models import load_config as load_legacy_config
from evaluate_manuscript_benchmarks import metrics as manuscript_metrics
from prepare_matched_panels import (
    build_direct_panel,
    exact_paper_mapping,
    prepare_paper_records,
    select_representative_oof,
)
from rat_kp_controls.models import CenteredTissueIntercept
from rat_kp_dvi.analysis import _aggregate_panel


class PublicSmokeTests(unittest.TestCase):
    def test_frozen_configs_load(self):
        self.assertEqual(load_legacy_config()["target"], "log10(Kp)")
        self.assertEqual(load_config(require_frozen=True)["target"], "log10(Kp)")

    def test_graph_model_forward_paths(self):
        for context_dim, position in (
            (0, "none"),
            (11, "early"),
            (11, "late"),
            (5, "early"),
            (5, "late"),
        ):
            context = np.zeros(context_dim, dtype=np.float32)
            graphs = [
                _pyg_graph("CCO", context, 0.0),
                _pyg_graph("c1ccccc1", context, 0.0),
            ]
            batch = Batch.from_data_list(graphs)
            first = graphs[0]
            for builder in (MolecularGCN, MolecularGINE, InjectionAttentiveFP):
                model = builder(
                    first.x.shape[1], first.edge_attr.shape[1], context_dim, position
                )
                output = model(batch)
                self.assertEqual(tuple(output.shape), (2,))
                self.assertTrue(torch.isfinite(output).all())

    def test_d_mpnn_parameterization(self):
        early = trainable_parameter_count(build_d_mpnn(11, "early"))
        late = trainable_parameter_count(build_d_mpnn(11, "late"))
        self.assertGreater(early, 0)
        self.assertGreater(late, 0)
        self.assertNotEqual(early, late)

    def test_statistical_helpers(self):
        self.assertEqual(exact_sign_flip_pvalue(np.ones(10)), 2 / (2**10))
        low, high = percentile_seed_ci(np.linspace(-0.1, -0.01, 10), replicates=200)
        self.assertLess(low, high)

    def test_processed_manuscript_benchmark_metrics(self):
        observed = np.array([0.0, 1.0, -1.0])
        predicted = np.array([0.0, 1.0 + np.log10(2.0), -1.0 - np.log10(4.0)])
        result = manuscript_metrics(observed, predicted)
        self.assertAlmostEqual(result["f2"], 2 / 3)
        self.assertEqual(result["f4"], 1.0)
        self.assertTrue(np.isfinite(result["rmse_log10"]))

    def test_additional_context_and_dvi_helpers(self):
        intercept = CenteredTissueIntercept(3)
        with torch.no_grad():
            intercept.raw_effect.copy_(torch.tensor([1.0, 2.0, 4.0]))
        centered = intercept(torch.eye(3))
        self.assertAlmostEqual(float(centered.sum().detach()), 0.0, places=6)

        frame = pd.DataFrame(
            {
                "parent_group_id": ["PG1", "PG1"],
                "identity_unit_id": ["IU1", "IU1"],
                "Drug": ["synthetic", "synthetic"],
                "SMILES": ["CCO", "CCO"],
                "Tissue": ["liver", "lung"],
                "y_true": [0.0, np.log10(2.0)],
                "y_pred": [np.log10(1.5), np.log10(2.5)],
            }
        )
        result = _aggregate_panel(frame, {"liver": 10.0, "lung": 5.0}, 250.0)
        self.assertEqual(len(result), 1)
        self.assertGreater(float(result.loc[0, "observed_dvi_true_L_per_kg"]), 0.0)

    def test_value_free_matched_panel_builder(self):
        processed = pd.DataFrame(
            {
                "row_index": [1],
                "parent_group_id": ["PG1"],
                "identity_unit_id": ["IU1"],
                "Drug": ["synthetic"],
                "SMILES": ["CCO"],
                "Tissue": ["liver"],
            }
        )
        paper = prepare_paper_records(
            pd.DataFrame(
                {
                    "workbook_excel_row": [10],
                    "workbook_drug": ["synthetic"],
                    "Tissue": ["liver"],
                    "canonical_smiles": ["CCO"],
                    "paper_exp_kp": [1.0],
                    "baseline_pred_kp": [1.2],
                }
            ),
            "PT",
        )
        mapping = exact_paper_mapping(paper, processed)
        predictions = select_representative_oof(
            pd.DataFrame(
                {
                    "split_type": ["parent_group"],
                    "split_seed": [0],
                    "model": ["gine"],
                    "condition": ["onehot_early"],
                    "row_index": [1],
                    "Tissue": ["liver"],
                    "y_pred": [0.1],
                }
            )
        )
        seed_level, pooled = build_direct_panel({"PT": mapping}, predictions)
        self.assertEqual(len(seed_level), 1)
        self.assertEqual(len(pooled), 1)
        self.assertEqual(int(pooled.loc[0, "n_oof_split_seeds"]), 1)


if __name__ == "__main__":
    unittest.main()
