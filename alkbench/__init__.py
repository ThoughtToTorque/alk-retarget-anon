"""alkbench: core geometry pipeline for keypoint-anchored retargeting (ALK).

Pure numpy + scipy. Python >= 3.8.
"""

from alkbench.candidates import backproject, project, kmeans, compute_candidates, CandidateSet
from alkbench.alk import (halfplane_signed_distance, build_alk,
                          alk_from_candidates, lateral_extent_ratio,
                          SLENDER_RATIO_THRESHOLD, ALK_SUBSETS,
                          alk_subset_indices, select_alk_subset)
from alkbench.procrustes import (procrustes, transform_points,
                                 rotation_angle_deg, shortest_arc_rotation,
                                 two_point_alignment, align_keypoints)
from alkbench.registration import (chamfer_distance, bounded_registration,
                                   adaptive_bounds, adaptive_registration,
                                   SIGMA_RATIO_THRESHOLD)
from alkbench.retarget import retarget_waypoints, grasp_translation_correction, retarget
from alkbench.discrete import DiscreteChoice, DiscreteSolver, OracleSolver, VLMSolver

__all__ = [
    "backproject", "project", "kmeans", "compute_candidates", "CandidateSet",
    "halfplane_signed_distance", "build_alk", "alk_from_candidates",
    "lateral_extent_ratio", "SLENDER_RATIO_THRESHOLD",
    "ALK_SUBSETS", "alk_subset_indices", "select_alk_subset",
    "procrustes", "transform_points", "rotation_angle_deg",
    "shortest_arc_rotation", "two_point_alignment", "align_keypoints",
    "chamfer_distance", "bounded_registration",
    "adaptive_bounds", "adaptive_registration", "SIGMA_RATIO_THRESHOLD",
    "retarget_waypoints", "grasp_translation_correction", "retarget",
    "DiscreteChoice", "DiscreteSolver", "OracleSolver", "VLMSolver",
]
