# Handpicked (good working) parameters that made sense when thinking about it
# class HexGraphConfig(
#     neighbour_pos_tolerance: float = 22,
#     bidrectional_assignment: bool = False,
#     image_size: tuple[int, int] | None = None,
#     max_iterations: int = 15,
#     curve_aware_candidate_pred: bool = True,
#     cluster_vote_threshold: int = 2,
#     cluster_pred_eps: float = 16,
#     pred_merge_dist: float = 25,
#     cluster_conflict_pred_eps: float = 22,
#     min_dist_nodes: float = 40,
#     prefer_method: Literal['curve', 'lattice_vector'] | None = "curve",
#     debug: bool = False
# )


# - Group all final_predicted (dbscan and eps 22)
# 1st case (group length = 1):
#     - final_predicted is close to existing (25 or 27?):
#         - if there are multiple existing -> discard final_predicted -> set_predictors_edges_to_conflict for final_predicted
#         - existing has open edges in the direction of the predictors of final_predicted (assert_open_edges_in_directions)?
#             - yes: final_predicted is removed and existing takes over the predictors and also reverse at these edges
#             - no: final_predicted is removed and all edges of the predictors in this direction are set to CONFLICT.
#     - final_predicted is located within the extended radius of existing (25/27 to 40):
#         - discard final_predicted -> set_predictors_edges_to_conflict for final_predicted
#         - !!! this probably belongs to the final check
#     - No existing nearby -> normal point -> add to the nodes
# 2. Case (group length > 3):
#     - Discard all final_predicted -> set_predictors_edges_to_conflict for final_predicted
# 3. Case (2 <= group length <= 3):
#     - Calculate the centre point of the group (problem: with dbscan, the group can occupy a large area)
#     - Centre point of the group close to existing? (22?)
#         - yes: discard all final_predicted and set all edges of all predictors to CONFLICT
#         - no:
#             - all final_predicted have their predictors in different directions (assert_open_edges_in_directions)?
#                 - Yes: centre point is new point and link all predictors from all final_predicted to this
#                 - No: set_predictors_edges_to_conflict
# Final check for all remaining final_predicted:
#     - Does existing or final_predicted have any in the immediate vicinity? (< 40)
#         - yes: discard and set all predictors to CONFLICT (set_predictors_edges_to_conflict)
#         - no: check passed!

# Functions:
# - set_predictors_edges_to_conflict
#     - receives a list of predictors and their directions
#     - sets the directions to CONFLICT for each predictor

# - get_reverse_direction
#     - Receives a direction and returns the index for the opposite direction

# - assert_open_edges_in_directions
#     - For existing or final_predicted
#     - Receives a list of directions (if reverse, calculate beforehand)
#     - Returns True if all directions are still free, false if not.

# =========================================================================================================

# TODO: Function to remove two cells that are too close to each other, when building edges
# TODO: can it happen that A assigns B as neighbor but B does not do with A? relates to bidirectional assignment
# TODO: lattice vector fallback in position prediction is applied immediately. could also be applied later, if curve gets stuck and still missing neighbors
# TODO: prefer method in cluster_and_filter_candidates only filters out the prefered method and ignores the other. if the prefered are too few, it skips.
#   should be rather taking all candidates and just weight the ones that are prefered when calculating new point

# =========================================================================================================

# - if two final predictions are too close they are both removed and all valid cells around that have an open edge in this direction are marked at the edge as not_sure
# - or at least find the cells that predicted that predicton and assign the edges as not_sure
# - or two final predictions that are within a distance (eg 20) will form their mean (a super prediction) and all predictors will conntect to this guy. (will that work with build_edges?)
# - if the predicted point is too close to a real point, both could be removed. the real point just removed and the predicted also but also assign its predictors as not_sure
# - alternative: if predicted point too close to existing node, just assign the edges of the predictors to that existing node.
# - related to the previous. could also first check if the real point has a missing edges in that direction of the predictors. if not just mark as not suere
# - what can you do about the iterations. i dont know how long it takes until all cells are found (could wait until only few things are changing) but if there is an error

# possiblities:
# - two or three final_predicted fall together
#   - compute the average position and assign the edges to the predictors of each final_predicted
#   - check if all predictors predict their final_predicted to different directions.
#   - if two or more predictors predict to the same direction discard all final_predicted and mark the related missing edges of predictors as not_sure
# - more than three final_predicted fall together (each node has six neighbors and min two build a final_predicted)
#   - this should not happen. discard all and mark the edges of all predictors of all final_predicted as not_sure
# - one final_predicted is next/too close to an existing_node
#   - does this existing_node have missing edges in the direction of the predictors?
#   - do the predictors of the final_predicted have missing edges in that direction? (this is not relevant. should not happen, otherwise the would not predict)
#   - if yes merge.
#   - if no, discard and mark posssible open edges as not_sure
# - a final_predicted is next to another final_predicted and existing_node
#   - in this case discard all final_predicted and mark all edges of the responsible predictors as not_sure
# - two existing_nodes should be neighbors but are not connected, because the angle is too much off or they are too distant
#   - could increase the tolerance but don't know if this affects the rest of the procedure.
#   - leve it for now. maybe these missing edges can be ignored

#  =========================================================================================================
# Stopping Problem
# - create a flag that is set to true each time a node is added to the graph per loop. if the flag is true the loop continues.
# - still have a max_iterations number to prevent infinite loop in an unforseen case

# - gruppiere alle final_predicted (dbscan und eps 22)
# 1. Fall (group len = 1):
#     - final_predicted ist in der Nähe von existing (25 oder 27?):
#         - falls mehrer existing -> final_predicted verwerfen -> set_predictors_edges_to_conflict für final_predicted
#         - existing hat offene Kanten in Richtung der Predictors des final_predicted (assert_open_edges_in_directions)?
#             - ja: final_predicted wird entfernt und der existing übernimmt an diesen Kanten die Predictors und auch reverse
#             - nein: final_predicted wird entfernt und alle kanten der Predictors in diese Richtung auf CONFLICT gesetzt.
#     - final_predicted ist im erweiterten Radius von existing (25/27 bis 40):
#         final_predicted verwerfen -> set_predictors_edges_to_conflict für final_predicted
#         - !!! vermutlich gehört das zum final check
#     - Keinen existing in der Nähe -> normaler punkt -> hinzufügen
# 2. Fall (group len > 3):
#     - alle final_predicted verwerfen -> set_predictors_edges_to_conflict für final_predicted
# 3. Fall (2 <= group len <= 3):
#     - berechne den Mittelpunkt der Gruppe (Problem: da mit dbscan kann die Gruppe einen großen Bereich einnehmen)
#     - Mittelpunkt der Gruppe in der Nähe von existing? (22?)
#         - ja: alle final_predicted verwerfen und alle Kanten von allen Predictors auf CONFLICT setzen
#         - nein:
#             - alle final_predicted haben ihre Predictors in unterschiedlichen Richtungen (assert_open_edges_in_directions)?
#                 - ja: Mittelpunkt ist neuer Punkt und alle Predictors von allen final_predicted verknüpfen zu diesem
#                 - nein: set_predictors_edges_to_conflict
# Final Check für alle übrigen final_predicted:
#     - Hat existing oder final_predicted im Näheren Umfeld? (< 40)
#         - ja: verwerfen und alle Predictors auf CONFLICT setzen (set_predictors_edges_to_conflict)
#         - nein: Check bestanden!

# Funktionen:
# - set_predictors_edges_to_conflict
#     - bekommt liste von Predictors und deren Directions
#     - setzt für jeden Predictor die Directions auf CONFLICT

# - get_reverse_direction
#     - bekommt eine Richtung und gibt den Index für die Gegenrichtung zurück

# - assert_open_edges_in_directions
#     - für existing oder final_predicted
#     - bekommt liste von Directions (falls reverse, vorher berechnen)
#     - gibt True falls alle Directions noch frei sind, false falls nicht.
