"""pipeline: one-shot retargeting closed loop in simulation.

perception  -> world-frame candidate keypoints from a saved capture
oracle      -> ground-truth answers to the discrete VLM questions
retarget_runner -> full demo->target transfer + execution for one pair
eval_quick  -> small-N evaluation over tasks x seeds x method variants
"""
