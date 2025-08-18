[1mdiff --git a/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/agents/gcdp_cfg.json b/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/agents/gcdp_cfg.json[m
[1mindex f313ab62..e0601150 100644[m
[1m--- a/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/agents/gcdp_cfg.json[m
[1m+++ b/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/agents/gcdp_cfg.json[m
[36m@@ -48,7 +48,7 @@[m
         "hdf5_normalize_obs": false,[m
         "hdf5_filter_key": "train",[m
         "hdf5_validation_filter_key": "test",[m
[31m-        "seq_length": 4, [m
[32m+[m[32m        "seq_length": 8,[m[41m [m
         "pad_seq_length": false,[m
         "frame_stack": 2,[m
         "pad_frame_stack": true,[m
[36m@@ -92,8 +92,8 @@[m
         },[m
         "horizon": {[m
             "observation_horizon": 2,[m
[31m-            "action_horizon": 2,[m
[31m-            "prediction_horizon": 4[m
[32m+[m[32m            "action_horizon": 4,[m
[32m+[m[32m            "prediction_horizon": 8[m
         },[m
         "unet": {[m
             "enabled": true,[m
[36m@@ -140,6 +140,7 @@[m
                     "object_lin_vel",[m
                     "object_ang_vel",[m
                     "last_action",[m
[32m+[m[32m                    "fingertip_contacts",[m
                     "goal_pose",[m
                     "goal_quat_diff"[m
                 ],[m
