"""Record SkeletonMuscle.walk mocap replay — guaranteed walking video.

Run on cluster with sufficient memory:
    condor_submit_bid 10 -i -append 'request_memory=16384'
    cd /lustre/fast/fast/lsivakumar/LEAPS
    MUJOCO_GL=osmesa JAX_PLATFORMS=cpu python3 record_walk.py
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MUJOCO_GL", "osmesa")

from pathlib import Path
from loco_mujoco.task_factories import ImitationFactory
from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf

out_dir = "results/videos"
Path(out_dir).mkdir(parents=True, exist_ok=True)

print("Creating SkeletonMuscle env...")
env = ImitationFactory.make(
    "SkeletonMuscle",
    DefaultDatasetConf("walk", "mocap"),
    headless=True,
    recorder_params={
        "path": out_dir,
        "tag": "",
        "video_name": "skeleton_walk_mocap",
        "fps": 50,
        "compress": True,
    },
)

print("Playing trajectory (1 episode, 500 steps)...")
env.play_trajectory(n_episodes=1, n_steps_per_episode=500, record=True)
print("Stopping env (finalizing video)...")
env.stop()
print(f"Done! Check {out_dir}/skeleton_walk_mocap.mp4")
