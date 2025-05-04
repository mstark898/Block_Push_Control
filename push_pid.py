#!/usr/bin/env python3
"""
push_fast_v4.py  – proportional cube pusher with timeout + CSV logging
"""
import os, sys, time, warnings, numpy as np, robosuite as suite
from robosuite import load_composite_controller_config
from collections import deque

warnings.filterwarnings("ignore")
os.environ.update({"PYTHONWARNINGS": "ignore",
                   "ROBOSUITE_SUPPRESS_WARNINGS": "1"})

# ─── timeout guard ────────────────────────────────────────────
MAX_TIME = 180.0        # seconds before we declare failure

# ─── metric buffers ───────────────────────────────────────────
t0          = time.time()
next_sample = 1.0        # sample every second
samples     = []         # (sec, err)
t_finish    = None       # when err ≤ 1 cm
status      = "timeout"  # default until success

# ─── build env ------------------------------------------------
cfg = load_composite_controller_config(controller="BASIC")
cfg["body_parts"]["right"]["type"] = "OSC_POSE"
env = suite.make("Lift","Panda",
                 controller_configs=cfg,
                 renderer="mujoco", has_renderer=True,
                 use_camera_obs=False, control_freq=60,
                 ignore_done=True)
obs = env.reset()

# ─── task constants ------------------------------------------
cube_xyz = obs["object-state"][:3]
GOAL_XY  = cube_xyz[:2] + np.array([0.15, -0.10])
TABLE_Z  = cube_xyz[2]
WRIST_Z  = TABLE_Z + 0.001   # press depth

# controller gains (unchanged) …
APPROACH_HEIGHT=0.15; BASE_OFFSET=0.05
PUSH_VEL_MAX=0.15; PUSH_VEL_MIN=0.05
FADE_DIST=0.001; EXPONENT=2
SPEED_XY=0.10; SPEED_Z=0.10
TOL_FINE=0.015
STALL_TIME=1.0; STALL_THRESH=0.002; REAPP_THRESH=0.15
RENDER_EVERY=100
DAMP = 0.15 

# ─── main loop -----------------------------------------------
phase="approach"; step=0
stall_buf=deque(maxlen=int(STALL_TIME*env.control_freq))

while True:
    elapsed = time.time() - t0
    if elapsed > MAX_TIME:                # timeout guard
        break

    cube_xy = obs["object-state"][:2]
    ee_pos  = obs["robot0_eef_pos"]
    err_vec = GOAL_XY - cube_xy
    err     = np.linalg.norm(err_vec)
    stall_buf.append(err)

    # log once per second
    while elapsed >= next_sample:
        samples.append((int(next_sample), err))
        next_sample += 1.0
    if t_finish is None and err <= 0.01:
        t_finish = elapsed

    # success
    if err < TOL_FINE:
        status = "success"
        break

    # ----- phase machine --------------------------
    if phase=="approach":
        dir_u   = err_vec/(err+1e-9)
        tgt_xy  = cube_xy - dir_u*BASE_OFFSET
        tgt_z   = TABLE_Z + APPROACH_HEIGHT
        if np.linalg.norm(ee_pos - np.r_[tgt_xy,tgt_z]) < 0.005:
            phase="lower"
    elif phase=="lower":
        tgt_xy, tgt_z = ee_pos[:2], WRIST_Z + 0.001
        if abs(ee_pos[2]-tgt_z) < 0.0008:
            phase="push"; stall_buf.clear()
    else:  # push
        dir_u = err_vec/(err+1e-9)
        lateral = np.linalg.norm(cube_xy - (ee_pos[:2]+dir_u*BASE_OFFSET))
        if lateral>REAPP_THRESH: phase="approach"; continue
        if len(stall_buf)==stall_buf.maxlen and (stall_buf[0]-err)<STALL_THRESH:
            phase="approach"; continue

        fade = np.clip((err/FADE_DIST)**EXPONENT + DAMP,0,1)
        fwd  = max(PUSH_VEL_MAX*fade, PUSH_VEL_MIN if err>FADE_DIST else 0)
        dxy  = dir_u*fwd - 0.5*(cube_xy-ee_pos[:2])
        dz   = np.clip(WRIST_Z - ee_pos[2], -SPEED_Z, SPEED_Z)
        dpos = np.clip(np.r_[dxy,dz], -SPEED_XY, SPEED_XY)
        act  = np.zeros(env.action_dim); act[:3]=dpos; act[6]=1
        obs,*_=env.step(act); step+=1
        if step%RENDER_EVERY==0: env.render()
        continue

    # fallback for approach / lower
    tgt = np.r_[tgt_xy,tgt_z]
    dpos= np.clip(tgt-ee_pos, -SPEED_XY, SPEED_XY)
    act = np.zeros(env.action_dim); act[:3]=dpos; act[6]=1
    obs,*_=env.step(act); step+=1
    if step%RENDER_EVERY==0: env.render()

env.close()

# ─── CSV output ----------------------------------------------
impl  = os.path.basename(__file__).replace(".py","")
trial = os.environ.get("TRIAL_IDX","0")
total = time.time() - t0
line  = [impl, trial, f"{total:.2f}",
         f"{t_finish:.2f}" if t_finish else "NaN",
         status]
for sec,er in samples:
    line.append(f"{sec}s:{er:.3f}")
print(",".join(line))
sys.stdout.flush()
