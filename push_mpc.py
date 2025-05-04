#!/usr/bin/env python3
"""
push_pid_mpc.py – MPC add‑on with simple re‑engage lift
────────────────────────────────────────────────────────────
• Keeps the original 4 phases: approach → lower → seek → mpc.
• When contact is lost during MPC, transition to **seek_lift**:
    1. seek_lift (new): raise the wrist 6 cm above cube top and
       re‑center 5 cm behind the cube.
    2. When lift pose reached → go back to normal seek (drop & slide).
Minimal delta to earlier working version; no complex sub‑phase loops.
"""
import numpy as np
import cvxpy as cp
import robosuite as suite
from robosuite import load_composite_controller_config

import time, warnings, os, sys
warnings.filterwarnings("ignore")              # silence python warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["ROBOSUITE_SUPPRESS_WARNINGS"] = "1"  # robosuite built‑in flag

# ─── metric buffers ────────────────────────────────────────────────
t0           = time.time()
next_sample  = 1.0            # first sample at 10 s
samples      = []              # (elapsed_sec, error_m) pairs
t_finish     = None            # when |error| ≤1 cm

# ───────── constants ─────────
APPROACH_H   = 0.15          # 15 cm high initial approach
LIFT_H       = 0.06          # 6 cm lift when re‑engaging
BASE_OFF     = 0.05          # 5 cm behind cube
PALM_Z_OFF   = -0.002        # palm 2 mm below cube top
SEEK_VEL     = 0.03          # 3 cm per step slide
SPEED_XY     = 0.10; SPEED_Z = 0.10
TOL_FINE     = 0.015
TIMEOUT_SEC  = 30.0    
RENDER_EVERY = 100; LOG_EVERY = 20

# MPC params
HORIZON=15; U_MAX=3; V_MAX=0.4; LAMBDA_U=1e-3; LAMBDA_T=5; DT=1/60

# ───────── environment ─────────
cfg = load_composite_controller_config(controller="BASIC")
cfg["body_parts"]["right"]["type"] = "OSC_POSE"
env = suite.make("Lift","Panda",controller_configs=cfg,
                 renderer="mujoco",has_renderer=True,use_camera_obs=False,
                 control_freq=60,ignore_done=True)
obs = env.reset()

cube_xyz = obs["object-state"][:3]
GOAL_XY  = cube_xyz[:2] + np.array([0.15,-0.10])
TABLE_Z  = cube_xyz[2]
WRIST_Z  = TABLE_Z + PALM_Z_OFF

# ───────── build small MPC ─────────
x=cp.Variable((2,HORIZON+1)); v=cp.Variable((2,HORIZON+1)); a=cp.Variable((2,HORIZON))
px,pv,pg = cp.Parameter(2),cp.Parameter(2),cp.Parameter(2)
cons=[x[:,0]==px, v[:,0]==pv]; cost=0
for k in range(HORIZON):
    cost+=cp.sum_squares(x[:,k]-pg)+LAMBDA_U*cp.sum_squares(a[:,k])
    cons+=[x[:,k+1]==x[:,k]+v[:,k]*DT,
           v[:,k+1]==v[:,k]+a[:,k]*DT,
           cp.norm_inf(a[:,k])<=U_MAX,
           cp.norm_inf(v[:,k+1])<=V_MAX]
cost+=LAMBDA_T*cp.sum_squares(x[:,HORIZON]-pg)
prob=cp.Problem(cp.Minimize(cost),cons)
px.value=cube_xyz[:2]; pv.value=np.zeros(2); pg.value=GOAL_XY
prob.solve(solver=cp.OSQP,warm_start=True)

# ───────── helpers ─────────
contact = lambda o: 0.0 if o.get("robot0_right_hand_touch_forces") is None else np.linalg.norm(o.get("robot0_right_hand_touch_forces"))

def back_pose(cube_xy,height):
    dir_u = (GOAL_XY - cube_xy) / (np.linalg.norm(GOAL_XY - cube_xy) + 1e-9)
    return np.r_[cube_xy - dir_u*BASE_OFF, TABLE_Z + height]

# ───────── state machine ─────────
state = "approach"       # approach → lower → seek → mpc ; plus seek_lift
prev_xy = cube_xyz[:2]
step = 0

status = "success"           # place immediately before while True
while True:
    cube_xy = obs["object-state"][:2]
    err     = np.linalg.norm(GOAL_XY - cube_xy)
    if err < TOL_FINE:
        status = "success"
        break

    ee      = obs["robot0_eef_pos"]
    cf      = contact(obs)
        
    # ─── metrics ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    # --- timeout guard ------------------------------------------
    if elapsed > TIMEOUT_SEC:
        status = "timeout"
        break

    if elapsed >= next_sample:                # sample every 10 s
        #print(f"err {err:.3f} m")
        samples.append((int(next_sample), err))
        next_sample += 1.0
    if t_finish is None and err <= 0.01: # reached 1 cm
        t_finish = elapsed

    
    if state == "approach":
        tgt = back_pose(cube_xy, APPROACH_H)
        dpos = np.clip(tgt - ee, -SPEED_XY, SPEED_XY)
        act = np.zeros(env.action_dim); act[:3]=dpos; act[6]=1
        obs,*_=env.step(act)
        if np.linalg.norm(tgt - ee) < 0.004:
            state = "lower"

    elif state == "lower":
        tgt = np.r_[ee[:2], WRIST_Z]
        dpos = np.clip(tgt - ee, -SPEED_XY, SPEED_Z)
        act = np.zeros(env.action_dim); act[:3]=dpos; act[6]=1
        obs,*_=env.step(act)
        if abs(ee[2]-tgt[2]) < 0.002:
            state = "seek"

    elif state == "seek_lift":
        tgt = back_pose(cube_xy, LIFT_H)
        dpos = np.clip(tgt - ee, -SPEED_XY, SPEED_Z)
        act = np.zeros(env.action_dim); act[:3]=dpos; act[6]=1
        obs,*_=env.step(act)
        if np.linalg.norm(tgt - ee) < 0.004:
            state = "seek"

    elif state == "seek":
        dir_u = (GOAL_XY - cube_xy) / (err + 1e-9)
        act = np.zeros(env.action_dim)
        act[:2] = dir_u * SEEK_VEL
        act[2]  = np.clip(WRIST_Z - ee[2], -SPEED_Z, SPEED_Z)
        act[6]  = 1
        obs,*_=env.step(act)
        if contact(obs) > 1e-4:
            state = "mpc"; prev_xy = cube_xy.copy(); #print("contact → mpc")

    else:  # mpc phase
        cube_v = (cube_xy - prev_xy)/DT; prev_xy = cube_xy.copy()
        px.value, pv.value, pg.value = cube_xy, cube_v, GOAL_XY
        prob.solve(warm_start=True, solver=cp.OSQP)
        v_cmd = v[:,1].value
        dxy   = np.clip(v_cmd*DT, -SPEED_XY, SPEED_XY)
        dz    = WRIST_Z - ee[2]
        act   = np.zeros(env.action_dim)
        act[:2] = dxy; act[2] = np.clip(dz, -SPEED_Z, SPEED_Z); act[6]=1
        obs,*_=env.step(act)
        if err < TOL_FINE:
            #print(f"🎯 success err={err:.4f}"); 
            break
        if contact(obs) < 1e-4:
            #print("lost contact → seek_lift"); 
            state = "seek_lift"

    if step % RENDER_EVERY == 0:
        env.render()
    step += 1

# ─── print CSV line ──────────────────────────────────────────────
# format: impl,trial,total_sec,t_finish,s10,e20,e30,…
impl   = os.path.basename(__file__).replace(".py","")  # script name
trial  = os.environ.get("TRIAL_IDX","0")
total  = time.time() - t0
line = [impl, trial,
        f"{total:.2f}",
        status,
        f"{t_finish:.2f}" if t_finish else "NaN"]
for sec,err in samples:
    line.append(f"{sec}s:{err:.3f}")
print(",".join(line))
sys.stdout.flush()


env.close()