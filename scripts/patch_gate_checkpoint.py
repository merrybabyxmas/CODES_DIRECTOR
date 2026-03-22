#!/usr/bin/env python3
"""Patch checkpoint gate parameters: convert from tanh-space to direct-space.

Old: gate_val = tanh(gate_param),  gate_param ≈ 0.5 → effective 0.4621
New: gate_val = gate_param,        gate_param = 0.4621 (same effective value)

Also patches optimizer state (momentum/variance) for the gate parameters.
"""
import sys
import math
import torch

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_10k/checkpoint_step_18000.pt"
out_path = ckpt_path.replace(".pt", "_patched.pt")

print(f"Loading {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Patch adapter gate values in model state
patched = 0
for key in list(ckpt.get("adapters", {}).keys()):
    state = ckpt["adapters"][key]
    if "gate" in state:
        old_val = state["gate"].item()
        new_val = math.tanh(old_val)
        state["gate"] = torch.tensor(new_val)
        print(f"  {key}.gate: {old_val:.6f} -> {new_val:.6f}")
        patched += 1

print(f"Patched {patched} gate parameters")

# Reset optimizer state for gate params (momentum/variance from old parameterization
# would be wrong for new parameterization). The gate param group is index 2.
if "optimizer" in ckpt and "state" in ckpt["optimizer"]:
    opt_state = ckpt["optimizer"]["state"]
    param_groups = ckpt["optimizer"]["param_groups"]

    # Find gate param group (group with weight_decay=0.0 and 5 params or LR matching gate)
    gate_group_idx = None
    for i, pg in enumerate(param_groups):
        if pg.get("weight_decay", -1) == 0.0 and i >= 2:
            gate_group_idx = i
            break

    if gate_group_idx is not None:
        # Get param indices for gate group
        gate_param_indices = param_groups[gate_group_idx]["params"]
        print(f"  Resetting optimizer state for gate group (idx={gate_group_idx}, {len(gate_param_indices)} params)")
        for pid in gate_param_indices:
            if pid in opt_state:
                # Reset Adam state (exp_avg, exp_avg_sq, step)
                for k in ["exp_avg", "exp_avg_sq"]:
                    if k in opt_state[pid]:
                        opt_state[pid][k].zero_()
                if "step" in opt_state[pid]:
                    opt_state[pid]["step"] = torch.tensor(0.0)
                print(f"    Reset optimizer state for param {pid}")

print(f"Saving to {out_path}...")
torch.save(ckpt, out_path)
print("Done!")
