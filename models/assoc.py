"""Position-difference inputs and residual prediction for the models other than the Transformer.

"<base>_resid_eq2" = the model <base> with
  - the association features of preprocess_data/make_assoc_features.py as 5 extra input channels of every agent at the
    target event (displacement to the previous event, time gap, displacement / gap; zeros at the earlier events),
  - the prediction = network output + the position-difference velocity (residual prediction), and
  - the permutation-equivariant layers of models/equivariant.py (make_equivariant). All slot-specific parts of these
    models are replaced there, so the model is fully equivariant inside a team ("eq2" of the Transformer).

The models read their input through items["cies"] (agents x channels, locations in rows 0..23: ball, ball end, 22 players).
The wrapper appends the 5 channels to every row of items["cies"] (n_input becomes 2 + 5); only the location rows of the
agents carry values, the velocity rows (the targets) get zeros there. restore() cuts them off again, the arrays that
the models store for the evaluation (loc_gt, v_rb) are cut back to x, y. The residual is added in
models.utils.add_difference (called where the models restore the predicted velocity).
"""

import torch

N_ASSOC = 5
N_LOC_ROWS = 24  # rows 0..23 of cies: ball, ball end, 22 players


def augment_cies(cies, assoc):
    """cies (B, nodes, C) or (T, B, nodes, C); assoc (B, 23, 5) or (1, B, 23, 5) -> same layout with C + 5 channels."""
    a = assoc.to(cies.device, cies.dtype)
    if a.dim() == 4:
        a = a[0]
    extra = torch.zeros(*cies.shape[:-1], N_ASSOC, dtype=cies.dtype, device=cies.device)
    target = extra[-1] if cies.dim() == 4 else extra  # (B, nodes, 5): the target event is the last of the window
    target[:, 0, :] = a[:, 0, :]  # ball
    target[:, 2:N_LOC_ROWS, :] = a[:, 1:, :]  # players
    return torch.cat([cies, extra], dim=-1)


def make_assoc(model):
    model.resid = True
    orig_forward, orig_sample, orig_restore = model.forward, model.sample, model.restore

    def restore(data, info):
        return orig_restore(data[..., :2], info)

    def with_assoc(items):
        items2 = dict(items)
        items2["cies"] = augment_cies(items["cies"], items["assoc"])
        return items2

    def forward(items, players_info, *args, **kwargs):  # training and validation loss
        return orig_forward(with_assoc(items), players_info, *args, **kwargs)

    def sample(items, players_info, *args, **kwargs):  # validation / test (and the stored predictions)
        out = orig_sample(with_assoc(items), players_info, *args, **kwargs)
        if isinstance(out, tuple) and len(out) == 3 and isinstance(out[2], dict):
            for key in ("loc_gt", "v_rb"):
                if key in out[2]:
                    out[2][key] = out[2][key][..., :2]
        return out

    model.restore = restore
    model.forward = forward
    model.sample = sample
    model.uses_assoc = True
    return model
