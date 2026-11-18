CLASSIFICATION: FIX

The experiment added Conv1d layers (q_conv, k_conv) to local attention layers. These have 3D weight tensors of shape (512, 1, 4). In `setup_optimizer`, the filter `p.ndim >= 2` routes them to Muon, but Muon's `_step_muon` and `muon_step_fused` assume strictly 2D parameter shapes. When stacked, the conv params become 4D tensors (num_params, 512, 1, 4), causing `second_momentum_buffer` shape (num_params, 1, 4) to be incompatible with `v_mean` shape (num_params, 512, 1, 4) in the `lerp_` call.

Fix: Changed `matrix_params` filter from `p.ndim >= 2` to `p.ndim == 2` and added a separate `conv_params` group (ndim >= 3) routed to AdamW instead of Muon.
