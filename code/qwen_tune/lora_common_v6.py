"""Shared LoRA settings for the v6 two-pass experiment."""

# NOTE: linear_attn projections (18 of 24 layers) cannot be converted by convert_lora_to_gguf (unmapped tensors), so they are excluded.
# language-layer linear modules only (attention, linear-attention and MLP); the vision tower is not matched
TARGET_REGEX = r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
