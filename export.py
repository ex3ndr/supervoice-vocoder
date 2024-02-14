import torch
from supervoice_vocoder.bigvsan import BigVSAN

# Load source model
checkpoint = torch.load('bigvsan.pt', map_location='cpu')
state_dict = checkpoint['generator']

# Remove weight normalization
keys_to_delete = []
for key in list(state_dict.keys()):
    if 'weight_v' in key:
        # Construct the corresponding weight_g key
        g_key = key.replace('weight_v', 'weight_g')

        # Check if both weight_g and weight_v exist
        if g_key in state_dict:
            weight_v = state_dict[key]
            weight_g = state_dict[g_key]

            # Compute the norm of weight_v
            norm_v = torch.norm(weight_v, p=2, dim=None, keepdim=True)

            # Recombine weight_g and weight_v to get the original weight
            recombined_weight = weight_g / norm_v * weight_v

            # Update the original weight key in the state dict
            original_weight_key = key.replace('.weight_v', '.weight')
            state_dict[original_weight_key] = recombined_weight

            # Mark weight_g and weight_v keys for deletion
            keys_to_delete.append(key)
            keys_to_delete.append(g_key)

# Remove the weight_g and weight_v keys from the state dict
for key in keys_to_delete:
    del state_dict[key]

# Try to load model
model = BigVSAN()
model.load_state_dict(state_dict)

# Export the model
model.eval()
torch.save(model.state_dict(), 'bigvsan_exported.pt')