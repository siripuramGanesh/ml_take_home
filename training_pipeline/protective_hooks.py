import torch
import torch.nn as nn

def setup_protective_hooks(model):
    """Setup gradient hooks to prevent catastrophic forgetting"""
    
    def protective_backward_hook(module, grad_input, grad_output):
        """Hook that filters destructive gradients"""
        if grad_output[0] is not None:
            # Only allow gradients that don't harm original classes
            protected_grad = grad_output[0] * model.original_class_mask
            return (protected_grad,) + grad_output[1:]
        return grad_output
    
    # Register hooks on sensitive layers
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight.requires_grad:
            if 'model.23' in name:  # Original detection head
                module.register_full_backward_hook(protective_backward_hook)
    
    print("✅ Protective gradient hooks installed on original layers")

class GradientNoiseInjection:
    """Add controlled noise to gradients to prevent overfitting to new data"""
    
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level
    
    def __call__(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            noise = torch.randn_like(grad_output[0]) * self.noise_level
            return (grad_output[0] + noise,) + grad_output[1:]
        return grad_output