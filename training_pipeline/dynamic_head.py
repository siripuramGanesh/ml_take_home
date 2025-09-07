from typing import Tuple

import torch
import torch.nn as nn


class DynamicYOLO(nn.Module):
    """
    Wrapper that adds new class detection capability to pretrained YOLO
    without modifying the original architecture.
    """
    
    def __init__(self, base_model, original_classes: int = 80, max_new_classes: int = 20):
        super().__init__()
        self.base_model = base_model
        self.original_classes = original_classes
        self.max_new_classes = max_new_classes
        self.total_classes = original_classes + max_new_classes
        
        # Freeze backbone and preserve original weights
        self._freeze_base_model()
        
        # Create parallel head for new classes
        self.new_class_head = self._create_new_head()
        
        # Learnable routing to combine outputs
        self.class_router = nn.Sequential(
            nn.Linear(self.total_classes, 128),
            nn.ReLU(),
            nn.Linear(128, self.total_classes)
        )
        
        # Protection parameters
        self.original_class_mask = torch.ones(self.total_classes)
        self.original_class_mask[:original_classes] = 1.0  # Protect original classes
        self.original_class_mask[original_classes:] = 0.5  # Encourage new classes

    def _freeze_base_model(self):
        """Freeze most of the base model, keep final layer trainable"""
        # Freeze all layers except the detection head
        for name, param in self.base_model.named_parameters():
            if 'model.23' not in name:  # YOLOv8 detection head is usually layer 23
                param.requires_grad = False
            else:
                param.requires_grad = True  # Allow fine-tuning detection head
        print("✅ Base model frozen, detection head remains trainable")

    def _create_new_head(self):
        """Create a parallel detection head for new classes"""
        # Get the original detection head structure
        original_head = self.base_model.model[-1]
        
        # Create identical structure but for new classes
        return nn.Sequential(
            nn.Conv2d(original_head.conv[0].in_channels, 
                     original_head.conv[0].out_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(original_head.conv[0].out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(original_head.conv[0].out_channels, 
                     self.max_new_classes * 5,  # 5 parameters per anchor
                     kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with dual-head architecture
        Returns: (boxes, scores, class_ids)
        """
        # Extract features using base model backbone
        features = self.base_model.model[:-1](x)  # All layers except head
        
        # Original head predictions
        orig_output = self.base_model.model[-1](features)
        orig_boxes, orig_scores = self._decode_output(orig_output, self.original_classes)
        
        # New head predictions  
        new_output = self.new_class_head(features)
        new_boxes, new_scores = self._decode_output(new_output, self.max_new_classes)
        
        # Combine predictions
        combined_boxes = torch.cat([orig_boxes, new_boxes], dim=1)
        combined_scores = torch.cat([orig_scores, new_scores], dim=2)
        
        # Apply learnable routing with protection
        routed_scores = self.class_router(combined_scores)
        protected_scores = routed_scores * self.original_class_mask.to(routed_scores.device)
        
        # Get final class predictions
        final_scores, class_ids = torch.max(protected_scores, dim=2)
        
        return combined_boxes, final_scores, class_ids

    def _decode_output(self, output: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode YOLO output into boxes and scores"""
        # Simplified decoding - real implementation would handle anchors, etc.
        batch_size, _, height, width = output.shape
        output = output.view(batch_size, 5 + num_classes, -1).permute(0, 2, 1)
        
        boxes = output[..., :4]  # x, y, w, h
        confidence = output[..., 4:5]  # object confidence
        class_scores = output[..., 5:5+num_classes]  # class scores
        
        # Combine confidence and class scores
        scores = confidence * class_scores
        
        return boxes, scores

    def get_trainable_parameters(self):
        """Get parameters that should be trained"""
        return [
            {'params': self.base_model.model[-1].parameters(), 'lr': 1e-5},  # Fine-tune original head
            {'params': self.new_class_head.parameters(), 'lr': 1e-4},        # Train new head
            {'params': self.class_router.parameters(), 'lr': 1e-4}           # Train router
        ]