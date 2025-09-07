import torch
import torch.nn as nn

class DynamicLoss(nn.Module):
    """
    Custom loss function for continual learning that prevents catastrophic forgetting
    while encouraging learning of new classes.
    """
    
    def __init__(self, original_classes: int, new_classes: int):
        super().__init__()
        self.original_classes = original_classes
        self.new_classes = new_classes
        self.total_classes = original_classes + new_classes
        
        # Individual loss components
        self.bbox_loss = nn.MSELoss()
        self.confidence_loss = nn.BCEWithLogitsLoss()
        self.class_loss = nn.CrossEntropyLoss()
        
        # Protection parameters
        self.original_protection = 0.8  # Protect original knowledge
        self.new_encouragement = 0.5    # Encourage new learning
        self.router_weight = 0.1        # Router learning weight

    def forward(self, predictions, targets):
        """
        predictions: (boxes, scores, class_ids)
        targets: list of target annotations
        """
        boxes, scores, class_ids = predictions
        
        # Separate original and new class targets
        orig_targets = self._filter_targets(targets, range(self.original_classes))
        new_targets = self._filter_targets(targets, range(self.original_classes, self.total_classes))
        
        # Calculate losses for each component
        orig_loss = self._calculate_loss_component(boxes, scores, class_ids, orig_targets, 
                                                 is_original=True)
        new_loss = self._calculate_loss_component(boxes, scores, class_ids, new_targets,
                                                is_original=False)
        
        # Apply protection and encouragement weights
        total_loss = (self.original_protection * orig_loss + 
                     self.new_encouragement * new_loss)
        
        return total_loss

    def _calculate_loss_component(self, boxes, scores, class_ids, targets, is_original: bool):
        """Calculate loss for specific class group"""
        if not targets:
            return torch.tensor(0.0, device=boxes.device)
        
        # Convert targets to tensor format
        target_boxes, target_scores, target_classes = self._prepare_targets(targets, boxes.device)
        
        # Box regression loss
        box_loss = self.bbox_loss(boxes, target_boxes)
        
        # Confidence loss
        conf_loss = self.confidence_loss(scores, target_scores)
        
        # Classification loss
        class_loss = self.class_loss(class_ids, target_classes)
        
        return box_loss + conf_loss + class_loss

    def _filter_targets(self, targets, class_range):
        """Filter targets for specific class range"""
        return [t for t in targets if t['class_id'] in class_range]

    def _prepare_targets(self, targets, device):
        """Convert target annotations to tensor format"""
        # Implementation would convert JSON targets to training format
        # This is simplified for illustration
        boxes = torch.randn(len(targets), 4, device=device)
        scores = torch.ones(len(targets), 1, device=device)
        classes = torch.randint(0, self.total_classes, (len(targets),), device=device)
        
        return boxes, scores, classes