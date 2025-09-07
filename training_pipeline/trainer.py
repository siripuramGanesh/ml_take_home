from .dynamic_head import DynamicYOLO
from .continual_loss import DynamicLoss
from .protective_hooks import setup_protective_hooks

class ContinuousTrainer:
    def __init__(self):
        # ... existing code ...
        self.dynamic_loss = DynamicLoss(original_classes=80, new_classes=20)
    
    async def train_with_new_classes(self, new_data, new_class_names):
        """Train model with new classes using dynamic architecture"""
        
        # Register new classes
        for class_name in new_class_names:
            self.model.add_new_class(class_name)
        
        # Phased training approach
        await self._phase_1_train_new_head(new_data)
        await self._phase_2_joint_fine_tuning(new_data)
        await self._phase_3_router_optimization(new_data)
    
    async def _phase_1_train_new_head(self, new_data):
        """Phase 1: Train only the new head"""
        print("🎯 Phase 1: Training new class head")
        # Freeze original head, train only new head
        # High regularization to prevent overfitting
    
    async def _phase_2_joint_fine_tuning(self, new_data):
        """Phase 2: Joint training with careful learning rates"""
        print("🎯 Phase 2: Joint fine-tuning")
        # Lower LR for original head, higher for new head
    
    async def _phase_3_router_optimization(self, new_data):
        """Phase 3: Optimize the class router"""
        print("🎯 Phase 3: Router optimization")
        # Learn optimal combination of old and new predictions