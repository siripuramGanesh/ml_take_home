import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

from .trainer import ContinuousTrainer


class CyclicalTrainingScheduler:
    """Manages cyclical training with progressive dataset mixing"""
    
    def __init__(self):
        self.phases = [
            {"name": "Phase 1", "new_ratio": 0.1, "coco_ratio": 0.9, "epochs": 5},
            {"name": "Phase 2", "new_ratio": 0.25, "coco_ratio": 0.75, "epochs": 8},
            {"name": "Phase 3", "new_ratio": 0.5, "coco_ratio": 0.5, "epochs": 10},
            {"name": "Phase 4", "new_ratio": 0.75, "coco_ratio": 0.25, "epochs": 12},
            {"name": "Phase 5", "new_ratio": 0.9, "coco_ratio": 0.1, "epochs": 15}
        ]
        
        self.current_phase_index = 0
        self.training_history = []
        self.next_training_time = datetime.now()
        self.training_interval = timedelta(hours=2)  # Train every 2 hours
    
    async def start_continuous_training(self, trainer: ContinuousTrainer, model):
        """Start continuous training process"""
        print("🔄 Starting continuous training scheduler")
        
        while True:
            try:
                # Check if it's time for training
                if datetime.now() >= self.next_training_time:
                    await self._run_training_phase(trainer, model)
                    self.next_training_time = datetime.now() + self.training_interval
                
                # Wait before checking again
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"❌ Training scheduler error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _run_training_phase(self, trainer: ContinuousTrainer, model):
        """Run a single training phase"""
        phase_config = self.phases[self.current_phase_index]
        
        print(f"🎯 Starting {phase_config['name']}")
        print(f"📊 Dataset mix: {phase_config['new_ratio']*100}% new data, {phase_config['coco_ratio']*100}% COCO")
        print(f"📈 Epochs: {phase_config['epochs']}")
        
        start_time = datetime.now()
        
        try:
            # Run training
            await trainer.train_model(model, self.current_phase_index, phase_config['epochs'])
            
            # Record training completion
            training_record = {
                "phase": phase_config['name'],
                "start_time": start_time,
                "end_time": datetime.now(),
                "duration": (datetime.now() - start_time).total_seconds(),
                "dataset_mix": {
                    "new_data": phase_config['new_ratio'],
                    "coco_data": phase_config['coco_ratio']
                },
                "epochs": phase_config['epochs'],
                "status": "completed"
            }
            
            self.training_history.append(training_record)
            
            # Move to next phase
            self.current_phase_index = (self.current_phase_index + 1) % len(self.phases)
            
            print(f"✅ {phase_config['name']} completed successfully")
            
        except Exception as e:
            print(f"❌ {phase_config['name']} failed: {e}")
            
            error_record = {
                "phase": phase_config['name'],
                "start_time": start_time,
                "end_time": datetime.now(),
                "error": str(e),
                "status": "failed"
            }
            
            self.training_history.append(error_record)
    
    def get_training_schedule(self) -> Dict:
        """Get current training schedule information"""
        current_phase = self.phases[self.current_phase_index]
        
        return {
            "current_phase": current_phase,
            "next_training_time": self.next_training_time.isoformat(),
            "training_interval_hours": self.training_interval.total_seconds() / 3600,
            "total_phases": len(self.phases),
            "training_history_summary": {
                "total_sessions": len(self.training_history),
                "completed": len([r for r in self.training_history if r['status'] == 'completed']),
                "failed": len([r for r in self.training_history if r['status'] == 'failed']),
                "last_training": self.training_history[-1] if self.training_history else None
            }
        }
    
    def trigger_immediate_training(self):
        """Trigger training immediately instead of waiting for schedule"""
        self.next_training_time = datetime.now()
        print("⏰ Immediate training triggered")
    
    def adjust_training_interval(self, hours: float):
        """Adjust training interval"""
        self.training_interval = timedelta(hours=hours)
        print(f"🕒 Training interval adjusted to {hours} hours")