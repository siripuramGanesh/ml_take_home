import asyncio
from torch.utils.data import DataLoader
from .dataset import MixedDataset
from .model import Detector

class Trainer:
    def __init__(self, batch_size=4, num_workers=0):
        self.detector = Detector()
        self.queue = asyncio.Queue()
        self.step = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.running = False
        self._train_task = None

    async def start(self):
        self.running = True
        # Start background training loop
        self._train_task = asyncio.create_task(self._train_loop())

    async def stop(self):
        self.running = False
        if self._train_task:
            await self._train_task

    async def enqueue_mixed_batch(self):
        """Add a signal to train a batch"""
        await self.queue.put(1)

    async def _train_loop(self):
        """Background async training loop"""
        while self.running:
            # Wait for signal to train batch
            await self.queue.get()
            try:
                dataset = MixedDataset(step=self.step)
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    drop_last=True
                )

                for imgs, boxes in loader:
                    # Train in a thread to avoid blocking event loop
                    loop = asyncio.get_running_loop()
                    loss = await loop.run_in_executor(None, self.detector.train_step, imgs, boxes)
                    print(f"[Trainer] Batch trained, loss={loss}")

                # Move to next co-training step
                self.step = (self.step + 1) % len(MixedDataset.RATIOS)
                print(f"[Trainer] Completed batch, moved to step {self.step}")
            except Exception as e:
                print("[Trainer] Training error:", e)
            finally:
                self.queue.task_done()

# Global singleton trainer
trainer = Trainer()
