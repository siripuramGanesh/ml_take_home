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

    async def start(self):
        self.running = True
        asyncio.create_task(self._train_loop())

    async def enqueue_mixed_batch(self):
        await self.queue.put(1)

    async def _train_loop(self):
        while self.running:
            await self.queue.get()
            dataset = MixedDataset(step=self.step)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            for imgs, boxes in loader:
                self.detector.train_step(imgs, boxes)
            self.step = (self.step + 1) % len(MixedDataset.RATIOS)
            print(f"Completed training batch, moved to stage {self.step}")

trainer = Trainer()
