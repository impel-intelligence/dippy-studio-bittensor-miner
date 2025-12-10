from tqdm import tqdm
import time


class ToolkitProgressBar(tqdm):
    """Simple wrapper around tqdm that adds pause/unpause methods"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paused = False

    def pause(self):
        """Pause the progress bar (no-op for now)"""
        self.paused = True

    def unpause(self):
        """Unpause the progress bar (no-op for now)"""
        self.paused = False

    def update(self, *args, **kwargs):
        """Update progress bar only if not paused"""
        if not self.paused:
            super().update(*args, **kwargs)
