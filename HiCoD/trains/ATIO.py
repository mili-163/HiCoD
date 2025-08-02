"""
ATIO -- All Trains in One
"""
from .singleTask import HiCoDTrainer

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'hicod': HiCoDTrainer,
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args.model_name](args)
    
    def do_train(self, model, dataloader, return_epoch_results=False):
        """Train the model"""
        args = model.args
        trainer = self.getTrain(args)
        return trainer.do_train(model, dataloader, return_epoch_results)
    
    def do_validate(self, model, dataloader, mode="VALIDATION"):
        """Validate the model"""
        args = model.args
        trainer = self.getTrain(args)
        return trainer.do_validate(model, dataloader, mode)
