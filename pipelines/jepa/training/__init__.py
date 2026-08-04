def __getattr__(name):
    if name == "Trainer":
        from pipelines.jepa.training.trainer import Trainer
        return Trainer
    if name == "JepaModule":
        from pipelines.jepa.training.trainer import JepaModule
        return JepaModule
    if name == "TrainingPipeline":
        from pipelines.jepa.training.pipeline import TrainingPipeline
        return TrainingPipeline
    if name == "SingleTrainRunner":
        from pipelines.jepa.training.pipeline import SingleTrainRunner
        return SingleTrainRunner
    raise AttributeError(f"module 'jepa.training' has no attribute '{name}'")


__all__ = [
    "Trainer",
    "JepaModule",
    "TrainingPipeline",
    "SingleTrainRunner",
]
