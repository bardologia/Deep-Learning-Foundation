def __getattr__(name):
    if name == "Trainer":
        from pipelines.profile_autoencoder.training.trainer import Trainer
        return Trainer
    if name == "TrainingPipeline":
        from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline
        return TrainingPipeline
    if name == "SingleTrainRunner":
        from pipelines.profile_autoencoder.training.pipeline import SingleTrainRunner
        return SingleTrainRunner
    raise AttributeError(f"module 'profile_autoencoder.training' has no attribute '{name}'")


__all__ = [
    "Trainer",
    "TrainingPipeline",
    "SingleTrainRunner",
]
