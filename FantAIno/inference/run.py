from omegaconf import DictConfig, OmegaConf
import hydra
import os

from FantAIno.constants import ROOT_DIR
from FantAIno.inference.train import train
from FantAIno.inference.predict import predict

@hydra.main(version_base=None, config_path=os.path.join(ROOT_DIR, "hydra_run_config"), config_name="OrdinalLogisticRegressionModel")
def FantAInoRun(cfg: DictConfig) -> None:
    """Run the FantAIno pipeline."""
    print(OmegaConf.to_yaml(cfg))

    if cfg.mode == "train":
        train(cfg=cfg)
    elif cfg.mode == "test":
        predict(cfg=cfg)

if __name__ == "__main__":
    FantAInoRun()
