from omegaconf import DictConfig, OmegaConf
import hydra
import os

from inference.train import train
from inference.predict import predict

import FantAIno

root_dir = os.path.dirname(os.path.abspath(FantAIno.__path__[0]))

@hydra.main(version_base=None, config_path=os.path.join(root_dir, "hydra_run_config"), config_name="RadiusNeighborsRegressionModel")
def FantAInoRun(cfg: DictConfig) -> None:
    """Run the FantAIno pipeline."""
    print(OmegaConf.to_yaml(cfg))

    if cfg.mode == "train":
        train(cfg=cfg)
    elif cfg.mode in ("predict", "evaluate"):
        predict(cfg=cfg)

if __name__ == "__main__":
    FantAInoRun()
