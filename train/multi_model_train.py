from train.train_suite import TrainSuite

class MultiModelTrainSignLang(TrainSuite):
    def __init__(self,
                 epochs: int = 30,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 train_set_size: float = 0.8,
                 device: str = "cpu"
                 ) -> None:
        super().__init__(epochs=epochs, 
                         lr=lr,
                         batch_size=batch_size,
                         train_set_size=train_set_size,
                         device=device)

        def init_data(self, 
                      image_dir: os.PathLike | str,
                      label_df: pd.DataFrame,
                      augment_data: bool = True,
                      sample_ratio: float = 1.0,
                      threshold: int = -1):
            pass


    def train(self, vocal=False):
        pass


    def evaluate(self, vocal=False):
        pass


    def _gen_data_info(self):
        pass


    def save_model(self, dir: str | os.PathLike, vocal=False):
        pass


    def save_info(self, info_dir: str | os.PathLike):
        pass
