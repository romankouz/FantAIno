from FantAIno.data.melondy_base import MelondyBaseDataset

class MelondyFragmentedMultimodalDataset(MelondyBaseDataset):
    """
    FantAIno dataset designed for classical machine learning algorithms.

    This dataset is designed for merging outputs from various models, some that can work
    with images and raw text with others that work on organized, tabular data.
    """

    def __init__(self):
        super().__init__()
        
        self.album_data_df = self.retrieve_tabular_album_data()
        self.lyrics_embeddings_df = self.retrieve_lyrics_embeddings()