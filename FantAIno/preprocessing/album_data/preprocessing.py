"""

Library of preprocessing modules for album data in the FantAIno project.

"""

import pandas as pd

from FantAIno.utils.data_utils import process_melondy_genre

class AlbumData_Preprocessor:
    """
    Classic preprocessor for album data.
    """

    def __call__(self, album_data_df: pd.DataFrame) -> pd.DataFrame:

        # remove bad ratings
        album_data_df = album_data_df[(album_data_df["rating"] >= -1) & (album_data_df["rating"] <= 10)]

        # 0 infill missing values
        album_data_df.fillna(0, inplace=True)

        return album_data_df