"""

Library of preprocessing modules for album data in the FantAIno project.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

        # normalize the data
        scaler = StandardScaler()
        columns_to_scale = [column_name for column_name in album_data_df.select_dtypes(include=[np.number]).columns if column_name != "rating"]
        album_data_df[columns_to_scale] = scaler.fit_transform(album_data_df[columns_to_scale])

        return album_data_df
