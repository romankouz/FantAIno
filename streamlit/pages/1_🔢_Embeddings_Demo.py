import boto3
from dotenv import load_dotenv
load_dotenv()
import json
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import os
import streamlit as st

from FantAIno.constants import S3_GENERAL_PURPOSE_BUCKET_NAME
from FantAIno.utils.genius_utils import get_album_lyrics
from FantAIno.utils.data_utils import sanitize_filename

client = OpenAI()
s3_client = boto3.client("s3")

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1, 4, 1])

with col2:

    st.write("# Album Embeddings Demo")

    st.markdown(
        """
        
        <div style="border-left: 4px solid #f59e0b; padding-left: 10px;">
            ⚠️❗ This demo computes values on the fly. Therefore, it may load in chunks. Please allow a few seconds for the full page to load.
        </div>
        <br>
        
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '''

        This demo indicates how we manage to capture the lyrics of an album
        and have them be a usable predictor for our FantAIno models.
        '''
    )

    st.markdown(
        '''
        ## OpenAI Embeddings

        We can extract a numerical representation of text (an embedding) by calling the OpenAI embeddings model API.
        '''
    )

    with st.echo():
        response_small = client.embeddings.create(
            input="Your text string goes here",
            model="text-embedding-3-small"
        )

    st.write(
        response_small.data[0].embedding
    )

    st.write(
        "You can also get some more signal by using the larger embedding model from OpenAI."
    )

    with st.echo():
        response_large = client.embeddings.create(
            input="Your text string goes here",
            model="text-embedding-3-large"
        )

    st.write(
    '''Note: Anthony Fantano has his finger on the pulse 
    of current events and politics. This means [the following 
    note from OpenAI](https://developers.openai.com/api/docs/guides/embeddings#do-v3-embedding-models-know-about-recent-events) highlights a possible limitation of over-relying 
    on these embeddings:
    
    "the text-embedding-3-large and text-embedding-3-small models lack 
    knowledge of events that occurred after September 2021."
    '''
    )

    st.markdown(
        '''
        ## Getting Embeddings from Lyrics

        Leveraging the Genius API, we can get the lyrics for an album.
        We utilize an internal utility function to accomplish this.
        '''
    )

    st.code(
        'logic_under_pressure = get_album_lyrics("Logic", "Under Pressure")'
    )

    st.markdown(
        '''

        To avoid 403 errors from the Genius API fetching lyrics for this demo, we will just load these lyrics from our
        S3 bucket to which we uploaded these lyrics already.
        '''
    )

    with st.echo():

        artist, album = "Logic", "Under Pressure"
        lyrics_filename = sanitize_filename(f"{artist}___{album}.jsonl")
        lyrics_response = s3_client.get_object(
            Bucket=S3_GENERAL_PURPOSE_BUCKET_NAME,
            Key=f"lyrics\\{lyrics_filename}"
        )
        logic_under_pressure = json.load(lyrics_response['Body'])

    with st.expander("Show Logic 'Under Pressure' Lyrics JSON", expanded=False):
        st.write(logic_under_pressure)

    st.markdown(
        '''
        Having retrieved the lyrics, we can now iterate throught the lyrics of each song and get the embeddings for each song.
        We store each embedding in a row of a numpy array (matrix).
        '''
    )

    with st.echo():
        under_pressure_embeddings = np.empty((len(logic_under_pressure['tracks']), len(response_small.data[0].embedding)))

        for i, (track, lyrics) in enumerate(logic_under_pressure['tracks'].items()):
            response_embedding = client.embeddings.create(
                input=lyrics,
                model="text-embedding-3-small"
            )
            under_pressure_embeddings[i] = response_embedding.data[0].embedding

        under_pressure_embeddings

    st.write(
        """Now we have the embeddings for each song! Let's visualize each of Logic's Under 
        Pressure song embeddings on a heatmap."""
    )

    fig, ax = plt.subplots(figsize=(25, 12))
    im = ax.imshow(under_pressure_embeddings, aspect='auto')
    ax.set_title("Under Pressure - Logic: Song Embeddings", fontsize=24)
    ax.set_yticks(np.arange(under_pressure_embeddings.shape[0]))
    ax.set_yticklabels(list(logic_under_pressure['tracks'].keys()), fontsize=12)

    # Insert a horizontal black dashed line as a divider between each song
    for y in range(1, under_pressure_embeddings.shape[0]):
        ax.axhline(y - 0.5, color='black', linestyle='--', linewidth=4)

    plt.colorbar(im, label='Embedding Magnitude')

    st.pyplot(fig)

    st.write(
        """
        It seems like a lot of the songs share similar positive or negative symbols along certain 
        embedding dimensions. We could create an "album-level embedding" by taking a row sum over 
        all song embeddings. This sum could contain relevant signal about how much this specific embedding 
        dimension is represented by the album.
        """
    )

    fig, ax = plt.subplots(figsize=(25, 5))
    im = ax.imshow(under_pressure_embeddings.sum(axis=0, keepdims=True), aspect='auto')
    ax.set_title("Under Pressure - Logic: Album Embeddings", fontsize=24)
    plt.colorbar(im, label='Embedding Magnitude')

    st.pyplot(fig)

    st.write(
        '''
        Voila! We have embeddings as predictors for an album!
        '''
    )

    st.markdown(
        '''
        ## Cosine Similarity Between Albums

        One interesting thing we can examine is if albums have similar embedding structures. 
        Let's compare "Under Pressure" by Logic to two albums. The first will be Logic's 
        "No Pressure", a follow up album that plays as the counterpart to Logic's "Under Pressure" 
        6 years prior. The next album will be Madonna's "Ray of Light", which to my knowledge is 
        fairly different from Logic's music.

        I hypothesize that the cosine similarity between the two Logic albums will be much higher than
        the cosine similarity between Under Pressure and Ray of Light. This would provide some 
        initial evidence that the lyrics embeddings are capturing relevant lyrical signal
        to use as predictors for our FantAIno models.
        ''',
    )

    with st.echo():

        artist, album = "Logic", "No Pressure"
        lyrics_filename = sanitize_filename(f"{artist}___{album}.jsonl")
        lyrics_response = s3_client.get_object(
            Bucket=S3_GENERAL_PURPOSE_BUCKET_NAME, 
            Key=f"lyrics\\{lyrics_filename}"
        )
        logic_no_pressure = json.load(lyrics_response['Body'])

        artist, album = "Madonna", "Ray of Light"
        lyrics_filename = sanitize_filename(f"{artist}___{album}.jsonl")
        lyrics_response = s3_client.get_object(
            Bucket=S3_GENERAL_PURPOSE_BUCKET_NAME, 
            Key=f"lyrics\\{lyrics_filename}"
        )
        madonna_ray_of_light = json.load(lyrics_response['Body'])
   

        no_pressure_embeddings = np.empty((len(logic_no_pressure['tracks']), len(response_small.data[0].embedding)))

        for i, (track, lyrics) in enumerate(logic_no_pressure['tracks'].items()):
            response_embedding = client.embeddings.create(
                input=lyrics,
                model="text-embedding-3-small"
            )
            no_pressure_embeddings[i] = response_embedding.data[0].embedding

        ray_of_light_embeddings = np.empty((len(madonna_ray_of_light['tracks']), len(response_small.data[0].embedding)))

        for i, (track, lyrics) in enumerate(madonna_ray_of_light['tracks'].items()):
            response_embedding = client.embeddings.create(
                input=lyrics,
                model="text-embedding-3-small"
            )
            ray_of_light_embeddings[i] = response_embedding.data[0].embedding

    st.markdown(
        r'''
        If we normalize each vector to be of unit length ($||A|| = 1$ and $||B|| = 1$),
        the cosine similarity boils down to just the dot product:
        '''
    )

    st.markdown(
        r'$$\cos(A,B) = \frac{A \cdot B}{||A||||B||} = A \cdot B$$',
        text_alignment="center"
    )

    with st.echo():
        under_pressure_embedding_total = under_pressure_embeddings.sum(axis=0, keepdims=True)
        under_pressure_embedding_total_normalized = under_pressure_embedding_total / np.linalg.norm(under_pressure_embedding_total)
        no_pressure_embedding_total = no_pressure_embeddings.sum(axis=0, keepdims=True)
        no_pressure_embedding_total_normalized = no_pressure_embedding_total / np.linalg.norm(no_pressure_embedding_total)
        ray_of_light_embedding_total = ray_of_light_embeddings.sum(axis=0, keepdims=True)
        ray_of_light_embedding_total_normalized = ray_of_light_embedding_total / np.linalg.norm(ray_of_light_embedding_total)

    with st.echo():
        # Logic Cosine Similarity
        # transpose needed to ensure consistent dimensions
        np.dot(under_pressure_embedding_total_normalized, no_pressure_embedding_total_normalized.T).item()
        
    st.write(
        np.dot(under_pressure_embedding_total_normalized, no_pressure_embedding_total_normalized.T).item()
    )

    with st.echo():
        # Logic v. Madonna cosine similarity
        np.dot(under_pressure_embedding_total_normalized, ray_of_light_embedding_total_normalized.T).item()

    st.write(
        np.dot(under_pressure_embedding_total_normalized, ray_of_light_embedding_total_normalized.T).item()
    )

    st.markdown(
        '''
        As expected, Logic's "No Pressure" has a MUCH higher cosine similarity with "Under Pressure" than "Ray of Light" had with "Under Pressure".
        However, because both 
        lyrics from Logic and Madonna are generally associated with something musical, there is likely 
        some overlap with 
        just music in general between the two. Even still, this is still indicative that there is 
        some relevant signal from the lyrics embeddings to train our FantAIno models!
        '''
    )
