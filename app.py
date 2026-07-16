import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import random
import urllib.parse
# ===================== ❤️ RECOMMENDATION PAGE (ULTIMATE PRO ML DASHBOARD) =====================
from sklearn.neighbors import NearestNeighbors

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("spotify (2).csv")

# ================= LOAD KNN MODEL =================

knn = joblib.load("spotify_knn.pkl")
scaler = joblib.load("spotify_scaler.pkl")
song_data = joblib.load("spotify_data.pkl")
#---spotify image-----
st.sidebar.image(
"https://upload.wikimedia.org/wikipedia/commons/8/84/Spotify_icon.svg",
width=125
)
#--sidebar title----
with st.sidebar:
    st.title("🎵 Spotify AI")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🎵 Songs",
            "🎤 Artists",
            "🏷 Genres",
            "📊 Analytics",
            "🏆 Top Charts",
            "🤖 AI Prediction",
            "❤️ Recommendation"
        ]
    )

st.sidebar.markdown("---")

st.sidebar.success("MCA Final Project")

st.sidebar.write("Python + Streamlit + ML")

# ---------------- HOME ---------------- #

if page == "🏠 Home":

    st.markdown("""
    <div style="
    background:linear-gradient(90deg,#000000,#1DB954);
    padding:30px;
    border-radius:20px;
    ">
    <h1 style="color:pink;">
    🎵 Spotify AI Dashboard
    </h1>

    <h4 style="color: red;">
    Machine Learning Based Music Recommendation System
    </h4>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🎵 Songs", len(df))

    with c2:
        st.metric("🎤 Artists", df["Artist"].nunique())

    with c3:
        st.metric("💿 Albums", df["Album"].nunique())

    with c4:
        st.metric("🎼 Genres", df["Genre"].nunique())

    st.markdown("---")

    st.subheader("🔥 Trending Songs")

    top = df.sort_values(
        "Popularity",
        ascending=False
    ).head(10)

    st.dataframe(top, use_container_width=True)

    

# ================= SONG PAGE =================

elif page == "🎵 Songs":

    st.title("🎵 Songs Library")
    st.write("Search and Explore Spotify Songs")

    # Search Song
    search = st.text_input("🔍 Search Song")

    # Search Filter
    if search:
        song_df = df[
            df["Track_Name"].str.contains(
                search,
                case=False,
                na=False
            )
        ]
    else:
        song_df = df.copy()

    # Empty Check
    if song_df.empty:
        st.warning("❌ No Song Found. Try Another Search.")

    else:

        # Select Song
        song = st.selectbox(
            "🎵 Choose Song",
            song_df["Track_Name"].unique(),
            key="song_select"
        )

        song_data = song_df[
            song_df["Track_Name"] == song
        ].iloc[0]

        st.markdown("---")

        # Song Details
        col1, col2 = st.columns([1,2])

        with col1:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/8/84/Spotify_icon.svg",
                width=180
            )

        with col2:

            st.info(f"""
🎵 **Song :** {song_data['Track_Name']}

🎤 **Artist :** {song_data['Artist']}

💿 **Album :** {song_data['Album']}

🎼 **Genre :** {song_data['Genre']}

⭐ **Popularity :** {song_data['Popularity']}
""")

        st.markdown("---")

        # Song Statistics
        st.subheader("📊 Song Statistics")

        if "Popularity" in df.columns:
            st.write("⭐ Popularity")
            st.progress(min(float(song_data["Popularity"])  /100, 1.0))

        if "Energy" in df.columns:
            st.write("⚡ Energy")
            st.progress(min(float(song_data["Energy"]), 1.0))

        if "Danceability" in df.columns:
            st.write("💃 Danceability")
            st.progress(min(float(song_data["Danceability"]), 1.1))

        if "Valence" in df.columns:
            st.write("😊 Valence")
            st.progress(min(float(song_data["Valence"]), 1.0))

        st.markdown("---")

        # Overall Song Quality
        quality = (
            float(song_data["Popularit" \
            "y"]) +
            float(song_data["Energy"]) * 100 +
            float(song_data["Danceability"]) * 100
        ) / 3

        st.metric("⭐ Overall Song Quality", f"{quality:.1f}/100")

        st.markdown("---")

        # Music Buttons
        spotify = (
            "https://open.spotify.com/search/"
            + song.replace(" ", "%20")
        )

        youtube = (
            "https://www.youtube.com/results?search_query="
            + song.replace(" ", "+")
        )

        c1, c2 = st.columns(2)

        with c1:
            st.link_button(
                "🎧 Play on Spotify",
                spotify,
            
            )

        with c2:
            st.link_button(
                "▶ Watch on YouTube",
                youtube,
                
            )

        st.success("✅ Song Loaded Successfully 🎵")
    # ================= ARTISTS PAGE =================

elif page == "🎤 Artists":

    st.title("🎤 Artists")

    artist = st.selectbox(
        "Choose Artist",
        sorted(df["Artist"].dropna().unique()),
        key="artist_page"
    )

    artist_df = df[df["Artist"] == artist]


    st.markdown("---")


    c1, c2, c3 = st.columns(3)


    with c1:
        st.metric(
            "🎵 Total Songs",
            len(artist_df)
        )


    with c2:
        st.metric(
            "⭐ Avg Popularity",
            round(
                artist_df["Popularity"].mean(),
                1
            )
        )


    with c3:
        st.metric(
            "💿 Albums",
            artist_df["Album"].nunique()
        )


    st.markdown("---")


    st.subheader("🔥 Top Songs")


    top = artist_df.sort_values(
        "Popularity",
        ascending=False
    )


    st.dataframe(
        top[
            [
                "Track_Name",
                "Album",
                "Popularity"
            ]
        ],
        use_container_width=True
    )


    st.markdown("---")


    st.subheader("📈 Song Popularity")


    fig, ax = plt.subplots(figsize=(8,4))


    ax.bar(
        top["Track_Name"],
        top["Popularity"]
    )


    plt.xticks(rotation=45)

    st.pyplot(fig)


    st.markdown("---")


    st.subheader("🎧 Listen on Spotify")


    spotify = (
        "https://open.spotify.com/search/"
        + artist.replace(" ","%20")
    )


    st.link_button(
        "🎵 Open Artist",
        spotify
    )
    # ================= GENRES PAGE =================

elif page == "🏷 Genres":

    st.title("🏷 Music Genres")

    genre = st.selectbox(
        "Choose Genre",
        sorted(df["Genre"].dropna().unique()),
    )

    genre_df = df[df["Genre"] == genre]

    st.markdown("---")


    c1, c2, c3 = st.columns(3)


    with c1:
        st.metric(
            "🎵 Songs",
            len(genre_df)
        )


    with c2:
        st.metric(
            "⭐ Avg Popularity",
            round(
                genre_df["Popularity"].mean(),
                1
            )
        )


    with c3:
        st.metric(
            "🎤 Artists",
            genre_df["Artist"].nunique()
        )


    st.markdown("---")


    st.subheader("🔥 Top Songs")


    top = genre_df.sort_values(
        "Popularity",
        ascending=False
    )


    st.dataframe(
        top[
            [
                "Track_Name",
                "Artist",
                "Popularity"
            ]
        ],
        use_container_width=True
    )


    st.markdown("---")


    st.subheader("📊 Genre Popularity")


    fig = px.bar(
        top.head(10),
        x="Track_Name",
        y="Popularity",
        color="Popularity",
        title="Top Songs in Genre"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.markdown("---")


    st.subheader("🥧 Genre Distribution")


    pie = px.pie(
        df,
        names="Genre",
        title="Overall Genre Distribution"
    )


    st.plotly_chart(
        pie,
        use_container_width=True
    )


    st.markdown("---")


    st.success(
        f"🎼 {genre} contains {len(genre_df)} songs."
    )



# ================= ANALYTICS PAGE =====
elif page == "📊 Analytics":

    st.title("🎧 Spotify analytics")


    st.markdown("---")

    # Metrics

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🎵 Songs", len(df))
    c2.metric("🎤 Artists", df["Artist"].nunique())
    c3.metric("💿 Albums", df["Album"].nunique())
    c4.metric("🎼 Genres", df["Genre"].nunique())


    st.markdown("---")


    # Artist Images

    st.subheader("🌟 Featured Artists")

    a,b = st.columns(2)

    with a:

        st.image(
            "https://images.unsplash.com/photo-1516280440614-37939bbacd81",
            use_container_width=True
        )

        st.markdown("### 🎤 Music Artist")

        st.link_button(
            "🎵 Open Spotify",
            "https://open.spotify.com"
        )


    with b:

        st.image(
            "https://images.unsplash.com/photo-1524368535928-5b5e00ddc76b",
            use_container_width=True
        )

        st.markdown("### 🎙 Music Podcast")

        st.link_button(
            "🎧 Listen Podcast",
            "https://open.spotify.com/genre/podcasts-web"
        )


    st.markdown("---")


    # Charts

    st.subheader("📊 Popularity Analysis")

    fig = px.histogram(
        df,
        x="Popularity",
        nbins=20,
        color_discrete_sequence=["#1DB954"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader("🎼 Genre Distribution")

    genre = df["Genre"].value_counts().reset_index()

    genre.columns = ["Genre","Songs"]

    fig = px.pie(
        genre,
        names="Genre",
        values="Songs",
        hole=0.5
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.markdown("---")


    # Music Links

    x,y,z = st.columns(3)

    with x:
        st.link_button(
            "🟢 Spotify",
            "https://open.spotify.com"
        )

    with y:
        st.link_button(
            "▶️ YouTube Music",
            "https://music.youtube.com"
        )

    with z:
        st.link_button(
            "🎙 Podcasts",
            "https://open.spotify.com/genre/podcasts-web"
        )


    st.success("✅ Analytics Generated Successfully")
    # ================= TOP CHARTS PAGE =================

elif page == "🏆 Top Charts":

    st.title("🏆 Spotify Top Charts")
    st.caption("🔥 Explore Trending Songs & Artists")

    st.markdown("---")

    # Genre Filter

    genre_select = st.selectbox(
        "🎼 Select Genre",
        ["All"] + sorted(df["Genre"].unique())
    )

    if genre_select != "All":
        chart_df = df[df["Genre"] == genre_select]
    else:
        chart_df = df.copy()


    top10 = chart_df.sort_values(
        "Popularity",
        ascending=False
    ).head(10)


    st.subheader("🔥 Global Top 10 Songs")


    for rank, (i,row) in enumerate(top10.iterrows(),1):

        c1,c2,c3 = st.columns([1,6,2])


        with c1:
            st.markdown(f"## #{rank}")


        with c2:

            st.markdown(
            f"""
### 🎵 {row['Track_Name']}

🎤 {row['Artist']}  
💿 {row['Album']}  
🎼 {row['Genre']}
"""
            )


        with c3:

            st.metric(
                "⭐ Score",
                row["Popularity"]
            )


        link = (
            "https://www.youtube.com/results?search_query="
            + row["Track_Name"].replace(" ","+")
        )


        st.link_button(
            "▶ Play Song",
            link,
            key=f"song_{i}"
        )


        st.markdown("---")


    # Top Artist Chart

    st.subheader("👑 Top Artists")


    artist = (
        chart_df.groupby("Artist")["Popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )


    fig = px.bar(
        artist,
        x="Artist",
        y="Popularity",
        color="Popularity",
        title="Most Popular Artists"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Trending Song Card

    st.markdown("---")

    best = top10.iloc[0]


    st.success(
    f"""
🔥 Today's #1 Trending Song

🎵 {best['Track_Name']}

🎤 {best['Artist']}

⭐ Popularity : {best['Popularity']}
"""
    )
    # ================= AI PREDICTION =================
elif page == "🤖 AI Prediction":

    st.title("🚀 AI Music Time Machine")

    st.write(
        "Predict your song's future with AI simulation 🎵"
    )

    st.divider()


    song = st.text_input(
        "🎧 Enter Song Name"
    )


    if st.button("🚀 Travel To Future"):


        with st.status(
            "🧠 AI Future Simulation Running...",
            expanded=True
        ) as status:


            st.write("🎵 Analyzing Rhythm...")
            st.write("🔥 Checking Viral Pattern...")
            st.write("🌍 Finding Future Audience...")
            st.write("🏆 Generating Music Award...")


            status.update(
                label="✅ Future Prediction Completed",
                state="complete"
            )


        hit = random.randint(70,100)

        listeners = random.randint(
            1,10
        )


        awards = random.choice(
            [
                "🏆 Global Trending Track",
                "🥇 Viral Music Award",
                "🌟 Future Superstar Song",
                "🎧 Listener Favorite"
            ]
        )


        artist = random.choice(
            [
                "🔥 Rising Artist",
                "🌎 Global Performer",
                "🎤 Music Influencer",
                "🎵 Independent Star"
            ]
        )


        growth = random.randint(
            120,250
        )


        st.markdown(
        f"""
        <div style="
        background:#191414;
        padding:30px;
        border-radius:20px;
        color:white;
        text-align:center;
        ">

        <h1>🚀 Future Music Report</h1>

        <h2>🎵 {song}</h2>

        <h2>{awards}</h2>

        <h3>🔥 Viral Probability : {hit}%</h3>

        <h3>🌍 Future Listeners : {listeners} Million+</h3>

        <h3>🎤 Artist Status : {artist}</h3>

        <h3>📈 Growth Prediction : +{growth}%</h3>

        </div>
        """,
        unsafe_allow_html=True
        )


        if hit >=85:

            st.balloons()

            st.success(
                "🎉 AI predicts a future hit!"
            )

        else:

            st.warning(
                "📈 Song has growth potential"

            )


    # ================= SEARCH SONG =================

    st.markdown("---")
    st.subheader("🔍 Search Any Song")

    song_name = st.text_input("Enter Song Name")
    artist_name = st.text_input("Artist Name (Optional)")

    col1, col2, col3 = st.columns(3)

    with col1:
        play = st.button("▶ Play Song")

    with col2:
        spotify = st.button("🟢 Spotify")

    with col3:
        youtube = st.button("📺 YouTube")

    if spotify:
        if song_name:
            query = urllib.parse.quote(song_name)
            st.link_button(
                "🎵 Open in Spotify",
                f"https://open.spotify.com/search/{query}"
            )
        else:
            st.warning("Please enter a song name.")

    if youtube:
        if song_name:
            query = urllib.parse.quote(song_name)
            st.link_button(
                "▶ Watch on YouTube",
                f"https://www.youtube.com/results?search_query={query}"
            )
        else:
            st.warning("Please enter a song name.")

    if play:
        if song_name:
            query = urllib.parse.quote(song_name)
            st.success(f"🎵 Searching for {song_name}")
            st.link_button(
                "▶ Play Now",
                f"https://www.youtube.com/results?search_query={query}"
            )
        else:
            st.warning("Please enter a song name.")

    # ================= SURPRISE SONG =================

    songs = [
        "Believer",
        "Shape of You",
        "Perfect",
        "Levitating",
        "Blinding Lights",
        "Senorita",
        "Closer",
        "Peaches"
    ]

    if st.button("🎁 Surprise Me"):
        surprise = random.choice(songs)
        st.success(f"🎵 Today's Surprise Song: {surprise}")

        query = urllib.parse.quote(surprise)

        st.link_button(
            "▶ Play Surprise Song",
            f"https://www.youtube.com/results?search_query={query}"
        )

    # ================= MUSIC QUIZ =================

    st.markdown("---")
    st.subheader("🎵 Music Quiz")

    answer = st.radio(
        "Who sang 'Shape of You'?",
        [
            "Justin Bieber",
            "Ed Sheeran",
            "Arijit Singh",
            "Bruno Mars"
        ]
    )

    if st.button("Submit Answer"):
        if answer == "Ed Sheeran":
            st.success("🎉 Correct Answer!")
            st.balloons()
        else:
            st.error("❌ Wrong Answer")

    # ================= LUCKY SONG =================

    st.markdown("---")
    st.subheader("🍀 Today's Lucky Song")

    if st.button("🍀 Reveal Lucky Song"):
        lucky = random.choice(songs)

        st.success(f"🎵 Lucky Song: {lucky}")

        query = urllib.parse.quote(lucky)

        st.link_button(
            "▶ Listen Lucky Song",
            f"https://www.youtube.com/results?search_query={query}"
        )

    # ===== Premium Button CSS =====
    st.markdown("""
    <style>

    /* All Buttons */
    div.stButton > button {
        width:100%;
        height:58px;
        border:none;
        border-radius:18px;
        color:white;
        font-size:17px;
        font-weight:700;
        background:linear-gradient(135deg,#1DB954,#159c46);
        box-shadow:0 6px 15px rgba(29,185,84,.35);
        transition:all .35s ease;
    }

    /* Hover */
    div.stButton > button:hover{
        transform:translateY(-4px) scale(1.03);
        box-shadow:0 10px 25px rgba(29,185,84,.6);
        letter-spacing:1px;
    }

    /* Click */
    div.stButton > button:active{
        transform:scale(.96);
    }

    /* Text Input */
    div[data-baseweb="input"] > div{
        border-radius:15px;
        border:2px solid #1DB954;
        box-shadow:0 0 10px rgba(29,185,84,.15);
    }

    /* Radio Box */
    div[role="radiogroup"]{
        padding:12px;
        border-radius:15px;
        border:1px solid #1DB954;
        background:rgba(29,185,84,.06);
    }

    /* Success Box */
    div[data-testid="stAlert"]{
        border-radius:18px;
    }

    </style>
    """, unsafe_allow_html=True)


# ===================== ❤️ RECOMMENDATION PAGE (MCA STANDARDIZED ML INTERFACE) =====================
elif page == "❤️ Recommendation":

    st.title("🧬 ML Recommendation Engine")
    st.write("This module calculates the multi-dimensional distance between audio feature vectors to isolate similar data points.")

    st.markdown("---")

    # 📊 SYSTEM METRICS
    st.markdown("### 📊 Dataset & Model Specifications")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Dataset Instances (N)", value=len(df))
    with col2:
        st.metric(label="Feature Dimensions (D)", value="5 Audio Vectors")
    with col3:
        st.metric(label="Distance Algorithm", value="Minkowski / Euclidean")

    # 🛠️ DATA PREPARATION & CLEANING
    feature_cols = ["Popularity", "Danceability", "Energy", "Valence", "Tempo"]
    # Drop duplicates and reset index to completely prevent IndexError
    ml_df = df.dropna(subset=feature_cols).drop_duplicates(subset=["Track_Name"]).reset_index(drop=True)

    st.markdown("---")

    # 🎛️ MODEL HYPERPARAMETERS
    st.markdown("### 🎛️ Model Hyperparameters")
    k_neighbors = st.slider("Select K-Value (Number of Neighbors)", min_value=3, max_value=10, value=5)
    metric_choice = st.selectbox("Select Distance Metric", ["euclidean", "manhattan"])

    # INPUT QUERY ANCHOR
    selected_song = st.selectbox(
        "🎯 Select Input Anchor Track (Query Vector):", 
        options=ml_df["Track_Name"].values
    )

    if st.button("🚀 Execute KNN Vector Search", type="primary", use_container_width=True):
        try:
            # 1. Fetch Target Row Index from Clean Dataframe
            target_index = ml_df[ml_df["Track_Name"] == selected_song].index[0]
            
            # 2. Extract and Transform Feature Array
            X_data = ml_df[feature_cols].values
            X_scaled = scaler.transform(X_data)  # Standard Scaling using your pre-loaded scaler
            
            # 3. Dynamic Model Compilation
            knn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, metric=metric_choice)
            knn_model.fit(X_scaled)
            
            # 4. Compute Vector Distance Queries
            query_vector = X_scaled[target_index].reshape(1, -1)
            distances, indices = knn_model.kneighbors(query_vector)
            
            st.success("🎯 Model execution complete. Similarity vector space convergence localized successfully.")
            
            # SPLIT INTERFACE INTO RESULTS & GRAPH
            left_col, right_col = st.columns([3, 2])
            
            with left_col:
                st.markdown("### 📋 KNN Prediction Output Matrix")
                
                # Loop starts from index 1 to skip the selected song itself
                for rank, i in enumerate(range(1, k_neighbors + 1), start=1):
                    node_idx = indices[0][i]
                    vector_gap = distances[0][i]
                    
                    matched_track = ml_df.iloc[node_idx]
                    similarity_percent = round((1 / (1 + vector_gap)) * 100, 2)
                    
                    # Search URLs
                    spot_link = f"https://open.spotify.com/search/{matched_track['Track_Name'].replace(' ', '%20')}"
                    yt_link = f"https://www.youtube.com/results?search_query={matched_track['Track_Name'].replace(' ', '+')}"
                    
                    # Native Clean Container Box
                    with st.container(border=True):
                        st.markdown(f"##### **#{rank} Match:** {matched_track['Track_Name']}")
                        st.write(f"**Artist:** {matched_track['Artist']} | **Genre:** {matched_track['Genre']}")
                        st.write(f"📐 Vector Distance ($\Delta$): `{round(vector_gap, 4)}` | 🧠 Confidence: `{similarity_percent}%`")
                        
                        # Streaming Redirect Buttons
                        btn1, btn2 = st.columns(2)
                        with btn1:
                            st.link_button("🎧 Open Spotify Asset", spot_link, use_container_width=True, key=f"sp_mca_{i}")
                        with btn2:
                            st.link_button("▶ Stream YouTube Node", yt_link, use_container_width=True, key=f"yt_mca_{i}")

            with right_col:
                st.markdown("### 📊 Vector Distance Graph")
                
                output_names = [ml_df.iloc[idx]['Track_Name'] for idx in indices[0][1:]]
                output_distances = distances[0][1:]
                
                plot_data = pd.DataFrame({
                    'Track Name': output_names,
                    'Distance Metric Value': output_distances
                }).sort_values(by='Distance Metric Value', ascending=False)
                
                st.bar_chart(data=plot_data, x='Track Name', y='Distance Metric Value', color="#1DB954")
                st.caption("Lower statistical distance implies tighter data-point proximity in Euclidean space.")
                
                # SHOW TARGET VECTORS
                st.markdown("---")
                st.markdown("### 📐 Target Node Feature Array")
                anchor_data = ml_df.iloc[target_index]
                
                st.json({
                    "Popularity": float(anchor_data['Popularity']),
                    "Danceability": float(anchor_data['Danceability']),
                    "Energy": float(anchor_data['Energy']),
                    "Valence": float(anchor_data['Valence']),
                    "Tempo (BPM)": float(anchor_data['Tempo'])
                })

        except Exception as error_msg:
            st.error(f"Execution Error: {str(error_msg)}")
            st.info("Verify that 'scaler' is globally initialized at the top level of the codebase script.")
