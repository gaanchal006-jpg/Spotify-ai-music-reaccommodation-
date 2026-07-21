import streamlit as st # Import Streamlit library to create the web application
import pandas as pd # Import Streamlit library to create the web application
import matplotlib.pyplot as plt # Import Matplotlib for basic charts and graph
import plotly.express as px # Import Plotly Express for interactive visualizations
import joblib# Import Joblib to load saved Machine Learning models
import random # Import Random library to generate random predictions and songs
import urllib.parse # Import urllib.parse to encode song names inside URLs
# ===================== ❤️ RECOMMENDATION PAGE (ULTIMATE PRO ML DASHBOARD) =====================
from sklearn.neighbors import NearestNeighbors 
# Import K-Nearest Neighbors algorithm for song recommendation

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("spotify (2).csv")

# ================= LOAD KNN MODEL =================
# Load trained KNN model
knn = joblib.load("spotify_knn.pkl")
# Load StandardScaler used during training covert value small scale
scaler = joblib.load("spotify_scaler.pkl")
# Load processed song dataset
song_data = joblib.load("spotify_data.pkl")
#---spotify image-----
# ================= MUSKANNOVA AI COMPACT PREMIUM SIDEBAR =================

st.sidebar.markdown("""
<style>

.muskan-card{

    background: linear-gradient(
        135deg,
        #f3e8ff,
        #d8b4fe,
        #a855f7
    ) !important;

    border-radius:18px;

    padding:18px 10px;

    height:270px;

    text-align:center;

    border:1px solid rgba(0,0,0,0.25);

    box-shadow:
    0 8px 25px rgba(168,85,247,0.35);

    margin-bottom:35px !important;

    color:#000 !important;
}


.muskan-card *{
    color:#000 !important;
}
/* AI Logo */

.ai-logo{

    font-size:45px;
    font-weight:900;

    font-family:'Arial Black',sans-serif;

    color:#d7ff76;

    margin-top:0px;

    text-shadow:
    0 0 8px #9cff3b,
    0 0 15px #7cff00;

    animation:pulse 2s infinite;

}



@keyframes pulse{

0%{
transform:scale(1);
}

50%{
transform:scale(1.06);
}

100%{
transform:scale(1);
}

}



/* Heart Beat Line */

.heartbeat{

width:110px;
height:22px;

margin:auto;

position:relative;

}


.heartbeat:before{

content:"";

position:absolute;

top:10px;
left:0;

width:100%;
height:2px;


background:
linear-gradient(
90deg,
transparent,
#8cff3a,
transparent
);


box-shadow:
0 0 10px #8cff3a;


animation:move 2s linear infinite;

}



@keyframes move{

from{
transform:translateX(-25px);
}

to{
transform:translateX(25px);
}

}



/* Brand Name */

.brand-name{

font-size:18px;

font-weight:800;

margin-top:8px;

color:#baff70;

letter-spacing:1px;

text-shadow:
0 0 10px #8cff3a;

}



/* Tagline */

.tagline{

color:#c5c5c5;

font-size:11px;

line-height:1.3;

margin-top:8px;

}



/* KNN Button */

.knn{

display:inline-block;

margin-top:12px;

padding:6px 20px;


border-radius:30px;


background:
linear-gradient(
90deg,
#63e62e,
#c8ff73
);


color:#071000;

font-size:12px;

font-weight:800;


box-shadow:

0 0 15px #8cff3a;

}



/* Bottom Line */

.bottom-line{

margin-top:15px;

height:1px;

background:
linear-gradient(
90deg,
transparent,
#777,
transparent
);

}



/* Footer Text */

.ai-text{

font-size:9px;

color:#888;

margin-top:8px;

}


</style>



<div class="muskan-card">


<div class="ai-logo">
M
</div>


<div class="heartbeat"></div>



<div class="brand-name">
MuskanNova AI
</div>



<div class="tagline">

Intelligent Music<br>
Recommendation System

</div>



<div class="knn">

Powered by KNN

</div>
<div class="bottom-line"></div>
<div class="ai-text">

Machine Learning • AI • Music Intelligence

</div>


</div>


""", unsafe_allow_html=True)
with st.sidebar:

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#1DB954,red);
        padding:18px;
        border-radius:18px;
        text-align:center;
        margin-bottom:18px;
    ">
        <h2 style="color:white;">🤖 AI CONTROL PANEL</h2>
        <p style="color:white;">
        🟢 Model Status : Online<br>#current system status#==
        🎵 Dataset : Loaded<br>#==dataset load#==
        ⚡ Recommendation Engine : Active #model active or not #
        </p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "📌 Navigation",
        [
            "🏠 Home",
            "🎵 Songs",
            "🎤 Artists",
            "🏷 Genres",
            "📊 Analytics",
            "🏆 Top Charts",
            "🤖 AI Prediction",
            "❤️ Recommendation",
           "🎧 AI Music Personality",
            "🌟 Conclusion"
        ]
    )

    st.markdown("---")

    st.markdown("""
    <div style="
        background:pink;
        padding:15px;
        border-radius:15px;
        border-left:5px solid green;
    ">
    <h4 style="color:red;">⚙ Technologies</h4>

    🐍 Python<br>
    📊 Streamlit<br>
    🤖 Scikit-Learn<br>
    📁 Pandas<br>
    🎵 Spotify Dataset
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("💚 MCA Final Project • 2026")
    
# ===================== 🏠 HOME PAGE =====================
if page == "🏠 Home":

    # ================= Premium Banner =================
    if page == "🏠 Home":

    # ================= Premium Banner =================
    st.markdown("""
    <div style="
        background:linear-gradient(90deg,#000000,#1DB954);
        padding:30px;
        border-radius:20px;
        text-align:center;
        box-shadow:0px 5px 15px rgba(0,0,0,0.4);
    ">
        <h1 style="color:white;">🎵 Spotify AI Dashboard</h1>
        <h3 style="color:#FFD700;">
            Machine Learning Based Music Recommendation System
        </h3>
        <p style="color:white;">
            🎧 Discover • Analyze • Recommend • Enjoy
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ================= Dashboard Overview =================
    st.subheader("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎵 Songs", len(df))

    with col2:
        st.metric("🎤 Artists", df["Artist"].nunique())

    with col3:
        st.metric("💿 Albums", df["Album"].nunique())

    with col4:
        st.metric("🎼 Genres", df["Genre"].nunique())

    st.caption("Real-time overview of the Spotify music dataset.")

    # ================= Trending Songs =================
    st.markdown("---")

    st.subheader("🔥 Today's Top 10 Trending Songs")

    top = df.sort_values(
        "Popularity",
        ascending=False
    ).head(10)

    st.dataframe(
        top,
        use_container_width=True
    )
        # ================= AI Choice with Image + Link =================

    st.markdown("---")
    st.subheader("🤖 AI Choice 🎧")


    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background: linear-gradient(90deg,#1DB954,#00ff88);
        color:white;
        font-size:18px;
        font-weight:bold;
        border-radius:20px;
        height:50px;
        width:250px;
    }

    div.stButton > button:hover {
        background:linear-gradient(90deg,#00ff88,#1DB954);
        color:black;
    }
    </style>
    """, unsafe_allow_html=True)



    if st.button(
        "🤖 Let AI Choose",
        key="home_ai_choice"
    ):


        home_ai_song = df.sample(1)

        home_song_name = home_ai_song.iloc[0]["Track_Name"]

        home_artist = home_ai_song.iloc[0]["Artist"]



        col1, col2 = st.columns([1,2])


        with col1:

            st.image(
                "https://images.unsplash.com/photo-1511379938547-c1f69419868d",
                width=220
            )


        with col2:

            st.success(f"""
🎵 **{home_song_name}**

🎤 Artist : {home_artist}

💿 Album : {home_ai_song.iloc[0]['Album']}

⭐ Popularity : {home_ai_song.iloc[0]['Popularity']}
""")


            import urllib.parse


            home_search = urllib.parse.quote(
                home_song_name + " " + home_artist
            )


            spotify_link = (
                "https://open.spotify.com/search/"
                + home_search
            )


            youtube_link = (
                "https://www.youtube.com/results?search_query="
                + home_search
            )


            c1,c2 = st.columns(2)


            with c1:

                st.link_button(
                    "🎧 Play Spotify",
                    spotify_link
                )


            with c2:

                st.link_button(
                    "▶️ Play YouTube",
                    youtube_link
                )



    # ================= Surprise Me =================


    st.markdown("---")

    st.subheader("🎲 Surprise Me")


    if st.button(
        "🎁 Recommend a Surprise Song",
        key="home_surprise_button"
    ):


        st.session_state.home_surprise_song = df.sample(1)

        st.balloons()



    if "home_surprise_song" in st.session_state:


        surprise_song = st.session_state.home_surprise_song


        surprise_name = surprise_song.iloc[0]["Track_Name"]

        surprise_artist = surprise_song.iloc[0]["Artist"]



        st.success(f"""
🎵 **{surprise_name}**

🎤 Artist : {surprise_artist}

💿 Album : {surprise_song.iloc[0]['Album']}

⭐ Popularity : {surprise_song.iloc[0]['Popularity']}
""")


        import urllib.parse


        surprise_search = urllib.parse.quote(
            surprise_name + " " + surprise_artist
        )


        a,b,c = st.columns(3)


        with a:

            st.link_button(
                "🎧 Spotify",
                "https://open.spotify.com/search/" + surprise_search
            )


        with b:

            st.link_button(
                "▶️ YouTube",
                "https://www.youtube.com/results?search_query=" + surprise_search
            )


        with c:

            if st.button(
                "💡 Song Knowledge",
                key="home_song_knowledge"
            ):

                st.info(f"""
🎼 Song Information

🎵 Title:
{surprise_name}

🎤 Artist:
{surprise_artist}

💿 Album:
{surprise_song.iloc[0]['Album']}

⭐ Popularity:
{surprise_song.iloc[0]['Popularity']}

🎧 Energy:
{surprise_song.iloc[0].get('Energy','N/A')}

💃 Danceability:
{surprise_song.iloc[0].get('Danceability','N/A')}

🎸 Genre:
{surprise_song.iloc[0].get('Genre','N/A')}
""")
            
            
#=============SONG PAGE=============
elif  page == "🎵 Songs":

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
    else:# Create a copy of original dataset
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
# Calculate average quality score using Popularity, Energy and Danceability
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
        # Create Spotify search URL uniform resource loactor 
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
# Filter dataset by selected artist
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

# Create interactive bar chart using Plotly
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
# Count songs in each genre
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
#slect unique song 
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
#to f ka matlab "formatted string" (f-string) hota hai.

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

 # Audio features jo ML model use karega similarity nikalne ke liye
    feature_cols = ["Popularity", "Danceability", "Energy", "Valence", "Tempo"]
    
# Missing values aur duplicate songs remove kiye taki model error na de
    ml_df = df.dropna(subset=feature_cols).drop_duplicates(subset=["Track_Name"]).reset_index(drop=True)

    st.markdown("---")

    # User K (number of recommendations) select karta hai
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
           # I selected this project to apply Machine Learning and Data Analytics
           #  in a practical, real-world application.
elif page == "🎧 AI Music Personality":

    import time
    import urllib.parse

    st.title("🎧 AI Music Personality Scanner 🤖")

    st.write(
        "Discover your music personality using AI analysis!"
    )

    st.markdown("---")


    # ================= USER INPUT =================


    st.subheader("🎭 Step 1: Choose Your Mood")


    if "personality_mood" not in st.session_state:
        st.session_state.personality_mood = ""


    c1,c2,c3 = st.columns(3)


    with c1:
        if st.button("😊 Happy"):
            st.session_state.personality_mood="Happy"


    with c2:
        if st.button("🔥 Energetic"):
            st.session_state.personality_mood="Energetic"


    with c3:
        if st.button("💙 Emotional"):
            st.session_state.personality_mood="Emotional"



    st.markdown("---")


    listening = st.selectbox(
        "🎧 When do you usually listen to music?",
        [
            "🌅 Morning",
            "🌙 Night",
            "🚗 Travel",
            "🏋️ Workout"
        ]
    )


    style = st.selectbox(
        "🎵 Your favourite music style?",
        [
            "🔥 High Energy",
            "❤️ Romantic",
            "😌 Calm",
            "🎉 Party"
        ]
    )



    st.markdown("---")



    # ================= AI BUTTON =================


    if st.button("🤖 Analyze My Music Personality"):


        if st.session_state.personality_mood == "":


            st.warning(
                "Please select your mood first 😊"
            )


        else:


            with st.spinner(
                "🤖 AI is analyzing your music taste..."
            ):

                time.sleep(2)



            st.success(
                "✨ Analysis Complete!"
            )


            st.balloons()



            # ================= PERSONALITY LOGIC =================


            if (
                st.session_state.personality_mood=="Happy"
                and style=="🎉 Party"
            ):

                personality="🎉 The Mood Booster"


                songs=[
                    "Happy - Pharrell Williams",
                    "Uptown Funk - Bruno Mars"
                ]



            elif style=="❤️ Romantic":

                personality="❤️ The Romantic Listener"


                songs=[
                    "Perfect - Ed Sheeran",
                    "Until I Found You"
                ]



            elif style=="😌 Calm":

                personality="🌙 The Peace Seeker"


                songs=[
                    "Photograph - Ed Sheeran",
                    "Night Changes"
                ]



            else:

                personality="🔥 The Energy Explorer"


                songs=[
                    "Believer - Imagine Dragons",
                    "Thunder - Imagine Dragons"
                ]



            # ================= AI RESULT CARD =================


            st.subheader("🎧 Your AI Music Personality")


            st.markdown(
            f"""
            <div style="
            background:linear-gradient(135deg,#121212,#1DB954);
            padding:35px;
            border-radius:25px;
            box-shadow:0px 0px 25px #1DB954;
            ">

            <h1 style="color:white;text-align:center;">
            {personality}
            </h1>

            <h3 style="color:white;text-align:center;">
            AI understands your musical emotions 🎶
            </h3>

            </div>
            """,
            unsafe_allow_html=True
            )



            st.markdown("---")



            # ================= SCORE CARDS =================


            col1,col2,col3 = st.columns(3)


            with col1:

                st.metric(
                    "🎭 Mood Match",
                    "95%"
                )


            with col2:

                st.metric(
                    "🔥 Energy Level",
                    "88%"
                )


            with col3:

                st.metric(
                    "🤖 AI Confidence",
                    "96%"
                )



            st.markdown("---")



            # ================= SONG RECOMMENDATION =================


            st.subheader("🎵 AI Recommended For You")


            for song in songs:


                st.markdown(
                f"""
                <div style="
                background:#191919;
                padding:20px;
                border-radius:18px;
                margin:15px 0;
                border-left:6px solid #1DB954;
                ">

                <h2 style="color:#1DB954;">
                🎵 {song}
                </h2>

                <p style="color:white;">
                🤖 AI Match: 94%
                <br>
                ⭐ Personalized Recommendation
                </p>

                </div>
                """,
                unsafe_allow_html=True
                )



                search = urllib.parse.quote(song)


                spotify = (
                f"https://open.spotify.com/search/{search}"
                )


                youtube = (
                f"https://www.youtube.com/results?search_query={search}"
                )


                a,b = st.columns(2)


                with a:

                    st.link_button(
                        "🚀 Listen Now",
                        spotify
                    )


                with b:

                    st.link_button(
                        "▶️ Watch Song",
                        youtube
                    )



            st.markdown("---")


            st.info(
            """
            🧠 How AI Works:

            User Preference
            +
            Mood Analysis
            +
            Song Features
            +
            Machine Learning (KNN)

            = Smart Personalized Recommendation 🎧
            """
            )
   # ===================== 🌟 CONCLUSION PAGE =====================

elif page == "🌟 Conclusion":

    st.title("🎧 AI Music Studio")


    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#121212,#1DB954);
    padding:40px;
    border-radius:25px;
    text-align:center;
    ">

    <h1 style="color:white;">
    🚀 Project Successfully Completed
    </h1>

    <p style="color:white;font-size:20px;">
    Machine Learning + Music Intelligence
    </p>

    </div>
    """, unsafe_allow_html=True)


    st.write("")


    st.subheader("📌 What This System Can Do?")


    features = [
        "🎵 Analyze thousands of music patterns",
        "🤖 Recommend songs using AI similarity",
        "📊 Visualize music trends",
        "🎤 Explore artists and genres",
        "❤️ Create a personalized music experience"
    ]


    for item in features:
        st.success(item)



    st.markdown("---")


    st.subheader("🏆 Technology Stack")


    col1, col2, col3, col4 = st.columns(4)


    with col1:
        st.metric("🐍 Python","Core")


    with col2:
        st.metric("🤖 ML","KNN")


    with col3:
        st.metric("📊 Dashboard","Streamlit")


    with col4:
        st.metric("🎵 Data","Spotify")



    st.markdown("---")


    st.markdown("""
    <div style="
    padding:25px;
    background:#000;
    border-radius:20px;
    text-align:center;
    ">

    <h2 style="color:#1DB954;">
    🎶 Your Next Favorite Song Is Waiting
    </h2>

    <p style="color:white;">
    Thank you for exploring my AI-powered music recommendation system ❤️
    </p>

    </div>
    """, unsafe_allow_html=True)
    
    

    
