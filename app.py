import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="IPLytics - IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide"
)

# ---------------- LOAD DATA ---------------- #

@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")
    return matches, deliveries

matches, deliveries = load_data()

# ---------------- HEADER ---------------- #

st.title("🏏 IPLytics – IPL Analytics & Match Prediction System")

st.markdown("""
### Real-time IPL Analytics Dashboard with Machine Learning Match Prediction
""")

st.markdown("---")

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📌 Dashboard Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Team Analytics",
        "Player Analytics",
        "Match Prediction"
    ]
)

# ---------------- HOME PAGE ---------------- #

if page == "Home":

    st.header("📊 IPL Dashboard Overview")

    total_matches = matches.shape[0]
    total_teams = pd.concat([matches['team1'], matches['team2']]).nunique()
    total_seasons = matches['Season'].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric("🏏 Total Matches", total_matches)
    col2.metric("👥 Total Teams", total_teams)
    col3.metric("📅 Seasons", total_seasons)

    st.markdown("---")

    st.subheader("🏆 IPL Winners By Season")

    winners = matches[['Season', 'winner']].dropna()

    fig = px.bar(
        winners,
        x='Season',
        y='winner',
        color='winner',
        title="IPL Season Winners"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- TEAM ANALYTICS ---------------- #

elif page == "Team Analytics":

    st.header("📈 Team Analytics")

    st.subheader("🏆 Team Win Counts")

    team_wins = matches['winner'].value_counts().reset_index()
    team_wins.columns = ['Team', 'Wins']

    fig1 = px.bar(
        team_wins,
        x='Team',
        y='Wins',
        color='Wins',
        text='Wins',
        title="Total Wins By Teams"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    st.subheader("🎯 Toss Winner Analysis")

    toss_wins = matches['toss_winner'].value_counts().reset_index()
    toss_wins.columns = ['Team', 'Toss Wins']

    fig2 = px.pie(
        toss_wins,
        names='Team',
        values='Toss Wins',
        title="Toss Wins Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("🏟️ Top Match Venues")

    venues = matches['venue'].value_counts().head(10).reset_index()
    venues.columns = ['Venue', 'Matches']

    fig3 = px.bar(
        venues,
        x='Venue',
        y='Matches',
        color='Matches',
        text='Matches',
        title="Top IPL Venues"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ---------------- PLAYER ANALYTICS ---------------- #

elif page == "Player Analytics":

    st.header("🏏 Player Analytics")

    # ---------------- TOP BATSMEN ---------------- #

    st.subheader("🔥 Top Run Scorers")

    batsman_runs = deliveries.groupby('batsman')['batsman_runs'].sum().sort_values(
        ascending=False
    ).head(10).reset_index()

    batsman_runs.columns = ['Player', 'Runs']

    fig4 = px.bar(
        batsman_runs,
        x='Player',
        y='Runs',
        color='Runs',
        text='Runs',
        title="Top 10 IPL Run Scorers"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ---------------- TOP BOWLERS ---------------- #

    st.subheader("🎯 Top Wicket Takers")

    wickets = deliveries.dropna(subset=['player_dismissed'])

    top_bowlers = wickets['bowler'].value_counts().head(10).reset_index()
    top_bowlers.columns = ['Bowler', 'Wickets']

    fig5 = px.bar(
        top_bowlers,
        x='Bowler',
        y='Wickets',
        color='Wickets',
        text='Wickets',
        title="Top 10 Wicket Takers"
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # ---------------- PLAYER OF MATCH ---------------- #

    st.subheader("🌟 Most Player of the Match Awards")

    pom = matches['player_of_match'].value_counts().head(10).reset_index()
    pom.columns = ['Player', 'Awards']

    fig6 = px.bar(
        pom,
        x='Player',
        y='Awards',
        color='Awards',
        text='Awards',
        title="Most Player of the Match Awards"
    )

    st.plotly_chart(fig6, use_container_width=True)

# ---------------- MATCH PREDICTION ---------------- #

# ---------------- MATCH PREDICTION ---------------- #

elif page == "Match Prediction":

    st.header("🤖 Match Winner Prediction")

    st.markdown("Predict IPL match winners using Machine Learning.")

    # ---------------- DATA PREPARATION ---------------- #

    prediction_data = matches[
        [
            'team1',
            'team2',
            'toss_winner',
            'toss_decision',
            'venue',
            'winner'
        ]
    ].dropna()

    # Remove matches with no result
    prediction_data = prediction_data[
        prediction_data['winner'].isin(
            prediction_data['team1']
        ) | prediction_data['winner'].isin(
            prediction_data['team2']
        )
    ]

    # ---------------- ENCODING ---------------- #

    from sklearn.preprocessing import LabelEncoder

    encoders = {}

    for column in prediction_data.columns:
        le = LabelEncoder()
        prediction_data[column] = le.fit_transform(
            prediction_data[column]
        )
        encoders[column] = le

    # ---------------- FEATURES ---------------- #

    X = prediction_data[
        [
            'team1',
            'team2',
            'toss_winner',
            'toss_decision',
            'venue'
        ]
    ]

    y = prediction_data['winner']

    # ---------------- TRAIN TEST SPLIT ---------------- #

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------- MODEL ---------------- #

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    st.success(
        f"✅ Prediction Model Accuracy: {round(accuracy * 100, 2)}%"
    )

    st.markdown("---")

    # ---------------- USER INPUT ---------------- #

    teams = sorted(matches['team1'].dropna().unique())

    venues = sorted(matches['venue'].dropna().unique())

    team1 = st.selectbox("Select Team 1", teams)

    team2 = st.selectbox(
        "Select Team 2",
        [team for team in teams if team != team1]
    )

    toss_winner = st.selectbox(
        "Select Toss Winner",
        [team1, team2]
    )

    toss_decision = st.selectbox(
        "Toss Decision",
        ['bat', 'field']
    )

    venue = st.selectbox(
        "Select Venue",
        venues
    )

    # ---------------- PREDICTION ---------------- #

    if st.button("Predict Winner"):

        input_data = pd.DataFrame({
            'team1': [team1],
            'team2': [team2],
            'toss_winner': [toss_winner],
            'toss_decision': [toss_decision],
            'venue': [venue]
        })

        for column in input_data.columns:
            input_data[column] = encoders[column].transform(
                input_data[column]
            )

        prediction = model.predict(input_data)

        winner = encoders['winner'].inverse_transform(
            prediction
        )

        st.balloons()

        st.success(f"🏆 Predicted Winner: {winner[0]}")