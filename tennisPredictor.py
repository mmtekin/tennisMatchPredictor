# Importing necessary libraries
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def get_player_stats(data):
    """
    Aggregate player statistics into a single DataFrame.

    Args:
        data (pandas.DataFrame): Raw match data.
    Returns:
        pandas.DataFrame: Player statistics.
    """
    # Extract winner stats
    winner_stats = data[
        [
            "winner_name",
            "winner_rank",
            "winner_age",
            "winner_ht",
            "w_ace",
            "w_df",
            "w_bpSaved",
            "w_bpFaced",
        ]
    ].rename(
        columns={
            "winner_name": "player_name",
            "winner_rank": "player_rank",
            "winner_age": "player_age",
            "winner_ht": "player_ht",
            "w_ace": "player_ace",
            "w_df": "player_df",
            "w_bpSaved": "player_bpSaved",
            "w_bpFaced": "player_bpFaced",
        }
    )

    # Extract loser stats
    loser_stats = data[
        [
            "loser_name",
            "loser_rank",
            "loser_age",
            "loser_ht",
            "l_ace",
            "l_df",
            "l_bpSaved",
            "l_bpFaced",
        ]
    ].rename(
        columns={
            "loser_name": "player_name",
            "loser_rank": "player_rank",
            "loser_age": "player_age",
            "loser_ht": "player_ht",
            "l_ace": "player_ace",
            "l_df": "player_df",
            "l_bpSaved": "player_bpSaved",
            "l_bpFaced": "player_bpFaced",
        }
    )

    # Combine winner and loser stats
    player_stats = pd.concat([winner_stats, loser_stats], ignore_index=True)

    # Handle missing values
    player_stats = player_stats.dropna(
        subset=["player_rank", "player_age", "player_ht"]
    )

    # Convert stats to numeric
    numeric_cols = [
        "player_rank",
        "player_age",
        "player_ht",
        "player_ace",
        "player_df",
        "player_bpSaved",
        "player_bpFaced",
    ]
    player_stats[numeric_cols] = player_stats[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    player_stats[numeric_cols] = player_stats[numeric_cols].fillna(0)

    # Get the latest stats for each player
    player_stats = player_stats.sort_values(by="player_rank")
    player_stats = player_stats.groupby("player_name").first().reset_index()

    return player_stats


# Step 1: Load the dataset
def load_data(url):
    """
    Fetch ATP match data from a GitHub URL.

    Args:
        url (str): URL to the raw CSV file on GitHub.
    Returns:
        pandas.DataFrame: Processed match data.
    """
    response = requests.get(url)
    if response.status_code == 200:
        csv_data = StringIO(response.text)
        data = pd.read_csv(csv_data)
        return data
    else:
        raise Exception(
            f"Failed to fetch data. HTTP Status Code: {response.status_code}"
        )


# Step 2: Preprocess the data
def preprocess_data(data):
    """
    Clean and prepare the data for training.

    Args:
        data (pandas.DataFrame): Raw match data.
    Returns:
        tuple: Features (X) and labels (y).
    """
    # Ensure necessary columns are present
    required_columns = [
        "winner_name",
        "loser_name",
        "winner_rank",
        "loser_rank",
        "winner_age",
        "loser_age",
        "winner_ht",
        "loser_ht",
        "surface",
        "w_ace",
        "l_ace",
        "w_df",
        "l_df",
        "w_bpSaved",
        "l_bpSaved",
        "w_bpFaced",
        "l_bpFaced",
    ]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean the data by dropping rows with missing values
    data = data.dropna(subset=required_columns)

    # One-hot encode the surface types
    data = pd.get_dummies(data, columns=["surface"], prefix="surface")

    # Ensure all possible 'surface_*' columns are present
    surface_cols = ["surface_Clay", "surface_Grass", "surface_Hard"]
    for col in surface_cols:
        if col not in data.columns:
            data[col] = 0

    # Create two copies of the data: winner as Player 1 and loser as Player 1
    data_win = data.copy()
    data_win["player_1_name"] = data_win["winner_name"]
    data_win["player_1_rank"] = data_win["winner_rank"]
    data_win["player_1_age"] = data_win["winner_age"]
    data_win["player_1_ht"] = data_win["winner_ht"]
    data_win["player_1_ace"] = data_win["w_ace"]
    data_win["player_1_df"] = data_win["w_df"]
    data_win["player_1_bpSaved"] = data_win["w_bpSaved"]
    data_win["player_1_bpFaced"] = data_win["w_bpFaced"]

    data_win["player_2_name"] = data_win["loser_name"]
    data_win["player_2_rank"] = data_win["loser_rank"]
    data_win["player_2_age"] = data_win["loser_age"]
    data_win["player_2_ht"] = data_win["loser_ht"]
    data_win["player_2_ace"] = data_win["l_ace"]
    data_win["player_2_df"] = data_win["l_df"]
    data_win["player_2_bpSaved"] = data_win["l_bpSaved"]
    data_win["player_2_bpFaced"] = data_win["l_bpFaced"]

    data_win["target"] = 1  # Player 1 wins

    data_lose = data.copy()
    data_lose["player_1_name"] = data_lose["loser_name"]
    data_lose["player_1_rank"] = data_lose["loser_rank"]
    data_lose["player_1_age"] = data_lose["loser_age"]
    data_lose["player_1_ht"] = data_lose["loser_ht"]
    data_lose["player_1_ace"] = data_lose["l_ace"]
    data_lose["player_1_df"] = data_lose["l_df"]
    data_lose["player_1_bpSaved"] = data_lose["l_bpSaved"]
    data_lose["player_1_bpFaced"] = data_lose["l_bpFaced"]

    data_lose["player_2_name"] = data_lose["winner_name"]
    data_lose["player_2_rank"] = data_lose["winner_rank"]
    data_lose["player_2_age"] = data_lose["winner_age"]
    data_lose["player_2_ht"] = data_lose["winner_ht"]
    data_lose["player_2_ace"] = data_lose["w_ace"]
    data_lose["player_2_df"] = data_lose["w_df"]
    data_lose["player_2_bpSaved"] = data_lose["w_bpSaved"]
    data_lose["player_2_bpFaced"] = data_lose["w_bpFaced"]

    data_lose["target"] = 0  # Player 1 loses

    # Combine the data
    data_all = pd.concat([data_win, data_lose], ignore_index=True)

    # Ensure all 'surface_*' columns are present in data_all
    for col in surface_cols:
        if col not in data_all.columns:
            data_all[col] = 0

    # List of columns to convert to numeric
    numeric_cols = [
        "player_1_rank",
        "player_2_rank",
        "player_1_age",
        "player_2_age",
        "player_1_ht",
        "player_2_ht",
        "player_1_ace",
        "player_2_ace",
        "player_1_df",
        "player_2_df",
        "player_1_bpSaved",
        "player_2_bpSaved",
        "player_1_bpFaced",
        "player_2_bpFaced",
    ]
    # Convert to numeric and handle errors
    data_all[numeric_cols] = data_all[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Fill NaN values with 0
    data_all[numeric_cols] = data_all[numeric_cols].fillna(0)

    # Calculate feature differences (Player 1 - Player 2)
    data_all["rank_diff"] = data_all["player_1_rank"] - data_all["player_2_rank"]
    data_all["age_diff"] = data_all["player_1_age"] - data_all["player_2_age"]
    data_all["height_diff"] = data_all["player_1_ht"] - data_all["player_2_ht"]
    data_all["ace_diff"] = data_all["player_1_ace"] - data_all["player_2_ace"]
    data_all["df_diff"] = data_all["player_1_df"] - data_all["player_2_df"]
    data_all["bp_saved_diff"] = (
        data_all["player_1_bpSaved"] - data_all["player_2_bpSaved"]
    )
    data_all["bp_faced_diff"] = (
        data_all["player_1_bpFaced"] - data_all["player_2_bpFaced"]
    )

    # Create interaction terms
    data_all["rank_surface_clay"] = data_all["rank_diff"] * data_all["surface_Clay"]
    data_all["rank_surface_grass"] = data_all["rank_diff"] * data_all["surface_Grass"]
    data_all["rank_surface_hard"] = data_all["rank_diff"] * data_all["surface_Hard"]

    # Update the features list
    features = [
        "rank_diff",
        "age_diff",
        "height_diff",
        "ace_diff",
        "df_diff",
        "bp_saved_diff",
        "bp_faced_diff",
        "surface_Clay",
        "surface_Grass",
        "surface_Hard",
        "rank_surface_clay",
        "rank_surface_grass",
        "rank_surface_hard",
    ]

    X = data_all[features]
    y = data_all["target"]

    return X, y


# Step 3: Split the data into training and testing sets
def split_data(X, y):
    """
    Split data into training and testing sets.

    Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Labels.
    Returns:
        tuple: Training and testing datasets.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Step 4: Train the model
def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier.

    Args:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training labels.
    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model


# Step 5: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (pandas.DataFrame): Testing features.
        y_test (pandas.Series): Testing labels.
    """
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


def predict_winner(player_stats, model, player_1_name, player_2_name, surface):
    """
    Predict the winner of a match between two players on a specific surface.

    Args:
        player_stats (pandas.DataFrame): DataFrame containing player statistics.
        model (RandomForestClassifier): Trained machine learning model.
        player_1_name (str): Name of Player 1.
        player_2_name (str): Name of Player 2.
        surface (str): Court type (e.g., "Hard", "Clay", "Grass").
    Returns:
        str: Predicted winner's name or an error message.
    """
    # Retrieve player stats
    player_1 = player_stats[player_stats["player_name"] == player_1_name]
    player_2 = player_stats[player_stats["player_name"] == player_2_name]

    if player_1.empty or player_2.empty:
        return "Error: One or both players not found in the dataset."

    # Calculate feature differences
    rank_diff = player_1["player_rank"].values[0] - player_2["player_rank"].values[0]
    age_diff = player_1["player_age"].values[0] - player_2["player_age"].values[0]
    height_diff = player_1["player_ht"].values[0] - player_2["player_ht"].values[0]
    ace_diff = player_1["player_ace"].values[0] - player_2["player_ace"].values[0]
    df_diff = player_1["player_df"].values[0] - player_2["player_df"].values[0]
    bp_saved_diff = (
        player_1["player_bpSaved"].values[0] - player_2["player_bpSaved"].values[0]
    )
    bp_faced_diff = (
        player_1["player_bpFaced"].values[0] - player_2["player_bpFaced"].values[0]
    )

    # One-hot encode the surface type
    surface_types = ["Clay", "Grass", "Hard"]
    surface_features = {f"surface_{s}": int(surface == s) for s in surface_types}

    # Ensure all surface features are present
    for col in ["surface_Clay", "surface_Grass", "surface_Hard"]:
        if col not in surface_features:
            surface_features[col] = 0

    # Interaction terms
    rank_surface_clay = rank_diff * surface_features["surface_Clay"]
    rank_surface_grass = rank_diff * surface_features["surface_Grass"]
    rank_surface_hard = rank_diff * surface_features["surface_Hard"]

    # Prepare input features
    match_features = pd.DataFrame(
        [
            {
                "rank_diff": rank_diff,
                "age_diff": age_diff,
                "height_diff": height_diff,
                "ace_diff": ace_diff,
                "df_diff": df_diff,
                "bp_saved_diff": bp_saved_diff,
                "bp_faced_diff": bp_faced_diff,
                **surface_features,
                "rank_surface_clay": rank_surface_clay,
                "rank_surface_grass": rank_surface_grass,
                "rank_surface_hard": rank_surface_hard,
            }
        ]
    )

    # Ensure feature columns match the training features
    expected_features = [
        "rank_diff",
        "age_diff",
        "height_diff",
        "ace_diff",
        "df_diff",
        "bp_saved_diff",
        "bp_faced_diff",
        "surface_Clay",
        "surface_Grass",
        "surface_Hard",
        "rank_surface_clay",
        "rank_surface_grass",
        "rank_surface_hard",
    ]
    match_features = match_features[expected_features]

    # Handle missing values
    match_features = match_features.fillna(0)

    # Make prediction
    prediction = model.predict(match_features)

    # Return the predicted winner's name
    return player_1_name if prediction[0] == 1 else player_2_name


# Step 6: Main program
def main():
    # Load your dataset
    url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv"
    data = load_data(url)
    print("Dataset loaded successfully.")
    print(f"The data is composed of {data.shape[0]} rows and {data.shape[1]} columns.")

    # Preprocess the data
    X, y = preprocess_data(data)
    # print("Features (X) sample:")
    # print(X.head())
    # print("Target (y) sample:")
    # print(y.head())

    # Get player stats
    player_stats = get_player_stats(data)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Repeatedly ask for player names and predict
    print(
        "\nMatch Predictor - Enter player names and court type to predict the winner."
    )
    print("Type 'exit' to quit the program.")
    while True:
        player_1_name = input("Enter Player 1's name: ").strip()
        if player_1_name.lower() == "exit":
            print("Exiting program. Goodbye!")
            break

        player_2_name = input("Enter Player 2's name: ").strip()
        if player_2_name.lower() == "exit":
            print("Exiting program. Goodbye!")
            break

        court_type = (
            input("Enter court type (Hard, Clay, Grass): ").strip().capitalize()
        )
        if court_type.lower() == "exit":
            print("Exiting program. Goodbye!")
            break

        # Predict the winner
        result = predict_winner(
            player_stats, model, player_1_name, player_2_name, court_type
        )
        print(f"Prediction: {result}\n")


if __name__ == "__main__":
    main()
