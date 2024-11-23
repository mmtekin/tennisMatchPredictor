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
        ["winner_name", "winner_rank", "winner_age", "winner_ht"]
    ].rename(
        columns={
            "winner_name": "player_name",
            "winner_rank": "player_rank",
            "winner_age": "player_age",
            "winner_ht": "player_ht",
        }
    )

    # Extract loser stats
    loser_stats = data[["loser_name", "loser_rank", "loser_age", "loser_ht"]].rename(
        columns={
            "loser_name": "player_name",
            "loser_rank": "player_rank",
            "loser_age": "player_age",
            "loser_ht": "player_ht",
        }
    )

    # Combine winner and loser stats
    player_stats = pd.concat([winner_stats, loser_stats], ignore_index=True)

    # Handle missing values
    player_stats = player_stats.dropna(
        subset=["player_rank", "player_age", "player_ht"]
    )

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
        raise Exception(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

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
    ]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean the data by dropping rows with missing values
    data = data.dropna(subset=required_columns)

    # Map surface types to numerical values
    surface_map = {"Hard": 1, "Clay": 2, "Grass": 3}
    data["surface_type"] = data["surface"].map(surface_map)

    # Drop rows with missing or unmapped surface values
    data = data.dropna(subset=["surface_type"])

    # Create two copies of the data: one where the winner is Player 1, and one where the loser is Player 1
    data_win = data.copy()
    data_win["player_1_name"] = data_win["winner_name"]
    data_win["player_1_rank"] = data_win["winner_rank"]
    data_win["player_1_age"] = data_win["winner_age"]
    data_win["player_1_ht"] = data_win["winner_ht"]
    data_win["player_2_name"] = data_win["loser_name"]
    data_win["player_2_rank"] = data_win["loser_rank"]
    data_win["player_2_age"] = data_win["loser_age"]
    data_win["player_2_ht"] = data_win["loser_ht"]
    data_win["target"] = 1  # Player 1 wins

    data_lose = data.copy()
    data_lose["player_1_name"] = data_lose["loser_name"]
    data_lose["player_1_rank"] = data_lose["loser_rank"]
    data_lose["player_1_age"] = data_lose["loser_age"]
    data_lose["player_1_ht"] = data_lose["loser_ht"]
    data_lose["player_2_name"] = data_lose["winner_name"]
    data_lose["player_2_rank"] = data_lose["winner_rank"]
    data_lose["player_2_age"] = data_lose["winner_age"]
    data_lose["player_2_ht"] = data_lose["winner_ht"]
    data_lose["target"] = 0  # Player 1 loses

    # Combine the data
    data_all = pd.concat([data_win, data_lose], ignore_index=True)

    # Calculate feature differences (Player 1 - Player 2)
    data_all["rank_diff"] = data_all["player_1_rank"] - data_all["player_2_rank"]
    data_all["age_diff"] = data_all["player_1_age"] - data_all["player_2_age"]
    data_all["height_diff"] = data_all["player_1_ht"] - data_all["player_2_ht"]

    # Select features
    features = ["rank_diff", "age_diff", "height_diff", "surface_type"]
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

    # Calculate feature differences (Player 1 - Player 2)
    rank_diff = player_1["player_rank"].values[0] - player_2["player_rank"].values[0]
    age_diff = player_1["player_age"].values[0] - player_2["player_age"].values[0]
    height_diff = player_1["player_ht"].values[0] - player_2["player_ht"].values[0]

    # Map surface type to a numerical value
    surface_map = {"Hard": 1, "Clay": 2, "Grass": 3}
    surface_type = surface_map.get(surface, None)
    if surface_type is None:
        return "Error: Invalid surface type. Please enter 'Hard', 'Clay', or 'Grass'."

    # Prepare input feature for the model
    match_features = pd.DataFrame(
        [
            {
                "rank_diff": rank_diff,
                "age_diff": age_diff,
                "height_diff": height_diff,
                "surface_type": surface_type,
            }
        ]
    )

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
    print("Features (X) sample:")
    print(X.head())
    print("Target (y) sample:")
    print(y.head())

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