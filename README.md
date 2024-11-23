# Tennis Match Predictor

This Python program predicts the winner of a tennis match based on historical player statistics and match data from 2024. The model uses a Random Forest classifier to make predictions.

The data is being pulled from the following repository:  
[Jeff Sackmann's Tennis GitHub](https://github.com/JeffSackmann/tennis_atp/blob/master/atp_matches_2024.csv).

### Customizing the Data Source

If you want to change the dataset being used, simply modify the `url` variable in the `main` function to point to the desired CSV file.

## Features

- Predicts match outcomes between two players on different court surfaces (Hard, Clay, Grass).
- Supports user input for player names and court type.
- Trains a Random Forest model using historical ATP match data.

---

## Installation

Follow these steps to set up and run the program:

### Step 1: Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/mmtekin/tennisMatchPredictor
cd tennisMatchPredictor
```

### Step 2: Create a Virtual Environment

Set up a Python virtual environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Usage

Run the `tennisPredictor.py` script to start the program:

```bash
python tennisPredictor.py
```

Follow the on-screen prompts to enter player names and court type to predict the winner of a tennis match. Type 'exit' to quit the program.

Example:

```bash
Enter Player 1's name: Alexander Zverev
Enter Player 2's name: Rafael Nadal
Enter court type (Hard, Clay, Grass): Clay
Prediction: Alexander Zverev
```