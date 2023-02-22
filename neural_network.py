import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("international_matches.csv")

# Create a new column to represent whether the home team won or not
data["host_team_won"] = data["home_team_result"] == "Win"
features = ["home_team_fifa_rank", "away_team_fifa_rank"]
target = "host_team_won"
train_data, test_data, train_target, test_target = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Define the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=2, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(train_data, train_target, epochs=50, validation_data=(test_data, test_target))

# Evaluate the model
_, accuracy = model.evaluate(test_data, test_target)
print(f"Accuracy of model: {accuracy}")

# win of home team
host_wins = test_target[test_target == True]
host_win_percentage = len(host_wins) / len(test_target) * 100

# win of away team
away_wins = test_target[test_target == False]
away_win_percentage = len(away_wins) / len(test_target) * 100

print(f"Percentage of games won by the host team: {host_win_percentage:.2f}%")
print(f"Percentage of games won by the away team: {away_win_percentage:.2f}%")

test_data["lower_rank_team_won"] = ((test_data["home_team_fifa_rank"] > test_data["away_team_fifa_rank"]) & (
        test_target == False)) \
                                   | ((test_data["home_team_fifa_rank"] < test_data["away_team_fifa_rank"]) & (
        test_target == True))

lower_rank_wins = len(test_data[test_data["lower_rank_team_won"] == True])

print(f"The team with the lower FIFA rank won {lower_rank_wins} times in the test data.")
FIFA_correct_percentage = lower_rank_wins / len(test_data) * 100
print(f"Percentage of times FIFA was correct in the test data: {FIFA_correct_percentage:.2f}%")
print(f"Percentage of times FIFA was incorrect in the test data: {100 - FIFA_correct_percentage:.2f}%")

# Plot the model accuracy over number of epochs
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy over Number of Epochs")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()
