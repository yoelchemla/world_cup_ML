import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the CSV file
data = pd.read_csv("international_matches.csv")

# Create a new column to represent whether the home team won or not
data["host_team_won"] = data["home_team_result"] == "Win"
features = ["home_team_fifa_rank", "away_team_fifa_rank"]
target = "host_team_won"
train_data, test_data, train_target, test_target = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# logistic regression classifier
clf = LogisticRegression()
clf.fit(train_data, train_target)
predictions = clf.predict(test_data)

# accuracy of the test in relation to prediction
accuracy = accuracy_score(test_target, predictions)
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

# create a scatter plot of the data
probs = clf.predict_proba(test_data[features])[:, 1]
plt.scatter(test_data["home_team_fifa_rank"], probs)
plt.xlabel("FIFA rank of home team")
plt.ylabel("Probability of home team winning")
plt.show()
