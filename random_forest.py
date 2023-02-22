import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# Random Forest classifier
clf = RandomForestClassifier(n_estimators=50, random_state=42)
accuracy_list = []
for i in range(1, 51):
    clf.set_params(n_estimators=i)
    clf.fit(train_data, train_target)
    predictions = clf.predict(test_data)
    accuracy = accuracy_score(test_target, predictions)
    accuracy_list.append(accuracy)

# Plot the accuracy over the number of trees
plt.plot(range(1, 51), accuracy_list)
plt.xlabel("Number of trees")
plt.ylabel("Accuracy of model")
plt.title("Random Forest Classifier")
plt.show()

# Plot the feature importances
importances = clf.feature_importances_
plt.bar(features, importances)
plt.title("Feature importances")
plt.show()

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
