import pandas as pd 

circuits = pd.read_csv("circuits.csv")
constructorResults = pd.read_csv("constructorResults.csv")
constructors = pd.read_csv("constructors.csv")
constructorStandings = pd.read_csv("constructorStandings.csv")
drivers = pd.read_csv("drivers.csv")
lapTimes = pd.read_csv("lapTimes.csv")
pitStops = pd.read_csv("pitStops.csv")
qualifying = pd.read_csv("qualifying.csv")
races = pd.read_csv("races.csv")
results = pd.read_csv("results.csv")
seasons = pd.read_csv("seasons.csv")
status = pd.read_csv("status.csv")

print(qualifying.head())
print(qualifying.iloc[0])
print(qualifying.describe())

"""
Create a function that gets different stints from a certain driver during a certain race.
The length in laps of those stints may allow one to find which tyre compound was used.
"""

def getStints(race,driver):
	return 0
	#Returns an array which lengths contains the number of laps of each stint.
	#For a two-stop strategy, we get something like [15,35,10]
	#We can deduce that the first and third stints were done using the soft compound while the second used medium tyres.

#Implement the tyre compound classifier

"""
Now that we know, for a given car, circuit, driver, how many laps do each compound last, 
we can predict what strategy is the best for the race.
We can then train a model to give the probability of a driver winning the race 
taking into account every driver on track and their tyres & performance.
"""

#Implement a logistic regression model
