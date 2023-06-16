import pickle

with open('history_simulation.pickle', 'rb') as file:
    history = pickle.load(file)

print(history)