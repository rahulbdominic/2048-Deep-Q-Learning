import json
from keras.models import Sequential
from keras.layers.core import Dense

from play import *

if __name__ == "__main__":

    # Constants
    num_actions = 4  # ["left", "up", "right", "down"]
    epoch = 100
    hidden_size = 64
    grid_size = 4

    model = Sequential()
    model.add(Dense(16, input_dim=16, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(loss="mse",
                  optimizer="sgd")
    print(model.summary())

    # model.load_weights("model.h5")

    for e in range(epoch):
        # Make initial input
        get_g_values = make_get_q_values(model)

        experience_replay = ExperienceReplay(model)
        (exps, max_score) = experience_replay.collect(strategy=highest_reward_strategy,
                                                      epochs=100, verbose=False)
        # (exps, max_score) = experience_replay.collect(strategy=make_epsilon_greedy_strategy(make_get_q_values(model), 0),
        #                                               epochs=100, verbose=False)

        model = experience_replay.get_model()
        train_X = []
        train_Y = []
        (train_X, train_Y, action) = experience_replay.experiences_to_batches()
        print ("Epoch: {0}\tMax Score: {1}".format(e, max_score))

    # Save trained model
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
