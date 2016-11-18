"""Algorithms and strategies to play 2048 and collect experience."""

from __future__ import division
from __future__ import print_function
from game import Game, ACTION_NAMES
from random import randint

import numpy as np

NORMALIZING_FACTOR = 15


class Experience(object):
    """Holds a single experience"""

    def __init__(self, state, action, reward, next_state, game_over,
                 not_available, next_state_available_actions):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.game_over = game_over
        self.not_available = not_available
        self.next_state_available_actions = next_state_available_actions

    def __str__(self):
        return str((self.state, self.action, self.reward, self.next_state,
                    self.game_over, self.next_state_available_actions))

    def __repr__(self):
        return self.__str__()


class ExperienceReplay(object):
    """ Class to encapsulate functions acting on a batch of experiences. """

    memory = 1000

    def __init__(self, model, experiences=[]):
        """ Constructor
        Args:
             experiences: Batch of experiences collected over playing multiple games
             model: The Keras model
        """
        self.experiences = experiences
        self.model = model

    def collect(self, strategy, epochs=10, verbose=False):
        """ Plays the game for a number of epochs and then return accumulated
            experiences. """
        max_score = 0

        for i in range(epochs):
            score, experience_set = play(strategy, verbose, False)
            self.experiences += experience_set

            max_score = max(max_score, score)

            if len(self.experiences) > self.memory:
                for j in range(len(self.experiences) - self.memory):
                    self.experiences.pop(0)
            print("Game {0} completed: Experiences - {1}; Score - {2}"
                   .format(i, len(experience_set), score))
            self.train()
        return self.experiences, max_score

    def experiences_to_batches(self):
        """Computes state_batch, targets, actions."""
        experiences = self.experiences

        batch_size = len(experiences)
        state_batch = np.zeros((batch_size, 16), dtype=np.float)
        next_state_batch = np.zeros((batch_size, 16), dtype=np.float)
        actions = np.zeros((batch_size,), dtype=np.int)
        reward_batch = np.zeros((batch_size,), dtype=np.float)
        bad_action_batch = np.zeros((batch_size,), dtype=np.bool)
        available_actions_batch = np.zeros((batch_size, 4), dtype=np.bool)
        merged = np.zeros((batch_size,), dtype=np.float)

        for i, experience in enumerate(experiences):
            state_batch[i, :] = (experience.state.flatten())
            next_state_batch[i, :] = (experience.next_state.flatten())
            actions[i] = experience.action
            reward_batch[i] = experience.reward
            bad_action_batch[i] = experience.game_over or experience.not_available
            available_actions_batch[i, experience.next_state_available_actions] = True
            merged[i] = (np.count_nonzero(experience.state) -
                         np.count_nonzero(experience.next_state) + 1)

        targets = compute_targets(reward_batch, state_batch, next_state_batch,
                                  actions, merged, self.model)

        return state_batch, targets, actions

    def train(self):
        (train_x, train_y, action) = self.experiences_to_batches()
        # model.fit(train_x, train_y, nb_epoch=1, batch_size=16)
        self.model.train_on_batch(np.divide(train_x, NORMALIZING_FACTOR),
                                  np.divide(train_y, NORMALIZING_FACTOR))

    def get_model(self):
        return self.model


def play(strategy, verbose=False, allow_unavailable_action=True):
    """Plays a single game, using a provided strategy.
    Args:
      strategy: A function that takes as argument a state and a list of available
          actions and returns an action from the list.
      allow_unavailable_action: Boolean, whether strategy is passed all actions
          or just the available ones.
      verbose: If true, prints game states, actions and scores.
    Returns:
      score, experiences where score is the final score and experiences is the
          list Experience instances that represent the collected experience.
    """

    game = Game()

    state = game.state().copy()
    game_over = game.game_over()
    experiences = []

    while not game_over:
        if verbose:
            print("Score:", game.score())
            game.print_state()

        old_state = state
        next_action = strategy(
            old_state, range(4) if allow_unavailable_action
            else game.available_actions())

        if game.is_action_available(next_action):

            reward = game.do_action(next_action)
            state = game.state().copy()
            game_over = game.game_over()

            if verbose:
                print("Action:", ACTION_NAMES[next_action])
                print("Reward:", reward)

            experiences.append(Experience(old_state, next_action, reward, state,
                                          game_over, False, game.available_actions()))

        else:
            experiences.append(Experience(state, next_action, -1, state, False, True,
                                          game.available_actions()))

    if verbose:
        print("Score:", game.score())
        game.print_state()
        print("Game over.")

    return game.score(), experiences


def random_strategy(_, actions):
    """Strategy that always chooses actions at random."""
    return np.random.choice(actions)


def static_preference_strategy(_, actions):
    """Always prefer left over up over right over top."""

    return min(actions)


def highest_reward_strategy(state, actions, epsilon=0.05):
    """Strategy that always chooses the action of highest immediate reward.
    If there are any ties, the strategy prefers left over up over right over down.
    """

    do_random_action = np.random.choice([True, False], p=[epsilon, 1 - epsilon])
    if do_random_action:
        return random_strategy(state, actions)

    sorted_actions = np.sort(actions)[::-1]
    rewards = map(lambda action: Game(np.copy(state)).do_action(action),
                  sorted_actions)
    action_index = np.argsort(rewards, kind="mergesort")[-1]
    return sorted_actions[action_index]

# Returns function for Greedy  Strategy
def make_greedy_strategy(get_q_values, verbose=False):
    """Makes greedy_strategy."""

    def greedy_strategy(state, actions):
        """Strategy that always picks the action of maximum Q(state, action)."""
        q_values = get_q_values(state)
        if verbose:
            print("State:")
            print(state)
            print("Q-Values:")
            for action, q_value, action_name in zip(range(4), q_values, ACTION_NAMES):
                not_available_string = "" if action in actions else "(not available)"
                print("%s:\t%.2f %s" % (action_name, q_value, not_available_string))
        sorted_actions = np.argsort(q_values)
        action = [a for a in sorted_actions if a in actions][-1]
        if verbose:
            print("-->", ACTION_NAMES[action])
        return action

    return greedy_strategy

# Returns function for Greedy Epsilon Strategy
def make_epsilon_greedy_strategy(get_q_values, epsilon):
    """Makes epsilon_greedy_strategy."""

    greedy_strategy = make_greedy_strategy(get_q_values)

    def epsilon_greedy_strategy(state, actions):
        """Picks random action with prob. epsilon, otherwise greedy_strategy."""
        do_random_action = np.random.choice([True, False], p=[epsilon, 1 - epsilon])
        if do_random_action:
            return random_strategy(state, actions)
        return greedy_strategy(state, actions)

    return epsilon_greedy_strategy


def make_get_q_values(model):
    """Make get_q_values() function for given session and model."""

    def get_q_values(state):
        """Run inference on a single (4, 4) state matrix."""
        state_vector = state.flatten()
        state_batch = np.array([state_vector])
        q_values_batch = model.predict(np.divide(state_batch, NORMALIZING_FACTOR))
        return q_values_batch[0]

    return get_q_values


def compute_targets(rewards, state_batch, next_state_batch,
                    actions, merged, model):
    GAMMA = 0.9
    (batch_size,) = rewards.shape
    targets = np.zeros((batch_size, 4))
    for i in range(batch_size):
        targets[i] = model.predict(np.divide(state_batch, NORMALIZING_FACTOR))[0]
        Q_sa = np.max(model.predict(np.divide(next_state_batch, NORMALIZING_FACTOR))[0])
        if np.math.isnan(Q_sa):
            targets[i, actions[i]] = rewards[i]
        else:
            # targets[i, actions[i]] = rewards[i] + 0 * Q_sa
            targets[i, actions[i]] = rewards[i] + GAMMA * Q_sa
    return targets
