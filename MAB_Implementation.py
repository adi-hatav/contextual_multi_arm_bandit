import numpy as np
from copy import copy


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        self.users_distribution = np.array(users_distribution)
        self.arms_thresh = np.array(arms_thresh)
        self.num_users = num_users
        self.phase_len = phase_len
        self.num_arms = num_arms
        self.num_rounds = num_rounds
        # count for each user, how many times he picked each arm
        self.user_arm_choose_count = np.zeros((num_users, num_arms))
        # the sum of retrieved data of reward for every user for every arm
        self.reward_user_arm = np.zeros((num_users, num_arms))
        self.radius_reward = np.zeros((num_users, num_arms))
        self.deactivated = np.zeros((num_arms, 1))
        self.current_round = 0
        # for each phase, countdown of rounds per arm
        self.phase_countdown = np.array(arms_thresh)
        self.current_user = None
        self.current_arm = None
        # array of 0 and 1. 1 means the arm is worth saving
        self.worth_to_save = np.ones((num_arms, 1))

    def update(self, arm, user):
        """
        after each round, will update our data based on the chosen arm
        :param arm: the chosen arm for this round
        :param user: the given user in this round
        """
        self.phase_countdown[arm] -= 1
        self.user_arm_choose_count[user][arm] += 1
        self.current_arm = arm
        self.radius_reward[user][arm] = \
            np.sqrt(0.5 * np.log(self.num_rounds) / self.user_arm_choose_count[user][arm])

        # if this is the end of the phase
        if self.current_round % self.phase_len == 0:
            # calculate the best permutation of arms for this phase
            self.calc_worth_to_save()
            # update deactivate arms
            for i, arm_countdown in enumerate(self.phase_countdown):
                if arm_countdown > 0:
                    self.deactivated[i] = 1
                    self.arms_thresh[i] = 0
                    # update the user's reward and radius to be 0 for deactivated arm (for further calculations)
                    for j in range(self.num_users):
                        self.reward_user_arm[j][i] = 0
                        self.radius_reward[j][i] = 0
            # initialize phase countdown for each round
            self.phase_countdown = np.array(self.arms_thresh)

    def explore_chose_arm(self, user):
        """
        will be used in explore phase, the arm is chosen by its own countdown.
        and if all countdowns as satisfied (reach 0) then will choose the arm that the user have explored the least.
        :param user: the sampled user
        :return: chosen arm
        """
        # first choose from user needs and then from producer needs (countdown)
        for arm in np.argsort(self.user_arm_choose_count[user]):
            if self.phase_countdown[arm] > 0:
                self.update(arm, user)
                return arm
        # if all countdowns are zero or below then choose from user needs
        arm = np.argmin(self.user_arm_choose_count[user])
        self.update(arm, user)
        return arm

    def exploit_chose_arm(self, user):
        """
        first, check what is the best arm (based on reward) for the given user.
        if this arm's countdown is positive then return this arm.

        second, calculate the rank of the user for the arm.
        rank = number of users if this user gets the best rewards for this arm
        rank = 1 if he will have the least reward between all users for this arm

        third, check if there is any arm that needs to be saved.
        if so, iterate over the arms, starting with arm with the highest ucb, and if it is worth saving and activated
        then return it.

        :param user: the sampled user
        :return: the chosen arm
        """
        ucb = np.zeros((self.num_arms, 1))
        for i, arm_count in enumerate(self.user_arm_choose_count[user]):
            if self.deactivated[i] == 0:
                ucb[i] = self.reward_user_arm[user][i] / arm_count + self.radius_reward[user][i]
        arm = np.argmax(ucb)
        # return best arm if its countdown is not yet satisfied (reached 0)
        if self.phase_countdown[arm] > 0:
            self.update(arm, user)
            return arm

        # calculate user's rank for the arm (in comparison to other users)
        rank = np.where(np.argsort(self.reward_user_arm.T[arm]) == user)[0][0] + 1
        sorted_ucb = np.argsort(ucb.T)[::-1]
        # is there an arm that needs to be saved
        if (0.7 + 0.3 / rank) * (self.phase_len - (self.current_round % self.phase_len)) < sum(
                self.phase_countdown[self.phase_countdown > 0]):
            for i in sorted_ucb[0]:
                if self.phase_countdown[i] > 0 and self.worth_to_save[i] and self.deactivated[i] == 0:
                    self.update(i, user)
                    return i

        # if there are no arms worth saving return the best arm for user
        self.update(arm, user)
        return arm

    def calc_total_reward(self, permutation):
        """
        calculates the total reward based on the given permutation or activate arms.
        calc reward as if we gave each user its best arm.
        :param: permutation: A list or array representing the permutation of arms.
        :return: The total reward obtained from the given permutation.
        """
        total_reward = 0
        user_countdown = self.users_distribution * self.phase_len
        # changing thresh to 0 for deactivate arms
        thresh_current_countdown = np.array(self.arms_thresh) * permutation
        for i in np.argsort(thresh_current_countdown)[::-1]:
            best_users = np.argsort(self.reward_user_arm.T[i])[::-1]
            for user in best_users:
                if thresh_current_countdown[i] > 0:
                    if user_countdown[user] > 0:
                        num_to_subtract = min(user_countdown[user], thresh_current_countdown[i])
                        thresh_current_countdown[i] -= num_to_subtract
                        user_countdown[user] -= num_to_subtract
                        total_reward += num_to_subtract * self.reward_user_arm[user][i]

        # if there are users with leftover countdown
        for user in range(self.num_users):
            if user_countdown[user] > 0:
                total_reward += user_countdown[user] * np.max(self.reward_user_arm[user] * permutation)

        return total_reward

    def calc_worth_to_save(self):
        """
        calculate expected reward for all possible permutations of arm activations.

        :return: A numpy array of the best permutations of arms, 1 represents worth to save and 0 otherwise.
        """
        all_permutations = [[0]]
        if self.deactivated[0] == 0:
            all_permutations.append([1])

        # generate all possible permutations of arm activations
        for i in range(1, self.num_arms):
            for j in range(len(all_permutations)):
                if self.deactivated[i] == 0:
                    temp = copy(all_permutations[j])
                    temp.append(1)
                    all_permutations.append(temp)
                all_permutations[j].append(0)

        max_reward = 0
        max_permutation = []
        # iterate over all permutations and find the one with maximum reward
        for permutation in all_permutations:
            reward = self.calc_total_reward(np.array(permutation))
            if reward > max_reward:
                max_reward = reward
                max_permutation = permutation

        self.worth_to_save = np.array(max_permutation)

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        self.current_user = user_context
        self.current_round += 1

        # Explore phase
        if 0.0045 * self.num_rounds >= self.current_round:
            return self.explore_chose_arm(user_context)

        # Exploit phase
        else:
            return self.exploit_chose_arm(user_context)

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # sum the rewards for each user and arm
        self.reward_user_arm[self.current_user][self.current_arm] += reward

    def get_name(self):
        return "Adi Hatav simulation"
