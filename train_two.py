
import torch
import torch.nn.functional as F
import time
import random
import numpy as np
from collections import deque
import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os
#from tabulate import tabulate
# Some Functions
def sensor_front_sig(data):
    player = []
    sensor_data = []
    for sensor in range(33):
        player.append(data[8 * sensor:(8 * sensor) + 8])

    for stack in range(3):
        sensor_data.append(player[11 * stack:(11 * stack) + 11])

    return sensor_data

def sensor_back_sig(data):
    player = []
    sensor_data = []
    for sensor in range(9):
        player.append(data[8 * sensor:(8 * sensor) + 8])

    for stack in range(3):
        sensor_data.append(player[3 * stack:(3 * stack) + 3])

    return sensor_data


from model import QNetwork
from replay_memory import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, behavior_name, index_player, replay_memory_size=1e4, batch_size=128, gamma=0.99,
                 learning_rate = 1e-4, target_tau=1e-3, update_rate=4, seed=0):
        self.state_size = state_size
        self.current_state = []
        self.action_size = action_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)
        self.behavior_name = behavior_name
        self.index_player = index_player
        self.close_ball_reward = 0
        self.touch_ball_reward = 0

        """
        Now we define two models: 
        (a) one netwoek will be updated every (step % update_rate == 0),
        (b) A target network, with weights updated to equal to equal to the network (a) at a slower (target_tau) rate.
        """

        self.network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network =  QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr= self.learn_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # Initialize time step ( for updating every UPDATE_EVERY steps)
        self.t_step = 0
    def load_model(self, path_model, path_target = None):
        params = torch.load(path_model)
        #self.network.set_params(params)
        self.network.load_state_dict(torch.load(path_model))
        if path_target != None:
            self.target_network.load_state_dict(torch.load(path_target))
    def model_step(self, state, action, reward, next_state):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state)

        # learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1)%1003 #% self.update_rate
        if self.t_step% self.update_rate == 0:
            print('LEAR HERE')
            # if enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma,self.t_step)

    def choose_action(self, state, eps = 0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) # return a number from 0 to action_size
        else:
            return random.choice(np.arange(self.action_size)) # return a number from 0 to action_size

    def learn(self, experiences, gamma,stp):
        states, actions, rewards, next_states = experiences

        # Get Q values from current observations (s,a) using model network
        # get max Q values for (s', a') from target model
        self.network.train()
        Q_sa = self.network(states).gather(1, actions)
        #print(Q_sa)
        Q_sa_prime_target_values = self.target_network(next_states).max(1)[0].to(device).float().detach()
        #Q_sa_prime_targets = Q_sa_prime_target_values.max(1)[0].unsqueeze(1)
        #print(Q_sa_prime_target_values)

        # compute Q targets for current states
        #print(rewards)

        Q_sa_targets = rewards + gamma * Q_sa_prime_target_values.unsqueeze(1)
        #print(Q_sa_targets)
        #input('train')

        #Q_sa_targets = Q_sa_targets.unsqueeze(1)
        


        # Compute loss (error)
        criterion = torch.nn.MSELoss(reduction='sum')
        loss = criterion(Q_sa.to(device),Q_sa_targets.to(device))#F.mse_loss(Q_sa, Q_sa_targets)


        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        if(stp%1000==0):
            self.soft_update(self.network, self.target_network, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    def Read(self):
        decision_steps, terminal_steps = env.get_steps(self.behavior_name)
        try:
            signal_front = np.array(sensor_front_sig(decision_steps.obs[0][self.index_player, :]))  # 3 x 11 x 8
            signal_back = np.array(sensor_back_sig(decision_steps.obs[1][self.index_player, :]))  # 3 x 3 x 8
            pre_state = []
            #table = tabulate(signal_front[0], tablefmt="fancy_grid",floatfmt=".2f")
            #print(table)
            #table = tabulate(signal_back[0], tablefmt="fancy_grid", floatfmt=".2f")
            #print(table)
            #print('ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss')
            signal_front =np.array(signal_front)
            #print(signal_front.shape)
            #print(signal_back.shape)
            r = np.concatenate((signal_front,signal_back),axis=1)
            #print(r.shape)
            #input('ff')
            #pre_state.extend(list(np.array(signal_front).flatten()))
            #pre_state.extend(list(np.array(signal_back).flatten()))
            #state = np.array(pre_state)
            self.current_state = r
            count_close_to_ball = 0
            count_touch_ball = 0
            count_back_touch = 0
            count_back_close = 0
            self.rew_d_to_our_post =0
            self.rew_for_ball_dist = -0.1
            for i in range(len(signal_front[0])):
                if signal_front[0][i][0] == 1.0:
                    print('baaallllll')
                    count_close_to_ball+= 1
                    self.rew_for_ball_dist = max(0.3*(1 - signal_front[0][i][7]),self.rew_for_ball_dist)

                    if signal_front[0][i][7] <= 0.03:
                        count_touch_ball += 1

                if signal_front[0][i][1] == 1.0:
                    self.rew_d_to_our_post =-0.1
                if signal_front[0][i][2] == 1.0:
                    self.rew_d_to_our_post =0.1


            for i in range(len(signal_back[0])):
                if signal_back[0][i][0] == 1.0:
                    count_back_close+= 1
                    if signal_back[0][i][7] <= 0.03:
                        count_back_touch += 0
            self.back_touch = 1 if count_back_touch>0 else 0
            self.back_close = 1 if count_back_close>0 else 0
            # add reward if kick the ball
            self.touch_ball_reward = 2.5 if count_touch_ball > 0 else 0
            #if count_back_touch>0:
            #    self.touch_ball_reward= -0.3
            
            # penalize if the ball is not in view
            self.close_ball_reward = -0.15 if count_close_to_ball == 0 else 0.25
            if count_back_close >0:
                self.close_ball_reward = 0.1
            #if count_back_close >0:
            #    self.close_ball_reward = -0.15

            return self.current_state
        except:
            self.touch_ball_reward = 0
            self.close_ball_reward = 0
        return self.current_state
    def upd_after_goal(self, n_upds):
        self.memory.upd_goal(n_upds)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma,self.t_step)
    def we_goll(self):
        self.memory.we_goll()
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma,self.t_step)
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma,self.t_step)
    def us_goll(self):
        self.memory.us_goll()
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma,self.t_step)
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma,self.t_step)


def transfer_6_striker(n):
    list_actions = [(1,0,0), (2,0,0), (0,1,0), (0,2,0), (0,0,1), (0,0,2)]
    return list_actions[n]
def transfer_4_defender(n):
    list_actions = [(1,0,0), (2,0,0), (0,1,0), (0,2,0)]
    return list_actions[n]

def process_reward_list_agressive_striker(reward_list):
    if reward_list[0] > 0:
        return 100.0
    elif reward_list[0] == -1.0 or reward_list[0] == -2.0: # get scored on
        return -0.01
    else:  # apply existing penalty
        return -0.01

def process_reward_list_defender(reward_list):
    if reward_list[0] == -1.0 or reward_list[0] == -2.0:
        return -100.0
    elif reward_list[0] > 0:
        return 0.1
    else: # apply existing reward
        return 0.01

num_episodes = 10000
num_game_each_episode = 600
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99
scores = [[],[]]
best_avg_score = 0
scores_average_window = 5
striker_solved_score = 1
defender_solved_score = -1

# configure model
action_size_striker = 6
action_size_defender = 4
state_size = 14*8*3



np.set_printoptions(precision=3)



channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name = './final_proj/CoE202', side_channels=[channel])
channel.set_configuration_parameters(time_scale = 4)
env.reset()

behavior_name_pink = list(env.behavior_specs)[0]
# behavior_name_pink = SoccerTwos?team=0
behavior_name_blue = list(env.behavior_specs)[1]
print(behavior_name_pink,behavior_name_blue )
#input('bes')
# behavior_name_blue = SoccerTwos?team=1

agent_b1 = Agent(state_size = state_size, action_size = action_size_striker, behavior_name= behavior_name_blue, index_player= 0)
agent_b1.load_model("./res/p1_act_last_for22234","./res/p1_target_last_for22234")
agent_b2 = Agent(state_size = state_size, action_size = action_size_defender, behavior_name = behavior_name_blue, index_player= 1)
agent_b2.load_model("./res/b2_act_lat_for22234","./res/b2_target_last_for22234")
agent_p1 = Agent(state_size = state_size, action_size = action_size_striker, behavior_name = behavior_name_pink, index_player= 0)
agent_p1.load_model("./res/b1_act_last_for22234","./res/b1_target_last_for22234")
agent_p2 = Agent(state_size = state_size, action_size = action_size_defender, behavior_name = behavior_name_pink, index_player= 1)
agent_p2.load_model("./res/p2_act_lat_for22234","./res/p2_target_last_for22234")

#sub_last_episode_dirs = os.listdir("paper_save_model/one_team/last_episode")
#new_last_episode_dir = "paper_save_model/one_team/last_episode/version" + str(len(sub_last_episode_dirs) + 1)
new_last_episode_dir = './res_mod'
#os.makedirs(new_last_episode_dir)
epsss=1
for i_episode in range(1, num_episodes + 1):
    env.reset()
    tmp_score = [0,0]
    state_p1 = agent_p1.Read()  # torch.tensor(112-array)
    state_p2 = agent_p2.Read()  # torch.tensor(112-array)
    
    state_b1 = agent_b1.Read()  # torch.tensor(112-array)
    state_b2 = agent_b2.Read()
    list_reward_p1 = []
    list_reward_p2 = []
    for i_game in range(num_game_each_episode):
        print(i_episode, i_game)
        #array_action_b1 = transfer_6_striker(random.randint(0,5))
        #array_action_b1 = [0,0,0]
        #array_action_b2 = transfer_4_defender(random.randint(0,3))
        #array_action_b2 =[0,0,0]

        action_b1 = agent_b1.choose_action(agent_b1.current_state,epsss) # a number from 0 to 5
        array_action_b1 = transfer_6_striker(action_b1)

        action_b2 = agent_b2.choose_action(agent_b2.current_state,epsss) # a number from 0 to 3
        array_action_b2 = transfer_4_defender(action_b2)
        if(i_episode<500):
            epsss = 0.5 -i_episode*0.001



        action_p1 = agent_p1.choose_action(agent_p1.current_state,epsss) # a number from 0 to 5
        array_action_p1 = transfer_6_striker(action_p1)

        action_p2 = agent_p2.choose_action(agent_p2.current_state,epsss) # a number from 0 to 3
        array_action_p2 = transfer_4_defender(action_p2)


        epsss = epsss - 0.01
        if epsss< 0.3 -i_episode*0.01  :
            epsss =1 -i_episode*0.01
        if (i_episode>500 and random.random()>0.95 ):
            epsss =1
        elif(i_episode>500):
            epsss = 0



        
        env.set_actions(behavior_name_blue, np.array([array_action_b1, array_action_b2]))
        env.set_actions(behavior_name_pink, np.array([array_action_p1, array_action_p2]))


        env.step()

        next_state_p1 = agent_p1.Read()
        next_state_p2 = agent_p2.Read()

        next_state_b1 = agent_b1.Read()
        next_state_b2 = agent_b2.Read()

        dec_p, terminal_p = env.get_steps(behavior_name_pink)
        dec_b, terminal_b = env.get_steps(behavior_name_blue)
        ####
        goal_reward_p1 = process_reward_list_agressive_striker(dec_p.reward)
        goal_reward_p2 = process_reward_list_defender(dec_p.reward)

        goal_reward_b1 = process_reward_list_agressive_striker(dec_b.reward)
        goal_reward_b2 = process_reward_list_defender(dec_b.reward)
        #print(dec_p)
        #print(dec_p.reward)
        #print(goal_reward_p1)
        #print(goal_reward_p2)
        #input('hui')
        # p1 is striker
        #reward_p1 = agent_p1.close_ball_reward + agent_p1.touch_ball_reward + goal_reward_p1
        reward_p1 =  agent_p1.touch_ball_reward + goal_reward_p1# - 0.001*i_game #agent_p1.close_ball_reward - 0.001*i_game# +agent_p1.rew_d_to_our_post# +agent_p1.rew_d_to_our_post
        reward_b1 =  agent_b1.touch_ball_reward + goal_reward_b1# - 0.001*i_game #agent_b1.close_ball_reward - 0.001*i_game
        if(agent_p1.touch_ball_reward>0):
            agent_p1.upd_after_goal((min(45,i_game)))
        if(agent_p2.touch_ball_reward>0):
            agent_p2.upd_after_goal((min(45,i_game))) 
        if(goal_reward_p1>1):
            agent_p1.we_goll()
            #agent_p2.we_goll()
        if(goal_reward_p2<-1):
            #agent_p1.us_goll()
            agent_p2.us_goll()


        #########for B########

        if(agent_b1.touch_ball_reward>0):
            agent_b1.upd_after_goal((min(40,i_game)))
        if(agent_b2.touch_ball_reward>0):
            agent_b2.upd_after_goal((min(40,i_game))) 
        if(goal_reward_b1>1):
            agent_b1.we_goll()
            #agent_b2.we_goll()
        if(goal_reward_b2<-1):
            #agent_b1.us_goll()
            agent_b2.us_goll()

        # p2 is defender
        #reward_p2 = goal_reward_p2 + agent_p2.close_ball_reward
        reward_p2 =  agent_p2.touch_ball_reward + goal_reward_p2# + goal_reward_p2 #+ agent_p2.close_ball_reward #+ agent_p2.rew_for_ball_dist #+agent_p2.rew_d_to_our_post
        reward_b2 =  agent_b2.touch_ball_reward + goal_reward_b2# + goal_reward_b2 #+ agent_b2.close_ball_reward
        print(reward_p1)
        print(reward_p2)
        #input('rew')
        list_reward_p1.append(reward_p1)
        list_reward_p2.append(reward_p2)

        agent_p1.model_step(state_p1, action_p1, reward_p1, next_state_p1)
        agent_p2.model_step(state_p2, action_p2, reward_p2, next_state_p2)

        agent_b1.model_step(state_b1, action_b1, reward_b1, next_state_b1)
        agent_b2.model_step(state_b2, action_b2, reward_b2, next_state_b2)

        state_p1 = next_state_p1
        state_p2 = next_state_p2

        state_b1 = next_state_b1
        state_b2 = next_state_b2

        tmp_score[0] += goal_reward_p1
        tmp_score[1] += goal_reward_p2

        # check there is goal
        if dec_p.reward[0] != 0:
            env.reset()
            print("goal, reset")
            print("dec_p.reward: ", dec_p.reward)

    scores[0].append(tmp_score[0])
    scores[1].append(tmp_score[1])
    average_scores = [ np.mean(score[i_episode-min(i_episode,scores_average_window):i_episode+1]) for score in scores]

    # Decrease epsilon for epsilon-greedy policy by decay rate
    epsilon = max(epsilon_min, epsilon_decay*epsilon)

    # print average score
    print("Episode ", i_episode, " - average_scores: ", average_scores, "; reward p1: ", np.sum(list_reward_p1), " / reward p2 : ", np.sum(list_reward_p2))

    if average_scores[0] > striker_solved_score and average_scores[1] > defender_solved_score:
        print("Environment solved in episodes ", i_episode, " \ average_scores = ", average_scores)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        nn_p1_filename = './res/p1_mod_for2'#"paper_save_model/one_team/above_standard/Trained_p1_" + timestr + "_" + str(i_episode) + ".pth"
        nn_p2_filename = './res/p2_mod_for2'#"paper_save_model/one_team/above_standard/Trained_p2_" + timestr + "_" + str(i_episode) + ".pth"
        torch.save(agent_p1.network.state_dict(), nn_p1_filename)
        torch.save(agent_p2.network.state_dict(), nn_p2_filename)

    if i_episode % 50 == 0:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        nn_last_p1_filename = './res/p1_act_last_for222345' #new_last_episode_dir + "/p1_episode_" + str(i_episode) + ".pth"
        nn_last_target_p1_filename = "./res/p1_target_last_for222345"#new_last_episode_dir + "/target_p1_episode_" + str(i_episode) + ".pth"

        nn_last_p2_filename = "./res/p2_act_lat_for222345"#new_last_episode_dir + "/p2_episode_" + str(i_episode) + ".pth"
        nn_last_target_p2_filename = "./res/p2_target_last_for222345"#new_last_episode_dir + "/target_p2_episode_" + str(i_episode) + ".pth"

        #torch.save(agent_p1.network.state_dict(), nn_last_p1_filename)
        #torch.save(agent_p2.network.state_dict(), nn_last_p2_filename)
        #torch.save(agent_p1.target_network.state_dict(), nn_last_target_p1_filename)
        #torch.save(agent_p2.target_network.state_dict(), nn_last_target_p2_filename)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        nn_last_p1_filename = './res/b1_act_last_for222345' #new_last_episode_dir + "/p1_episode_" + str(i_episode) + ".pth"
        nn_last_target_p1_filename = "./res/b1_target_last_for22234"#new_last_episode_dir + "/target_p1_episode_" + str(i_episode) + ".pth"

        nn_last_p2_filename = "./res/b2_act_lat_for22234"#new_last_episode_dir + "/p2_episode_" + str(i_episode) + ".pth"
        nn_last_target_p2_filename = "./res/b2_target_last_for222345"#new_last_episode_dir + "/target_p2_episode_" + str(i_episode) + ".pth"

        #torch.save(agent_b1.network.state_dict(), nn_last_p1_filename)
        #torch.save(agent_b2.network.state_dict(), nn_last_p2_filename)
        #torch.save(agent_b1.target_network.state_dict(), nn_last_target_p1_filename)
        #torch.save(agent_b2.target_network.state_dict(), nn_last_target_p2_filename)


env.close()









