from unityagents import UnityEnvironment
import numpy as np

class EnvWrapper():
    """
    Wrapper for unity framework to match OpenAI environment interface
    """
    def __init__(self, filename):
        self.env = UnityEnvironment(file_name=filename)
        self.brain_name = self.env.brain_names[0]
        
        brain = self.env.brains[self.brain_name]
        
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        self.nA = brain.vector_action_space_size
        print('Number of actions:', self.nA)
        
        # number of agents in the environment
        print('Number of agents:', len(env_info.agents))
        # examine the state space 
        state = env_info.vector_observations[0]
        print('States look like:', state)
        self.nS = len(state)
        print('State space dimension:', self.nS)

    def reset(self):
        """
        Returns the state, as OpenAI env would
        """
        env_info = self.env.reset(train_mode = True)[self.brain_name]
        return np.array(env_info.vector_observations)
    
    def step(self, action):
        """
        Updates the environment with action and sends feedback to the agent
        """
        env_info = self.env.step(action)[self.brain_name]
        next_state, reward, done = np.array(env_info.vector_observations), np.array(env_info.rewards), np.array(env_info.local_done)
        if np.any(done):
            next_state = self.reset()
            
        return next_state, reward, done

    def close(self):
        self.env.close()