import model as m
import gym

cart_pole = gym.make("CartPole-v1", render_mode="human")
agent = m.Agent(cart_pole, cart_pole.observation_space.shape[0], cart_pole.action_space.n)

m.train(agent)
