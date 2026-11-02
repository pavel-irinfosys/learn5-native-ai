
import gymnasium as gym
import torch
from model_rl import ModelRL

D_TYPE = torch.float16
D_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model):
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    done = False
    
    while not done:
        obs_t = torch.as_tensor(obs, dtype=D_TYPE, device=D_DEVICE)
        action = model(obs_t).argmax().item()
        print(f"{obs_t} : {action}")
        obs, _, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
    env.close()

def main():
    model = ModelRL()
    test(model)



if __name__ == "__main__":
    main()