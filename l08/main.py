
import sys
from model_rl import ModelRL
import gymnasium as gym

def main():
    model = ModelRL()
    if "train" in sys.argv:
        print("start training")
        envs = [gym.make("CartPole-v1") for _ in range(10)]
        model.train_ml(envs, 1000)
        for env in envs:
            env.close()
    model.eval()

if __name__ == "__main__":
    main()
