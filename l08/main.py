
import sys
from model_rl import ModelRL
import gymnasium as gym

def main():
    model = ModelRL()
    if "train" in sys.argv:
        print("start training")
        env = gym.make("CartPole-v1")
        model.train_ml(env)
        env.close()
    model.eval()

if __name__ == "__main__":
    main()
