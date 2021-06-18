import argparse

import cv2
from graphic_pomme_env.wrappers import NUM_STACK
from stable_baselines3 import PPO

from scripts.agents import make_actor, ACTORS
from scripts.env import GraphicPommerEnv

parser = argparse.ArgumentParser(description="Export a learned model to ONNX")
parser.add_argument("--model", help="Path to your model .zip file")


def make_video(observations, prefix):
    images = observations
    height, width, layer = images[0].shape
    video_name = f'{prefix}-video.avi'
    video = cv2.VideoWriter(video_name, 0, 3, (width, height))
    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    args = parser.parse_args()
    env = GraphicPommerEnv(num_stack=NUM_STACK,
                           start_pos=0,
                           opponent_actor=make_actor(ACTORS.simple),
                           board="GraphicOVOCompact-v0")
    model = PPO.load(path=args.model)

    # test the learned model
    num_win = 0
    num_tie = 0
    num_lose = 0
    total = 50  # number of playouts
    for i_episode in range(total):
        obs = env.reset()
        done = False
        info = None
        observations = []
        while not done:
            action_training, _states = model.predict(obs)
            obs, reward, done, info = env.step(action_training)
            observations.append(env.get_rgb_img())
        print("Episode {} finished".format(i_episode))

        # make a video
        make_video(observations, f"videos/{i_episode}")

        if (info["result"].value == 0):
            if (0 in info["winners"]):
                num_win += 1
            else:
                num_lose += 1
        elif (info["result"].value == 2):
            num_tie += 1

    print(f"Win {num_win} / {total} games")
    print(f"Tie {num_tie} / {total} games")
    print(f"Lose {num_lose} / {total} games")
