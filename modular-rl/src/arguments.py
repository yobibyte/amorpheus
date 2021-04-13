import argparse
import datetime


def get_args():

    running_start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--seed", default=0, type=int, help="sets Gym, PyTorch and Numpy seeds"
    )
    parser.add_argument(
        "--debug", default=0, type=int, help="Do not log to sacred if on"
    )
    parser.add_argument(
        "--save_buffer",
        default=0,
        type=int,
        help="If set to 1, save the buffer together with the model.",
    )
    parser.add_argument(
        "--label", type=str, default="", help="label of a sacred experiment"
    )
    parser.add_argument(
        "--morphologies",
        nargs="*",
        type=str,
        default=["walker"],
        help="which morphology env to run (walker, hopper, etc)",
    )
    parser.add_argument(
        "--custom_xml",
        type=str,
        default=None,
        help="path to MuJoCo xml files (can be either one file or a directory containing multiple files)",
    )
    parser.add_argument(
        "--start_timesteps",
        default=1e4,
        type=int,
        help="How many time steps purely random policy is run for?",
    )
    parser.add_argument(
        "--max_timesteps", type=int, default=10e6, help="number of timesteps to train"
    )
    parser.add_argument(
        "--expl_noise",
        default=0.126,
        type=float,
        help="std of Gaussian exploration noise",
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="batch size for both actor and critic",
    )
    parser.add_argument("--discount", default=0.99, type=float, help="discount factor")
    parser.add_argument(
        "--tau", default=0.046, type=float, help="target network update rate"
    )
    parser.add_argument(
        "--policy_noise",
        default=0.2,
        type=float,
        help="noise added to target policy during critic update",
    )
    parser.add_argument(
        "--noise_clip",
        default=0.5,
        type=float,
        help="range to clip target policy noise",
    )
    parser.add_argument(
        "--policy_freq", default=2, type=int, help="frequency of delayed policy updates"
    )
    parser.add_argument(
        "--video_length",
        default=10,
        type=int,
        help="length of video to generate (in seconds)",
    )
    parser.add_argument(
        "--msg_dim",
        default=32,
        help="message dimension when trained modularly with message passing",
    )
    parser.add_argument(
        "--disable_fold",
        action="store_true",
        help="disable the use of pytorch fold (used for accelerating training)",
    )
    parser.add_argument(
        "--lr", default=0.0005, type=float, help="learning rate for Adam"
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=1000,
        help="maximum number of timesteps allowed in one episode",
    )
    parser.add_argument(
        "--save_freq",
        default=5e5,
        type=int,
        help="How often (time steps) we save the model and the replay buffer?",
    )
    parser.add_argument(
        "--td", action="store_true", help="enable top down message passing"
    )
    parser.add_argument(
        "--bu", action="store_true", help="enable bottom up message passing"
    )
    parser.add_argument(
        "--rb_max",
        type=int,
        default=10e6,
        help="maximum replay buffer size across all morphologies",
    )
    parser.add_argument(
        "--max_children",
        type=int,
        default=None,
        help="maximum number of children allowed at each node (optional; facilitate model loading if max_children is different at training time)",
    )
    parser.add_argument(
        "--actor_type",
        type=str,
        default="smp",
        choices=["smp", "transformer"],
        help="Type of the actor to use",
    )
    parser.add_argument(
        "--critic_type",
        type=str,
        default="smp",
        choices=["smp", "transformer"],
        help="Type of the critic to use",
    )
    parser.add_argument(
        "--grad_clipping_value",
        type=float,
        default=-1,
        help="Clip grad by this value both for actor and critic. If < 0, do not clip",
    )

    parser.add_argument(
        "--expID",
        type=str,
        default=None,
        help="Clip grad by this value both for actor and critic. If < 0, do not clip",
    )

    parser.add_argument(
        "--condition_decoder_on_features",
        default=0,
        type=int,
        help="Concat input to the decoder with the features of the joint",
    )

    parser.add_argument(
        "--attention_layers",
        default=1,
        type=int,
        help="How many attention layers to stack",
    )

    parser.add_argument(
        "--attention_heads",
        default=1,
        type=int,
        help="How many attention heads to stack",
    )

    parser.add_argument(
        "--transformer_norm", default=0, type=int, help="Use layernorm",
    )

    parser.add_argument(
        "--observation_graph_type",
        type=str,
        default="morphology",
        choices=["morphology", "tree", "line"],
        help="How is the observation graph built",
    )

    parser.add_argument(
        "--attention_hidden_size",
        type=int,
        default=128,
        help="Hidden units in an attention block",
    )

    parser.add_argument(
        "--attention_embedding_size",
        type=int,
        default=128,
        help="Hidden units in an attention block",
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="How much to drop if drop in transformers",
    )

    args = parser.parse_args()
    if args.expID is None:
        args.expID = str(running_start_time.strftime("%Y_%m_%d-%X"))
    else:
        import json
        import os

        eval_exp_id = args.expID
        envs2eval = None
        custom_xml = None
        max_children = args.max_children
        if not args.custom_xml:
            envs2eval = args.morphologies
        else:
            custom_xml = args.custom_xml
        with open(os.path.join("results", args.expID, "args.txt"), "r") as f:
            args.__dict__ = json.load(f)
        args.__dict__["expID"] = eval_exp_id
        if envs2eval:
            args.__dict__["morphologies"] = envs2eval
            args.__dict__["custom_xml"] = None
        else:
            args.__dict__["custom_xml"] = custom_xml

        args.__dict__["max_children"] = max_children
        args.__dict__["seed"] = 42

    return args
