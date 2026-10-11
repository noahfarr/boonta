from . import super_mario_bros

games = {"super_mario_bros": super_mario_bros.make}


def make(env_id, **kwargs):
    return games[env_id](**kwargs)
