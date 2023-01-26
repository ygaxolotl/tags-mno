import json

def get_trained_model_cfg(digest, dir_out):
    """
    Returns trained model cfg given hex digest code
    """
    with (dir_out / "{}_cfg.json".format(digest)).open() as jf:
        cfg = json.load( jf )
    return cfg
