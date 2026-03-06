"""Misc tools
"""

def convert_to_yaml(overrides):
    """Convert args to yaml for overrides
    https://speechbrain.readthedocs.io/en/stable/_modules/speechbrain/core.html#parse_arguments

    This library includes code from SpeechBrain
    SpeechBrain is released under the Apache License, version 2.0. 
    """
    yaml_string = ""

    # Handle '--arg=val' type args
    joined_args = "=".join(overrides)
    split_args = joined_args.split("=")

    for arg in split_args:
        if arg.startswith("--"):
            yaml_string += "\n" + arg[len("--") :] + ":"
        else:
            yaml_string += " " + arg

    return yaml_string.strip()
