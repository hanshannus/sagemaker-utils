import os
import json
from pathlib import Path
from typing import List, Dict

# SageMaker default input paths
SM_INPUT_DIR = os.environ.get("SM_INPUT_DIR", ".")
SM_CHECKPOINT_DIR = "/opt/ml/checkpoints" if SM_INPUT_DIR != "." else "."
SM_OUTPUT_DIR = os.environ.get("SM_OUTPUT_DIR", "./outputs")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "./outputs/model")


def get_channel_names() -> List[str]:
    """Read the channel names from the environment variable 'SM_CHANNELS'.

    The environment variable is expected to be a comma-separated list of
    strings which is encoded as a single string, i.e.:
        SM_CHANNELS = '["<channel name 1>", ..., "<channel name n>"]'

    SM_CHANNELS is loaded with json.loads() and returned as a list of strings.

    Returns
    -------
    List[str]
        List of SageMaker channel names. The list is empty if Python is not
        running in a SageMaker environment or if no SageMaker channels are
        defined.

    """
    return json.loads(os.environ.get("SM_CHANNELS", "[]"))


def get_channel_environment_variable_names(
    channel_names: str = None,
) -> List[str]:
    """Read the channel names from the environment variable 'SM_CHANNELS'.

    Get the list of channel names from the environment variable 'SM_CHANNELS'
    with :func:`get_sagemaker_channel_names` and prepend 'SM_CHANNEL_' to each
    channel name, i.e.:
        [SM_CHANNEL_"<channel name 1>", ..., SM_CHANNEL_"<channel name n>"]

    Parameters
    ----------
    channel_names : str, optional
        Channel names, by default None.
        Return all channels that are defined by environment variables if None.

    Returns
    -------
    List[str]
        List of SageMaker channel environment variables. The list is empty if
        Python is not running in a SageMaker environment or if no SageMaker
        channels are defined.
    """
    all_names = get_channel_names()
    print("all_names", all_names)
    if channel_names is None:
        return [f"SM_CHANNEL_{s.upper()}" for s in all_names]
    #
    return [f"SM_CHANNEL_{s.upper()}" for s in channel_names if s in all_names]


def get_channel_paths(channel_names: str = None) -> List[Path]:
    """Get sagemaker channel paths defined in environment variable 'SM_CHANNELS'.

    Get the list of channel environment variable names with
    :func:`get_sagemaker_channel_environment_variable_names` and return a list
    of channel paths, i.e.:
        ["/opt/ml/input/data/<channel name 1>", ...,
         "/opt/ml/input/data/<channel name n>"]

    Parameters
    ----------
    channel_names : str, optional
        Channel names, by default None.
        Return all channels that are defined by environment variables if None.

    Returns
    -------
    List[str]
        List of SageMaker channel paths. The list is empty if Python is not
        running in a SageMaker environment or if no SageMaker channels are
        defined.
    """
    env = os.environ
    func = get_channel_environment_variable_names
    return [Path(env[i]) for i in func(channel_names=channel_names)]


def get_channels() -> Dict[str, Path]:
    """Get sagemaker channel paths from environment variables.

    Get the list of channel path with :func:`get_sagemaker_channel_paths` and
    return a dictionary of channel name and channel paths as key and value
    respectively, i.e.:
        {
            "<channel name 1>": "/opt/ml/input/data/<channel name 1>",
            ...,
            "<channel name n>": "/opt/ml/input/data/<channel name n>",
        }

    Returns
    -------
    Dict[str, str]
        Dictionary of channel name and channel paths. The dictionary is empty if
        Python is not running in a SageMaker environment or if no SageMaker
        channels are defined.
    """
    return {p.stem: p for p in get_channel_paths()}
