"""Settings for the analysis module."""
from snorefox_med.utils.env_config import get_env_var_value, load_params_to_environment_variables, str_to_bool

parameter_names_to_env_vars = {
    # Don't forget the leading slash!
    "is-logging-level-debug": "IS_LOGGING_LEVEL_DEBUG",
    "sqs-queue-url": "SQS_QUEUE_URL",
    "/api/url": "API_URL",
    "/api/token": "API_TOKEN",
}

load_params_to_environment_variables(parameter_names_to_env_vars=parameter_names_to_env_vars)

SQS_QUEUE_URL = get_env_var_value("SQS_QUEUE_URL")
RETRIES_SQS_MESSAGE: int = 5
IS_LOGGING_LEVEL_DEBUG = str_to_bool(get_env_var_value("IS_LOGGING_LEVEL_DEBUG"))

API_URL = get_env_var_value("API_URL")
API_TOKEN = get_env_var_value("API_TOKEN")
HTTPX_RETRIES: int = 5
