"""Pass and store parameters (dicts) in decorated functions"""

from functools import wraps

err_msg = lambda k: f"Cannot override, `{k}`, choose a different key!"

# --- Decorator with arguments
def inherit(defined_dict):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            custom_dict = {}
            customs = func(*args, **kwargs)

            for k, v in defined_dict.items():
                custom_dict[k] = v

            for k, v in customs.items():
                assert k not in defined_dict, err_msg(k)
                custom_dict[k] = v

            return custom_dict

        return wrapper

    return decorator
