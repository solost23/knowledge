from typing import Any, Tuple

def success(message: str, data: Any) -> Tuple[dict, int]:
    return _response(code=0, success=True, message=message, data=data), 200


def error(code: int, err: str, data: Any = None) -> Tuple[dict, int]:
    return _response(code=code, success=False, message=err, data=data), code


def _response(code: int, success: bool, message: str, data: Any) -> dict:
    return {
        'code': code,
        'success': success,
        'message': message,
        'data': data,
    }
