from typing import Any, Tuple

from flask import jsonify


def success(message: str, data: Any) -> Tuple[dict, int]:
    return _response(code=0, success=True, message=message, data=data), 200


def error(code: int, err: str) -> Tuple[dict, int]:
    return _response(code=code, success=False, message=err, data=None), code


def _response(code: int, success: bool, message: str, data: Any) -> dict:
    return {
        'code': code,
        'success': success,
        'message': message,
        'data': data,
    }
