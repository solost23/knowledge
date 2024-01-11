from flask import jsonify


def success(message: str, data: any) -> {}:
    return response(
        code=0,
        success=True,
        message=message,
        data=data
    )


def error(code: int, err: str) -> {}:
    return response(
        code=code,
        success=False,
        message=err,
        data=None,
    )


def response(
        code: int,
        success: bool,
        message: str,
        data: any,
) -> {}:
    return {
        'code': code,
        'success': success,
        'message': message,
        'data': data,
    }
