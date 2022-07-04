import requests


def send_line_notify(notification_message: str, token: str):
    """
    LINEに通知する
    """
    line_notify_token = token
    line_notify_api = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {"message": f"{notification_message}"}
    requests.post(line_notify_api, headers=headers, data=data)
