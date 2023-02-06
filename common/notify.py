import requests


def send_notification(notification_message: str, token: str):
    """Send nortification.

    Send message after ending all pipeline processes.
    Replace your prefer notification tools.
    """
    line_notify_token = token
    line_notify_api = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {"message": f"{notification_message}"}
    requests.post(line_notify_api, headers=headers, data=data)
