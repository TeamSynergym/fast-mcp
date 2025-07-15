# app/services/notification.py
"""
사용자에게 알림(이메일, SMS 등)을 보내는 기능을 담당합니다.
"""
def send_email_notification(subject: str, body: str, to_email: str):
    """(시뮬레이션) 목표 달성 축하 이메일을 전송합니다."""
    print("\n" + "=" * 40)
    print("🏆 보상 알림 전송 (Email Simulation)")
    print(f"  [받는 사람]: {to_email}")
    print(f"  [제      목]: {subject}")
    print(f"  [내      용]:\n{body}")
    print("=" * 40)
    return "알림 전송 완료"