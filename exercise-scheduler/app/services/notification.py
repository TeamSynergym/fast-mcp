# app/services/notification.py
"""
사용자에게 알림(이메일, SMS 등)을 보내는 기능을 담당합니다.
"""
from typing import Optional, Dict

def send_email_notification(subject: str, body: str, to_email: str, badge_info: Optional[Dict[str, str]] = None):
    """
    (시뮬레이션) 목표 달성 축하 이메일을 전송합니다.
    뱃지 정보가 있는 경우 함께 전송합니다.
    """
    email_content = f"{body}"

    if badge_info:
        badge_section = (
            f"\n\n{'='*20}\n"
            f"🌟 획득한 뱃지: {badge_info.get('badge_name')}\n"
            f"📜 설명: {badge_info.get('badge_description')}\n"
            f"{'='*20}\n"
            f"\n이 뱃지는 '나만의 목표 트로피방'에 전시됩니다!"
        )
        email_content += badge_section

    print("\n" + "=" * 40)
    print("🏆 보상 알림 전송 (Email Simulation)")
    print(f"  [받는 사람]: {to_email}")
    print(f"  [제      목]: {subject}")
    print(f"  [내      용]:\n{email_content}")
    print("=" * 40)
    return "알림 전송 완료"
