'''
此脚本用于生成基于机器码和有效期的 License Key。

重要警告:
当前的 "加密" 方式 (Base64) 极不安全，仅用于演示目的。
在生产环境中，您必须使用真正的强加密算法 (例如 cryptography 库的 Fernet)
来替换这里的 Base64 编码和解码逻辑。
'''
import platform
import subprocess
import base64
from datetime import datetime, timedelta
import argparse
import os
import psutil # 需要 pip install psutil

def get_machine_id_for_generation() -> str:
    """
    获取机器唯一标识符 (与 setting_api.py 中的 get_machine_id 逻辑保持一致或兼容)。
    这里我们直接复用 setting_api.py 中的大部分逻辑，以便生成的 key 能被其正确验证。
    """
    system = platform.system()
    try:
        if system == "Windows":
            result = subprocess.check_output(
                ['wmic', 'csproduct', 'get', 'uuid'], 
                universal_newlines=True,
                stderr=subprocess.DEVNULL
            )
            machine_id = result.split('\n')[1].strip()
            if machine_id:
                return machine_id
        elif system == "Linux":
            for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        machine_id = f.read().strip()
                        if machine_id:
                            return machine_id
            for interface, snics in psutil.net_if_addrs().items():
                for snic in snics:
                    if snic.family == psutil.AF_LINK and snic.address and snic.address != "00:00:00:00:00:00":
                        return snic.address.replace(":", "").replace("-", "").lower()
        elif system == "Darwin": # macOS
            result = subprocess.check_output(
                ['ioreg', '-rd1', '-c', 'IOPlatformExpertDevice'], 
                universal_newlines=True,
                stderr=subprocess.DEVNULL
            )
            for line in result.split('\n'):
                if "IOPlatformUUID" in line:
                    machine_id = line.split('"')[-2]
                    if machine_id:
                        return machine_id
    except Exception as e:
        print(f"获取机器码时出错 ({system}): {e}")
    
    print("警告: 无法确定唯一的机器ID，将尝试使用主机名作为备用（不推荐）。")
    return platform.node()

def generate_license_key(machine_id: str, days_valid: int) -> str:
    """
    生成 License Key。
    当前实现使用 Base64，极不安全，仅为演示。
    真实场景下，这里应该是强加密过程。
    """
    expiry_date = datetime.now() + timedelta(days=days_valid)
    payload = f"{machine_id}|{expiry_date.isoformat()}"
    # 使用 URL安全的 Base64 编码
    encoded_key = base64.urlsafe_b64encode(payload.encode()).decode()
    return encoded_key

def main():
    parser = argparse.ArgumentParser(description="生成 License Key.")
    parser.add_argument(
        "--machine-id", 
        type=str, 
        help="目标机器的机器码。如果未提供，则使用当前机器的机器码。"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=30, 
        help="License Key 的有效期天数 (默认为 30 天)。"
    )

    args = parser.parse_args()

    target_machine_id = args.machine_id
    if not target_machine_id:
        print("未提供 machine-id 参数，将使用当前机器的机器码...")
        target_machine_id = get_machine_id_for_generation()
        if not target_machine_id:
            print("错误：无法获取当前机器的机器码。请手动指定 --machine-id。")
            return
        print(f"成功获取到当前机器码: {target_machine_id}")

    days_valid = args.days
    if days_valid <= 0:
        print("错误：有效期天数必须为正整数。")
        return

    license_key = generate_license_key(target_machine_id, days_valid)
    
    expiry_date_display = (datetime.now() + timedelta(days=days_valid)).strftime("%Y-%m-%d")

    print("\n======================================================")
    print(f"  为机器码: {target_machine_id}")
    print(f"  有效期至: {expiry_date_display} ({days_valid} 天)")
    print("------------------------------------------------------")
    print(f"  生成的 License Key: ")
    print(f"  {license_key}")
    print("======================================================\n")
    print("请将此 License Key 配置为 `setting_api.py` 应用的环境变量 `SYSTEM_LICENSE_KEY`。")
    print("例如，在 .env 文件中添加:")
    print(f"SYSTEM_LICENSE_KEY={license_key}")
    print("\n重要警告: 当前的 Key 生成方式 (Base64) 极不安全，仅用于演示。")
    print("生产环境中必须替换为真正的强加密算法。")

if __name__ == "__main__":
    main() 