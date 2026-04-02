# test_servo_gpiozero_simple.py
from gpiozero import Servo
import time


def test_servo_simple():
    """简单测试舵机"""
    print("=" * 50)
    print("舵机测试 (gpiozero)")
    print("=" * 50)

    print("\n请确保:")
    print("1. 水平舵机连接到 GPIO 18")
    print("2. 垂直舵机连接到 GPIO 27")
    print("3. 舵机有独立 5V 电源")
    print("4. 舵机信号线、电源线、地线都正确连接")

    input("\n按 Enter 开始测试...")

    try:
        # 初始化水平舵机
        print("\n1. 测试水平舵机 (GPIO 18)")
        servo_pan = Servo(18, min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000)

        # 测试不同位置
        positions = [
            ("最左 (-1)", -1),
            ("偏左 (-0.5)", -0.5),
            ("中心 (0)", 0),
            ("偏右 (0.5)", 0.5),
            ("最右 (1)", 1)
        ]

        for name, value in positions:
            print(f"  移动到: {name}")
            servo_pan.value = value
            time.sleep(1)

        servo_pan.detach()
        print("  ✅ 水平舵机测试完成")

        time.sleep(1)

        # 测试垂直舵机
        print("\n2. 测试垂直舵机 (GPIO 27)")
        servo_tilt = Servo(27, min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000)

        for name, value in positions:
            print(f"  移动到: {name}")
            servo_tilt.value = value
            time.sleep(1)

        servo_tilt.detach()
        print("  ✅ 垂直舵机测试完成")

        print("\n✅ 所有测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("\n故障排除:")
        print("1. 检查舵机电源 (需要 5V 2A)")
        print("2. 检查接线 (信号线、电源线、地线)")
        print("3. 尝试用 sudo 运行: sudo python3 test_servo_gpiozero_simple.py")


if __name__ == "__main__":
    test_servo_simple()