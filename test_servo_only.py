# test_servo_only.py
import time
import sys

sys.path.append('.')
from servo_controller import ServoController
from config import Config


def test_servo():
    print("=" * 50)
    print("舵机测试程序")
    print("=" * 50)

    try:
        # 初始化舵机
        print("\n初始化舵机...")
        servo = ServoController()
        time.sleep(1)

        print("\n开始测试舵机...")
        print("注意：舵机将缓慢移动，请确保没有障碍物")

        # 测试1：中心位置
        print("\n1. 移动到中心位置 (90°)")
        servo.set_target(Config.SERVO_CENTER_ANGLE, Config.SERVO_CENTER_ANGLE)
        time.sleep(2)

        # 测试2：向左移动
        print("\n2. 向左移动 (45°)")
        servo.set_target(20, Config.SERVO_CENTER_ANGLE)
        for _ in range(30):
            servo.update(0.02)
            time.sleep(0.02)
        time.sleep(1)

        # 测试3：向右移动
        print("\n3. 向右移动 (135°)")
        servo.set_target(45, Config.SERVO_CENTER_ANGLE)
        for _ in range(30):
            servo.update(0.02)
            time.sleep(0.02)
        time.sleep(1)

        # 测试4：回到中心
        print("\n4. 回到中心 (90°)")
        servo.set_target(Config.SERVO_CENTER_ANGLE, Config.SERVO_CENTER_ANGLE)
        for _ in range(30):
            servo.update(0.02)
            time.sleep(0.02)
        time.sleep(1)

        print("\n✅ 舵机测试完成")

        # 提示用户观察
        print("\n请观察舵机是否正常转动")
        print("如果舵机没有转动或发出异常声音，请立即停止")

        input("按 Enter 继续...")

        # 测试5：上下移动
        print("\n5. 上下移动测试")
        print("向上移动 (135°)")
        servo.set_target(Config.SERVO_CENTER_ANGLE, 135)
        for _ in range(30):
            servo.update(0.02)
            time.sleep(0.02)
        time.sleep(1)

        print("向下移动 (45°)")
        servo.set_target(Config.SERVO_CENTER_ANGLE, 45)
        for _ in range(30):
            servo.update(0.02)
            time.sleep(0.02)
        time.sleep(1)

        print("回到中心")
        servo.set_target(Config.SERVO_CENTER_ANGLE, Config.SERVO_CENTER_ANGLE)
        for _ in range(30):
            servo.update(0.02)
            time.sleep(0.02)

        print("\n✅ 所有测试完成")

    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        servo.cleanup()
        print("\n舵机已关闭")


if __name__ == "__main__":
    test_servo()