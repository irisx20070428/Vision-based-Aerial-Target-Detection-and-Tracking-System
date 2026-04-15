# calibrate_center.py - 交互式舵机中值校准工具
import time
import sys
from gpiozero import Servo


def calibrate_center():
    """交互式校准舵机中值"""
    print("=" * 60)
    print("舵机中值校准工具")
    print("=" * 60)
    print("\n操作说明：")
    print("  w/s - 调整水平舵机 (左/右)")
    print("  i/k - 调整垂直舵机 (上/下)")
    print("  r   - 重置到90度")
    print("  q   - 保存并退出")
    print("  x   - 不保存退出")
    print("\n当前舵机角度会实时显示")

    try:
        # 初始化舵机
        pan_servo = Servo(18, min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000)
        tilt_servo = Servo(27, min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000)

        # 当前角度
        pan_angle = 90
        tilt_angle = 90
        pan_offset = 0
        tilt_offset = 0

        # 步进值
        step = 1

        def update_servo():
            """更新舵机位置"""
            pan_value = (pan_angle - 90) / 90
            tilt_value = (tilt_angle - 90) / 90
            pan_servo.value = pan_value
            tilt_servo.value = tilt_value

        # 初始位置
        update_servo()
        time.sleep(0.5)

        print(f"\n初始位置: Pan={pan_angle}°, Tilt={tilt_angle}°")
        print("请调整舵机到您认为的\"正前方\"位置")

        import sys
        import tty
        import termios

        def get_key():
            """获取键盘输入（无需回车）"""
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        while True:
            # 显示当前状态
            print(
                f"\rPan: {pan_angle:3d}° (偏移:{pan_offset:+3d}°) | Tilt: {tilt_angle:3d}° (偏移:{tilt_offset:+3d}°)   ",
                end="")

            key = get_key()

            if key == 'w':
                pan_angle = min(180, pan_angle + step)
                pan_offset = pan_angle - 90
                update_servo()
            elif key == 's':
                pan_angle = max(0, pan_angle - step)
                pan_offset = pan_angle - 90
                update_servo()
            elif key == 'i':
                tilt_angle = min(180, tilt_angle + step)
                tilt_offset = tilt_angle - 90
                update_servo()
            elif key == 'k':
                tilt_angle = max(0, tilt_angle - step)
                tilt_offset = tilt_angle - 90
                update_servo()
            elif key == 'r':
                pan_angle = 90
                tilt_angle = 90
                pan_offset = 0
                tilt_offset = 0
                update_servo()
                print("\n🔄 已重置到90度")
            elif key == 'q':
                print(f"\n\n✅ 保存校准参数:")
                print(f"   PAN_OFFSET = {pan_offset}")
                print(f"   TILT_OFFSET = {tilt_offset}")
                print("\n请将以下配置添加到 config.py 中：")
                print(f"   SERVO_PAN_OFFSET = {pan_offset}   # 水平偏移")
                print(f"   SERVO_TILT_OFFSET = {tilt_offset}  # 垂直偏移")
                break
            elif key == 'x':
                print("\n\n❌ 未保存，退出校准")
                break
            elif key == '+':
                step = min(10, step + 1)
                print(f"\n步进值: {step}°", end="")
            elif key == '-':
                step = max(1, step - 1)
                print(f"\n步进值: {step}°", end="")

        pan_servo.detach()
        tilt_servo.detach()

    except Exception as e:
        print(f"\n❌ 错误: {e}")
    finally:
        print("\n校准完成")


if __name__ == "__main__":
    calibrate_center()