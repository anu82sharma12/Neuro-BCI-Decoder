import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
pwm = GPIO.PWM(18, 50)
pwm.start(0)

def move_servo(intent):
    angles = [30, 90, 150, 120]  # LEFT, RIGHT, FEET, TONGUE
    duty = angles[intent] / 18 + 2
    pwm.ChangeDutyCycle(duty)
