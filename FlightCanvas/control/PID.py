from FlightCanvas.control.siso_system import SISOSystem


class PID(SISOSystem):
    """
    Crates PID transfer function
    """
    def __init__(self, Kp: float, Ki: float, Kd: float, tau: float):
        """
        Initialise PID transfer function
        :param Kp: Proportional gain
        :param Ki: Integral gain
        :param Kd: Derivative gain
        :param tau: Time constant (cutoff frequency is 1/tau rad/s)
        """
        b2 = Kd + Kp * tau
        b1 = Kp * Ki * tau
        b0 = Ki

        a2 = tau
        a1 = 1
        a0 = 0

        num = [b2, b1, b0]
        den = [a2, a1, a0]
        super().__init__(num, den)
