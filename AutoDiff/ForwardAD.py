import numpy as np
import math


class Var:
    valid_types = (int, float, np.int, np.float)

    def __init__(self, val):
        self._val = val
        self._jacobian = {self: 1}

    def get_value(self):
        return self._val

    def get_jacobian(self):
        return list(self._jacobian.values())

    def get_derivative_of(self, var):
        if isinstance(var, Var):
            return self._jacobian[var]
        else:
            raise TypeError('Invalid input type. ' +
                            'var must be any of the following types: Var.')

    def __add__(self, other):
        if isinstance(other, Var):
            new_val = self._val + other._val
            new_jacobian = {}

            # Obtain a new variable set. For example, if self has {x, y}, and other has {y, z},
            # then the new variable set would be {x, y, z}
            new_vars = set(self._jacobian.keys()) | set(other._jacobian.keys())

            # Loop through new variables in the new variable set
            # For each variable calculate the partial derivative.
            # if the dictionary does not contain the key/variable it will return None.
            # float(None or 0) = 0.0; float(a real number or 0) = a real number (e.g. float(5 or 0) = 5.0)
            for var in new_vars:
                new_jacobian[var] = float(self._jacobian.get(var) or 0) + float(other._jacobian.get(var) or 0)
        elif isinstance(other, Var.valid_types):
            new_val = self._val + other
            new_jacobian = self._jacobian
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def __radd__(self, other):
        if isinstance(other, Var.valid_types):
            new_val = other + self._val
            new_jacobian = self._jacobian
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def __sub__(self, other):
        if isinstance(other, Var):
            new_val = self._val - other._val
            new_jacobian = {}

            new_vars = set(self._jacobian.keys()) | set(other._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = float(self._jacobian.get(var) or 0) - float(other._jacobian.get(var) or 0)
        elif isinstance(other, Var.valid_types):
            new_val = self._val - other
            new_jacobian = self._jacobian
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def __rsub__(self, other):
        if isinstance(other, Var.valid_types):
            new_val = other - self._val
            new_jacobian = {}

            new_vars = set(self._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = -self._jacobian.get(var)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def __mul__(self, other):
        if isinstance(other, Var):
            new_val = self._val * other._val
            new_jacobian = {}

            new_vars = set(self._jacobian.keys()) | set(other._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = float(self._jacobian.get(var) or 0) * other._val + self._val * float(other._jacobian.get(var) or 0)
        elif isinstance(other, Var.valid_types):
            new_val = self._val * other
            new_jacobian = {}

            new_vars = self._jacobian.keys()
            for var in new_vars:
                new_jacobian[var] = self._jacobian.get(var) * other
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def __rmul__(self, other):
        if isinstance(other, Var.valid_types):
            new_val = other * self._val
            new_jacobian = {}

            new_vars = self._jacobian.keys()
            for var in new_vars:
                new_jacobian[var] = self._jacobian.get(var) * other
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def __truediv__(self, other):
        if isinstance(other, Var):
            if other._val == 0:
                raise ZeroDivisionError("Denominator cannot be 0.")
            new_val = self._val / other._val
            new_jacobian = {}

            new_vars = set(self._jacobian.keys()) | set(other._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = (float(self._jacobian.get(var) or 0) * other._val
                                - self._val * float(other._jacobian.get(var) or 0))\
                               /(other._val ** 2)
        elif isinstance(other, Var.valid_types):
            if other == 0:
                raise ZeroDivisionError("Denominator cannot be 0.")
            new_val = self._val / other
            new_jacobian = {}

            new_vars = set(self._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = self._jacobian.get(var) / other
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def __rtruediv__(self, other):
        if self._val == 0:
            raise ZeroDivisionError("Denominator cannot be 0.")
        if isinstance(other, Var.valid_types):
            new_val = other / self._val
            new_jacobian = {}

            new_vars = set(self._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = -other/(self._val ** 2) * self._jacobian.get(var)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def __abs__(self):
        """

        :return:

        EXAMPLES
        =========
        >>> x = Var(5.0)
        >>> f = abs(x)
        >>> print(f.get_value())
        5.0
        >>> print(f.get_jacobian())
        [1.0]
        """
        if self._val == 0:
            raise ValueError('Derivative of abs() is not defined at 0.')
        elif self._val < 0:
            new_val = -self._val
            new_der = {}

            new_vars = set(self._jacobian.keys())
            for var in new_vars:
                new_der[var] = -self._jacobian.get(var)
        elif self._val > 0:
            new_val = self._val
            new_der = self._jacobian

        new_var = Var(new_val)
        new_var._jacobian = new_der
        return new_var

    def __neg__(self):
        """

         :return:

         EXAMPLES
         =========
         >>> x = Var(2.0)
         >>> f = -x
         >>> print(f.get_value())
         -2.0
         >>> print(f.get_jacobian())
         [-1.0]
         """
        new_val = -self._val
        new_jacobian = {}

        new_vars = set(self._jacobian.keys())
        for var in new_vars:
            new_jacobian[var] = -self._jacobian.get(var)

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def __pow__(self, power, modulo=None):
        if isinstance(power, Var):
            if self._val < 0:
                raise ValueError("The derivative of x ** y is not defined on x < 0.")
            new_val = self._val ** power._val
            new_jacobian = {}

            new_vars = set(self._jacobian.keys()) | set(power._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = new_val * \
                                    (float(power._jacobian.get(var) or 0) * np.log(self._val) +
                                     power._val * float(self._jacobian.get(var) or 0) / self._val)
        elif isinstance(power, Var.valid_types):
            new_val = self._val ** power
            new_jacobian = {}

            new_vars = set(self._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = power * self._val ** (power - 1) * self._jacobian.get(var)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, np.int.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def __rpow__(self, other):
        if isinstance(other, Var.valid_types):
            if other < 0:
                raise ValueError("The derivative of b ** x, b**x * ln(b), is not defined on b < 0.")
            new_val = other ** self._val
            new_jacobian = {}

            new_vars = set(self._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = (other ** self._val) * np.log(other) * self._jacobian.get(var)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def exp(self):
        """

        :return:

        EXAMPLES
        =========
        >>> x = Var(5.0)
        >>> f = Var.exp(x)
        >>> print(f.get_value())
        148.41
        >>> print(np.round(f.get_jacobian(), 2))
        [148.41]
        """
        new_val = np.exp(self._val)
        new_jacobian = {}

        new_vars = set(self._jacobian.keys())
        for var in new_vars:
            new_jacobian[var] = np.exp(self._val) * self._jacobian.get(var)

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def log(self, b):
        """

         :return:

         EXAMPLES
         =========
         >>> x = Var(1000)
         >>> f = Var.log(x, 10)
         >>> print(f.get_value())
         3.0
         >>> print(np.round(f.get_jacobian(), 4))
         [0.0004]
         """
        # b is the base. The default is e (natural log).
        if not isinstance(b, (int, np.int)):
            raise TypeError("Invalid input type. b should be any of the following type: int and numpy.int.")
        if self._val <= 0:
            raise ValueError("log(x) is not defined on x <= 0.")

        new_val = math.log(self._val, b)
        new_jacobian = {}

        new_vars = set(self._jacobian.keys())
        for var in new_vars:
            new_jacobian[var] = 1 / (self._val * np.log(b)) * self._jacobian.get(var)

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def sqrt(self):
        """
         :return:

         EXAMPLES
         =========
         >>> x = Var(9)
         >>> f = Var.sqrt(x)
         >>> print(f.get_value())
         3.0
         >>> print(np.round(f.get_jacobian(), 2))
         0.17
         """
        # TODO should we check self type?
        if self._val < 0:
            raise ValueError("srqt(x) is not not defined on x < 0.")
        elif self._val == 0:
            raise ZeroDivisionError("Zero division occurs when derivative is calculated. " +
                                    "The derivative of sqrt(x), 1/2 * 1/sqrt(x), is undefined on x = 0.")
        new_val = np.sqrt(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = 1/2 * self._val**(-1/2) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def sin(self):
        """
        :return:
        EXAMPLES
        =========
        >>> x = Var(np.pi)
        >>> f = 10e16 * Var.sin(x)
        >>> print(np.round(f.get_value(), 2))
        12.25
        >>> print(np.round(f.get_jacobian(), 2))
        [-1.e+17]
        """
        new_val = np.sin(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = np.cos(self._val) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def arcsin(self):
        """
        :return:
        EXAMPLES
        =========
        >>> x = Var(0)
        >>> f = Var.arcsin(x)
        >>> print(f.get_value())
        0
        >>> print(f.get_jacobian())
        [1.0]
        """
        if abs(self._val) > 1:
            raise ValueError("Invalid value input. arcsine is not define on |x| > 1 for real output.")
        elif self._val == 1:
            raise ZeroDivisionError("Zero division occurs when derivative is calculated. " +
                                    "The derivative of arcsin(x), 1/sqrt(1 - x^2), " +
                                    "is undefined on x = 1.")

        new_val = np.arcsin(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = 1 / np.sqrt(1 - self._val**2) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def cos(self):
        """
        :return:
        EXAMPLES
        =========
        >>> x = Var(np.pi)
        >>> f = 10e16 * Var.cos(x)
        >>> print(np.round(f.get_value(), 2))
        -1.e+17
        >>> print(np.round(f.get_jacobian(), 2))
        [-12.25]
        """
        new_val = np.cos(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = -np.sin(self._val) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def arccos(self):
        """
        :return:
        EXAMPLES
        =========
        >>> x = Var(0)
        >>> f = Var.arccos(x)
        >>> print(np.round(f.get_value(), 2))
        1.57
        >>> print(np.round(f.get_jacobian(), 2))
        [-1.0]
        """
        if abs(self._val) > 1:
            raise ValueError("Invalid value input. arcsin(x) is not defined on |x| > 1 for real output.")
        elif self._val == 1:
            raise ZeroDivisionError("Zero division occurs when derivative is calculated. " +
                                    "The derivative of arccos(x), -1/sqrt(1 - x^2), " +
                                    "is undefined on x = 1.")
        new_val = np.arccos(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = -1 / np.sqrt(1 - self._val**2) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def tan(self):
        """
        :return:
        EXAMPLES
        =========
        >>> x = Var(np.pi / 3)
        >>> f = Var.tan(x)
        >>> print(np.round(f.get_value(), 2))
        1.73
        >>> print(np.round(f.get_jacobian(), 2))
        [4.0]
        """
        if self._val % (np.pi/2) == 0 and (self._val / (np.pi/2)) % 2 != 0:
            raise ValueError("Invalid value input. tan(x) is not defined on x = (2n+1)*pi/2.")
        new_val = np.tan(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = (1 / np.cos(self._val))**2 * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def arctan(self):
        """
        :return:
        EXAMPLES
        =========
        >>> x = Var(1)
        >>> f = Var.arctan(x)
        >>> print(np.round(f.get_value(), 2))
        0.79
        >>> print(np.round(f.get_jacobian(), 2))
        [0.5]
        """
        new_val = np.arctan(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = 1 / (self._val**2 + 1) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def sinh(self):
        """
        :return:
        EXAMPLES
        =========
        >>> x = Var(1)
        >>> f = Var.arcsin(x)
        >>> print(np.round(f.get_value(), 2))
        1.18
        >>> print(np.round(f.get_jacobian(), 2))
        [1.54]
        """
        new_val = np.sinh(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = np.cosh(self._val) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def cosh(self):
        """
        :return:
        EXAMPLES
        =========
        >>> x = Var(1)
        >>> f = Var.cosh(x)
        >>> print(np.round(f.get_value(), 2))
        1.54
        >>> print(np.round(f.get_jacobian(), 2))
        [1.18]
        """
        new_val = np.cosh(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = np.sinh(self._val) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def tanh(self):
        """
        :return:
        EXAMPLES
        =========
        >>> x = Var(1)
        >>> f = Var.tanh(x)
        >>> print(np.round(f.get_value(), 2))
        0.76
        >>> print(np.round(f.get_jacobian(), 2))
        [0.42]
        """
        new_val = np.tanh(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = (1 / np.cosh(self._val))**2 * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

# x = Var(3)
# y = Var(3)
# f = Var.sqrt(x + 2 * y)
# print(f.get_value())
# print(type(f.get_value()))
# print(f.get_jacobian())
# print(type(f.get_jacobian()))

# x = Var(3*np.pi/2)
# f = Var.tan(x)
# print(f.get_value())

# x = Var(100)
# f = Var.log(x, b=10)
# print(f.get_value())
# print(f.get_jacobian())

# x = Var(2)
# f = np.exp(1) ** x
# print(f.get_value())
# print(f.get_jacobian())


