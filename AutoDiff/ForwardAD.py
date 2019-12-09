import numpy as np
import math


class Var:
    valid_types = (int, float, np.int, np.float)

    def __init__(self, val):
        """Initializes a Var object with three private attributes: self._val, self._var_list and self._der

        INPUTS
        =======
        self: a Var object
        val: a real number (int, float, np.int ot np.float)

        RETURNS
        =======
        None

        Raises TypeError when val is not a real number
        """
        if not isinstance(val, Var.valid_types):
            raise TypeError('Invalid input type. ' +
                            'Val must be any of the following types: int, float, np.int, np.float.')

        # the current evaluation of the Var object
        self._val = val

        # a list storing the current variables
        self._var_list = [self]

        # a list storing the partial derivatives of the Var with respect to the current variables
        self._der = [1.0]

    def get_value(self):
        """Returns the _val attribute of the Var object

        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        self.val: a real number

        EXAMPLES
        =======
        # >>> x = Var(5.0)
        # >>> x.get_value()
        5.0
        """
        return self._val

    def _set_value(self, value):
        """Sets the _val attribute of the Var object equal to value and _der attribute to [1.0]
        INPUTS
        =======
        self: a Var object
        value : a real number

        RETURNS
        =======
        None

        EXAMPLES
        =======

        Raises TypeError if value is not a real number
        """
        if not isinstance(value, Var.valid_types):
            raise TypeError('Invalid input type. ' +
                            'Value must be any of the following types: int, float, np.int, np.float.')
        self._val = value
        self._var_list = [self]
        self._der = [1.0]
        return None

    def get_der(self, var_list=None):
        """Gets the _der attribute of the Var object
        INPUTS
        =======
        self: a Var object
        var_list: a list of Var objects

        RETURNS
        =======
        der: a list of real numbers; the i-th number is the partial derivative with respect to the i-th Var in var_list

        EXAMPLES
        =======
        # >>> x = Var(1)
        # >>> y = Var(1)
        # >>> f = x + 2*y
        # >>> f.get_der([x, y])
        [1.0, 2.0]
        """
        if var_list is None:
            return self._der.copy()

        der = [None]*len(var_list)
        for i, var in enumerate(var_list):
            der[i] = self._get_derivative_of(var)
        return der

    def _get_derivative_of(self, var):
        """Returns the partial derivative of the Var object self with respect to one of its variables var

        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        self._val: a real number, which is the partial derivative of self with respect to the variable var

        EXAMPLES
        =======

        Raises TypeError if var is not a Var object
        """
        if isinstance(var, Var):
            idx = self._get_var_index(var, self._var_list)
            if idx != -1:
                return self._der[idx]
            else:
                return 0.0
        else:
            raise TypeError('Invalid input type. ' +
                            'var must be any of the following types: Var.')

    def _get_var_index(self, var, var_list):
        """Returns the index of var in var_list
        INPUTS
        =======
        self: a Var object
        var: a Var object
        var_list: a list of Var objects

        RETURNS
        =======
        an integer: the index of var if var is in var_list or -1 if var is not in var_list

        EXAMPLES
        =======
        """
        for i in range(len(var_list)):
            if var is var_list[i]:
                return i
        return -1

    def __eq__(self, other):
        """Returns true if self and other have the same value and same variables with same partial derivatives

        INPUTS
        =======
        self: a Var object
        other: a Var object

        RETURNS
        =======
        a boolean, indicating if self and other have the same value and same variables with same partial derivatives

        # TODO
        EXAMPLES
        =======

        """
        if isinstance(other, Var) and self._val == other._val and len(self._var_list) == len(other._var_list):
            for var, der in zip(self._var_list, self._der):
                idx = other._get_var_index(var, other._var_list)
                if idx == -1 or other._der[idx] != der:
                    return False
            return True
        return False

    def __ne__(self, other):
        """Returns true if __eq__ returns False

        INPUTS
        =======
        self: a Var object
        other: a Var object

        RETURNS
        =======
        a boolean, indicating if __eq__ returns False

        # TODO
        EXAMPLES
        =======

        """
        return not self == other

    def __add__(self, other):
        """Returns a Var object that results from adding the two inputs (self + other)

        INPUTS
        =======
        self: a Var object
        other: a float or integer number, or a Var object

        RETURNS
        =======
        new_var: a Var object
            If other is a Var object, returns a Var object with:
            - _val equal to the sum of the _val attribute of self and other
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to the sum of partial derivatives of self and other


            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - _val equal to the sum of the self._val and the number
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to self._der

        EXAMPLES
        =======
        # >>> x = Var(5.0)
        # >>> f = x+3.0
        # >>> f.get_value()
        8.0
        # >>> f.get_der()
        [1.0]

        Raises TypeError when other is not a Var object or a real number
        """
        if isinstance(other, Var):
            new_val = self._val + other._val

            new_var_list = self._var_list.copy()
            for var in other._var_list:
                if self._get_var_index(var, self._var_list) == -1:
                    new_var_list.append(var)

            new_der = [None] * len(new_var_list)

            for i, var in enumerate(new_var_list):
                new_der[i] = self._get_derivative_of(var) + other._get_derivative_of(var)
        elif isinstance(other, Var.valid_types):
            return self + _Constant(other)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._var_list = new_var_list
        new_var._der = new_der
        return new_var

    def __radd__(self, other):
        """Returns a Var object that results from adding the two inputs (other + self)

        INPUTS
        =======
        self: a Var object
        other: a float or integer number

        RETURNS
        =======
        new_var: a Var object.
            Other cannot be a Var object, as this case falls under __add__

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - _val equal to the sum of the self._val and the number
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to self._der

        EXAMPLES
        =======
        # >>> x = Var(5.0)
        # >>> f = 3.0+x
        # >>> f.get_value()
        8.0
        # >>> f.get_der()
        [1.0]

        Raises TypeError when other is not a real number
        """
        if isinstance(other, Var.valid_types):
            return _Constant(other) + self
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

    def __sub__(self, other):
        """Returns a Var object that results from subtracting the two inputs (self - other)

        INPUTS
        =======
        self: a Var object
        other: a float or integer number, or a Var object

        RETURNS
        =======
        new_var: a Var object.
            If other is a Var object, returns a Var object with:
            - _val equal to self._val - other._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to the difference of partial derivatives between self and other

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - _val equal to self._val - other
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to self._der

        EXAMPLES
        =======
        # >>> x = Var(5.0)
        # >>> f = x-3.0
        # >>> f.get_value()
        2.0
        # >>> f.get_der()
        [1.0]

        Raises TypeError when other is not a Var object or a real number
        """
        if isinstance(other, Var):
            new_val = self._val - other._val

            new_var_list = self._var_list.copy()
            for var in other._var_list:
                if self._get_var_index(var, self._var_list) == -1:
                    new_var_list.append(var)

            new_der = [None] * len(new_var_list)

            for i, var in enumerate(new_var_list):
                new_der[i] = self._get_derivative_of(var) - other._get_derivative_of(var)
        elif isinstance(other, Var.valid_types):
            return self - _Constant(other)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._var_list = new_var_list
        new_var._der = new_der
        return new_var

    def __rsub__(self, other):
        """Returns a Var object that results from subtracting the two inputs (other - self)

        INPUTS
        =======
        self: a Var object
        other: a float or integer number

        RETURNS
        =======
        new_var: a Var object.
            Other cannot be a Var object, as this case falls under __sub__

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - _val equal to other - self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to self._der

        EXAMPLES
        =======
        # >>> x = Var(5.0)
        # >>> f = 3.0-x
        # >>> f.get_value()
        -2.0
        # >>> f.get_der()
        [1.0]

        Raises TypeError when other is not a real number
        """
        if isinstance(other, Var.valid_types):
            return _Constant(other) - self
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

    def __mul__(self, other):
        """Returns a Var object that results from multiplying the two inputs (self*other)

        INPUTS
        =======
        self: a Var object
        other: a float or integer number, or a Var object

        RETURNS
        =======
        new_var: a Var object.
            If other is a Var object, returns a Var object with:
            - _val equal to self._val * other._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule of d(uv) = udv + vdu

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - _val equal to self._val * other
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to self._der multiplied by other

        EXAMPLES
        =======
        # >>> x = Var(5.0)
        # >>> f = x*3.0
        # >>> f.get_value()
        15.0
        # >>> f.get_der()
        [3.0]

        Raises TypeError when other is not a Var object or a real number
        """
        if isinstance(other, Var):
            new_val = self._val * other._val

            new_var_list = self._var_list.copy()
            for var in other._var_list:
                if self._get_var_index(var, self._var_list) == -1:
                    new_var_list.append(var)

            new_der = [None] * len(new_var_list)

            for i, var in enumerate(new_var_list):
                new_der[i] = self._get_derivative_of(var) * other._val + self._val * other._get_derivative_of(var)
        elif isinstance(other, Var.valid_types):
            return self * _Constant(other)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._var_list = new_var_list
        new_var._der = new_der
        return new_var

    def __rmul__(self, other):
        """Returns the var object that results from multiplying the two inputs (other*self)

        INPUTS
        =======
        self: a Var object
        other: a float or integer number

        RETURNS
        =======
        new_var: a Var object.
            Other cannot be a Var object, as this case falls under __mul__

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - _val equal to other * self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to self._der multiplied by other

        EXAMPLES
        =======
        # >>> x = Var(5.0)
        # >>> f = 3.0*x
        # >>> f.get_value()
        15.0
        # >>> f.get_der()
        [3.0]

        Raises TypeError when other is not a real number
        """
        if isinstance(other, Var.valid_types):
            return _Constant(other) * self
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

    def __truediv__(self, other):
        """Returns a Var object that results from dividing the two inputs (self/other)

        INPUTS
        =======
        self: a Var object
        other: a float or integer number, or a Var object

        RETURNS
        =======
        new_var: a Var object.
            If other is a Var object, returns a Var object with:
            - _val equal to self._val / other._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule of d(u/v) = (udv - vdu)/v^2

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - _val equal to other / other
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to self._der divided by other

        EXAMPLES
        =======
        # >>> x = Var(5.0)
        # >>> f = x/2.0
        # >>> f.get_value()
        2.5
        # >>> f.get_der()
        [0.5]

        Raises TypeError when other is not a Var object or a real number
        Raises ZeroDivisionError when other._val or other is equal to zero
        """
        if isinstance(other, Var):
            if other._val == 0:
                raise ZeroDivisionError("Denominator cannot be 0.")
            new_val = self._val / other._val

            new_var_list = self._var_list.copy()
            for var in other._var_list:
                if self._get_var_index(var, self._var_list) == -1:
                    new_var_list.append(var)

            new_der = [None] * len(new_var_list)

            for i, var in enumerate(new_var_list):
                new_der[i] = (self._get_derivative_of(var) * other._val - self._val * other._get_derivative_of(var)) / (other._val ** 2)
        elif isinstance(other, Var.valid_types):
            return self / _Constant(other)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._var_list = new_var_list
        new_var._der = new_der
        return new_var

    def __rtruediv__(self, other):
        """Returns the var object that results from dividing the two inputs (other/self)

        INPUTS
        =======
        self: object of Var
        other: a float or integer number

        RETURNS
        =======
        new_var: a Var object.
            Other cannot be a Var object, as this case falls under __truediv__

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - _val equal to other / self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule of d(1/u) = -1/u^2

        EXAMPLES
        =======
        # >>> x = Var(5.0)
        # >>> f = 10.0/x
        # >>> f.get_value()
        2.0
        # >>> f.get_der()
        [0.4]

        Raises TypeError when other is no real number
        Raises ZeroDivisionError when self._val is equal to zero
        """
        if self._val == 0:
            raise ZeroDivisionError("Denominator cannot be 0.")
        if isinstance(other, Var.valid_types):
            return _Constant(other) / self
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

    def __abs__(self):
        """Returns a Var object whose _val is the absolute value of self._val

        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object
            When self.val > 0, returns a Var object with:
            - _val equal to self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to self._der

            When self.val < 0, returns a Var object with:
            - _val equal to -self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to -self._der

        EXAMPLES
        =========
        # >>> x = Var(-5.0)
        # >>> f = abs(x)
        # >>> print(f.get_value())
        5.0
        # >>> print(f.get_der())
        [-1.0]

        Raises ValueError when self._val = 0, as the derivative is then undefined
        """
        if self._val == 0:
            raise ValueError('Derivative of abs() is not defined at 0.')

        if self._val < 0:
            new_val = -self._val
            new_der = []

            for der in self._der:
                new_der.append(-der)
        else:
            new_val = self._val
            new_der = self._der.copy()

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def __neg__(self):
        """Returns a Var object whose _val is the negative value of self._val

        INPUTS
        =======
        self: a Var object
        other: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to -self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives equal to -self._der

         EXAMPLES
         =========
         # >>> x = Var(2.0)
         # >>> f = -x
         # >>> print(f.get_value())
         -2.0
         # >>> print(f.get_der())
         [-1.0]
         """
        new_val = -self._val
        new_der = []

        for der in self._der:
            new_der.append(-der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def __pow__(self, power, modulo=None):
        """Returns a Var object that results from taking self to the power of power (self**power)

        INPUTS
        =======
        self: a Var object
        other: a real number (int, float, np.int, np.float), or a Var object

        RETURNS
        =======
        new_var: a Var object.
            If power is a Var object, returns a Var object with:
            - _val equal to self._val^power._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule of d(u^v) = u^(v-1)(dv*log(u)*u + v*du)

            If power is a real number (int, float, np.int, np.float), returns a Var object with:
            - _val equal to self.val^power
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule of d(u^power) = power*u^(power-1)*du

        EXAMPLES
        =========
        # >>> x = Var(5.0)
        # >>> f = x**2
        # >>> print(f.get_value())
        25.0
        # >>> print(f.get_der())
        [10.0]

        Raises TypeError when power is not a Var object or a real number
        Raises ValueError when self._val < 0
        """
        if isinstance(power, Var):
            if (not isinstance(power, _Constant)) and self._val < 0:
                raise ValueError("The derivative of x ** y is not defined on x < 0.")
            new_val = self._val ** power._val

            new_var_list = self._var_list.copy()
            for var in power._var_list:
                if self._get_var_index(var, self._var_list) == -1:
                    new_var_list.append(var)

            new_der = [None] * len(new_var_list)

            for i, var in enumerate(new_var_list):
                if power._get_derivative_of(var) == 0.0:
                    new_der[i] = self._val ** (power._val-1) * (power._val * self._get_derivative_of(var))
                else:
                    new_der[i] = self._val ** (power._val-1) * (power._get_derivative_of(var) * np.log(self._val) * self._val +
                                     power._val * self._get_derivative_of(var))
        elif isinstance(power, Var.valid_types):
            return self ** _Constant(power)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, np.int.')

        new_var = Var(new_val)
        new_var._var_list = new_var_list
        new_var._der = new_der
        return new_var

    def __rpow__(self, other):
        """Returns a Var object that results from taking other to the power of self (other**self)

        INPUTS
        =======
        self: a Var object
        other: a real number (int, float, np.int, np.float)

        RETURNS
        =======
        new_var: a Var object.
            Other cannot be a Var object, as this case falls under __pow__

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - _val equal to power^self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule of d(other^u) = other^u*log(other)du

        EXAMPLES
        =========
        # # >>> x = Var(5.0)
        # # >>> f = 2.0**x
        # # >>> print(f.get_value())
        # 32.0
        # # >>> np.round(f.get_der(), 8) == 22.18070978

        Raises TypeError when other is not a Var object or a real number
        Raises ValueError when other < 0
        """
        if isinstance(other, Var.valid_types):
            return _Constant(other) ** self
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

    def exp(self):
        """Returns a Var object that results from taking the exponent of self

        INPUTS
        =======
        self: a Var object
        other: a real number (int, float, np.int, np.float), or a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to exp(self._val)
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule d(exp(u)) = exp(u)*du

        EXAMPLES
        =========
        # >>> x = Var(5.0)
        # >>> f = Var.exp(x)
        # >>> print(f.get_value())
        148.41
        # >>> print(np.round(f.get_der(), 2))
        [148.41]
        """
        new_val = np.exp(self._val)
        new_der = []

        for der in self._der:
            new_der.append(np.exp(self._val) * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def log(self, b=np.e):
        """
        INPUTS
        =======
        self: a Var object
        b: an integer, the base of the logarithm

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the base b logarithm of self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule d(log(u, b)) = 1/(u*log(b))*du

         EXAMPLES
         =========
         # >>> x = Var(1000)
         # >>> f = Var.log(x, 10)
         # >>> print(f.get_value())
         3.0
         # >>> print(np.round(f.get_der(), 4))
         [0.0004]

         Raises TypeError when b is not an int, float, np.int, np.float
         Raises ValueError when b <= 0
         Raises ValueError when self._val <= 0
         """
        # b is the base. The default is e (natural log).
        if not isinstance(b, Var.valid_types):
            raise TypeError("Invalid input type. b must be any of the following types: int, float, np.int, np.float.")
        if b <= 0:
            raise ValueError("Invalid input. b must <= 0.")
        if self._val <= 0:
            raise ValueError("log(x) is not defined on x <= 0.")

        new_val = math.log(self._val, b)
        new_der = []

        for der in self._der:
            new_der.append(1 / (self._val * np.log(b)) * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def sqrt(self):
        """
        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the square root of self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule d(sqrt(u)) = 1/2*(u)^(-1/2)*du

        EXAMPLES
        =========
        # >>> x = Var(9)
        # >>> f = Var.sqrt(x)
        # >>> print(f.get_value())
        3.0
        # >>> print(np.round(f.get_der(), 2))
        0.17

        Raises ValueError when self._val < 0
        Raises ZeroDivisionError when self._val = 0
        """
        if self._val < 0:
            raise ValueError("srqt(x) is not not defined on x < 0.")
        elif self._val == 0:
            raise ZeroDivisionError("Zero division occurs when derivative is calculated. " +
                                    "The derivative of sqrt(x), 1/2 * 1/sqrt(x), is undefined on x = 0.")
        new_val = np.sqrt(self._val)
        new_der = []

        for der in self._der:
            new_der.append(1/2 * self._val**(-1/2) * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def sin(self):
        """
        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the sine of self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule d(sin(u)) = cos(u)*du

        EXAMPLES
        =========
        # >>> x = Var(np.pi)
        # >>> f = 10e16 * Var.sin(x)
        # >>> print(np.round(f.get_value(), 2))
        12.25
        # >>> print(np.round(f.get_der(), 2))
        [-1.e+17]
        """
        new_val = np.sin(self._val)
        new_der = []

        for der in self._der:
            new_der.append(np.cos(self._val) * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def arcsin(self):
        """
        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the arcsine of self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule d(arcsin(u)) = 1/sqrt(1-u^2)*du

        EXAMPLES
        =========
        # >>> x = Var(0)
        # >>> f = Var.arcsin(x)
        # >>> print(f.get_value())
        0
        # >>> print(f.get_der())
        [1.0]

        Raises ValueError when abs(self._val) > 1
        Raises ZeroDivisionError when self._val = 1
        """
        if abs(self._val) > 1:
            raise ValueError("Invalid value input. arcsine is not define on |x| > 1 for real output.")
        elif self._val == 1:
            raise ZeroDivisionError("Zero division occurs when derivative is calculated. " +
                                    "The derivative of arcsin(x), 1/sqrt(1 - x^2), " +
                                    "is undefined on x = 1.")

        new_val = np.arcsin(self._val)
        new_der = []

        for der in self._der:
            new_der.append(1 / np.sqrt(1 - self._val**2) * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def cos(self):
        """
        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the cosine of self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatvies following the rule d(cos(u)) = -sin(u)*du

        EXAMPLES
        =========
        # >>> x = Var(np.pi)
        # >>> f = 10e16 * Var.cos(x)
        # >>> print(np.round(f.get_value(), 2))
        -1.e+17
        # >>> print(np.round(f.get_der(), 2))
        [-12.25]
        """
        new_val = np.cos(self._val)
        new_der = []

        for der in self._der:
            new_der.append(-np.sin(self._val) * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def arccos(self):
        """
        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the arccosine of self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule d(arccos(u)) = -1/sqrt(1-u^2)*du

        EXAMPLES
        =========
        # >>> x = Var(0)
        # >>> f = Var.arccos(x)
        # >>> print(np.round(f.get_value(), 2))
        1.57
        # >>> print(np.round(f.get_der(), 2))
        [-1.0]

        Raises ValueError when abs(self._val) > 1
        Raises ZeroDivisionError when self._val = 1
        """
        if abs(self._val) > 1:
            raise ValueError("Invalid value input. arcsin(x) is not defined on |x| > 1 for real output.")
        elif self._val == 1:
            raise ZeroDivisionError("Zero division occurs when derivative is calculated. " +
                                    "The derivative of arccos(x), -1/sqrt(1 - x^2), " +
                                    "is undefined on x = 1.")
        new_val = np.arccos(self._val)
        new_der = []

        for der in self._der:
            new_der.append(-1 / np.sqrt(1 - self._val**2) * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def tan(self):
        """
        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the tangent of self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule d(tan(u)) = 1/(cos(u)^2)*du

        EXAMPLES
        =========
        # >>> x = Var(np.pi / 3)
        # >>> f = Var.tan(x)
        # >>> print(np.round(f.get_value(), 2))
        1.73
        # >>> print(np.round(f.get_der(), 2))
        [4.0]

        Raises ValueError when self._val = (2n+1)*pi/2
        """
        if self._val % (np.pi/2) == 0 and (self._val / (np.pi/2)) % 2 != 0:
            raise ValueError("Invalid value input. tan(x) is not defined on x = (2n+1)*pi/2.")
        new_val = np.tan(self._val)
        new_der = []

        for der in self._der:
            new_der.append((1 / np.cos(self._val))**2 * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def arctan(self):
        """
        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the arctangent of self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule d(arctan(u)) = 1/(u^2+1)*du

        EXAMPLES
        =========
        # >>> x = Var(1)
        # >>> f = Var.arctan(x)
        # >>> print(np.round(f.get_value(), 2))
        0.79
        # >>> print(np.round(f.get_der(), 2))
        [0.5]
        """
        new_val = np.arctan(self._val)
        new_der = []

        for der in self._der:
            new_der.append(1 / (self._val**2 + 1) * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def sinh(self):
        """
        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the hyperbolic sine of self._val
            - _var_list being a list of current variables
            - _der being a list partial derivatives following the rule d(sinh(u)) = cosh(u)*du

        EXAMPLES
        =========
        # >>> x = Var(1)
        # >>> f = Var.arcsin(x)
        # >>> print(np.round(f.get_value(), 2))
        1.18
        # >>> print(np.round(f.get_der(), 2))
        [1.54]
        """
        new_val = np.sinh(self._val)
        new_der = []

        for der in self._der:
            new_der.append(np.cosh(self._val) * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def cosh(self):
        """
        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the hyperbolic cosine of self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule d(sinh(u)) = sinh(u)*du

        EXAMPLES
        =========
        # >>> x = Var(1)
        # >>> f = Var.cosh(x)
        # >>> print(np.round(f.get_value(), 2))
        1.54
        # >>> print(np.round(f.get_der(), 2))
        [1.18]
        """
        new_val = np.cosh(self._val)
        new_der = []

        for der in self._der:
            new_der.append(np.sinh(self._val) * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def tanh(self):
        """
        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the hyperbolic tangent of self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule d(tanh(u)) = 1/(cosh(u)^2)*du

        EXAMPLES
        =========
        # >>> x = Var(1)
        # >>> f = Var.tanh(x)
        # >>> print(np.round(f.get_value(), 2))
        0.76
        # >>> print(np.round(f.get_der(), 2))
        [0.42]
        """
        new_val = np.tanh(self._val)
        new_der = []

        for der in self._der:
            new_der.append((1 / np.cosh(self._val))**2 * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var

    def logistic(self):
        """
        INPUTS
        =======
        self: a Var object

        RETURNS
        =======
        new_var: a Var object with:
            - _val equal to the logistic of self._val
            - _var_list being a list of current variables
            - _der being a list of partial derivatives following the rule d(logistic(u)) = e^(-u)/(1+e^(-u))^2*du

        # TODO
        EXAMPLES
        =======

        """
        new_val = 1 / (1 + np.exp(-self._val))
        new_der = []

        for der in self._der:
            new_der.append(np.exp(-self._val) / (1 + np.exp(-self._val))**2 * der)

        new_var = Var(new_val)
        new_var._var_list = self._var_list.copy()
        new_var._der = new_der
        return new_var


class MultiFunc:
    def __init__(self, func_list):
        """
        Constructor of class MultiFunc

        Keyword arguments:
        -- func_list: a list or np.array containing Var objects
            TypeError is raised when func_list is not a list or np.array
            TypeError is raised when not each element in func_list is a Var object

        Initializes a MultiFunc object with one attribute:
        -- func_list: a list or np.array containing all Var objects
                type : list or np.array
                initial value corresponds to the input variable func_list
        """
        if not isinstance(func_list, (list, np.ndarray)):
            raise TypeError('Invalid input type. func_list must be a list or np.ndarray')

        for f in func_list:
            if not isinstance(f, Var):
                raise TypeError('Invalid input type. All elements in F must be Var objects.')

        self._func_list = func_list

    def get_value(self):
        """Returns a list of all val attributes corresponding to the Var objects in
        contained in the MultiFunc object self

        INPUTS
        =======
        self: object of MultiFunc

        RETURNS
        =======
        val: a list of attributes Var.val, or thus a list of real numbers

        # TODO
        EXAMPLES
        =======

        """
        val = []
        for f in self._func_list:
            val.append(f.get_value())
        return val

    def get_der(self, var_list=None):
        """
        INPUTS
        =======
        self: object of MultiFunc
        var_list: a list of Var

        RETURNS
        =======
        der: a matrix containing the partial derivatives of each function in self.func_list
                  with respect to the Var objects listed in var_list
                  So the element in the der with index (i,j) corresponds to the partial derivative
                  of the i-th element in self.func_list with respect to the j-th Var in var_list.
                  This matrix only contains real values

        # TODO
        EXAMPLES
        =======

        """
        der = []

        for f in self._func_list:
            der.append(f.get_der(var_list))
            
        return der

    def __eq__(self, other):
        """Returns true if each two corresponding entries in self and other are equal

        INPUTS
        =======
        self: object of MultiFunc
        other: object of MultiFunc

        RETURNS
        =======
        a boolean, indicating if each two corresponding entries in self and other are equal

        # TODO
        EXAMPLES
        =======
        """
        if isinstance(other, MultiFunc):
            for f1, f2 in zip(self._func_list, other._func_list):
                if f1 != f2:
                    return False
            return True
        return False

    def __ne__(self, other):
        """Returns true if __eq__ returns false

        INPUTS
        =======
        self: object of MultiFunc
        other: object of MultiFunc

        RETURNS
        =======
        a boolean, indicating if __eq__returns false

        # TODO
        EXAMPLES
        =======
        """
        return not self == other

    def __len__(self):
        """
        INPUTS
        =======
        self: object of MultiFunc

        RETURNS
        =======
        the length of the attribute func_list, or the dimension of the multifunction object self

        EXAMPLES
        =======
        """
        return len(self._func_list)

    def __add__(self, other):
        """Returns the MultiFunc object that results from adding the two inputs (self + other)

        INPUTS
        =======
        self: object of MultiFunc
        other: a float or integer number, a Var object or a Multifunc object

        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            If other is a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of Var objects added to the corresponding
              Var objects in other - so element-wise

            If other is not a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of the Var objects in self.func_list added to other

        # TODO
        EXAMPLES
        =========

        Raises ValueError when other is a MultiFunc object with other dimensions than self
        """
        new_func_list = []
        if isinstance(other, MultiFunc):
            if len(self) == len(other):
                for i in range(len(self)):
                    new_func_list.append(self._func_list[i] + other._func_list[i])
            else:
                raise ValueError("Dimensions of the MultiFunc objects are not equal.")
        else:
            for i in range(len(self)):
                new_func_list.append(self._func_list[i] + other)

        return MultiFunc(new_func_list)

    def __radd__(self, other):
        """Returns the MultiFunc object that results from adding the two inputs (other + self)

        INPUTS
        =======
        self: object of MultiFunc
        other: a float or integer number or a Var object

        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            Other cannot be a MultiFunc object, as this case falls under __add__

            If other is not a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of the Var objects in self.func_list added to other

        # TODO
        EXAMPLES
        =========
        """
        new_func_list = []
        for i in range(len(self)):
            new_func_list.append(other + self._func_list[i])

        return MultiFunc(new_func_list)

    def __sub__(self, other):
        """Returns the MultiFunc object that results from subtracting the two inputs (self - other)

        INPUTS
        =======
        self: object of MultiFunc
        other: a float or integer number, a Var object or a Multifunc object

        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            If other is a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of Var objects minus the corresponding
              Var objects in other - so element-wise

            If other is not a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of the Var objects in self.func_list added minus other

        # TODO
        EXAMPLES
        =========

        Raises ValueError when other is a MultiFunc object with other dimensions than self
        """
        new_func_list = []
        if isinstance(other, MultiFunc):
            if len(self) == len(other):
                for i in range(len(self)):
                    new_func_list.append(self._func_list[i] - other._func_list[i])
            else:
                raise ValueError("Dimensions of the MultiFunc objects are not equal.")
        else:
            for i in range(len(self)):
                new_func_list.append(self._func_list[i] - other)

        return MultiFunc(new_func_list)

    def __rsub__(self, other):
        """Returns the MultiFunc object that results from subtracting the two inputs (other - self)

        INPUTS
        =======
        self: object of MultiFunc
        other: a float or integer number or a Var object

        # TODO
        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            Other cannot be a MultiFunc object, as this case falls under __sub__

            If other is not a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of other minus the Var objects in self.func_list

        EXAMPLES
        =========
        """
        new_func_list = []
        for i in range(len(self)):
            new_func_list.append(other - self._func_list[i])

        return MultiFunc(new_func_list)

    def __mul__(self, other):
        """Returns the MultiFunc object that results from multiplying the two inputs (self*other)

        INPUTS
        =======
        self: object of MultiFunc
        other: a float or integer number, a Var object or a Multifunc object

        # TODO
        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            If other is a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of Var objects multiplied by the corresponding
              Var objects in other - so element-wise

            If other is not a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of the Var objects in self.func_list mulitplied by other

        EXAMPLES
        =========

        Raises ValueError when other is a MultiFunc object with other dimensions than self
        """
        new_func_list = []
        if isinstance(other, MultiFunc):
            if len(self) == len(other):
                for i in range(len(self)):
                    new_func_list.append(self._func_list[i] * other._func_list[i])
            else:
                raise ValueError("Dimensions of the MultiFunc objects are not equal.")
        else:
            for i in range(len(self)):
                new_func_list.append(self._func_list[i] * other)

        return MultiFunc(new_func_list)

    def __rmul__(self, other):
        """Returns the MultiFunc object that results from multiplying the two inputs (other*self)

        INPUTS
        =======
        self: object of MultiFunc
        other: a float or integer number or a Var object

        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            Other cannot be a MultiFunc object, as this case falls under __mul__

            If other is not a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of other multiplied by the Var objects in self.func_list

        # TODO
        EXAMPLES
        =========
        """
        new_func_list = []
        for i in range(len(self)):
            new_func_list.append(other * self._func_list[i])

        return MultiFunc(new_func_list)

    def __truediv__(self, other):
        """Returns the MultiFunc object that results from dividing the two inputs (self/other)

        INPUTS
        =======
        self: object of MultiFunc
        other: a float or integer number, a Var object or a Multifunc object

        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            If other is a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of Var objects divided by the corresponding
              Var objects in other - so element-wise

            If other is not a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of the Var objects in self.func_list divided by other

        # TODO
        EXAMPLES
        =========

        Raises ValueError when other is a MultiFunc object with other dimensions than self
        """
        new_func_list = []
        if isinstance(other, MultiFunc):
            if len(self) == len(other):
                for i in range(len(self)):
                    new_func_list.append(self._func_list[i] / other._func_list[i])
            else:
                raise ValueError("Dimensions of the MultiFunc objects are not equal.")
        else:
            for i in range(len(self)):
                new_func_list.append(self._func_list[i] / other)

        return MultiFunc(new_func_list)

    def __rtruediv__(self, other):
        """Returns the MultiFunc object that results from dividing the two inputs (other/self)

        INPUTS
        =======
        self: object of MultiFunc
        other: a float or integer number or a Var object

        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            Other cannot be a MultiFunc object, as this case falls under __truediv__

            If other is not a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of other divided by the Var objects in self.func_list

        # TODO
        EXAMPLES
        =========
        """
        new_func_list = []
        for i in range(len(self)):
            new_func_list.append(other / self._func_list[i])

        return MultiFunc(new_func_list)

    def __abs__(self):
        """Returns the MultiFunc object that contains all the absolute values of the Var objects in self.func_list

        INPUTS
        =======
        self: object of MultiFunc

        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            Returns a MultiFunc object with:
            - for func_list an array of the absolute values of all individual Var objects in self.func_list

        # TODO
        EXAMPLES
        =========
        """

        new_func_list = []
        for i in range(len(self)):
            new_func_list.append(abs(self._func_list[i]))

        return MultiFunc(new_func_list)

    def __neg__(self):
        """Returns the MultiFunc object that contains all the opposite values of the Var objects in self.func_list

        INPUTS
        =======
        self: object of MultiFunc

        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            Returns a MultiFunc object with:
            - for func_list an array of the opposite values of all individual Var objects in self.func_list

        # TODO
        EXAMPLES
        =========
        """
        new_func_list = []
        for i in range(len(self)):
            new_func_list.append(-self._func_list[i])

        return MultiFunc(new_func_list)

    def __pow__(self, power):
        """Returns the MultiFunc object that results from taking self to the power of power (self**power)

        INPUTS
        =======
        self: object of MultiFunc
        power: a real number (int, float, np.int, np.float), a Var object or a MultiFunc object

        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            If power is a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of Var objects of self raised to the corresponding
              Var objects in power - so element-wise

            If power is not a MultiFunc object, returns a MultiFunc object with:
            - for func_list an array of Var objects of self raised to the specified power

        # TODO
        EXAMPLES
        =========

        Raises ValueError when power is a MultiFunc object with other dimensions than self
        """

        new_func_list = []
        if isinstance(power, MultiFunc):
            if len(self) == len(power):
                for i in range(len(self)):
                    new_func_list.append(self._func_list[i] ** power._func_list[i])
            else:
                raise ValueError("Dimensions of the MultiFunc objects are not equal.")
        else:
            for i in range(len(self)):
                new_func_list.append(self._func_list[i] ** power)

        return MultiFunc(new_func_list)

    def __rpow__(self, other):
        """Returns the MultiFunc object that results from taking other to the power of self (other**self)

        INPUTS
        =======
        self: object of MultiFunc
        other: a real number (int, float, np.int, np.float), a Var object or a MultiFunc object

        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            Other cannot be a MultiFunc object, as this case falls under __pow__

            Returns a MultiFunc object with:
            - for func_list an array of Var objects of self raised to the specified power

        # TODO
        EXAMPLES
        =========

        """
        new_func_list = []
        for i in range(len(self)):
            new_func_list.append(other ** self._func_list[i])

        return MultiFunc(new_func_list)

    def apply(self, func, *args):
        """Returns the MultiFunc object that results from element-wisely apply the function func to all
        elements in MultiFunc

        INPUTS
        =======
        self: object of MultiFunc
        func: a non-dunder mtehod of the class Var

        Includes: Var.exp, Var.sqrt, Var.sin, Var.arcsin, Var.cos, Var.arccos
                  Var.tan, Var.arctan, Var.sinh, Var.cosh, Var.tanh, Var.logistic

                  Note that when a variable b should be specified in Var.log, the user should
                  input two arguments: Var.log and b

        RETURNS
        =======
        MultiFunc(new_func_list): an object of MultiFunc

            Returns a MultiFunc object with:
            - for func_list an array of the function func applied to all individual Var objects in self.func_list

        # TODO
        EXAMPLES
        =========

        """

        new_func_list = []
        for i in range(len(self)):
            if len(args) == 0:
                new_func_list.append(func(self._func_list[i]))
            else:
                new_func_list.append(func(self._func_list[i], *args))

        return MultiFunc(new_func_list)


class _Constant(Var):
    def __init__(self, val):
        """Initialized an object of the private Constant class, which is inherited from the Var class with three private
        attributes: self._val, self._var_list and self._der
        """
        super().__init__(val)
        self._var_list = []
        self._der = []


# x = Var(1)
# y = Var(1)
# z = Var(1)
# func_list = [x+y, x+y+z]
# multi_func = MultiFunc(func_list)
# print(multi_func.get_values())
# print(multi_func.get_der([x,y,z]))
#
# new_f = multi_func.apply(Var.sin)
# print(new_f.get_values())
# print(new_f.get_der([x,y,z]))

# # x = Var(2)
# y = Var(-5)
# f1 = - 3 * y
# print(f1.get_value())
# print(f1.get_der())
# # f2 = -f1
# # print(f2.get_value())
# # print(f2.get_der())
# f3 = abs(f1)
# print(f3.get_value())
# print(f3.get_der())

# x = Var(1.0)
# y = Var(2.0)
# z = Var(3.0)
# print(x, y, z)
# f = x + y + z
# print(f._var_list)
# print(f)
# f1 = Var.logistic(f)
# print(f1)
# print(f1._var_list)
# print(f1.get_der([x, y, z]))
# print(f1.get_value())
# print(np.round(f1.get_der(), 3))

# x = Var(5.0)
# f = MultiFunc([x + 2, x ** 3, 2 * x])
# f1 = f.apply(Var.log)
# print(f1.get_value())
# print(f1.get_der())

# x = Var(1)
# y = Var(2)
# f1 = x + y
# f2 = 2*x + y
# print(f1 == f2) # should be true because f1 and f2 have the same value and partial derivative w/ respect to x and y.

# x = Var(1)
# y = Var(1)
# print(x == y) # should be false because x and y have different variables

# x = Var(3.0)
# y = Var(2.0)
# z = MultiFunc([x, y])
# z1 = 2*z
# print(z1.get_der([x]))
# assert z1.get_der([x]) == [[2.0], [0.0]]

