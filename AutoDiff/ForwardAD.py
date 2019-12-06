import numpy as np
import math


class Var:
    valid_types = (int, float, np.int, np.float)

    def __init__(self, val):
        """
        Constructor of class Var

        Keyword arguments:
        -- val: A real number (int, float, np.int ot np.float)
            TypeError is raised when val is not a real number

        Initializes a Var object with two attributes:
        -- val: represents the current evaluation of the Var object
                type : real number
                initial value corresponds to the input variable val (real number)
        -- der: represents the partial derivative of the Var object with respect to all variables
                     type : dictionary with variables as keys and partial derivatives as values
                     initial value corresponds to a dictionary with 1 key, being the object itself and value 1
        """
        if not isinstance(val, Var.valid_types):
            raise TypeError('Invalid input type. ' +
                            'Val must be any of the following types: int, float, np.int, np.float.')
        self._val = val
        self._der = {self: 1.0}

    def get_value(self):
        """Returns the val attribute of the Var object

        INPUTS
        =======
        self: object of Var

        RETURNS
        =======
        self.val: a real number

        EXAMPLES
        =======
        >>> x = Var(5.0)
        >>> x.get_value()
        5.0
        """
        return self._val

    def get_der(self, var_list=None):
        """
        INPUTS
        =======
        self: obejct of Var
        var_list: a list of Var

        RETURNS
        =======
        der: a list of real numbers, whose i-th number is the derivative with respect to the i-th Var in var_list.

        EXAMPLES
        =======
        >>> x = Var(1)
        >>> y = Var(1)
        >>> f = x + 2*y
        >>> f.get_der([x, y])
        [1.0, 2.0]
        """
        if var_list is None:
            return list(self._der.values())
        
        der = [None]*len(var_list)
        for i, var in enumerate(var_list):
            der[i] = self._get_derivative_of(var)
        return der

    def _get_derivative_of(self, var):
        """Returns the partial derivative of the Var object self with respect to one of its
        variables var

        INPUTS
        =======
        self: object of Var

        RETURNS
        =======
        self.val: a real number, which is the partial derivaties of self with respect to the variable var.

        EXAMPLES
        =======
        >>> x = Var(5.0)
        >>> y = Var(2.0)
        >>> f = x**2+y
        >>> f.get_derivative_of(x)
        10.0
        """
        if isinstance(var, Var):
            if var in self._der.keys():
                return self._der[var]
            else:
                return 0.0
        else:
            raise TypeError('Invalid input type. ' +
                            'var must be any of the following types: Var.')

    def __eq__(self, other):
        pass


    def __ne__(self, other):
        pass

    def __add__(self, other):
        """Returns the var object that results from adding the two inputs (self + other)

        INPUTS
        =======
        self: obejct of Var
        other: a float or integer number, or a Var object

        RETURNS
        =======
        new_var: an object of Var.
            If other is a Var object, returns a Var object with:
            - A value equal to the sum of each val attribute of self and other
            - A der equal to the sum of the ders
            Note that when both Var objects contain different variables the der is expanded

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - A value equal to the sum of the val attribute of and the number
            - A der equal to the der of self

        EXAMPLES
        =======
        >>> x = Var(5.0)
        >>> f = x+3.0
        >>> x.get_value()
        8.0
        >>> x.get_der()
        [1.0]

        Raises TypeError when other is no Var object or real number
        """
        if isinstance(other, Var):
            new_val = self._val + other._val
            new_der = {}

            # Obtain a new variable set. For example, if self has {x, y}, and other has {y, z},
            # then the new variable set would be {x, y, z}
            new_vars = set(self._der.keys()) | set(other._der.keys())

            # Loop through new variables in the new variable set
            # For each variable calculate the partial derivative.
            # if the dictionary does not contain the key/variable it will return None.
            # float(None or 0) = 0.0; float(a real number or 0) = a real number (e.g. float(5 or 0) = 5.0)
            for var in new_vars:
                new_der[var] = float(self._der.get(var) or 0) + float(other._der.get(var) or 0)
        elif isinstance(other, Var.valid_types):
            new_val = self._val + other
            new_der = self._der
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def __radd__(self, other):
        """Returns the var object that results from adding the two inputs (other + self)

        INPUTS
        =======
        self: obejct of Var
        other: a float or integer number

        RETURNS
        =======
        new_var: an object of Var.
            Other cannot be a Var object, as this case falls under __add__

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - A value equal to the sum of the val attribute of and the number
            - A der equal to the der of self

        EXAMPLES
        =======
        >>> x = Var(5.0)
        >>> f = 3.0+x
        >>> x.get_value()
        8.0
        >>> x.get_der()
        [1.0]

        Raises TypeError when other is no real number
        """
        if isinstance(other, Var.valid_types):
            new_val = other + self._val
            new_der = self._der
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def __sub__(self, other):
        """Returns the var object that results from subtracting the two inputs (self - other)

        INPUTS
        =======
        self: obejct of Var
        other: a float or integer number, or a Var object

        RETURNS
        =======
        new_var: an object of Var.
            If other is a Var object, returns a Var object with:
            - A value equal to self.val - other.val
            - A der equal to the difference between the ders of self and other
            Note that when both Var objects contain different variables the der is expanded

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - A value equal to self._val - other
            - A der equal to the der of self

        EXAMPLES
        =======
        >>> x = Var(5.0)
        >>> f = x-3.0
        >>> x.get_value()
        2.0
        >>> x.get_der()
        [1.0]

        Raises TypeError when other is no Var object or real number
        """
        if isinstance(other, Var):
            new_val = self._val - other._val
            new_der = {}

            new_vars = set(self._der.keys()) | set(other._der.keys())
            for var in new_vars:
                new_der[var] = float(self._der.get(var) or 0) - float(other._der.get(var) or 0)
        elif isinstance(other, Var.valid_types):
            new_val = self._val - other
            new_der = self._der
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def __rsub__(self, other):
        """Returns the var object that results from subtracting the two inputs (other - self)

        INPUTS
        =======
        self: obejct of Var
        other: a float or integer number

        RETURNS
        =======
        new_var: an object of Var.
            Other cannot be a Var object, as this case falls under __sub__

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - A value equal to other - self._val
            - A der equal to the negative der of self

        EXAMPLES
        =======
        >>> x = Var(5.0)
        >>> f = 3.0-x
        >>> x.get_value()
        -2.0
        >>> x.get_der()
        [1.0]

        Raises TypeError when other is no real number
        """
        if isinstance(other, Var.valid_types):
            new_val = other - self._val
            new_der = {}

            new_vars = set(self._der.keys())
            for var in new_vars:
                new_der[var] = -self._der.get(var)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def __mul__(self, other):
        """Returns the var object that results from multiplying the two inputs (self*other)

        INPUTS
        =======
        self: obejct of Var
        other: a float or integer number, or a Var object

        RETURNS
        =======
        new_var: an object of Var.
            If other is a Var object, returns a Var object with:
            - A value equal to self._val*other._val
            - A der following the rule of d(uv) = udv + vdu
            Note that when both Var objects contain different variables the der is expanded

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - A value equal to self._val*other
            - A der equal to the der of self multiplied by other

        EXAMPLES
        =======
        >>> x = Var(5.0)
        >>> f = x*3.0
        >>> x.get_value()
        15.0
        >>> x.get_der()
        [3.0]

        Raises TypeError when other is no Var object or real number
        """
        if isinstance(other, Var):
            new_val = self._val * other._val
            new_der = {}

            new_vars = set(self._der.keys()) | set(other._der.keys())
            for var in new_vars:
                new_der[var] = float(self._der.get(var) or 0) * other._val + self._val * float(other._der.get(var) or 0)
        elif isinstance(other, Var.valid_types):
            new_val = self._val * other
            new_der = {}

            new_vars = self._der.keys()
            for var in new_vars:
                new_der[var] = self._der.get(var) * other
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def __rmul__(self, other):
        """Returns the var object that results from multiplying the two inputs (other*self)

        INPUTS
        =======
        self: obejct of Var
        other: a float or integer number

        RETURNS
        =======
        new_var: an object of Var.
            Other cannot be a Var object, as this case falls under __mul__

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - A value equal to other*self._val
            - A der equal to the der of self multiplied by other

        EXAMPLES
        =======
        >>> x = Var(5.0)
        >>> f = 3.0*x
        >>> x.get_value()
        15.0
        >>> x.get_der()
        [3.0]

        Raises TypeError when other is no real number
        """
        if isinstance(other, Var.valid_types):
            new_val = other * self._val
            new_der = {}

            new_vars = self._der.keys()
            for var in new_vars:
                new_der[var] = self._der.get(var) * other
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def __truediv__(self, other):
        """Returns the var object that results from dividing the two inputs (self/other)

        INPUTS
        =======
        self: obejct of Var
        other: a float or integer number, or a Var object

        RETURNS
        =======
        new_var: an object of Var.
            If other is a Var object, returns a Var object with:
            - A value equal to self._val/other._val
            - A der following the rule of d(u/v) = (udv - vdu)/v^2
            Note that when both Var objects contain different variables the der is expanded

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - A value equal to self.val/other
            - A der equal to the der of self divided by other

        EXAMPLES
        =======
        >>> x = Var(5.0)
        >>> f = x/2.0
        >>> x.get_value()
        2.5
        >>> x.get_der()
        [0.5]

        Raises TypeError when other is no Var object or real number
        Raises ZeroDivisionError when other._val or other is equal to zero
        """
        if isinstance(other, Var):
            if other._val == 0:
                raise ZeroDivisionError("Denominator cannot be 0.")
            new_val = self._val / other._val
            new_der = {}

            new_vars = set(self._der.keys()) | set(other._der.keys())
            for var in new_vars:
                new_der[var] = (float(self._der.get(var) or 0) * other._val
                                     - self._val * float(other._der.get(var) or 0))\
                               /(other._val ** 2)
        elif isinstance(other, Var.valid_types):
            if other == 0:
                raise ZeroDivisionError("Denominator cannot be 0.")
            new_val = self._val / other
            new_der = {}

            new_vars = set(self._der.keys())
            for var in new_vars:
                new_der[var] = self._der.get(var) / other
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def __rtruediv__(self, other):
        """Returns the var object that results from dividing the two inputs (other/self)

        INPUTS
        =======
        self: obejct of Var
        other: a float or integer number

        RETURNS
        =======
        new_var: an object of Var.
            Other cannot be a Var object, as this case falls under __truediv__

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - A value equal to other/self._val
            - A der equal to other multiplied by the der that follows the rule d(1/u) = -1/u^2

        EXAMPLES
        =======
        >>> x = Var(5.0)
        >>> f = 10.0/x
        >>> x.get_value()
        2.0
        >>> x.get_der()
        [0.4]


        Raises TypeError when other is no real number
        Raises ZeroDivisionError when self._val is equal to zero
        """
        if self._val == 0:
            raise ZeroDivisionError("Denominator cannot be 0.")
        if isinstance(other, Var.valid_types):
            new_val = other / self._val
            new_der = {}

            new_vars = set(self._der.keys())
            for var in new_vars:
                new_der[var] = -other/(self._val ** 2) * self._der.get(var)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def __abs__(self):
        """Returns the var object whose val is the absolute value of self._val

        INPUTS
        =======
        self: obejct of Var
        other: a Var object

        RETURNS
        =======
        new_var: an object of Var.
            When self.val > 0, returns a Var object with:
            - A value equal to self._val
            - A der equal self._der

            When self.val < 0, returns a Var object with:
            - A value equal to -self._val
            - A der equal -self._der

        EXAMPLES
        =========
        >>> x = Var(-5.0)
        >>> f = abs(x)
        >>> print(f.get_value())
        5.0
        >>> print(f.get_der())
        [-1.0]

        Raises ValueError when self._val = 0, as the derivative is then undefined
        """
        if self._val == 0:
            raise ValueError('Derivative of abs() is not defined at 0.')
        elif self._val < 0:
            new_val = -self._val
            new_der = {}

            new_vars = set(self._der.keys())
            for var in new_vars:
                new_der[var] = -self._der.get(var)
        elif self._val > 0:
            new_val = self._val
            new_der = self._der

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def __neg__(self):
        """Returns a Var object whose val is the negative value of self._val

        INPUTS
        =======
        self: obejct of Var
        other: a Var object

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to -self._val
            - A der equal -self._der

         EXAMPLES
         =========
         >>> x = Var(2.0)
         >>> f = -x
         >>> print(f.get_value())
         -2.0
         >>> print(f.get_der())
         [-1.0]
         """
        new_val = -self._val
        new_der = {}

        new_vars = set(self._der.keys())
        for var in new_vars:
            new_der[var] = -self._der.get(var)

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def __pow__(self, power, modulo=None):
        """Returns the var object that results from taking self to the power of power (self**power)

        INPUTS
        =======
        self: obejct of Var
        other: a real number (int, float, np.int, np.float), or a Var object

        RETURNS
        =======
        new_var: an object of Var.
            If power is a Var object, returns a Var object with:
            - A value equal to self._val**power._val
            - A der following the rule of d(u^v) = u^v(dv*log(u) + v*du/u)
            Note that when both Var objects contain different variables the der is expanded

            If power is a real number (int, float, np.int, np.float), returns a Var object with:
            - A value equal to self.val^power
            - A der following the rule d(u^power) = power*u^(power-1)*du

        EXAMPLES
        =========
        >>> x = Var(5.0)
        >>> f = x**2
        >>> print(f.get_value())
        25.0
        >>> print(f.get_der())
        [10.0]

        Raises TypeError when power is no Var object or real number
        Raises ValueError when self._val < 0
        """
        if isinstance(power, Var):
            if self._val < 0:
                raise ValueError("The derivative of x ** y is not defined on x < 0.")
            new_val = self._val ** power._val
            new_der = {}

            new_vars = set(self._der.keys()) | set(power._der.keys())
            for var in new_vars:
                new_der[var] = new_val * \
                                    (float(power._der.get(var) or 0) * np.log(self._val) +
                                     power._val * float(self._der.get(var) or 0) / self._val)
        elif isinstance(power, Var.valid_types):
            new_val = self._val ** power
            new_der = {}

            new_vars = set(self._der.keys())
            for var in new_vars:
                new_der[var] = power * self._val ** (power - 1) * self._der.get(var)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, np.int.')

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def __rpow__(self, other):
        """Returns the var object that results from taking other to the power of self (other**self)

        INPUTS
        =======
        self: obejct of Var
        other: a real number (int, float, np.int, np.float)

        RETURNS
        =======
        new_var: an object of Var.
            Other cannot be a Var object, as this case falls under __pow__

            If other is a real number (int, float, np.int, np.float), returns a Var object with:
            - A value equal to power^self._val
            - A der following the rule d(other^u) = other^u*log(other)du

        EXAMPLES
        =========
        >>> x = Var(5.0)
        >>> f = 2.0**x
        >>> print(f.get_value())
        32.0
        >>> np.round(f.get_der(), 8) == 22.18070978

        Raises TypeError when other is no Var object or real number
        Raises ValueError when other < 0
        """
        if isinstance(other, Var.valid_types):
            if other < 0:
                raise ValueError("The derivative of b ** x, b**x * ln(b), is not defined on b < 0.")
            new_val = other ** self._val
            new_der = {}

            new_vars = set(self._der.keys())
            for var in new_vars:
                new_der[var] = (other ** self._val) * np.log(other) * self._der.get(var)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def exp(self):
        """Returns the var object that results from taking the exponent of self

        INPUTS
        =======
        self: obejct of Var
        other: a real number (int, float, np.int, np.float), or a Var object

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to exp(self._val)
            - A der following the rule d(exp(u)) = exp(u)*du

        EXAMPLES
        =========
        >>> x = Var(5.0)
        >>> f = Var.exp(x)
        >>> print(f.get_value())
        148.41
        >>> print(np.round(f.get_der(), 2))
        [148.41]
        """
        new_val = np.exp(self._val)
        new_der = {}

        new_vars = set(self._der.keys())
        for var in new_vars:
            new_der[var] = np.exp(self._val) * self._der.get(var)

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def log(self, b):
        """
        INPUTS
        =======
        self: obejct of Var
        b: an integer, the base of the logarithm

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to the base b logarithm of self._val
            - A der following the rule d(log(u, b)) = 1/(u*log(b))*du

         EXAMPLES
         =========
         >>> x = Var(1000)
         >>> f = Var.log(x, 10)
         >>> print(f.get_value())
         3.0
         >>> print(np.round(f.get_der(), 4))
         [0.0004]

         Raises TypeError when b is not an integer
         Raises ValueError when self._val <= 0
         """
        # b is the base. The default is e (natural log).
        if not isinstance(b, (int, np.int)):
            raise TypeError("Invalid input type. b should be any of the following type: int and numpy.int.")
        if self._val <= 0:
            raise ValueError("log(x) is not defined on x <= 0.")

        new_val = math.log(self._val, b)
        new_der = {}

        new_vars = set(self._der.keys())
        for var in new_vars:
            new_der[var] = 1 / (self._val * np.log(b)) * self._der.get(var)

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def sqrt(self):
        """
        INPUTS
        =======
        self: obejct of Var

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to the square root of self._val
            - A der following the rule d(sqrt(u)) = 1/2*(u)^(-1/2)*du

        EXAMPLES
        =========
        >>> x = Var(9)
        >>> f = Var.sqrt(x)
        >>> print(f.get_value())
        3.0
        >>> print(np.round(f.get_der(), 2))
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
        new_der = {}

        new_vars = self._der.keys()
        for var in new_vars:
            new_der[var] = 1/2 * self._val**(-1/2) * self._der[var]

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def sin(self):
        """
        INPUTS
        =======
        self: obejct of Var

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to the sine of self._val
            - A der following the rule d(sin(u)) = cos(u)*du

        EXAMPLES
        =========
        >>> x = Var(np.pi)
        >>> f = 10e16 * Var.sin(x)
        >>> print(np.round(f.get_value(), 2))
        12.25
        >>> print(np.round(f.get_der(), 2))
        [-1.e+17]
        """
        new_val = np.sin(self._val)
        new_der = {}

        new_vars = self._der.keys()
        for var in new_vars:
            new_der[var] = np.cos(self._val) * self._der[var]

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def arcsin(self):
        """
        INPUTS
        =======
        self: obejct of Var

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to the arcsine of self._val
            - A der following the rule d(arcsin(u)) = 1/sqrt(1-u^2)*du

        EXAMPLES
        =========
        >>> x = Var(0)
        >>> f = Var.arcsin(x)
        >>> print(f.get_value())
        0
        >>> print(f.get_der())
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
        new_der = {}

        new_vars = self._der.keys()
        for var in new_vars:
            new_der[var] = 1 / np.sqrt(1 - self._val**2) * self._der[var]

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def cos(self):
        """
        INPUTS
        =======
        self: obejct of Var

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to the cosine of self._val
            - A der following the rule d(cos(u)) = -sin(u)*du

        EXAMPLES
        =========
        >>> x = Var(np.pi)
        >>> f = 10e16 * Var.cos(x)
        >>> print(np.round(f.get_value(), 2))
        -1.e+17
        >>> print(np.round(f.get_der(), 2))
        [-12.25]
        """
        new_val = np.cos(self._val)
        new_der = {}

        new_vars = self._der.keys()
        for var in new_vars:
            new_der[var] = -np.sin(self._val) * self._der[var]

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def arccos(self):
        """
        INPUTS
        =======
        self: obejct of Var

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to the arccosine of self._val
            - A der following the rule d(arccos(u)) = -1/sqrt(1-u^2)*du

        EXAMPLES
        =========
        >>> x = Var(0)
        >>> f = Var.arccos(x)
        >>> print(np.round(f.get_value(), 2))
        1.57
        >>> print(np.round(f.get_der(), 2))
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
        new_der = {}

        new_vars = self._der.keys()
        for var in new_vars:
            new_der[var] = -1 / np.sqrt(1 - self._val**2) * self._der[var]

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def tan(self):
        """
        INPUTS
        =======
        self: obejct of Var

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to the tangent of self._val
            - A der following the rule d(tan(u)) = 1/(cos(u)^2)*du

        EXAMPLES
        =========
        >>> x = Var(np.pi / 3)
        >>> f = Var.tan(x)
        >>> print(np.round(f.get_value(), 2))
        1.73
        >>> print(np.round(f.get_der(), 2))
        [4.0]

        Raises ValueError when self._val = (2n+1)*pi/2
        """
        if self._val % (np.pi/2) == 0 and (self._val / (np.pi/2)) % 2 != 0:
            raise ValueError("Invalid value input. tan(x) is not defined on x = (2n+1)*pi/2.")
        new_val = np.tan(self._val)
        new_der = {}

        new_vars = self._der.keys()
        for var in new_vars:
            new_der[var] = (1 / np.cos(self._val))**2 * self._der[var]

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def arctan(self):
        """
        INPUTS
        =======
        self: obejct of Var

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to the arctangent of self._val
            - A der following the rule d(arctan(u)) = 1/(u^2+1)*du

        EXAMPLES
        =========
        >>> x = Var(1)
        >>> f = Var.arctan(x)
        >>> print(np.round(f.get_value(), 2))
        0.79
        >>> print(np.round(f.get_der(), 2))
        [0.5]
        """
        new_val = np.arctan(self._val)
        new_der = {}

        new_vars = self._der.keys()
        for var in new_vars:
            new_der[var] = 1 / (self._val**2 + 1) * self._der[var]

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def sinh(self):
        """
        INPUTS
        =======
        self: obejct of Var

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to the hyperbolic sine of self._val
            - A der following the rule d(sinh(u)) = cosh(u)*du

        EXAMPLES
        =========
        >>> x = Var(1)
        >>> f = Var.arcsin(x)
        >>> print(np.round(f.get_value(), 2))
        1.18
        >>> print(np.round(f.get_der(), 2))
        [1.54]
        """
        new_val = np.sinh(self._val)
        new_der = {}

        new_vars = self._der.keys()
        for var in new_vars:
            new_der[var] = np.cosh(self._val) * self._der[var]

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def cosh(self):
        """
        INPUTS
        =======
        self: obejct of Var

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to the hyperbolic cosine of self._val
            - A der following the rule d(sinh(u)) = sinh(u)*du

        EXAMPLES
        =========
        >>> x = Var(1)
        >>> f = Var.cosh(x)
        >>> print(np.round(f.get_value(), 2))
        1.54
        >>> print(np.round(f.get_der(), 2))
        [1.18]
        """
        new_val = np.cosh(self._val)
        new_der = {}

        new_vars = self._der.keys()
        for var in new_vars:
            new_der[var] = np.sinh(self._val) * self._der[var]

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var

    def tanh(self):
        """
        INPUTS
        =======
        self: obejct of Var

        RETURNS
        =======
        new_var: an object of Var.
            Returns a Var object with:
            - A value equal to the hyperbolic tangent of self._val
            - A der following the rule d(tanh(u)) = 1/(cosh(u)^2)*du

        EXAMPLES
        =========
        >>> x = Var(1)
        >>> f = Var.tanh(x)
        >>> print(np.round(f.get_value(), 2))
        0.76
        >>> print(np.round(f.get_der(), 2))
        [0.42]
        """
        new_val = np.tanh(self._val)
        new_der = {}

        new_vars = self._der.keys()
        for var in new_vars:
            new_der[var] = (1 / np.cosh(self._val))**2 * self._der[var]

        new_var = Var(new_val)
        new_var._der = new_der
        return new_var


class MultiFunc:
    def __init__(self, func_list):
        if not isinstance(func_list, (list, np.ndarray)):
            raise TypeError('Invalid input type. func_list must be a list or np.ndarray')

        for f in func_list:
            if not isinstance(f, Var):
                raise TypeError('Invalid input type. All elements in F must be Var objects.')

        self._func_list = func_list

    def get_values(self):
        val = []
        for f in self._func_list:
            val.append(f.get_value())
        return val

    def get_jacobian(self, var_list):
        jacobian = []
        for f in self._func_list:
            jacobian.append(f.get_der(var_list))
        return jacobian

    def __eq__(self, other):
        pass

    def __ne__(self, other):
        pass

    def __len__(self):
        return len(self._func_list)

    def __add__(self, other):
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
        new_func_list = []
        for i in range(len(self)):
            new_func_list.append(other + self._func_list[i])

        return MultiFunc(new_func_list)

    def __sub__(self, other):
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
        new_func_list = []
        for i in range(len(self)):
            new_func_list.append(other - self._func_list[i])

        return MultiFunc(new_func_list)

    def __mul__(self, other):
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
        new_func_list = []
        for i in range(len(self)):
            new_func_list.append(other * self._func_list[i])

        return MultiFunc(new_func_list)

    def __truediv__(self, other):
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
        new_func_list = []
        for i in range(len(self)):
            new_func_list.append(other / self._func_list[i])

        return MultiFunc(new_func_list)


# x = Var(1)
# y = Var(1)
# z = Var(1)
# func_list = [x+y, x+y+z]
# multi_func = MultiFunc(func_list)
# print(multi_func.get_values())
# print(multi_func.get_jacobian([x,y,z]))
