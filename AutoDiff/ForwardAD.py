import numpy as np


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
            new_val = self._val + other
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
            new_val = self._val * other
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
            new_val = self._val / other._val
            new_jacobian = {}

            new_vars = set(self._jacobian.keys()) | set(other._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = (float(self._jacobian.get(var) or 0) * other._val
                                - self._val * float(other._jacobian.get(var) or 0))\
                               /(other._val ** 2)
        elif isinstance(other, Var.valid_types):
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
            new_val = self._val ** power._val
            new_jacobian = {}

            new_vars = set(self._jacobian.keys()) | set(power._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = power._val * self._val ** (power._val - 1) * float(self._jacobian.get(var) or 0) \
                               + (self._val ** power._val) * np.log(self._val) * float(power._jacobian.get(var) or 0)
        elif isinstance(power, Var.valid_types):
            new_val = self._val ** power
            new_jacobian = {}

            new_vars = set(self._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = power * self._val ** (power - 1) * self._jacobian.get(var)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: Var, int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def __rpow__(self, power, modulo=None):
        if isinstance(power, Var):
            new_val = self ** power._val
            new_jacobian = {}

            new_vars = set(self._jacobian.keys())
            for var in new_vars:
                new_jacobian[var] = (self ** power._val) * np.log(power._val) * self._jacobian.get(var)
        else:
            raise TypeError('Invalid input type. ' +
                            'Other must be any of the following types: int, float, np.int, np.float.')

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def exp(self):
        new_val = np.exp(self._val)
        new_jacobian = {}

        new_vars = set(self._jacobian.keys())
        for var in new_vars:
            new_jacobian[var] = np.exp(self._val) * self._jacobian.get(var)

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def log(self):
        new_val = np.log(self._val)
        new_jacobian = {}

        new_vars = set(self._jacobian.keys())
        for var in new_vars:
            new_jacobian[var] = 1/self._val * self._jacobian.get(var)

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def sqrt(self):
        # TODO should we check self type?
        new_val = np.sqrt(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = 1 / 2 * np.power(self._val, -1 / 2) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def sin(self):
        new_val = np.sin(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = np.cos(self._val) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def arcsin(self):
        new_val = np.arcsin(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = 1 / np.sqrt(1 - np.power(self._val, 2)) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def cos(self):
        new_val = np.cos(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = -np.sin(self._val) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def arccos(self):
        new_val = np.arccos(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = -1 / np.sqrt(1 - np.power(self._val, 2)) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def tan(self):
        new_val = np.tan(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = np.power(1 / np.cos(self._val), 2) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def arctan(self):
        new_val = np.arctan(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = 1 / (np.power(self._val, 2) + 1) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def sinh(self):
        new_val = np.sinh(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = np.cosh(self._val) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def cosh(self):
        new_val = np.cosh(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = np.sinh(self._val) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var

    def tanh(self):
        new_val = np.tanh(self._val)
        new_jacobian = {}

        new_vars = self._jacobian.keys()
        for var in new_vars:
            new_jacobian[var] = np.power(1 / np.cosh(self._val), 2) * self._jacobian[var]

        new_var = Var(new_val)
        new_var._jacobian = new_jacobian
        return new_var


x = Var(3)
y = Var(3)
f = Var.sqrt(x + 2 * y)
print(f.get_value())
print(type(f.get_value()))
print(f.get_jacobian())
print(type(f.get_jacobian()))
