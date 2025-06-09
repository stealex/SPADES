from numba import njit
from spades import ph

# Notation as in Nitescu  Universe 2021, 7(5), 147; https://doi.org/10.3390/universe7050147


@njit
def small_a(e1: float, e2: float, total_ke: float):
    return total_ke - e1 - e2


@njit
def small_b(e1: float, e2: float):
    return e1 - e2


@njit
def integral_order_0_00(e1: float, e2: float, total_ke: float):
    return 1./30. * small_a(e1, e2, total_ke)**5.0


@njit
def integral_order_2_00(e1: float, e2: float, total_ke: float):
    a = small_a(e1, e2, total_ke)
    b = small_b(e1, e2)
    return 1./(1680.*ph.electron_mass**2.0) *\
        a**5 * (a**2 + 7.*b**2)


@njit
def integral_order_22_00(e1: float, e2: float, total_ke: float):
    a = small_a(e1, e2, total_ke)
    b = small_b(e1, e2)
    return 1./(161280.*ph.electron_mass**4.0) *\
        a**5 * (a**4 - 6*(a**2) * (b**2) + 21.*b**4)


@njit
def integral_order_4_00(e1: float, e2: float, total_ke: float):
    a = small_a(e1, e2, total_ke)
    b = small_b(e1, e2)
    return 1./(80640.*ph.electron_mass**4) *\
        a**5 * (a**4 + 18.*(a**2)*(b**2) + 21*b**4)


@njit
def integral_order_22_02(e1: float, e2: float, total_ke: float):
    a = small_a(e1, e2, total_ke)
    b = small_b(e1, e2)
    return (a**7) * (b**2)/(3360.*(ph.electron_mass**4.0))


@njit
def integral_order_6_02(e1: float, e2: float, total_ke: float):
    a = small_a(e1, e2, total_ke)
    b = small_b(e1, e2)
    return (a**7) * (b**2)/(40320.*(ph.electron_mass**6.0)) * (a**2.0+3*b**2.0)


@njit
def integral_order(e1: float, e2: float, total_ke: float, order: ph.TaylorOrders, transition_type: ph.TransitionTypes):
    if transition_type == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS:
        if order == ph.TaylorOrders.TWOTWO:
            return integral_order_22_02(e1, e2, total_ke)
        elif order == ph.TaylorOrders.SIX:
            return integral_order_6_02(e1, e2, total_ke)
        else:
            raise NotImplementedError
    else:
        if order == ph.TaylorOrders.ZERO:
            res = integral_order_0_00(e1, e2, total_ke)
            print(res)
            return res
        elif order == ph.TaylorOrders.TWO:
            return integral_order_2_00(e1, e2, total_ke)
        elif order == ph.TaylorOrders.TWOTWO:
            return integral_order_22_00(e1, e2, total_ke)
        elif order == ph.TaylorOrders.FOUR:
            return integral_order_4_00(e1, e2, total_ke)
        else:
            raise NotImplementedError
