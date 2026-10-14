def runge_kutta(y_n, v_n, u_1, u_2, h, f, model):
    k_1_y = v_n
    k_1_v = f(y_n, v_n, u_1, model)
    v_2 = v_n + k_1_v*h/2
    y_2 = y_n + k_1_y*h/2
    k_2_y = v_2
    k_2_v = f(y_2, v_2, (u_1+u_2)/2, model)
    v_3 = v_n + k_2_v*h/2
    y_3 = y_n + k_2_y*h/2
    k_3_y = v_3
    k_3_v = f(y_3, v_3, (u_1+u_2)/2, model)
    v_4 = v_n + k_3_v*h
    y_4 = y_n + k_3_y*h
    k_4_y = v_4
    k_4_v = f(y_4, v_4, u_2, model)

    v_n = v_n + h/6 * (k_1_v + 2*k_2_v + 2*k_3_v + k_4_v)
    y_n = y_n + h/6 * (k_1_y + 2*k_2_y + 2*k_3_y + k_4_y)

    return [y_n, v_n]



