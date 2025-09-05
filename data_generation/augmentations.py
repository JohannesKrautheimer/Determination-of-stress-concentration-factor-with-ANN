import numpy as np
import numbers

def add_noise_fn(x, y, point_distance_mm):
    point_distance_micro_meter = point_distance_mm * 1000
    # Linear Interpolation of the std. dev. by using the values from the scanner https://www.micro-epsilon.de/download/products/dat--surfaceCONTROL-3D--de.pdf
    # The values of "Auflösung" & "Wiederholpräzision" are used 
    x1 = 40 #mikrometer
    x2 = 60 #mikrometer
    y1 = 0.4 #mikrometer
    y2 = 0.8 #mikrometer
    std_dev_mikro_meter = y1 + (y2 - y1)/(x2 - x1)*(point_distance_micro_meter - x1)
    # set noise to a minimum value
    if std_dev_mikro_meter < 0.4:
        std_dev_mikro_meter = 0.4
    noise_micro_meter = np.random.normal(loc=0.0, scale=std_dev_mikro_meter, size=y.shape)
    noise_cm = noise_micro_meter / 10000
    return x, y + noise_cm

def dropout_points_fn(x, y, point_drop_percentage):
    percentage_left = 1 - point_drop_percentage
    range_x = np.arange(len(x))
    selected_indices = sorted(np.random.choice(range_x, int(len(x)*percentage_left), replace=False))
    return x[selected_indices], y[selected_indices]

def dropout_cluster_fn(x, y, dropout_cluster_width_mm, dropout_point, interesting_points):
    #if dropout_point is a number take it as the number and not out of the intersting points
    if isinstance(dropout_point, numbers.Number):
        x_dropout_point = dropout_point
    else:
        if "div" in dropout_point:
            split = dropout_point.split("div")
            x_dropout_point = interesting_points[split[0]][0]
            x_dropout_point /= int(split[1])
        else:
            x_dropout_point = interesting_points[dropout_point][0]

    dropout_cluster_width_cm = dropout_cluster_width_mm / 10
    remaining_indices = np.logical_or(x < x_dropout_point - dropout_cluster_width_cm/2, x > x_dropout_point + dropout_cluster_width_cm/2)
    return x[remaining_indices], y[remaining_indices]