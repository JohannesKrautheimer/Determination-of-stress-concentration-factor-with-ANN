import numpy as np
import matplotlib.pyplot as plt
import json
import os

from augmentations import add_noise_fn, dropout_points_fn, dropout_cluster_fn

def plot_point_cloud_fn(x, y, interesting_points, plot_interesting_points=True):
    P1 = interesting_points["P1"]
    P2 = interesting_points["P2"]
    Pr = interesting_points["Pr"]
    Pb = interesting_points["Pb"]
    Pr2 = interesting_points["Pr2"]

    all_points_plot = plt.scatter(x, y, c= "Lightblue")
    if plot_interesting_points:
        p1_plot = plt.scatter(P1[0], P1[1], c="red")
        p2_plot = plt.scatter(P2[0], P2[1], c="green")
        pr_plot = plt.scatter(Pr[0], Pr[1], c="black")
        # pb_plot = plt.scatter(Pb[0], Pb[1], c="orange")
        pr2_plot = plt.scatter(Pr2[0], Pr2[1], c="yellow")

        # plt.legend((p1_plot, p2_plot, pr_plot, pb_plot, pr2_plot),
        #         ('P1', 'P2', 'Pr', 'Pb', 'Pr2'))
        plt.legend((p1_plot, p2_plot, pr_plot, pr2_plot),('P1', 'P2', 'Pr', 'Pr2'))

    # plt.axis('equal')
    plt.xlabel("[cm]")
    plt.ylabel("[cm]")
    plt.show()

def create_point_cloud(r, h, alpha, point_distance_mm):
    P1 = np.array([h*np.sin(alpha)/(1-np.cos(alpha)), 0.0])
    P2 = np.array([(h/(1-np.cos(alpha))-r)*np.sin(alpha), r*(1-np.cos(alpha))])
    Pr = np.array([h*np.sin(alpha)/(1-np.cos(alpha)), r])
    Pb = np.array([h*np.sin(alpha)/(1-np.cos(alpha)) - r*np.sin(alpha)/(1+np.cos(alpha)), 0.0])
    r2 = h/(1-np.cos(alpha))-r
    Pr2 = np.array([0, -r2+h])

    interesting_points = {
        "P1": P1, "P2": P2, "Pr": Pr, "Pb": Pb, "r2": r2, "Pr2": Pr2
    }


    #60 mm is orientated by the sensor that we want to buy (SC3500-120)
    #It has a covered area of 120mmx75mm and since it looks only on one half of the weld --> 60 mm
    #https://www.micro-epsilon.de/download/products/dat--surfaceCONTROL-3D--de.pdf
    total_length_mm = 60 ##in mm
    total_length_cm = total_length_mm / 10 ##in cm
    x = np.linspace(0, total_length_cm, int(total_length_mm / point_distance_mm), endpoint=True)
    x_abs = np.abs(x)

    sect_1 = x_abs < P2[0]
    sect_2 = np.stack((x_abs >= P2[0], x_abs < P1[0])).all(axis=0)
    sect_3 = x_abs >= P1[0]

    y_1 = np.sqrt(r2**2 - (x - Pr2[0])**2) + Pr2[1]
    y_2 = -np.sqrt(r**2 - (np.abs(x) - Pr[0])**2) + Pr[1]
    y_3 = np.zeros(x_abs.shape)

    y = np.stack((np.nan_to_num(y_1*sect_1, 0.0), np.nan_to_num(y_2*sect_2, 0.0), np.nan_to_num(y_3*sect_3, 0.0))).max(axis=0)

    return x, y, interesting_points

def save_point_cloud_to_file(filename, x, y, scf):
    print(f"Save point cloud to {filename}")
    with open(filename, 'w') as f:
        f.write(f"kt={scf}\n")
        for x_cord, y_cord in zip(x,y):
            f.write(f"{x_cord} {y_cord} 0.000000\n")

if __name__ == "__main__":
    #General options
    result_files_dir = 'finished_abaqus_files_and_param_fields/param_field_0/result-files/'

    skip_faulty_scfs = True
    plot_point_cloud = True
    plot_interesting_points = True
    save_files = False

    #Augmentations
    add_noise = False
    ##Dropout points
    dropout_points = False
    dropout_points_percentage = 0.4
    ##Dropout cluster
    dropout_cluster = False
    dropout_cluster_width_mm = 1
    #choose from P1, P2, Pr, Pb, Pr2 or choose a number or e.g. P1div2 for half of P1 or other divisions 
    dropout_cluster_point = "P1div2" 

    #Random seed used for the augmentations
    rnd_seed = 123456
    np.random.seed(rnd_seed)

    for point_distance_mm in [0.025]: 
        with open(result_files_dir + 'models_list_results.json', 'r') as f:
            results_dict = json.load(f)
            for key in results_dict:
                data = results_dict[key]['data']
                width_cm = data['width'] / 10
                height_cm = data['height'] / 10
                radius_cm = data["radius"] / 10
                alpha = data["alpha"]
                scf = results_dict[key]['results']['eval_radius']['S_MAX_PRINCIPAL']['value']

                x, y, interesting_points = create_point_cloud(r=radius_cm, h=height_cm, alpha=np.deg2rad(alpha), point_distance_mm=point_distance_mm)
                if add_noise:
                    x, y = add_noise_fn(x, y, point_distance_mm)
                if dropout_points:
                    x, y = dropout_points_fn(x, y, dropout_points_percentage)
                if dropout_cluster:
                    x, y = dropout_cluster_fn(x, y, dropout_cluster_width_mm, dropout_cluster_point, interesting_points)
                if plot_point_cloud:
                    plot_point_cloud_fn(x, y, interesting_points, plot_interesting_points)

                dir_path = "finished_abaqus_files_and_param_fields/param_field_0/point_clouds/" + str(point_distance_mm) + "_mm"
                if add_noise:
                    dir_path += f"_with_noise_seed_{rnd_seed}"
                if dropout_points:
                    dir_path += f"_with_dropout_points_{int(dropout_points_percentage*100)}%_seed_{rnd_seed}"
                if dropout_cluster:
                    dir_path += f"_with_dropout_cluster_{int(dropout_cluster_width_mm)}mm_point_{dropout_cluster_point}"
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                filename = dir_path + "/" + key + ".asc"
                if skip_faulty_scfs and scf < 1:
                    print(f"Skipped {key} because it has SCF: {scf}")
                    continue
                else:
                    if save_files:
                        save_point_cloud_to_file(filename, x, y, scf)