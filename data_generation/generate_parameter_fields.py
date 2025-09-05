import itertools
import numpy as np
import json

def generate_param_field_butt_weld(file_name_save):
    #Parameter ranges
    radius_min = 0.1 #mm
    radius_max = 5 #mm
    # radius_interval = np.linspace(start=radius_min, stop=radius_max, num=50)
    radius_interval = np.logspace(start=np.log10(radius_min), stop=np.log10(radius_max), num=25)

    plate_thickness_interval = [10] #mm

    seam_width_min = 7 #mm
    seam_width_max = 18 #mm
    seam_width_interval = np.linspace(start=seam_width_min, stop=seam_width_max, num=11)

    alpha_min = 5 #degree
    alpha_max = 80 #degree
    alpha_interval = np.linspace(start=alpha_min, stop=alpha_max, num=25)

    all_param_intervals = [radius_interval, plate_thickness_interval, seam_width_interval, alpha_interval]
    all_allowed_param_combinations = []

    # These are use to find the actual min and max values of the params with respect to the limitations
    min_radius = 100000000
    max_radius = 0
    min_width = 1000000
    max_width = 0
    min_alpha = 100000000
    max_alpha = 0
    min_height = 10000000
    max_height = 0
    #Iterate through all possible param combinations and check for limitations
    num_comb = 0
    for r, t, w, alpha in itertools.product(*all_param_intervals):
        num_comb += 1
        alpha_rad = np.radians(alpha)
        seam_height = ( w / (2*np.sin(alpha_rad)) + r*np.tan(alpha_rad/2)/np.sin(alpha_rad) ) * ( 1 - np.cos(alpha_rad) )
        #Limitations
        #DIN EN ISO 5817 Bewertungsgruppe C
        max_seam_height = 1 + 0.15 * w
        if seam_height > max_seam_height:
            print("Maximal allowed seam height of " + str(max_seam_height) + " is exceeded!")
            print("Params:")
            print("r, t, w, alpha, h: ", r, t, w, alpha, seam_height)
            print("########################")
            continue
        #Limitation of radius for small angles. Defined by Matthias such that at least 5 elements fit on the radius
        r_min_allowed = (0.025 * 5) * (180 / np.pi) * 1 / alpha
        if r < r_min_allowed:
            print("Radius is smaller than minimum allowed radius of " + str(r_min_allowed) + "!")
            print("Params:")
            print("r, t, w, alpha, h: ", r, t, w, alpha, seam_height)
            print("########################")
            continue
        else:
            max_radius = np.max([max_radius, r])
            max_width = np.max([max_width, w])
            max_alpha = np.max([max_alpha, alpha])
            max_height = np.max([max_height, seam_height])
            min_radius = np.min([min_radius, r])
            min_width = np.min([min_width, w])
            min_alpha = np.min([min_alpha, alpha])
            min_height = np.min([min_height, seam_height])
            param_combination = {
                "radius": r,
                "thickness": t,
                "width": w,
                "alpha": alpha,
                "height": seam_height
            }
            all_allowed_param_combinations.append(param_combination)
    print(f"In total {num_comb} combinations were checked.")
    print(f"Only {len(all_allowed_param_combinations)} combinations are left after checking the limitations.")

    with open(file_name_save, 'w') as f:
        json.dump(all_allowed_param_combinations, f, indent=4)

    print("Max radius", max_radius)
    print("Max width", max_width)
    print("Max alpha", max_alpha)
    print("Max height", max_height)
    print("##########")
    print("Min radius", min_radius)
    print("Min width", min_width)
    print("Min alpha", min_alpha)
    print("Min height", min_height)

def generate_param_field_fillet_weld(file_name_save):
    #Parameter ranges
    radius_min = 0.1 #mm
    radius_max = 4 #mm
    # radius_interval = np.linspace(start=radius_min, stop=radius_max, num=5)
    radius_interval = np.logspace(start=np.log10(radius_min), stop=np.log10(radius_max), num=25)

    plate_thickness_interval = [10] #mm

    # a-Maß
    # Min & max were calculated using the formula from "Wie berechne ich eine optimale Kehlnaht"
    throat_min = 2.66 #mm
    throat_max = 7 #mm
    throat_interval = np.linspace(start=throat_min, stop=throat_max, num=10)

    alpha_min = 115 #degree
    alpha_max = 155 #degree
    alpha_interval = np.linspace(start=alpha_min, stop=alpha_max, num=25)

    all_param_intervals = [radius_interval, plate_thickness_interval, throat_interval, alpha_interval]
    all_allowed_param_combinations = []

    # These are use to find the actual min and max values of the params with respect to the limitations
    min_radius = 100000000
    max_radius = 0
    min_throat = 1000000
    max_throat = 0
    min_alpha = 100000000
    max_alpha = 0
    #Iterate through all possible param combinations and check for limitations
    num_comb = 0
    for r, t, throat, alpha in itertools.product(*all_param_intervals):
        num_comb += 1
        #Limitations
        #Limitation of radius for small angles. Defined by Matthias such that at least 5 elements fit on the radius
        r_min_allowed = (0.025 * 5) * (180 / np.pi) * 1 / (180 - alpha)
        if r < r_min_allowed:
            print("Radius is smaller than minimum allowed radius of " + str(r_min_allowed) + "!")
            print("Params:")
            print("r, t, throat, alpha: ", r, t, throat, alpha)
            print("########################")
            continue
        else:
            max_radius = np.max([max_radius, r])
            max_throat = np.max([max_throat, throat])
            max_alpha = np.max([max_alpha, alpha])
            min_radius = np.min([min_radius, r])
            min_throat = np.min([min_throat, throat])
            min_alpha = np.min([min_alpha, alpha])
            param_combination = {
                "radius": r,
                "thickness": t,
                "throat": throat,
                "alpha": alpha,
            }
            all_allowed_param_combinations.append(param_combination)
    print(f"In total {num_comb} combinations were checked.")
    print(f"Only {len(all_allowed_param_combinations)} combinations are left after checking the limitations.")

    with open(file_name_save, 'w') as f:
        json.dump(all_allowed_param_combinations, f, indent=4)

    print("Max radius", max_radius)
    print("Max throat", max_throat)
    print("Max alpha", max_alpha)
    print("##########")
    print("Min radius", min_radius)
    print("Min throat", min_throat)
    print("Min alpha", min_alpha)

if __name__ == "__main__":
    # generate_param_field_butt_weld(file_name_save='param_field_butt_weld.json')
    generate_param_field_fillet_weld(file_name_save='param_field_fillet_weld.json')
