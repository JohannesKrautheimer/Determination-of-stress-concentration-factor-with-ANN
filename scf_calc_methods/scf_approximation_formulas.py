""" Collection of approximation formulas for SCF of welds """
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time

def anthes1993_butt_joint_bending(r, t, phi):
    """Formula by Anthes for SCF on buttwelds under bending load

    Parameters
    ----------
    r : float
        weld toe radius in mm
    t : float
        sheet thickness in mm
    phi : float
        weld toe angle in radian

    Returns
    -------
    float
        SCF of weld toe
    boolean
        restrictions passed
    """

    #Restrictions for Anthes' method
    restrictions_passed = True
    if not 0 <= t/r <= 200:
        print(f"Restriction is 0 <= t/r <= 200. Value is: {t/r}")
        restrictions_passed = False
    phi_degree = phi * 180 / np.pi
    if not 0 <= phi_degree <= 90:
        print(f"Restriction is 0° <= phi <= 90°. Value is: {phi_degree}")
        restrictions_passed = False

    a0, a1, a2, a3 = 0.181, 1.207, -1.737, 0.689
    b1, b2 = -0.156, 0.207
    l1, l2, l3 = 0.2919, 0.3491, 3.283

    scf = (1 + b1 * (t/r)**b2) * (
            1 + (a0+a1 * np.sin(phi) + a2 * np.sin(phi)**2 + a3 * np.sin(phi)**3
            ) * (t/r)**(l1 + l2 * np.sin(phi + np.deg2rad(l3))))

    return scf, restrictions_passed

def anthes1993_cross_joint_bending(r, t, phi, a, s):
    """Formula by Anthes for SCF on cross joints under bending load

    Parameters
    ----------
    r : float
        weld toe radius in mm
    t : float
        sheet thickness in mm
    phi : float
        weld toe angle in radian
    a: float
        seam thickness in mm
    s: float
        root gap width (Wurzelspaltbreite) in mm

    Returns
    -------
    float
        SCF of weld toe
    boolean
        restrictions passed
    """

    #Restrictions for Anthes' method
    restrictions_passed = True
    if not 0 <= t/r <= 200:
        print(f"Restriction is 0 <= t/r <= 200. Value is: {t/r}")
        restrictions_passed = False
    phi_radian = phi * 180 / np.pi
    if not 30 <= phi_radian <= 60:
        print(f"Restriction is 30° <= phi <= 60°. Value is: {phi}")
        restrictions_passed = False
    if not 15 <= t/r <= 100:
        print(f"Restriction is 15 <= t/r <= 100. Value is: {t/r}")
        restrictions_passed = False
    if not 0.3 <= a/t <= 1:
        print(f"Restriction is 0.3 <= a/t <= 1. Value is: {a/t}")
        restrictions_passed = False
    if not (2*r/t <= s/t <= 1 or s/t == 0):
        print(f"Restriction is 2*r/t <= s/t <= 1 or s/t == 0: 2*r/t={2*r/t} || s/t={s/t}")
        restrictions_passed = False

    m0, m1, m2, m3 = 1.256, 0.023, 2.153, -3.738
    p1, p2, p3, p4, p5, p6  = -3.090, 2.412, 0.154, 0.481, 1.723, 0.172
    scf = m0 + (1 + m1*(a/t)**p1*(s/t)**p2 + m2*(t/r)**p3 + m3*(np.sin(phi))**p4)*(np.sin(phi))**p5*(t/r)**p6

    return scf, restrictions_passed

def rainer_butt_joint_bending(t, r, u, alpha):
    """Formula by Rainer for SCF on butt joints under bending load

    Parameters
    ----------
    t : float
        sheet thickness in mm
    r : float
        weld toe radius in mm
    u : float
        weld reinforcement (Nahthöhe) in mm
    alpha: float
        weld toe angle in radian

    Returns
    -------
    float
        SCF of weld toe
    boolean
        restrictions passed
    """
    #Restrictions of Rainer's method
    restrictions_passed = True
    if not 0 <= (t/r) <= 400:
        print(f"Restriction is 0 <= (t1/r) <= 400. Value is: {t/r}")
        restrictions_passed = False
    if not alpha == (45 * np.pi / 180):
        print(f"Restriction is alpha == 45°). Value (radian) is: {alpha}")
        restrictions_passed = False
    if not 0 <= u/t <= 2.5:
        print(f"Restriction is 0 <= u/t <= 2.5). Value is: {u/t}")
        restrictions_passed = False

    term1 = 0.4 / ( (u/r)**0.66 )
    term2 = 3.8 * ( (1 + t/(2*r)) / (t / (2*r) * (t/(2*r))**0.5) )**2.25
    term3 = 0.2 * (t/(2*r)) / ((t/(2*r) + u/r)*(u/r)**1.33)
    scf = 1 + (term1 + term2 + term3)**(-0.5)
    return scf, restrictions_passed

def rainer_cross_joint_bending(y, t1, r, alpha):
    """Formula by Rainer for SCF on cross joints under bending load

    Parameters
    ----------
    y: float
        ratio of leg length to sheet thickness
    t1 : float
        sheet thickness in mm
    r : float
        weld toe radius in mm
    alpha: float
        weld toe angle in radian

    Returns
    -------
    float
        SCF of weld toe
    boolean
        restrictions passed
    """
    #Restrictions of Rainer's method
    restrictions_passed = True
    if not 0 <= (t1/r) <= 400:
        print(f"Restriction is 0 <= (t1/r) <= 400. Value is: {t1/r}")
        restrictions_passed = False
    if not alpha == (45 * np.pi / 180):
        print(f"Restriction is alpha == 45 degree). Value (radian) is: {alpha}")
        restrictions_passed = False
    if not 0.1 <= y * np.sin(alpha) <= 0.9:
        print(f"Restriction is 0.1 <= y * sin(alpha) <= 0.9). Value is: {y*np.sin(alpha)}")
        restrictions_passed = False

    term1 = 0.4 / ((y*t1*np.sin(alpha)/(2**0.5 * r))**0.66)
    term2 = 3.8 * (((2*r/t1)**3 + (2*r/t1))**0.5)**2.25
    term3 = 0.2 * t1 / (2*r*(t1/(2*r)+y*t1*np.sin(alpha)/(2**0.5*r)*(y*t1*np.sin(alpha)/(2**0.5*r))**1.33))
    scf = 1 + (term1 + term2 + term3)**(-0.5)
    return scf, restrictions_passed

def kiyak_butt_and_cross_joint_bending(alpha, h, T, rho):
    """Formula by Kiyak for SCF on butt and cross joints under bending load

    Parameters
    ----------
    alpha: float
        weld toe angle in radian
    h: float
        seam height in mm for butt joints. according to kiyak paper h is leg length for cruciform joints
    T : float
        sheet thickness in mm
    rho : float
        weld toe radius in mm

    Returns
    -------
    float
        SCF of weld toe
    boolean
        restrictions passed
    """
    restrictions_passed = True
    alpha_degree = alpha * 180 / np.pi
    if not 10 <= alpha_degree <= 60:
        print(f"Restriction is 10° <= alpha <= 60°. Value is: {alpha_degree}")
        restrictions_passed = False
    if not 0.01 <= rho/T <= 0.4:
        print(f"Restriction is 0.01 <= rho/T <= 0.4. Value is: {rho/T}")
        restrictions_passed = False

    p1 = 1.1399
    p2 = 0.2062
    p3 = 1.0670
    p4 = 1.6775
    p5 = 0.4711
    term1 = p1 * (h/T)**(p2*alpha)
    term2 = alpha**p3*np.exp(-p4*alpha)*(rho/T)**(-0.295*alpha)
    term3 = (0.021 + rho/T)**(-p5)
    scf = 1 + term1*term2*term3
    return scf, restrictions_passed

def prepare_internet_driver():
    url = "http://rother.userweb.mwn.de/Page6/Page6.html"
    driver_path = "C:/Projekte/spannungskonzentrationen-mit-ki/src/resources/edgedriver_win64/msedgedriver.exe"
    driver = webdriver.Edge(executable_path=driver_path, service_log_path='NUL')
    driver.get(url)

    return driver

def get_oswald_scf_from_website(driver, r, t, f, beta):

    radius_selection_str = str(r)
    # The :g removes all trailing unneccesarry zeroes. This is needed because the website cant deal with trailing zeroes
    sheet_thickness_str = f'{t:g}'
    f_str = f'{f:g}'
    beta_str = f'{beta:g}'

    input1_element = driver.find_element(By.ID, "input1")
    input2_element = driver.find_element(By.ID, "input2")
    input4_element = driver.find_element(By.ID, "input4")
    input6_element = driver.find_element(By.ID, "input6")
    calculate_button = driver.find_element(By.XPATH, '//button[contains(text(), "CALCULATE SCFs")]')
    output1_element = driver.find_element(By.ID, "output1")

    # Fill in the input elements
    input1_element.clear()
    input2_element.clear()
    input4_element.clear()
    input1_element.send_keys(sheet_thickness_str)
    input2_element.send_keys(f_str)
    input4_element.send_keys(beta_str)

    # Select the dropdown option
    select = Select(input6_element)
    select.select_by_visible_text(radius_selection_str)

    # Click the button
    calculate_button.click()

    # Wait for the result to appear
    time.sleep(2)

    scf_bending = output1_element.text
    if not scf_bending == "":
        scf_bending = float(scf_bending)

    return scf_bending

def oswald_dy_butt_joint_bending(driver, r, t, w, beta):
    """Calculation of SCF on buttwelds under bending load after Oswald's method

    Parameters
    ----------
    r : string
        weld toe radius in mm (possible values: '0.05', '0.30', '1.00')
    t : float
        sheet thickness in mm
    w : float
        weld width in mm
    beta: float 
        angle in degree

    Returns
    -------
    float
        SCF of weld toe
    boolean
        restrictions passed
    """

    possible_r_values = ['0.05', '0.30', '1.00']
    if not r in possible_r_values:
        raise ValueError(f'The selected radius is {r} but only these values are allowed {possible_r_values}')

    f = w / t

    #Oswald restrictions
    restrictions_passed = True
    if r == 0.05:
        if not 0.5 <= t <= 7.5:
            print(f"Restriction for radius 0.05 mm is: 0.5 mm <= t <= 7.5 mm . Value is: {t}")
            restrictions_passed = False
    if r == 0.30:
        if not 3.6 <= t <= 25:
            print(f"Restriction for radius 0.30 mm is: 3.6 mm <= t <= 25 mm . Value is: {t}")
            restrictions_passed = False
    if r == 1.00:
        if not 10 <= t <= 100:
            print(f"Restriction for radius 1 mm is: 10 mm <= t <= 100 mm . Value is: {t}")
            restrictions_passed = False
    if not 0.15 <= f <= 15:
            print(f"Restriction is 0.15 <= w/t <= 15. Value is: {f}")
            restrictions_passed = False
    if not 5 <= beta <= 80:
            print(f"Restriction is 5° <= beta <= 80°. Value is: {beta}°")
            restrictions_passed = False

    scf = get_oswald_scf_from_website(driver, r, t, f, beta)

    return scf, restrictions_passed