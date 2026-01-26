import subprocess
import shlex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scattnlay import scattnlay, fieldnlay
import re
from pathlib import Path
import json

global exe

exe = "/home/david/miniconda3/bin/optool"

OPTOOL_SETTINGS = {
    "lmin": 0.1,     # micron
    "lmax": 1000,    # micron
    "nl": 100,
    "method": 'mie',
    "NANG": 180 #fix this to 180
}

# TODO: right now, the shell radii are allowed to be smaller than a0 
#       -> this causes the monomer density to be exceeded. 
#       Change this (maybe in the way you define Rn so we keep the other properties)!
#       Quick fix: i set it to 2*a0

# TODO: Optool is ran twice when working with angles and matrices (ones for the kappas and then again for the matrices)
#       This is because i first made a different file reading function. for performace we could make it so it only runs once!

# TODO: add a renormalization for the matrix elements and test it

class Particle:
    """_summary_

        Args:
            D (float): fractal dimension, unitless
            kf (float): fractal prefactor, unitless
            a0 (float): radius of a single monomer, (micron)
            rho_mono (float): denity of a single monomer, (gr cm^-3)
            Rc (float): characteristic radius of the particle, (micron)
            N (int): number of shells the particle is subdivided in, Defaults to 1
            material (str): Optool interpretable string containing the material of the particle, Defaults to "pyr-mg70 0.70 c 0.30" 
    """
  
  
    def __init__(self, D, kf, a0, Rc, N = 1, material = "pyr-mg70 0.70 c 0.30", matrix = False, from_file = None):
        """
        Initilializes particle
        """ 
        self.D  = float(D)
        self.kf = float(kf)
        self.a0 = float(a0)
        self.material = material
        self.rho_mono = self.get_monomer_density()
        self.Rc = float(Rc)
        self.Rg = self.Rc / (np.sqrt(5/3))
        self.C  = self.get_C(kf, D, a0, self.rho_mono) #now in micron
        self.mass = self.get_mass(D, Rc, self.C)
        self.volume = 4/3 * np.pi* Rc**3 #in micron^3
        self.rho_bar = self.mass/(self.volume * 1e-12) #to compare to optools gr/cm
        self.f  = self.rho_bar / self.rho_mono
        self.porosity = 1 - self.f
        self.N  = N
        if self.N == 1:
            self.Rn = self.Rc
        else:
            self.Rn = np.linspace(2*a0, Rc, N) # radii of the shells 
        self.Rho_n = self.rho_shell()
        
        #set if we should also get the matrix elements per angle
        self.returnmatrix = matrix
        self.Optool_results = self.get_optool_kappas()
        #calculate the results
        if not self.returnmatrix:
            self.Scatt_results = self.get_scattnlay_kappas()
        else:
            self.Scatt_results = self.get_scattnlay_kappas(self.Optool_results[2])

        
        
    def get_C(self, kf, D, a0, rho_mono):
        """Gets the integration constant C used in the mass calculation of the dust aggregate (M_encl = 4 pi C Rc^D / D)
        Args:
            kf (float): fractal prefactor, unitless
            D (float): fractal dimension, unitless
            a0 (float): radius of a single monomer (micron)
            rho_mono (float): density of a single monomer (gr cm^-3)
        returns:
            C (gr cm^-D)
        """
        
        C  = rho_mono * 1e-12 #convert to gram/micron^3
        C *= a0**(3-D) #a0 in micron
        C *= kf
        C *= D/3
        C *= np.sqrt(3/5)
        C *= np.sqrt(5/3)**(3-D)
        
        return C
    
    def get_mass(self, D, Rc, C):
        """calc particle mass from D, Rc and C

        Args:
            D (float): fractal dimension, unitless
            Rc (float): particle (characteristic) radius, (cm)
            C (float): Integration constant from get_C(), (gr cm^-D)

        Returns:
            float: particle mass, (gr)
        """
        M_encl = 4*np.pi
        M_encl *= (Rc)**(D)
        M_encl *= 1/D
        M_encl *= C
        return M_encl
    
    def mass_enclosed(self, R):
        """calculates M(<R). The mass enclosed by a shell with radius R.
        
        Args:
            R (float): Radius of the shell, (cm)
            
        Returns:
            float: shell mass, (gr)
        """
        
        return 4*np.pi * self.C * (R**self.D) / self.D

    def rho_bar_shell(self, R):
        """
        
        """
        
        M = self.mass_enclosed(R)
        V = (4/3) * np.pi * (R**3)
        return M / V
    
    
        
    def describe(self, verbose=True, kappas = False):
        """Prints or returns a summary of the particle's physical properties."""
        if not kappas:
            description = (
                f"Fractal particle properties:\n"
                f"-----------------------------\n"
                f"Fractal dimension (D):      {self.D:.3f}\n"
                f"Fractal prefactor (kf):     {self.kf:.3f}\n"
                f"Monomer radius (a0):        {self.a0:.3e} micron\n"
                f"Monomer density (ρ_mono):   {self.rho_mono:.3e} g/cm³\n"
                f"Characteristic radius (Rc): {self.Rc:.3e} micron\n"
                f"Radius of gyration (Rg):    {self.Rg:.3e} micron\n"
                f"Integration constant (C):   {self.C:.3e} g·micron⁻{self.D:.2f}\n"
                f"Mass:                       {self.mass:.3e} g\n"
                f"Volume:                     {self.volume:.3e} micron³\n"
                f"Mean density (ρ̄):          {self.rho_bar:.3e} g/cm³\n"
            )
        else: 
                description = (
                f"-----------------------------\n"
                f"Characteristic radius (Rc): {self.Rc:.3e} micron\n"
                f"wavelenght: {self.Scatt_results[0]['wavelength_um'].values}\n"
                f"kabs: {self.Scatt_results[0]['kabs_cm2g'].values}\n"
                f"ksca: {self.Scatt_results[0]['ksca_cm2g'].values}\n"
                f"g   : {self.Scatt_results[0]['g'].values}\n"
                
            )
            
        if verbose:
            print(description)
        else:
            return description
        
    
    def get_optool_kappas(self):
        """Runs optool for a single shell given the particle properties (material)
            and the optool_setting lmin, lmax and nl.
            This function assumes the existence of a dictionary called OPTOOL_SETTING
            with these parameters and a boolean called "method"
        
        """
        if not self.returnmatrix:
            lmin = OPTOOL_SETTINGS["lmin"]
            lmax = OPTOOL_SETTINGS["lmax"]
            nl   = OPTOOL_SETTINGS["nl"]
            method  = OPTOOL_SETTINGS["method"]
            if method == 'mie':
                cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {self.Rc} -p {self.porosity} -mie" 
                )
            if method == 'mmf': #TODO: fix mmf
                if self.D < 1:
                    cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {self.Rc} -mmf {self.a0} {self.f} {self.kf}" 
                )
                else:    
                    cmd = (
                        f"optool {self.material} "
                        f"-lmin {lmin} -lmax {lmax} "
                        f"-nl {nl} -a {self.Rc} -mmf {self.a0} {self.D} {self.kf}" 
                    )
                    
            # else: 
            #     cmd = (
            #         f"optool {self.material} "
            #         f"-lmin {lmin} -lmax {lmax} "
            #         f"-nl {nl} -a {self.Rc} -p {self.porosity}" 
            #     )
            run_optool(cmd)
            self.cmd = cmd
            return read_dustkappa()
        else:
            lmin = OPTOOL_SETTINGS["lmin"]
            lmax = OPTOOL_SETTINGS["lmax"]
            nl   = OPTOOL_SETTINGS["nl"]
            method  = OPTOOL_SETTINGS["method"]
            nang = OPTOOL_SETTINGS["NANG"]
            
            if method == 'mie':
                cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {self.Rc} -p {self.porosity} -mie -s {nang}"
                )
                
            if method == 'mmf': 
                if self.D < 1:
                    cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {self.Rc} -mmf {self.a0} {self.f} {self.kf} -s {nang}" 
                )
                else:    
                    cmd = (
                        f"optool {self.material} "
                        f"-lmin {lmin} -lmax {lmax} "
                        f"-nl {nl} -a {self.Rc} -mmf {self.a0} {self.D} {self.kf} -s {nang}" 
                    )
            # else: 
            #     cmd = (
            #         f"optool {self.material} "
            #         f"-lmin {lmin} -lmax {lmax} "
            #         f"-nl {nl} -a {self.Rc} -p {self.porosity} -s {nang}"
            #     )
            run_optool(cmd)
            self.cmd = cmd
            return read_optool_file()
        
    def get_monomer_density(self):
        """gets the density of a single monomer given the material specified using optool.

        Returns:
            rho(float) : density of the monomer (g/cm^3)
        """
        cmd = (
                f"optool {self.material} -print lnk"
            )
        rho, lam, n_l, k_l = run_optool_lnk(cmd)
        return rho
            
        
    def get_refractive_index(self, porosity = None):
        """gets the refractive index using optool given the particle properties (material)
            and the optool_setting lmin, lmax and nl.
            This function assumes the existence of a dictionary called OPTOOL_SETTING
            with these parameters and a boolean called "method"

        Returns:
            df : wavelength, refractive index 
        """
        if porosity == None:
            porosity = self.porosity
        
        lmin = OPTOOL_SETTINGS["lmin"]
        lmax = OPTOOL_SETTINGS["lmax"]
        nl   = OPTOOL_SETTINGS["nl"]
        method  = OPTOOL_SETTINGS["method"]
        
        if method == 'mie':
            cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {self.Rc} -p {porosity} -mie -print lnk" 
            )
        else: 
            cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {self.Rc} -p {porosity} -print lnk" 
            )
            
        rho, lam, n_l, k_l = run_optool_lnk(cmd)
        df = pd.DataFrame({'wavelength' : lam,
                      'index': [complex(i,j) for i,j in zip(n_l, k_l)]}
                     )
        return df
    
    def get_dV(self):
        """calculates the volume element dV from two consecutive shells
        """
        S = (4.0/3.0) * np.pi
        if self.N == 1:
            return S * (self.Rc**3)
        else: 
            borders = self.Rn
            dV = S * (borders[1:]**3 - borders[:-1]**3)
            Vcore = S * (borders[0]**3)
            return np.concatenate(([Vcore], dV))
    
    def get_dM(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.N == 1:
            return self.mass
        else: 
            borders = self.Rn
            dM = self.mass_enclosed(borders[1:]) - self.mass_enclosed(borders[:-1])
            Mcore = self.mass_enclosed(borders[0])
            return np.concatenate(([Mcore], dM))
        
        
    def rho_shell(self):
        """
        differs from rho_bar_shell because this uses a differential method instead of using M_encl by the entire shell
        """
        dM = self.get_dM()
        dV = self.get_dV()
        return dM/ (dV*1e-12)
        
        
    def build_run_df(self):
        """
        Create one long DataFrame with wavelength, index, porosity, radius_cm, and size parameter x.
        it is used to run scattnlay
        """
        if self.N != 1:
            # shell densities and porosities
            rho_shell = self.rho_shell()
            porosities = 1 - rho_shell / self.rho_mono
                # check for non-physical porosities (≤ 0)
            if np.any(porosities < 0.0):
                bad_idx = np.where(porosities < 0.0)[0]
                print(
                    f"⚠️  Warning: some porosity values are ≤ 0!\n"
                    f"   Problematic indices: {bad_idx}\n"
                    f"   Porosity values: {porosities[bad_idx]}\n"
                    f"   → These values have been replaced with 0."
                )
                porosities[bad_idx] = 0.0

            frames = []
            for R_i, p in zip(np.asarray(self.Rn, float), np.asarray(porosities, float)):
                pdf = self.get_refractive_index(porosity=float(p))   # columns wavelength, index
                l = pdf["wavelength"].to_numpy(dtype=float)
                x = x_size_param(R_i, l)

                frames.append(pd.DataFrame({
                    "wavelength": l,
                    "index": pdf["index"].to_numpy(),
                    "porosity": float(p),
                    "radius": float(R_i),
                    "x": x
                }))

            run_df = pd.concat(frames, ignore_index=True)
            return run_df
        else:
            # shell densities and porosities
            rho_shell = self.rho_bar
            porosities = 1 - rho_shell / self.rho_mono
                # check for non-physical porosities (≤ 0)
            if np.any(porosities < 0.0):
                bad_idx = np.where(porosities < 0.0)[0]
                print(
                    f"⚠️  Warning: some porosity values are ≤ 0!\n"
                    f"   Problematic indices: {bad_idx}\n"
                    f"   Porosity values: {porosities[bad_idx]}\n"
                    f"   → These values have been replaced with 0."
                )
                porosities[bad_idx] = 0.0
            
            frames = []
            pdf = self.get_refractive_index(porosity=float(porosities))   # columns wavelength, index
            l = pdf["wavelength"].to_numpy(dtype=float)
            x = x_size_param(self.Rc, l)

            frames.append(pd.DataFrame({
                "wavelength": l,
                "index": pdf["index"].to_numpy(),
                "porosity": float(porosities),
                "radius": self.Rc,
                "x": x
            }))

            run_df = pd.concat(frames, ignore_index=True)
            return run_df
    
    def get_scattnlay_kappas(self,angles=None, printing = True):
        """runs scattnlay to get the opacities
        #TODO: FIX see todos at top
        """
        if angles is None:
            run_df = self.build_run_df()
            records = []
            for λ in run_df["wavelength"].unique():
                df_at_lambda = run_df[run_df["wavelength"] == λ]

                # Inputs for scattnlay
                x_vals = df_at_lambda["x"].to_numpy()
                m_vals = df_at_lambda["index"].to_numpy()

                terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(x_vals, m_vals)
                
                records.append({
                    "wavelength_um": λ,
                    "Qabs": Qabs,
                    "Qsca": Qsca,
                    "g": g,
                    "S1 S2": [S1, S2]
                })
            df_scattnlay = pd.DataFrame.from_records(records)
        
            df_scattnlay['kabs_cm2g'] = kappa_from_Q(df_scattnlay['Qabs'].values, self.Rc, self.rho_bar)
            df_scattnlay['ksca_cm2g'] = kappa_from_Q(df_scattnlay['Qsca'].values, self.Rc, self.rho_bar)
            df_scattnlay.drop(['Qabs', 'Qsca'], axis=1, inplace = True)
            return df_scattnlay, None, None
        else:
            run_df = self.build_run_df()
            
            records  = [] #one for opacities
            records2 = [] #one for scatt matrix
            for λ in run_df["wavelength"].unique():
                df_at_lambda = run_df[run_df["wavelength"] == λ]

                # Inputs for scattnlay
                x_vals = df_at_lambda["x"].to_numpy()
                m_vals = df_at_lambda["index"].to_numpy()
                terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(x_vals, m_vals, theta = [i * np.pi / 180 for i in angles]) #expected in radians
                records.append({
                    "wavelength_um": λ,
                    "Qabs": Qabs,
                    "Qsca": Qsca,
                    "g": g,
                })
                # Get elements and normalize!
                
                F11, F12, F22, F33, F34, F44 = get_scattering_matrix(S1, S2)
                factor = HoveniersFactor([i * np.pi / 180 for i in angles], F11)
                F11 *= factor
                F12 *= factor
                F22 *= factor
                F33 *= factor
                F34 *= factor
                F44 *= factor


                # ---- flatten by angle for dataframe compatibility ----
                records2.extend([
                    {
                        "wavelength_um": λ,
                        "angle_deg": theta,
                        "F11": f11,
                        "F12": f12,
                        "F22": f22,
                        "F33": f33,
                        "F34": f34,
                        "F44": f44,
                    }
                    for theta, f11, f12, f22, f33, f34, f44 
                    in zip(angles, F11, F12, F22, F33, F34, F44)
                ])
            df_scattnlay = pd.DataFrame.from_records(records)
            df_matrix_elements = pd.DataFrame.from_records(records2)
            for col in df_matrix_elements.columns:
                try:
                    df_matrix_elements[col] = df_matrix_elements[col].apply(lambda z: z.real) # all are real due to identities
                except Exception: # or they were not complex to start
                    pass

            
            df_scattnlay['kabs_cm2g'] = kappa_from_Q(df_scattnlay['Qabs'].values, self.Rc, self.rho_bar)
            df_scattnlay['ksca_cm2g'] = kappa_from_Q(df_scattnlay['Qsca'].values, self.Rc, self.rho_bar)
            df_scattnlay.drop(['Qabs', 'Qsca'], axis=1, inplace = True)
            return df_scattnlay, df_matrix_elements, angles   

    @classmethod
    def from_folder(cls, folder: str | Path) -> "Particle":
        """
        Create a Particle instance from a saved folder without running __init__.
        Expects the files written by save_particle:
            metadata.json
            optool_kappas.csv
            optool_matrix.csv optional
            optool_angles.csv optional
            scattnlay_kappas.csv
            scattnlay_matrix.csv optional
            scattnlay_angles.csv optional
        """
        folder = Path(folder)

        meta_path = folder / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {folder}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        def read_csv(name: str) -> pd.DataFrame | None:
            p = folder / name
            return pd.read_csv(p) if p.exists() else None

        def read_angles(name: str) -> list[float] | None:
            df = read_csv(name)
            if df is None:
                return None
            if "angle_deg" not in df.columns:
                return None
            return df["angle_deg"].astype(float).to_list()

        # key line: allocate object without calling __init__
        self = cls.__new__(cls)

        # fill core parameters from metadata
        self.D = float(meta["D"])
        self.kf = float(meta["kf"])
        self.a0 = float(meta["a0_micron"])
        self.Rc = float(meta["Rc_micron"])
        self.N = int(meta["N_shells"])
        self.material = meta["material"]
        self.returnmatrix = bool(meta.get("returnmatrix", False))

        # derived values saved in metadata
        self.rho_mono = float(meta.get("rho_mono_gcm3", np.nan))
        self.rho_bar = float(meta.get("rho_bar_gcm3", np.nan))
        self.porosity = float(meta.get("porosity", np.nan))
        self.f = float(meta.get("f_fill", np.nan))
        self.Rg = float(meta.get("Rg_micron", self.Rc / np.sqrt(5/3)))
        self.C = float(meta.get("C", np.nan))
        self.mass = float(meta.get("mass_g", np.nan))
        self.cmd = meta.get("cmd", None)

        # geometry fields
        self.volume = 4/3 * np.pi * self.Rc**3
        if self.N == 1:
            self.Rn = self.Rc
        else:
            self.Rn = np.linspace(2 * self.a0, self.Rc, self.N)

        # do not compute rho_shell here because that calls your physics
        # set to None or reconstruct only if you also saved it
        self.Rho_n = None

        # load results
        opt_kappas = read_csv("optool_kappas.csv")
        opt_matrix = read_csv("optool_matrix.csv")
        opt_angles = read_angles("optool_angles.csv")

        sc_kappas = read_csv("scattnlay_kappas.csv")
        sc_matrix = read_csv("scattnlay_matrix.csv")
        sc_angles = read_angles("scattnlay_angles.csv")

        self.Optool_results = (opt_kappas, opt_matrix, opt_angles)
        self.Scatt_results = (sc_kappas, sc_matrix, sc_angles)

        return self

        


def read_optool_file(filename = "dustkapscatmat.dat"):
    """
    Reads the entire optool file
    Please use this only for reading duskkapmat files, it expects a certain order, for reading in dustkappa files see read_dustkappa
    This code has been made using ChatGPT 5 on 10/11/2025.
    Besides the prompt, i gave it an example file and my read_dustkappa function and a frame for the read_optool_file function
    the prompt: 
    
        I think we can combine the two functions into one. 
        so the header says that the data first shows the format number, the number of wavelengths used and the number of angles. 
        The it shows the opacities and asymmetry parameters Then it shows the angles used and then, shows the matrix elements per wavelength and per angle.
        The last one is tricky as it loops over both the wavelengths and the angles. it does this in the following way:
        it starts at the first wavelength and then show N amount of rows with data where N is the amount of angles used. 
        Then it goes to the next wavelength. 
        I would like to store all this data in a df!
    
    I tested the code thrice and it seemed to work, except for using dustkappa files -> only use this for matrix files.
    also, i was not able to check the RaiseErrors that the function uses since i was not able to get those errors to trigger. 

    Args:
        filename (str): Name of the file

    Returns:
        df        : df containing the scattering matrix elements
        lam_df    : df containing the opacities and asymmetry param
        angle_data: df containging angles used
        
    """
    
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # --- 1️⃣ Find header with format, nlam, nang ---
    num_re = re.compile(r"^[\+\-0-9Ee\.]+$")
    header_vals = []
    for line in lines:
        if num_re.match(line):
            header_vals.append(float(line))
        if len(header_vals) == 3:
            break
    if len(header_vals) != 3:
        raise ValueError("Cannot find format, nlam, nang in the file header.")
    fmt, nlam, nang = map(int, header_vals)

    # --- 2️⃣ Find all wavelength-dependent data ---
    # next nlam lines each have 4 numbers (λ, kabs, ksca, g)
    lam_data_start = lines.index(str(int(nang))) + 1
    lam_data = []
    i = lam_data_start
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 4:
            lam_data.append([float(p) for p in parts])
            i += 1
            if len(lam_data) == nlam:
                break
        else:
            i += 1

    if len(lam_data) != nlam:
        raise ValueError(f"Expected {nlam} wavelength lines, found {len(lam_data)}.")

    lam_df = pd.DataFrame(lam_data, columns=["wavelength_um", "kabs_cm2g", "ksca_cm2g", "g"])

    # --- 3️⃣ Read angles ---
    angle_data = []
    while i < len(lines) and len(lines[i].split()) == 1:
        angle_data.append(float(lines[i]))
        i += 1
    if len(angle_data) != nang:
        raise ValueError(f"Expected {nang} angles, found {len(angle_data)}.")

    # --- 4️⃣ Read matrix data ---
    # nlam × nang blocks, each with 6 columns (F11–F44)
    matrix_data = []
    lam_indices = []
    angle_indices = []

    for ilam in range(nlam):
        for iang in range(nang):
            if i >= len(lines):
                raise ValueError("Unexpected end of file while reading matrix data.")
            parts = lines[i].split()
            if len(parts) != 6:
                raise ValueError(f"Expected 6 values for matrix line, got {len(parts)} on line {i}.")
            vals = [float(v) for v in parts]
            matrix_data.append(vals)
            lam_indices.append(lam_df.loc[ilam, "wavelength_um"])
            angle_indices.append(angle_data[iang])
            i += 1

    df = pd.DataFrame(matrix_data, columns=["F11", "F12", "F22", "F33", "F34", "F44"])
    df.insert(0, "angle_deg", angle_indices)
    df.insert(0, "wavelength_um", lam_indices)

    # --- 5️⃣ Optional metadata ---
    df.attrs.update({
        "format": fmt,
        "nlam": nlam,
        "nang": nang,
    })

    return lam_df, df, angle_data


def run_optool_lnk(cmd_str):
    """
    Run an optool CLI command that prints 'lnk' and return (lam, n, k) as floats or lists.
    Expects the CLI to output three whitespace-separated columns: lambda, n, k.
    """
    # Run the command in the shell safely
    # If you know the path to optool, include it in cmd_str.
    # e.g. "/home/david/miniconda3/bin/optool pyr-mg70 0.70 c 0.30 -l 200 -p 0.50 -print lnk"
    result = subprocess.run(
        shlex.split(cmd_str),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"optool failed (exit {result.returncode}).\n"
            f"STDERR:\n{result.stderr}"
        )
    rho = float(result.stdout.splitlines()[0].strip().split()[1]) #this extracts rho from the output -> based on the same logic as extracting n l k

    lam, n, k = [], [], []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            # Skip non-data lines gracefully
            continue
        try:
            L, N, K = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            # Skip lines that aren't pure numeric rows
            continue
        lam.append(L); n.append(N); k.append(K)

    # If it was a single wavelength, return scalars
    if len(lam) == 1:
        return rho, lam[0], n[0], k[0]
    return rho, lam, n, k

def run_optool(cmd_str):
    
    result = subprocess.run(
        shlex.split(cmd_str),
        capture_output=True,
        text=True,
        check=False)


def sweep_lnk(exe, materials, fracs, porosity, lambdas_nm):
    """
    Loop over wavelengths and collect lnk. Returns (lams, ns, ks) as lists.
    """
    assert len(materials) == len(fracs), "materials and fracs must match in length"
    mat_str = " ".join(sum(([m, f"{x:.6g}"] for m, x in zip(materials, fracs)), []))
    lams, ns, ks = [], [], []
    for L in lambdas_nm:
        cmd = f'{exe} {mat_str} -l {L:.6g} -p {porosity:.6g} -print lnk'
        Lout, N, K = run_optool_lnk(cmd)
        lams.append(Lout); ns.append(N); ks.append(K)
    return lams, ns, ks

def x_size_param(r, wavelength, n_m = 1):
    """Returns the size parameter of a given particle 

    Args:
        r (float): radius of the particle in the same units as wavelength
        wavelength (float): wavelength of the particle in  the same units as radius
        n_m (_type_): refractive index of the medium outside the particle 

    Returns:
        float: size parameter x (unitless)
    """
    return r*np.pi*2*n_m/wavelength

def kappa_from_Q(Q, r, rho):
    """Converts Q into kappa, for a spherical particle
    
    TODO: update the way rho is handled in this calculation

    Args:
        Q (float): scattering or absorption coefficient (unitless) 
        r (float): radius of the particle
        rho (float): density of the particle*

    Returns:
        kappa (float): extinction or absorption opacity 
    """
    
    kappa = (3*Q)/(4*(r*1e-4*rho))
    return kappa

def read_dustkappa(filename="dustkappa.dat"):
    with open(filename, "r") as f:
        lines = f.readlines()

    # Find where the data starts (after the "lambda[um]" header)
    start_idx = None
    for i, line in enumerate(lines):
        if "lambda[um]" in line:
            start_idx = i + 4
            break

    # Load the table into a DataFrame
    df = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=start_idx,
        names=["wavelength_um", "kabs_cm2g", "ksca_cm2g", "g"]
    )
    return df, None, None

def run_optool_kappa(cmd_str):
    """
    Run an optool CLI command that prints 'lnk' and return (lam, n, k) as floats or lists.
    Expects the CLI to output three whitespace-separated columns: lambda, n, k.
    """
    # Run the command in the shell safely
    # If you know the path to optool, include it in cmd_str.
    # e.g. "/home/david/miniconda3/bin/optool pyr-mg70 0.70 c 0.30 -l 200 -p 0.50 -print lnk"
    result = subprocess.run(
        shlex.split(cmd_str),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"optool failed (exit {result.returncode}).\n"
            f"STDERR:\n{result.stderr}"
        )
    k_abs = float(result.stdout.splitlines()[0].strip().split()[1])
    k_sca = float(result.stdout.splitlines()[0].strip().split()[2])
    return k_abs, k_sca


def get_scattering_matrix(S1, S2):
    """This function transforms the amplitude scattering matrix
    elements into the stokes scattering matrix elements
    The amplitude scattering matrix has form:
    [
        [S2, S3],
        [S4, S1]   
    ]
    NOTE: S2 and S1 are not a typo. S2 is the first element and S1 the fourth
    
    which is transformed into a 4x4 matrix.

    Args:
        S1 (complex): complex amplitude matrix element 1
        S2 (complex): complex amplitude matrix element 2
    Returns:
        np.matrix : 4x4 scattering matrix (stokes) 
    """
    aS1 = abs(S1)
    aS2 = abs(S2)
    
    
    s11 = 0.5 * (aS1**2 + aS2**2)
    s12 = 0.5 * (aS2**2 - aS1**2)
    s22 = s11.copy() # see scattnlay paper (first version)

    s33 = 0.5 * (S1 * S2.conjugate() + S1.conjugate() * S2)
    s34 = 0.5j* (S1 * S2.conjugate() - S1.conjugate() * S2)
    s44 = s33.copy() # see scattnlay paper (first version)
    
    # m = np.matrix([
    #     [s11, s12,   0  , 0 ],
    #     [s12, s11,   0  , 0 ],
    #     [ 0 ,  0 ,  s33 ,s34],
    #     [ 0 ,  0 , -s34 ,s33]    
    # ]) # sph. symmetry causes this form
    
    return s11, s12, s22, s33, -s34, s44

  


def HoveniersFactor(theta, F11):
    """
    This function is used to normalize F11 such that the integral over the full sphere equals 4 pi.

    Parameters
    ----------
    theta : array
        Scattering angles (degrees or radians)
    F11 : array
        Unnormalized F11(θ)

    Returns
    -------
    F11_norm : array
        Normalized F11 satisfying ∫ F11 dΩ = 4π
    """
    
    # differential element dθ
    dth = np.gradient(theta)

    # full solid angle element dΩ = sinθ dθ dφ = 2π sinθ dθ
    dOmega = 2*np.pi * np.sin(theta) * dth

    # current integral
    integral = np.sum(F11 * dOmega)

    # normalization factor
    factor = 4*np.pi / integral 
    

    # return normalized F11
    return factor

def BohrenAndHuffmanFactor(theta, F11, m_grain, ksca, wavelength):
    """
    This function is used to normalize F11 such that the integral over the full sphere equals kappa_scat m_grain (2pi/lambda)^2.

    Parameters
    ----------
    theta : array
        Scattering angles (degrees or radians)
    F11 : array
    

    Returns
    -------
    F11_norm : array
        Normalized F11 satisfying ∫ F11 dΩ = ksca * mgrain (2pi / lambda)**2
    """
    
    # differential element dθ
    dth = np.gradient(theta)

    # full solid angle element dΩ = sinθ dθ dφ = 2π sinθ dθ
    dOmega = 2*np.pi * np.sin(theta) * dth

    # current integral
    integral = np.sum(F11 * dOmega)

    # normalization factor
    factor = (ksca * m_grain * ((2*np.pi / wavelength)**2)) / integral 

    # return normalized F11
    return factor 

def MishchenkoFactor(theta, F11, m_grain, ksca):
    """
    This function is used to normalize F11 such that the integral over the full sphere equals kappa_scat m_grain (2pi/lambda)^2.

    Parameters
    ----------
    theta : array
        Scattering angles (degrees or radians)
    F11 : array
    

    Returns
    -------
    F11_norm : array
        Normalized F11 satisfying ∫ F11 dΩ = ksca * mgrain (2pi / lambda)**2
    """
    
    # differential element dθ
    dth = np.gradient(theta)

    # full solid angle element dΩ = sinθ dθ dφ = 2π sinθ dθ
    dOmega = 2*np.pi * np.sin(theta) * dth

    # current integral
    integral = np.sum(F11 * dOmega)

    # normalization factor
    factor = ksca * m_grain  / integral 

    # return normalized F11
    return factor 


def renormalize(particle, norm = 'b'):
    if norm == 'b':
        kappas, matrix, _ = particle.Scatt_results

        for wv in kappas["wavelength_um"].unique():
            # selecteer rijen met deze golflengte
            mask = matrix["wavelength_um"] == wv

            # ksca als scalar
            ksca_series = kappas.loc[kappas["wavelength_um"] == wv, "ksca_cm2g"]
            ksca = ksca_series.iloc[0]

            F11s = matrix.loc[mask, "F11"]
            thetas = matrix.loc[mask, "angle_deg"].to_numpy() * np.pi / 180

            factor = BohrenAndHuffmanFactor(thetas, F11s, particle.mass, ksca, wv*1e-4)

            cols = ["F11", "F12", "F22", "F33", "F34", "F44"]
            matrix.loc[mask, cols] = matrix.loc[mask, cols].mul(factor)
        
        kappas, matrix, _ = particle.Optool_results
        for wv in kappas["wavelength_um"].unique():
            # selecteer rijen met deze golflengte
            mask = matrix["wavelength_um"] == wv

            # ksca als scalar
            ksca_series = kappas.loc[kappas["wavelength_um"] == wv, "ksca_cm2g"]
            ksca = ksca_series.iloc[0]

            F11s = matrix.loc[mask, "F11"]
            thetas = matrix.loc[mask, "angle_deg"].to_numpy() * np.pi / 180

            factor = BohrenAndHuffmanFactor(thetas, F11s, particle.mass, ksca, wv*1e-4)

            cols = ["F11", "F12", "F22", "F33", "F34", "F44"]
            matrix.loc[mask, cols] = matrix.loc[mask, cols].mul(factor)
            
    if norm == 'm':
        kappas, matrix, _ = particle.Scatt_results

        for wv in kappas["wavelength_um"].unique():
            # selecteer rijen met deze golflengte
            mask = matrix["wavelength_um"] == wv

            # ksca als scalar
            ksca_series = kappas.loc[kappas["wavelength_um"] == wv, "ksca_cm2g"]
            ksca = ksca_series.iloc[0]

            F11s = matrix.loc[mask, "F11"]
            thetas = matrix.loc[mask, "angle_deg"].to_numpy() * np.pi / 180

            factor = MishchenkoFactor(thetas, F11s, particle.mass, ksca)

            cols = ["F11", "F12", "F22", "F33", "F34", "F44"]
            matrix.loc[mask, cols] = matrix.loc[mask, cols].mul(factor)
        
        kappas, matrix, _ = particle.Optool_results
        for wv in kappas["wavelength_um"].unique():
            # selecteer rijen met deze golflengte
            mask = matrix["wavelength_um"] == wv

            # ksca als scalar
            ksca_series = kappas.loc[kappas["wavelength_um"] == wv, "ksca_cm2g"]
            ksca = ksca_series.iloc[0]

            F11s = matrix.loc[mask, "F11"]
            thetas = matrix.loc[mask, "angle_deg"].to_numpy() * np.pi / 180

            factor = MishchenkoFactor(thetas, F11s, particle.mass, ksca)

            cols = ["F11", "F12", "F22", "F33", "F34", "F44"]
            matrix.loc[mask, cols] = matrix.loc[mask, cols].mul(factor)

    print("done")
    



################################################################################################
# Save the particle to a folder
################################################################################################


def particle_to_dict(p: "Particle") -> dict:
    """Makes a metadata dictionary, so you know what parameters were used as the input"""
    return {
        "D": p.D,
        "kf": p.kf,
        "a0_micron": p.a0,
        "Rc_micron": p.Rc,
        "N_shells": p.N,
        "material": p.material,
        "rho_mono_gcm3": float(p.rho_mono),
        "rho_bar_gcm3": float(p.rho_bar),
        "porosity": float(p.porosity),
        "f_fill": float(p.f),
        "Rg_micron": float(p.Rg),
        "C": float(p.C),
        "mass_g": float(p.mass),
        "cmd": getattr(p, "cmd", None),
        "returnmatrix": bool(p.returnmatrix),
        "optool_settings": dict(OPTOOL_SETTINGS),
    }

def _df_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Saves the dataframe to a csv file"""
    df2 = df.copy()

    # Complex kolommen naar twee kolommen: <col>_re en <col>_im
    for col in list(df2.columns):
        if df2[col].dtype == "complex128":
            df2[f"{col}_re"] = df2[col].apply(lambda z: z.real)
            df2[f"{col}_im"] = df2[col].apply(lambda z: z.imag)
            df2.drop(columns=[col], inplace=True)

    df2.to_csv(path, index=False)

def save_particle(p: "Particle", outdir: str, name: str | None = None) -> str:
    """
    saves the Particle to:
      outdir/<name>/
        metadata.json
        optool_kappas.csv
        optool_matrix.csv (optional)
        optool_angles.csv (optional)
        scattnlay_kappas.csv
        scattnlay_matrix.csv (optional)
        scattnlay_angles.csv (optional)
    """
    base = Path(outdir)
    base.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = f"particle_D{p.D:g}_kf{p.kf:g}_a0{p.a0:g}_Rc{p.Rc:g}_N{p.N}"

    folder = base / name
    folder.mkdir(parents=True, exist_ok=True)

    # metadata
    meta = particle_to_dict(p)
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # optool resultaten
    opt_kappas, opt_matrix, opt_angles = p.Optool_results
    _df_to_csv(opt_kappas, folder / "optool_kappas.csv")
    if opt_matrix is not None:
        _df_to_csv(opt_matrix, folder / "optool_matrix.csv")
    if opt_angles is not None:
        pd.DataFrame({"angle_deg": list(opt_angles)}).to_csv(folder / "optool_angles.csv", index=False)

    # scattnlay resultaten
    sc_kappas, sc_matrix, sc_angles = p.Scatt_results
    _df_to_csv(sc_kappas, folder / "scattnlay_kappas.csv")
    if sc_matrix is not None:
        _df_to_csv(sc_matrix, folder / "scattnlay_matrix.csv")
    if sc_angles is not None:
        pd.DataFrame({"angle_deg": list(sc_angles)}).to_csv(folder / "scattnlay_angles.csv", index=False)

    return str(folder)


