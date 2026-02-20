################################################################################################
# This file contains the bulk of the project. 
# Its structured as follows :
# 1. importing dependencies and defining model settings
# 2. Particle class
# 3. Helper functions
# 4. graphing functions
################################################################################################


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

exe = "/home/david/miniconda3/bin/optool" #legacy, direct pointer to optool

OPTOOL_SETTINGS = {
    "lmin": 0.1,     # micron
    "lmax": 1000,    # micron
    "nl": 100,       # nr of wavelength gridpoints
    "method": "mie", # legacy 
    "NANG": 180      # nr of angle gridpoints. 180 is fine!
}


################################################################################################
# particle class
################################################################################################

class Particle:
    """Particle class that couples Optool and Scattnlay into a single class.
    To build a particle, the structure needs to be defined using the arguments of the class (see below).
    Using this class, the opacities, asymmetry factor and mueller matrix elements can be calculated using MMF, EMT and 
    scattnlay!

        Args:
            D (float): fractal dimension, unitless
            kf (float): fractal prefactor, unitless
            a0 (float): radius of a single monomer, (micron)
            rho_mono (float): denity of a single monomer, (gr cm^-3)
            Rc (float): characteristic radius of the particle, (micron)
            N (int): number of shells the particle is subdivided in, Defaults to 1
            material (str): Optool interpretable string containing the material of the particle, Defaults to "pyr-mg70 0.70 c 0.30"
            matrix (bool): Boolean that triggers matrix element calculations
    """

    def __init__(self, D, kf, a0, Rc, N=1, material="pyr-mg70 0.70 c 0.30", matrix=False):
        
        self.D = float(D)
        self.kf = float(kf)
        self.a0 = float(a0)
        self.material = material
        self.rho_mono = self.get_monomer_density()
        self.Rc = float(Rc)
        self.Rg = self.Rc / (np.sqrt(5/3)) # radius of gyration
        self.C = self.get_C(kf, D, a0, self.rho_mono)
        self.mass = self.get_mass(D, Rc, self.C)
        self.volume = 4/3 * np.pi * Rc**3
        self.rho_bar = self.mass / (self.volume * 1e-12)
        self.f = self.rho_bar / self.rho_mono #volume filling factor
        self.porosity = 1 - self.f
        self.N = N
        if self.N == 1:
            self.Rn = self.Rc #if one shell -> that shell is the same size as the particle
        else:
            self.Rn = np.linspace(2*a0, Rc, N) # otherwise its at least twice the size of a single monomer up to Rc
            # this lower limit has to be set because otherwise the retrieval of refractive indexes runs into an error
            # trying to call data for unphysical particles!
        self.Rho_n = self.rho_shell() # denisty of the shells

        self.returnmatrix = matrix 
        
        self.Notes = None # notes about the particle

    def RunMie(self):
        '''
        Calls Optool to perform a -mie calculation on the particle 
        Stores it in self.Optool_results and .EMT_results
        '''
        self.Optool_results = self.get_optool_kappas()
        self.EMT_results = self.Optool_results

    def RunScatt(self):
        """Calls Scattnlay to perform a multilayered calculation on the particle 
        Stores it in self.Scatt_results"""
        if not self.returnmatrix:
            self.Scatt_results = self.get_scattnlay_kappas()
        else:
            self.Scatt_results = self.get_scattnlay_kappas(self.Optool_results[2])

    def RunMMF(self, force = False):
        """Calls Optool to perform a MMF calculation on the particle 
        Stores it in self.MMF_results
        
        force arguement forces calculations by optool by setting -mmfss
        defaults to False
        
        """
        if force == False:
            self.MMF_results = self.get_optool_kappas(method="mmf")
        else:
            self.MMF_results = self.get_optool_kappas(method="mmfss")

    def get_C(self, kf, D, a0, rho_mono):
        """gets the integration constant C used to calculate the mass and density of the particle given the structure params
        
        Args:
            kf (float): fractal prefactor
            D (float): dimension
            a0 (float): monomer radius in micron
            rho_mono (float): monomer denisty in g/cm^3

        Returns:
            float: integration constant
            
        note:
        made this before I realized asking for structure args is redundant here
        
        C = rho_mono * a0**(3-D) * (kf*D/3) * sqrt(3/5) * sqrt(5/3)**(3-D)
        could be simplified looking at the sqrts
        """
        C = rho_mono * 1e-12 # to cm
        C *= a0**(3-D) 
        C *= kf
        C *= D/3
        C *= np.sqrt(3/5)
        C *= np.sqrt(5/3)**(3-D)
        return C

    def get_compact_size(self):
        """calculates the compact size parameter for the -MMF method according to its definition in Optool

        Returns:
            float: compact size paramter a(micron)
        """
        a = self.kf * self.Rc**(self.D)
        a *= self.a0**(3-self.D)
        a *= (5/3)**(-self.D/2)
        return a**(1/3)

    def get_mass(self, D, Rc, C):
        '''
        calcs mass of the aggregate in grams
        '''
        M_encl = 4*np.pi
        M_encl *= (Rc)**(D)
        M_encl *= 1/D
        M_encl *= C
        return M_encl

    def mass_enclosed(self, R):
        """calcs aggregate mass enclosed by a sphere of radius R """
        return 4*np.pi * self.C * (R**self.D) / self.D

    def rho_bar_shell(self, R):
        """legacy"""
        M = self.mass_enclosed(R)
        V = (4/3) * np.pi * (R**3)
        return M / V

    def describe(self, verbose=True, kappas=False):
        """gives a description of the partcile

        Args:
            verbose (bool, optional): trigger for the printing. Defaults to True.
            kappas (bool, optional): prints kappas. Defaults to False.

        Returns:
            _type_: _description_
        """
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
                f"Mean density (ρ̄):           {self.rho_bar:.3e} g/cm³\n"
                f"Notes:                    \n{getattr(self, 'Notes', None)}\n " 
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

    def get_optool_kappas(self, method=None):
        """performs a Optool call to calc the kappas and asymmetry parameter
        """
        if not self.returnmatrix: # first we check if we need the matrix
            lmin = OPTOOL_SETTINGS["lmin"] # setting hyperparams locally
            lmax = OPTOOL_SETTINGS["lmax"]
            nl = OPTOOL_SETTINGS["nl"]

            if method is None: #standard -mie
                cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {self.Rc} -p {self.porosity} -mie"
                )
            if method == "mmf": #-mmf
                a = self.get_compact_size() # gets compact size param -a
                cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {a} -mmf {self.a0} {self.f} {self.kf}"
                )
            
            if method == "mmfss": #-mmf
                a = self.get_compact_size() # gets compact size param -a
                cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {a} -mmfss {self.a0} {self.f} {self.kf}"
                )
                
            #run_optool(cmd) # runs optool
            rc, out, err = run_optool(cmd)
            self.cmd = cmd # set cmd
            
            #check for wanrnings in the output
            warns = _parse_optool_warnings(out, err)
            if warns:
                for w in warns:
                    print(f"⚠ Optool warning: {w}")
                # add warning to notes
                if getattr(self, "Notes", None):
                    self.Notes += "\n" + "\n".join(warns)
                else:
                    self.Notes = "\n".join(warns)
                    
            return read_dustkappa() # returns results
        
        else: # identical to above but with matrix results
            lmin = OPTOOL_SETTINGS["lmin"]
            lmax = OPTOOL_SETTINGS["lmax"]
            nl = OPTOOL_SETTINGS["nl"]
            nang = OPTOOL_SETTINGS["NANG"]

            if method is None:
                cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {self.Rc} -p {self.porosity} -mie -s {nang}"
                )

            if method == "mmf":
                cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {self.Rc} -mmf {self.a0} {self.f} {self.kf} -s {nang}"
                )
            
            if method == "mmfss":
                cmd = (
                    f"optool {self.material} "
                    f"-lmin {lmin} -lmax {lmax} "
                    f"-nl {nl} -a {self.Rc} -mmfss {self.a0} {self.f} {self.kf} -s {nang}"
                )

            rc, out, err = run_optool(cmd)
            self.cmd = cmd #store cmd
            
            warns = _parse_optool_warnings(out, err)
            if warns:
                for w in warns:
                    print(f"⚠ Optool warning: {w}")
                #add warning to notes
                if getattr(self, "Notes", None):
                    self.Notes += "\n" + "\n".join(warns)
                else:
                    self.Notes = "\n".join(warns)
            
            return read_optool_file() # return data from run

    def get_monomer_density(self):
        """get the density of the individual monomers using Optool

        Returns:
            float: density in g/cm^3
        """
        cmd = f"optool {self.material} -print lnk" # using -lnk trigger
        rho, lam, n_l, k_l = run_optool_lnk(cmd)
        return rho

    def get_refractive_index(self, porosity=None):
        """gets the refractive index of a given material using optool

        Args:
            porosity (float, optional): porosity of the material. Defaults to None.

        Returns:
            df: df with refractive index per wavelength
        """
        # when getting the refractive index for the layers -> they have different porosities.
        # if None are given, we just want the index of the aggregate
        # if it is given, we want the index of the n-th layer
        if porosity is None: 
            porosity = self.porosity 

        lmin = OPTOOL_SETTINGS["lmin"]
        lmax = OPTOOL_SETTINGS["lmax"]
        nl = OPTOOL_SETTINGS["nl"]

        cmd = (
            f"optool {self.material} "
            f"-lmin {lmin} -lmax {lmax} "
            f"-nl {nl} -a {self.Rc} -p {porosity} -mie -print lnk" #calls -lnk for the given porosity and material
        ) # Rc technically does not matter

        rho, lam, n_l, k_l = run_optool_lnk(cmd)
        df = pd.DataFrame({
            "wavelength": lam,
            "index": [complex(i, j) for i, j in zip(n_l, k_l)]
        }) 
        return df #returns refractive index for each wavelength

    def get_dV(self):
        """gets volume for spherical integration
        """
        S = (4.0/3.0) * np.pi
        if self.N == 1:
            return S * (self.Rc**3)
        else:
            borders = self.Rn #using the borders of the layers
            dV = S * (borders[1:]**3 - borders[:-1]**3) # dV = 4/3*pi* (r_i+1-r_i)
            Vcore = S * (borders[0]**3) # inner one does not have a border
            return np.concatenate(([Vcore], dV)) #sum

    def get_dM(self):
        """gets mass of spherical integrtion elements
        """
        if self.N == 1:
            return self.mass
        else:
            borders = self.Rn
            dM = self.mass_enclosed(borders[1:]) - self.mass_enclosed(borders[:-1]) # difference between masses of spherei+1 and i
            Mcore = self.mass_enclosed(borders[0]) # one does not have a border
            return np.concatenate(([Mcore], dM)) #sum

    def rho_shell(self):
        "calcs the density of the shells in gr/cm^3 using dM and dV"
        dM = self.get_dM()
        dV = self.get_dV()
        return dM / (dV * 1e-12) #to cm from micron

    def build_run_df(self):
        """This function builds the structure that is used to run the scattnlay model. The scattnlay model requires 
        the size parameter x and the refractive index of the layers. This code uses the structure parameters given and the hyperparameter
        to construct a df that is then passed to scattnlay! 

        Returns:
            df :  run df passed to scattnlay containing size param x, refractive index (complex) and check values p, r and wavelength
        """
        if self.N != 1:
            rho_shell = self.rho_shell()
            porosities = 1 - rho_shell / self.rho_mono
            if np.any(porosities < 0.0):
                bad_idx = np.where(porosities < 0.0)[0] #security feature that can often be skipped. 
                # it's needed because one of the dependencies does not like very small numbers/negative numbers
                print(
                    f"⚠️  Warning: some porosity values are ≤ 0!\n"
                    f"   Problematic indices: {bad_idx}\n"
                    f"   Porosity values: {porosities[bad_idx]}\n"
                    f"   → These values have been replaced with 0."
                )
                porosities[bad_idx] = 0.0

            frames = []#storage
            for R_i, p in zip(np.asarray(self.Rn, float), np.asarray(porosities, float)): 
                # loops over the layers and their densities 
                pdf = self.get_refractive_index(porosity=float(p)) # to get the refractive index of each layer
                l = pdf["wavelength"].to_numpy(dtype=float)
                x = x_size_param(R_i, l)

                frames.append(pd.DataFrame({
                    "wavelength": l,
                    "index": pdf["index"].to_numpy(),
                    "porosity": float(p),
                    "radius": float(R_i),
                    "x": x
                })) # and store the results 

            run_df = pd.concat(frames, ignore_index=True) # the run df in the end is a nested df
            # for each layer we need the size parameter and the refractive index per wavelength
            # the porosity is not needed per se but its nice to store for checks!
            return run_df
        else: # if we have a single shell
            rho_shell = self.rho_bar
            porosities = 1 - rho_shell / self.rho_mono
            if np.any(porosities < 0.0):
                bad_idx = np.where(porosities < 0.0)[0] #see same warning above
                print(
                    f"⚠️  Warning: some porosity values are ≤ 0!\n"
                    f"   Problematic indices: {bad_idx}\n"
                    f"   Porosity values: {porosities[bad_idx]}\n"
                    f"   → These values have been replaced with 0."
                ) # if this warning is triggered -> do you even have a particle? 
                porosities[bad_idx] = 0.0

            pdf = self.get_refractive_index(porosity=float(porosities))
            l = pdf["wavelength"].to_numpy(dtype=float)
            x = x_size_param(self.Rc, l)

            run_df = pd.DataFrame({
                "wavelength": l,
                "index": pdf["index"].to_numpy(),
                "porosity": float(porosities),
                "radius": self.Rc,
                "x": x
            }) # see above
            return run_df

    def get_scattnlay_kappas(self, angles=None):
        '''
        this function runs the scattnlay model to find the optical properties! 
        
        args: 
         angles [list] : if angles is given, matrix calculations are performed as well as kappas and g!
         
        returns:
        
            3 dfs containing kappas and g (df_scattnlay), matrix elements (matrix_ df) and list of angles used 
        '''
        if angles is None: 
            run_df = self.build_run_df() # makes structure for scattnlay
            records = [] #storage
            for λ in run_df["wavelength"].unique(): #loop over every wavelength needed
                df_at_lambda = run_df[run_df["wavelength"] == λ] #select the rows needed (nested df)
                x_vals = df_at_lambda["x"].to_numpy() # size params of the layers
                m_vals = df_at_lambda["index"].to_numpy() #refractive index (complex) of layers
                terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(x_vals, m_vals) # run scattnlay

                records.append({
                    "wavelength_um": λ,
                    "Qabs": Qabs,
                    "Qsca": Qsca,
                    "g": g,
                    "S1 S2": [S1, S2]
                }) #store everything

            df_scattnlay = pd.DataFrame.from_records(records)# make df from records
            df_scattnlay["kabs_cm2g"] = kappa_from_Q(df_scattnlay["Qabs"].values, self.Rc, self.rho_bar) # convert Q to kappa
            df_scattnlay["ksca_cm2g"] = kappa_from_Q(df_scattnlay["Qsca"].values, self.Rc, self.rho_bar)
            df_scattnlay.drop(["Qabs", "Qsca"], axis=1, inplace=True)# drop terms
            return df_scattnlay, None, None
        else:
            run_df = self.build_run_df() # same as above
            records = [] #double storage
            records2 = []
            for λ in run_df["wavelength"].unique():
                df_at_lambda = run_df[run_df["wavelength"] == λ]
                x_vals = df_at_lambda["x"].to_numpy()
                m_vals = df_at_lambda["index"].to_numpy()

                terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(
                    x_vals, m_vals, theta=[i * np.pi / 180 for i in angles]
                )

                records.append({
                    "wavelength_um": λ,
                    "Qabs": Qabs,
                    "Qsca": Qsca,
                    "g": g
                })

                F11, F12, F22, F33, F34, F44 = get_scattering_matrix(S1, S2) # from amplitude scatter matrix to mueller
                factor = HoveniersFactor([i * np.pi / 180 for i in angles], F11) #normalize to hoveniers
                F11 *= factor
                F12 *= factor
                F22 *= factor
                F33 *= factor
                F34 *= factor
                F44 *= factor

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
                    df_matrix_elements[col] = df_matrix_elements[col].apply(lambda z: z.real) #every nonzero im value left is floating point err
                except Exception:
                    pass

            df_scattnlay["kabs_cm2g"] = kappa_from_Q(df_scattnlay["Qabs"].values, self.Rc, self.rho_bar)
            df_scattnlay["ksca_cm2g"] = kappa_from_Q(df_scattnlay["Qsca"].values, self.Rc, self.rho_bar)
            df_scattnlay.drop(["Qabs", "Qsca"], axis=1, inplace=True)
            return df_scattnlay, df_matrix_elements, angles

    ################################################################################################
    # Updated from_folder with robust optional loading for EMT, Scatt, MMF
    ################################################################################################
    @classmethod
    def from_folder(cls, folder: str | Path) -> "Particle":
        """
        
        code made with the help of Chatgpt 5.2! Jan 2026. Tested
        Create a Particle instance from a saved folder without running __init__.
        Loads whatever exists, and keeps missing parts as None.

        Files that may exist
            metadata.json

            optool_kappas.csv optional
            optool_matrix.csv optional
            optool_angles.csv optional

            scattnlay_kappas.csv optional
            scattnlay_matrix.csv optional
            scattnlay_angles.csv optional

            mmf_kappas.csv optional
            mmf_matrix.csv optional
            mmf_angles.csv optional
        """
        folder = Path(folder)

        meta_path = folder / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {folder}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        def read_csv(name: str) -> pd.DataFrame | None:
            p = folder / name
            if not p.exists():
                return None
            df = pd.read_csv(p)
            return df if not df.empty else None

        def read_angles(name: str) -> list[float] | None:
            df = read_csv(name)
            if df is None:
                return None
            if "angle_deg" not in df.columns:
                return None
            angles = df["angle_deg"].astype(float).to_list()
            return angles if len(angles) > 0 else None

        def read_triplet(stem: str):
            kappas = read_csv(f"{stem}_kappas.csv")
            matrix = read_csv(f"{stem}_matrix.csv")
            angles = read_angles(f"{stem}_angles.csv")
            return kappas, matrix, angles

        self = cls.__new__(cls)

        self.D = float(meta["D"])
        self.kf = float(meta["kf"])
        self.a0 = float(meta["a0_micron"])
        self.Rc = float(meta["Rc_micron"])
        self.N = int(meta["N_shells"])
        self.material = meta["material"]
        self.returnmatrix = bool(meta.get("returnmatrix", False))

        self.rho_mono = float(meta.get("rho_mono_gcm3", np.nan))
        self.rho_bar = float(meta.get("rho_bar_gcm3", np.nan))
        self.porosity = float(meta.get("porosity", np.nan))
        self.f = float(meta.get("f_fill", np.nan))
        self.Rg = float(meta.get("Rg_micron", self.Rc / np.sqrt(5/3)))
        self.C = float(meta.get("C", np.nan))
        self.mass = float(meta.get("mass_g", np.nan))
        self.cmd = meta.get("cmd", None)

        self.volume = 4/3 * np.pi * self.Rc**3
        if self.N == 1:
            self.Rn = self.Rc
        else:
            self.Rn = np.linspace(2 * self.a0, self.Rc, self.N)

        self.Rho_n = None

        self.Optool_results = read_triplet("optool")
        self.EMT_results = self.Optool_results
        self.Scatt_results = read_triplet("scattnlay")
        self.MMF_results = read_triplet("mmf")
        self.Notes = meta.get("notes", None)

        return self


def read_optool_file(filename="dustkapscatmat.dat"):
    '''
    This function is made to open/read dustkapscatmat.dat files
    and returns necessary data
    
    '''
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

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

    angle_data = []
    while i < len(lines) and len(lines[i].split()) == 1:
        angle_data.append(float(lines[i]))
        i += 1
    if len(angle_data) != nang:
        raise ValueError(f"Expected {nang} angles, found {len(angle_data)}.")

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

    df.attrs.update({
        "format": fmt,
        "nlam": nlam,
        "nang": nang,
    })

    return lam_df, df, angle_data


def run_optool_lnk(cmd_str):
    """
    runs optool to find and get data from .lnk files using a optool cmd string
    returns rho, lambda, n and k (where m = n + ik)
    """
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

    rho = float(result.stdout.splitlines()[0].strip().split()[1])

    lam, n, k = [], [], []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            L, N, K = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        lam.append(L)
        n.append(N)
        k.append(K)

    if len(lam) == 1:
        return rho, lam[0], n[0], k[0]
    return rho, lam, n, k


def run_optool(cmd_str):
    """runs optool, often used to generate a .dat file that is then read by this code using a reader function
    the return statement is added to capture warnings raised by optool
    """
    result = subprocess.run(
        shlex.split(cmd_str),
        capture_output=True,
        text=True,
        check=False
        
    )
    return result.returncode, result.stdout, result.stderr #used for capturing warnings

def _parse_optool_warnings(stdout: str, stderr: str) -> list[str]:
    text = (stdout or "") + "\n" + (stderr or "")
    lines = text.splitlines()

    warnings = []
    current_block = []

    for line in lines:
        stripped = line.strip()

        # new warning starts
        if stripped.startswith("WARNING:"):
            # store previous block if any
            if current_block:
                warnings.append(" ".join(current_block))
                current_block = []

            current_block.append(stripped)

        # continuation line (indented or not empty but not a new header)
        elif current_block and stripped != "":
            current_block.append(stripped)

        # blank line ends block
        elif current_block:
            warnings.append(" ".join(current_block))
            current_block = []

    # catch last block
    if current_block:
        warnings.append(" ".join(current_block))

    return warnings

def sweep_lnk(exe, materials, fracs, porosity, lambdas_nm):
    """legacy code that swept a range of wavelenghts to get all lnk data associated with those wavelengths.
    """
    assert len(materials) == len(fracs), "materials and fracs must match in length"
    mat_str = " ".join(sum(([m, f"{x:.6g}"] for m, x in zip(materials, fracs)), []))
    lams, ns, ks = [], [], []
    for L in lambdas_nm:
        cmd = f"{exe} {mat_str} -l {L:.6g} -p {porosity:.6g} -print lnk"
        Lout, N, K = run_optool_lnk(cmd)
        lams.append(Lout)
        ns.append(N)
        ks.append(K)
    return lams, ns, ks


def x_size_param(r, wavelength, n_m=1):
    '''
    calcs size param x using radius r and wavelength 
    note that those have to be in identical units
    '''
    return r * np.pi * 2 * n_m / wavelength


def kappa_from_Q(Q, r, rho):
    """converts Q to kappa
    where k is in [cm^2/g] for a given Q and radius r and density rho. 
    """
    kappa = (3 * Q) / (4 * (r * 1e-4 * rho)) #note we need to convert to cm from micron
    return kappa


def read_dustkappa(filename="dustkappa.dat"):
    """reads and stores dustkappa files for storing in the particle class
    """
    
    with open(filename, "r") as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if "lambda[um]" in line:
            start_idx = i + 4
            break

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=start_idx,
        names=["wavelength_um", "kabs_cm2g", "ksca_cm2g", "g"]
    )
    return df, None, None


def run_optool_kappa(cmd_str):
    """
    legacy code that ran optool and read the results in command center
    """
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
    """
    Using amplitude matrix elements S1 and S2, calc F11 F12, F22, F33, F34, F44
    see bohren and huffman 1983 page 65 for exact list:
    https://archive.org/details/bohren_huffman_scattering_light_by_small/page/64/mode/2up
    
    amplitude matrix:
                    [S1, S3]
                    [S4, S2] where S3 and S4 == 0 for spherical particles
                    Si are complex
                    ---->
                    [F11, F12, F13, F14]
                    [F21, .............]
                    [..................]
                    [............., F44]
    note that due to spherical symmetry
    F12 = F21, F34 = -F43
    
    Throughout the code, i make reference to Sxx and Fxx -> refer to the same thing
    Sx -> complex amplitude matrix elements
    Sxx ~ Fxx -> mueller matrix elements
    
    Normalization is done after retrieving the elements
                    
    """
    aS1 = abs(S1)
    aS2 = abs(S2)

    s11 = 0.5 * (aS1**2 + aS2**2)
    s12 = 0.5 * (aS2**2 - aS1**2)
    s22 = s11.copy()

    s33 = 0.5 * (S1 * S2.conjugate() + S1.conjugate() * S2)
    s34 = 0.5j * (S1 * S2.conjugate() - S1.conjugate() * S2)
    s44 = s33.copy()

    return s11, s12, s22, s33, -s34, s44


def HoveniersFactor(theta, F11):
    """Calcs Normalization factor for the Hoveniers convention 
    4pi = integral(F11(theta) domega)
    Args:
        theta (list): list of angles used 
        F11 (list): first matrix element

    Returns:
       float: normalization factor
    """
    dth = np.gradient(theta)
    dOmega = 2 * np.pi * np.sin(theta) * dth
    integral = np.sum(F11 * dOmega)
    factor = 4 * np.pi / integral
    return factor


def BohrenAndHuffmanFactor(theta, F11, m_grain, ksca, wavelength):
    """Calcs Normalization factor for the Bohren/Huffman convention 
    ksca * mgrain * (2pi/lambda)**2 = integral(F11(theta) domega)
    Args:
        theta (list): list of angles used 
        F11 (list): first matrix element

    Returns:
       float: normalization factor
    """
    dth = np.gradient(theta)
    dOmega = 2 * np.pi * np.sin(theta) * dth
    integral = np.sum(F11 * dOmega)
    factor = (ksca * m_grain * ((2 * np.pi / wavelength)**2)) / integral
    return factor


def MishchenkoFactor(theta, F11, m_grain, ksca):
    """Calcs Normalization factor for the Mischenko convention 
    ksca * mgrain  = integral(F11(theta) domega)
    Args:
        theta (list): list of angles used 
        F11 (list): first matrix element

    Returns:
       float: normalization factor"""
    dth = np.gradient(theta)
    dOmega = 2 * np.pi * np.sin(theta) * dth
    integral = np.sum(F11 * dOmega)
    factor = ksca * m_grain / integral
    return factor


def renormalize(particle, norm="b"):
    """performs the renormalization

    Args:
        particle (Particle class): particle you want to renormalize
        norm (str, optional): _description_. Defaults to "b".
                            Normalization wanted (Mischenko: 'm', Bohren:'b',
                            Hoveniers: 'h')
    """
    if norm == "b":
        kappas, matrix, _ = particle.Scatt_results
        if kappas is not None and matrix is not None and (not matrix.empty):
            for wv in kappas["wavelength_um"].unique():
                mask = matrix["wavelength_um"] == wv
                ksca_series = kappas.loc[kappas["wavelength_um"] == wv, "ksca_cm2g"]
                ksca = ksca_series.iloc[0]
                F11s = matrix.loc[mask, "F11"]
                thetas = matrix.loc[mask, "angle_deg"].to_numpy() * np.pi / 180
                factor = BohrenAndHuffmanFactor(thetas, F11s, particle.mass, ksca, wv * 1e-4)
                cols = ["F11", "F12", "F22", "F33", "F34", "F44"]
                matrix.loc[mask, cols] = matrix.loc[mask, cols].mul(factor)

        kappas, matrix, _ = particle.Optool_results
        if kappas is not None and matrix is not None and (not matrix.empty):
            for wv in kappas["wavelength_um"].unique():
                mask = matrix["wavelength_um"] == wv
                ksca_series = kappas.loc[kappas["wavelength_um"] == wv, "ksca_cm2g"]
                ksca = ksca_series.iloc[0]
                F11s = matrix.loc[mask, "F11"]
                thetas = matrix.loc[mask, "angle_deg"].to_numpy() * np.pi / 180
                factor = BohrenAndHuffmanFactor(thetas, F11s, particle.mass, ksca, wv * 1e-4)
                cols = ["F11", "F12", "F22", "F33", "F34", "F44"]
                matrix.loc[mask, cols] = matrix.loc[mask, cols].mul(factor)

    if norm == "m":
        kappas, matrix, _ = particle.Scatt_results
        if kappas is not None and matrix is not None and (not matrix.empty):
            for wv in kappas["wavelength_um"].unique():
                mask = matrix["wavelength_um"] == wv
                ksca_series = kappas.loc[kappas["wavelength_um"] == wv, "ksca_cm2g"]
                ksca = ksca_series.iloc[0]
                F11s = matrix.loc[mask, "F11"]
                thetas = matrix.loc[mask, "angle_deg"].to_numpy() * np.pi / 180
                factor = MishchenkoFactor(thetas, F11s, particle.mass, ksca)
                cols = ["F11", "F12", "F22", "F33", "F34", "F44"]
                matrix.loc[mask, cols] = matrix.loc[mask, cols].mul(factor)

        kappas, matrix, _ = particle.Optool_results
        if kappas is not None and matrix is not None and (not matrix.empty):
            for wv in kappas["wavelength_um"].unique():
                mask = matrix["wavelength_um"] == wv
                ksca_series = kappas.loc[kappas["wavelength_um"] == wv, "ksca_cm2g"]
                ksca = ksca_series.iloc[0]
                F11s = matrix.loc[mask, "F11"]
                thetas = matrix.loc[mask, "angle_deg"].to_numpy() * np.pi / 180
                factor = MishchenkoFactor(thetas, F11s, particle.mass, ksca)
                cols = ["F11", "F12", "F22", "F33", "F34", "F44"]
                matrix.loc[mask, cols] = matrix.loc[mask, cols].mul(factor)
        
    if norm == "h":
    # Scatt
        kappas, matrix, _ = particle.Scatt_results
        if kappas is not None and matrix is not None and (not matrix.empty):
            for wv in kappas["wavelength_um"].unique():
                mask = matrix["wavelength_um"] == wv
                F11s = matrix.loc[mask, "F11"].to_numpy(dtype=float)
                thetas = matrix.loc[mask, "angle_deg"].to_numpy(dtype=float) * np.pi / 180
                factor = HoveniersFactor(thetas, F11s)
                cols = ["F11", "F12", "F22", "F33", "F34", "F44"]
                matrix.loc[mask, cols] = matrix.loc[mask, cols].mul(factor)

        # Optool
        kappas, matrix, _ = particle.Optool_results
        if kappas is not None and matrix is not None and (not matrix.empty):
            for wv in kappas["wavelength_um"].unique():
                mask = matrix["wavelength_um"] == wv
                F11s = matrix.loc[mask, "F11"].to_numpy(dtype=float)
                thetas = matrix.loc[mask, "angle_deg"].to_numpy(dtype=float) * np.pi / 180
                factor = HoveniersFactor(thetas, F11s)
                cols = ["F11", "F12", "F22", "F33", "F34", "F44"]
                matrix.loc[mask, cols] = matrix.loc[mask, cols].mul(factor)
        
    print("done")


################################################################################################
# Save the particle to a folder
################################################################################################

def particle_to_dict(p: "Particle") -> dict:
    """makes a dictionary of the particle structure params/class used for storage
    """
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
        "notes": getattr(p, "Notes", None),
    }


def _df_to_csv(df: pd.DataFrame, path: Path) -> None:
    """converts a df to csv file used for storage
    """
    df2 = df.copy()

    for col in list(df2.columns):
        if df2[col].dtype == "complex128":
            df2[f"{col}_re"] = df2[col].apply(lambda z: z.real)
            df2[f"{col}_im"] = df2[col].apply(lambda z: z.imag)
            df2.drop(columns=[col], inplace=True)

    df2.to_csv(path, index=False)


################################################################################################
# Updated save_particle with robust optional saving for EMT, Scatt, MMF
################################################################################################

def save_particle(p: "Particle", outdir: str, name: str | None = None) -> str:
    """
    Saves the Particle to:
      outdir/<name>/
        metadata.json

        optool_kappas.csv optional
        optool_matrix.csv optional
        optool_angles.csv optional

        scattnlay_kappas.csv optional
        scattnlay_matrix.csv optional
        scattnlay_angles.csv optional

        mmf_kappas.csv optional
        mmf_matrix.csv optional
        mmf_angles.csv optional

    It only writes files that exist and are not empty.
    """
    base = Path(outdir)
    base.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = f"particle_D{p.D:g}_kf{p.kf:g}_a0{p.a0:g}_Rc{p.Rc:g}_N{p.N}"

    folder = base / name
    folder.mkdir(parents=True, exist_ok=True)

    meta = particle_to_dict(p)
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _save_triplet(triplet, stem: str):
        kappas, matrix, angles = _safe_triplet(triplet)

        if _is_nonempty_df(kappas):
            _df_to_csv(kappas, folder / f"{stem}_kappas.csv")

        if _is_nonempty_df(matrix):
            _df_to_csv(matrix, folder / f"{stem}_matrix.csv")

        if _is_nonempty_angles(angles):
            pd.DataFrame({"angle_deg": list(angles)}).to_csv(
                folder / f"{stem}_angles.csv",
                index=False
            )

    _save_triplet(getattr(p, "Optool_results", None), "optool")
    _save_triplet(getattr(p, "Scatt_results", None), "scattnlay")
    _save_triplet(getattr(p, "MMF_results", None), "mmf")

    return str(folder)

################################################################################################
# Helpers for robust save and load of optional results
################################################################################################

def _safe_triplet(obj):
    """
    Normalizes to (kappas_df, matrix_df, angles_list).
    Accepts None or tuple or list of length 3.
    """
    if obj is None:
        return None, None, None
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        return obj[0], obj[1], obj[2]
    return None, None, None

def _is_nonempty_df(x):
    return isinstance(x, pd.DataFrame) and (not x.empty)

def _is_nonempty_angles(x):
    try:
        return x is not None and len(x) > 0
    except Exception:
        return False
    
    
    
################################################################################################
# Plotting
################################################################################################

# controlling the output
legend_size = 16
axis_size = 20
ticksize = 14
lw = 3




def plot_kappas(particle, save=None):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    def _kappa_arrays(df):
        lam = df["wavelength_um"].to_numpy(dtype=float)
        kabs = df["kabs_cm2g"].to_numpy(dtype=float)
        ksca = df["ksca_cm2g"].to_numpy(dtype=float)
        return lam, kabs, ksca

    # Helper: fetch attribute safely and ensure it has at least one element
    def _get_first_result(obj, attr_name):
        value = getattr(obj, attr_name, None)  # None if attribute doesn't exist
        if value is None:
            return None
        if hasattr(value, "__len__") and len(value) == 0:
            return None
        try:
            return value[0]
        except Exception:
            return None

    # Keep references for residuals
    lam_ref = kabs_ref = ksca_ref = None

    # EMT reference curves
    emt0 = _get_first_result(particle, "EMT_results")
    if emt0 is not None:
        lam_ref, kabs_ref, ksca_ref = _kappa_arrays(emt0)

        ax1.plot(lam_ref, kabs_ref, color="black", linewidth=lw + 1)
        ax1.plot(lam_ref, ksca_ref, color="black", linewidth=lw + 1)
        ax1.plot(lam_ref, kabs_ref, label=r"EMT $\kappa_{\text{abs}}$", linewidth=lw)
        ax1.plot(lam_ref, ksca_ref, label=r"EMT $\kappa_{\text{sca}}$", linewidth=lw)

    # Scattering results (ML EMT)
    scatt0 = _get_first_result(particle, "Scatt_results")
    if scatt0 is not None:
        lam_s, kabs_s, ksca_s = _kappa_arrays(scatt0)

        ax1.scatter(lam_s, kabs_s, color="black", s=35, zorder=3)
        ax1.scatter(lam_s, ksca_s, color="black", s=35, zorder=3)
        ax1.scatter(lam_s, kabs_s, label=r"ML-EMT $\kappa_{\text{abs}}$", s=25, zorder=4)
        ax1.scatter(lam_s, ksca_s, label=r"ML-EMT $\kappa_{\text{sca}}$", s=25, zorder=4)

        # Residuals only if we have EMT reference available
        if kabs_ref is not None and ksca_ref is not None and lam_ref is not None:
            # If grids differ, interpolate reference onto scattering wavelengths
            if not np.array_equal(lam_ref, lam_s):
                kabs_ref_i = np.interp(lam_s, lam_ref, kabs_ref)
                ksca_ref_i = np.interp(lam_s, lam_ref, ksca_ref)
            else:
                kabs_ref_i = kabs_ref
                ksca_ref_i = ksca_ref

            # Avoid divide-by-zero warnings
            with np.errstate(divide="ignore", invalid="ignore"):
                diff_kabs = (kabs_ref_i - kabs_s) / kabs_ref_i * 100.0
                diff_ksca = (ksca_ref_i - ksca_s) / ksca_ref_i * 100.0

            ax2.scatter(lam_s, diff_kabs, color="black", s=35)
            ax2.scatter(lam_s, diff_ksca, color="black", s=35)
            ax2.scatter(lam_s, diff_kabs, label=r"Residual $\kappa_{\text{abs}}$", s=25)
            ax2.scatter(lam_s, diff_ksca, label=r"Residual $\kappa_{\text{sca}}$", s=25)

    # MMF curves
    mmf0 = _get_first_result(particle, "MMF_results")
    if mmf0 is not None:
        lam_m, kabs_m, ksca_m = _kappa_arrays(mmf0)

        ax1.plot(lam_m, kabs_m, color="black", linestyle="--", linewidth=lw)
        ax1.plot(lam_m, ksca_m, color="black", linestyle="--", linewidth=lw)
        ax1.plot(lam_m, kabs_m, label=r"MMF $\kappa_{\text{abs}}$", linestyle="--", linewidth=lw)
        ax1.plot(lam_m, ksca_m, label=r"MMF $\kappa_{\text{sca}}$", linestyle="--", linewidth=lw)

    # Formatting
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(1e0, 1e5)
    ax1.set_ylabel(r"$\kappa$ (cm$^2$ g$^{-1}$)", fontsize=axis_size)
    ax1.legend(fontsize=legend_size, ncol=3, frameon=False)
    ax1.tick_params(axis="both", which="major", labelsize=ticksize)

    ax2.axhline(0, color="black", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("Wavelength (Micron)", fontsize=axis_size)
    ax2.set_ylabel("Residual (%)", fontsize=axis_size)
    ax2.tick_params(axis="both", which="major", labelsize=ticksize)

    if save is not None:
        fig.savefig(save, bbox_inches="tight", dpi=300)

    return fig, (ax1, ax2)

def mmf_bad_lambda_um(particle):
    """
    Returns a float lambda_limit_um if MMF warnings contain 'lam<='
    otherwise returns None.
    Looks in particle.Optool_warnings first, then Notes.
    """
    texts = []

    w = getattr(particle, "Optool_warnings", None)
    if w:
        texts.extend([str(x) for x in w])

    notes = getattr(particle, "Notes", None)
    if notes:
        texts.append(str(notes))

    blob = "\n".join(texts)

    m = re.search(r"lam\s*<=\s*([0-9]*\.?[0-9]+)", blob)
    if not m:
        return None

    try:
        return float(m.group(1))
    except Exception:
        return None

def plot_mmf_with_warning_range(x, y, lam_bad_um=None, is_wavelength_x=True, **plot_kwargs):
    """
    If lam_bad_um is set and x is wavelength, plot y red where x <= lam_bad_um.
    Otherwise plot normally.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if lam_bad_um is None or not is_wavelength_x:
        plt.plot(x, y, **plot_kwargs)
        return

    bad = x <= float(lam_bad_um)
    good = ~bad

    # plot good segment in the requested style
    if np.any(good):
        plt.plot(x[good], y[good], **plot_kwargs)

    # plot bad segment in red, keep the same linestyle and linewidth if provided
    bad_kwargs = dict(plot_kwargs)
    bad_kwargs["color"] = "red"
    bad_kwargs["label"] = "Flagged range MMF"
    if np.any(bad):
        plt.plot(x[bad], y[bad], **bad_kwargs)

def plot_g(particle, logx=True, title="Asymmetry parameter g"):

    plt.figure(figsize=(10, 6))
    plt.rc("font", family="serif", size=12)

    def _get_first_result(obj, attr):
        value = getattr(obj, attr, None)
        if value is None:
            return None
        if hasattr(value, "__len__") and len(value) == 0:
            return None
        return value[0]

    def _extract_arrays(df):
        if hasattr(df, "columns"):
            if "wavelength_um" not in df.columns or "g" not in df.columns:
                return None, None
            lam = df["wavelength_um"].to_numpy(dtype=float)
            gval = df["g"].to_numpy(dtype=float)
        else:
            if "wavelength_um" not in df or "g" not in df:
                return None, None
            lam = np.asarray(df["wavelength_um"], dtype=float)
            gval = np.asarray(df["g"], dtype=float)
        return lam, gval

    plotted = False

    # Optool
    df = _get_first_result(particle, "Optool_results")
    if df is not None:
        lam, gval = _extract_arrays(df)
        if lam is not None:
            plt.plot(lam, gval, linewidth=2.5, label="Optool")
            plotted = True

    # Scatt
    df = _get_first_result(particle, "Scatt_results")
    if df is not None:
        lam, gval = _extract_arrays(df)
        if lam is not None:
            plt.scatter(lam, gval, s=25, label="ML EMT", zorder=3)
            plotted = True

    # MMF
    lam_bad = mmf_bad_lambda_um(particle) # checks bad ranges
    
    # MMF
    df = _get_first_result(particle, "MMF_results")
    if df is not None:
        lam, gval = _extract_arrays(df)
        if lam is not None:
            plot_mmf_with_warning_range(
                lam, gval,
                lam_bad_um=lam_bad,
                is_wavelength_x=True,
                linestyle="--",
                linewidth=2.0,
                label="MMF",
            )
            plotted = True

    if not plotted:
        raise ValueError("No valid g data found in this particle")

    if logx:
        plt.xscale("log")

    plt.xlabel("Wavelength micrometer")
    plt.ylabel("Asymmetry parameter g")
    plt.title(title)

    plt.grid(which="both", linestyle=":", linewidth=0.8, alpha=0.6)
    plt.minorticks_on()
    plt.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.show()


def _pick_nearest_wavelength(df, wl_target, wl_col="wavelength_um"):
    wls = np.asarray(df[wl_col].unique(), float)
    if wls.size == 0:
        raise ValueError("No wavelengths found in dataframe.")
    i = int(np.argmin(np.abs(wls - float(wl_target))))
    return float(wls[i])

def _select_wavelength(df, wl_target, wl_col="wavelength_um", tol=None):
    """
    Returns (subset_df, wl_used)

    If tol is given: tries isclose match to wl_target first.
    If that yields empty: snaps to nearest wavelength present in df.
    """
    arr = df[wl_col].to_numpy(dtype=float)

    if tol is not None:
        mask = np.isclose(arr, float(wl_target), rtol=0.0, atol=float(tol))
        out = df.loc[mask]
        if not out.empty:
            return out, float(wl_target)

    wl_use = _pick_nearest_wavelength(df, wl_target, wl_col=wl_col)
    mask = np.isclose(arr, wl_use, rtol=0.0, atol=0.0)
    out = df.loc[mask]
    return out, wl_use

def _get_result_df(particle, attr_name, idx=1):
    """
    Returns results[idx] if it exists else None.
    """
    res = getattr(particle, attr_name, None)
    if res is None:
        return None
    try:
        if len(res) <= idx:
            return None
        return res[idx]
    except Exception:
        return None

def plot_matrix(
    particle,
    element="F11",
    wavelength=None,
    tol_um=1e-6,
    title=None,
    logy=True,
    save=None,
):
    """
    Plot a Mueller matrix element vs scattering angle for a single Particle.

    Uses, if present:
      particle.Optool_results[1]
      particle.Scatt_results[1]
      particle.MMF_results[1]

    Each dataframe should contain:
      wavelength_um, angle_deg, and the requested element column.
    """

    df_opt = _get_result_df(particle, "Optool_results", idx=1)
    df_sca = _get_result_df(particle, "Scatt_results", idx=1)
    df_mmf = _get_result_df(particle, "MMF_results", idx=1)

    if df_opt is None and df_sca is None and df_mmf is None:
        raise ValueError("No matrix dataframe found. Need Optool_results[1], Scatt_results[1], or MMF_results[1].")

    # choose reference wavelength source
    df_ref = df_opt if df_opt is not None else (df_sca if df_sca is not None else df_mmf)

    if wavelength is None:
        wl_ref = float(np.sort(df_ref["wavelength_um"].unique())[0])
    else:
        wl_ref = float(wavelength)
        
    lam_bad = mmf_bad_lambda_um(particle)

    def _prepare(df):
        if df is None:
            return None, None, None
        needed = {"wavelength_um", "angle_deg", element}
        if hasattr(df, "columns") and not needed.issubset(set(df.columns)):
            return None, None, None
        m, wl_used = _select_wavelength(df, wl_ref, tol=tol_um)
        if m.empty:
            return None, None, None
        m = m.sort_values("angle_deg")
        angles = m["angle_deg"].to_numpy(dtype=float)
        F = m[element].to_numpy(dtype=float)
        return angles, F, wl_used

    ang_opt, F_opt, wl_opt = _prepare(df_opt)
    ang_sca, F_sca, wl_sca = _prepare(df_sca)
    ang_mmf, F_mmf, wl_mmf = _prepare(df_mmf)

    # decide base angles from first available dataset
    base_angles = ang_opt if ang_opt is not None else (ang_sca if ang_sca is not None else ang_mmf)
    if base_angles is None:
        raise ValueError(f"No data found near λ={wl_ref:g} µm for element {element}.")

    def _align(angles, F):
        if angles is None or F is None:
            return None
        if angles.size == base_angles.size and np.allclose(angles, base_angles):
            return F
        return np.interp(base_angles, angles, F)

    F_opt_a = _align(ang_opt, F_opt)
    F_sca_a = _align(ang_sca, F_sca)
    F_mmf_a = _align(ang_mmf, F_mmf)

    if F_opt_a is None and F_sca_a is None and F_mmf_a is None:
        raise ValueError(f"Nothing to plot for element {element} at λ≈{wl_ref:g} µm.")

    plt.figure(figsize=(15, 4.5))
    plt.rc("font", family="serif", size=12)

    labeled = {"optool": False, "scatt": False, "mmf": False}

    # Optool
    if F_opt_a is not None:
        plt.plot(base_angles, F_opt_a, color="black", lw=3.0, zorder=1)
        plt.plot(
            base_angles, F_opt_a,
            color="tab:blue", lw=2.4, zorder=2,
            label=("EMT" if not labeled["optool"] else None),
        )
        labeled["optool"] = True

    # Scatt
    if F_sca_a is not None:
        plt.scatter(base_angles, F_sca_a, color="black", s=35, zorder=3)
        plt.scatter(
            base_angles, F_sca_a,
            s=25, color="tab:blue", zorder=4,
            label=("ML EMT" if not labeled["scatt"] else None),
        )
        labeled["scatt"] = True

    # MMF
# MMF
    if F_mmf_a is not None:
        mmf_color = "tab:red" if (lam_bad is not None and wl_ref <= lam_bad) else "tab:orange"
        mmf_label = "Flagged MMF" if (lam_bad is not None and wl_ref <= lam_bad) else "MMF"
        plt.plot(
            base_angles, F_mmf_a,
            color=mmf_color, lw=2.0, linestyle="--", zorder=3,
            label=(mmf_label if not labeled["mmf"] else None),
        )
        labeled["mmf"] = True

    if logy:
        # switch to linear automatically if any plotted values are non positive
        plotted_vals = []
        if F_opt_a is not None:
            plotted_vals.append(F_opt_a)
        if F_sca_a is not None:
            plotted_vals.append(F_sca_a)
        if F_mmf_a is not None:
            plotted_vals.append(F_mmf_a)
        vals = np.concatenate(plotted_vals) if plotted_vals else np.array([1.0])
        if np.any(~np.isfinite(vals)) or np.any(vals <= 0):
            pass
        else:
            plt.yscale("log")

    plt.xlabel("Scattering angle (deg)", fontsize=axis_size)
    plt.ylabel(f"{element}(θ)", fontsize=axis_size)
    plt.title(title if title is not None else f"{element} at λ≈{wl_ref:g} µm", fontsize=axis_size)

    plt.tick_params(axis="both", which="major", labelsize=ticksize)
    plt.grid(which="both", linestyle=":", linewidth=0.8, alpha=0.6)
    plt.minorticks_on()
    plt.legend(fontsize=legend_size, frameon=False)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
   
   
################################################################################################
# As part of the project, i made sizx panel figures 
# I added them as reference so they could be used to make your own figures
# but did not want to include them as actual functions!
################################################################################################


# def _nearest_wavelength(df, wavelength_um):
#     lams = np.sort(df["wavelength_um"].unique().astype(float))
#     idx = int(np.argmin(np.abs(lams - float(wavelength_um))))
#     return float(lams[idx])


# def _get_triplet(particle, source):
#     if source == "emt":
#         return particle.Optool_results
#     if source == "scatt":
#         return particle.Scatt_results
#     if source == "mmf":
#         return particle.MMF_results
#     raise ValueError("source must be emt, scatt, or mmf")


# def _plot_matrix_element_three_sources_on_ax(
#     ax,
#     particle,
#     element="F11",
#     wavelength_um=1.65,
#     title=None,
#     show_legend=True,
#     normalize_by_F11=False,
#     ylog=False,
#     which=("emt", "scatt", "mmf"),
# ):
#     # style mapping: all black, different linestyles and markers
#     styles = {
#         "emt":   dict(color="black", lw=lw, ls="-",  marker=None, label="EMT"),
#         "scatt": dict(color="blue", lw=0,  ls="None", marker="o", ms=4.5, label="ML-EMT"),
#         "mmf":   dict(color="black", lw=lw, ls="--", marker=None, label="MMF"),
#     }

#     any_plotted = False
#     chosen_wv = {}

#     for src in which:
#         kappas_df, matrix_df, _ = _get_triplet(particle, src)

#         if matrix_df is None or matrix_df.empty:
#             continue
#         if element not in matrix_df.columns:
#             continue

#         wv = _nearest_wavelength(matrix_df, wavelength_um)
#         dfw = matrix_df.loc[matrix_df["wavelength_um"] == wv].copy()

#         ang = dfw["angle_deg"].to_numpy(dtype=float)
#         y = dfw[element].to_numpy(dtype=float)

#         if normalize_by_F11 and element != "F11":
#             denom = dfw["F11"].to_numpy(dtype=float)
#             y = y / denom

#         order = np.argsort(ang)
#         ang = ang[order]
#         y = y[order]

#         st = styles[src]
#         if st["marker"] is None:
#             ax.plot(ang, y, st["ls"], color=st["color"], lw=st["lw"],
#                     label=f'{st["label"]}')
#         else:
#             ax.plot(ang, y, linestyle="None", marker=st["marker"],
#                     color=st["color"], markersize=st["ms"],
#                     label=f'{st["label"]} ')

#         any_plotted = True
#         chosen_wv[src] = wv

#     if not any_plotted:
#         raise ValueError("Geen matrix data gevonden om te plotten voor de gekozen bronnen.")

#     ax.set_xlabel("Scattering angle (deg)", fontsize=ticksize + 2)
#     ax.tick_params(axis="both", which="major", labelsize=ticksize)

#     if ylog:
#         ax.set_yscale("log")

#     if title:
#         ax.set_title(title, fontsize=ticksize + 2)

#     if show_legend:
#         ax.legend(fontsize=legend_size, frameon=False)


# def plot_matrix_element_six_panel_compare(
#     particles,
#     wavelength_um,
#     element="F11",
#     titles=None,
#     which=("emt", "scatt", "mmf"),
#     normalize_by_F11=False,
#     ylog=False,
#     save=None,
#     sharex=True,
#     sharey=True,
#     legend_mode="first",  # first, all, none
# ):
#     if len(particles) != 6:
#         raise ValueError(f"Expected exactly 6 particles, got {len(particles)}")

#     if titles is None:
#         titles = [f"Particle {i+1}" for i in range(6)]
#     if len(titles) != 6:
#         raise ValueError(f"Expected exactly 6 titles, got {len(titles)}")

#     fig, axes = plt.subplots(
#         nrows=3,
#         ncols=2,
#         figsize=(12, 14),
#         sharex=sharex,
#         sharey=sharey,
#         constrained_layout=True,
#     )

#     axes_flat = axes.ravel()

#     for i, (ax, p, t) in enumerate(zip(axes_flat, particles, titles)):
#         if legend_mode == "all":
#             show_legend = True
#         elif legend_mode == "none":
#             show_legend = False
#         else:
#             show_legend = (i == 0)

#         _plot_matrix_element_three_sources_on_ax(
#             ax,
#             p,
#             element=element,
#             wavelength_um=wavelength_um,
#             title=t,
#             show_legend=show_legend,
#             normalize_by_F11=normalize_by_F11,
#             ylog=ylog,
#             which=which,
#         )

#     fig.suptitle(
#         f"Matrix element {element} comparison at {wavelength_um:g} um",
#         fontsize=axis_size
#     )

#     if save:
#         fig.savefig(save, dpi=300, bbox_inches="tight")

#     plt.show()
#     return fig, axes


# def _plot_ksca_on_ax(ax, particle, title=None, show_legend=True):
#     def _ksca_arrays(df):
#         lam = df["wavelength_um"].to_numpy(dtype=float)
#         ksca = df["ksca_cm2g"].to_numpy(dtype=float)
#         return lam, ksca

#     if getattr(particle, "EMT_results", None) and particle.EMT_results[0] is not None:
#         df = particle.EMT_results[0]
#         lam, y = _ksca_arrays(df)
#         ax.plot(lam, y, label=r"EMT $\kappa_{\mathrm{sca}}$", linewidth=lw, color="black")

#     if getattr(particle, "Scatt_results", None) and particle.Scatt_results[0] is not None:
#         df = particle.Scatt_results[0]
#         lam, y = _ksca_arrays(df)
#         ax.scatter(lam, y, color="black", s=35, zorder=3)
#         ax.scatter(lam, y, label=r"ML EMT $\kappa_{\mathrm{sca}}$", s=25, zorder=4)

#     if getattr(particle, "MMF_results", None) and particle.MMF_results[0] is not None:
#         df = particle.MMF_results[0]
#         lam, y = _ksca_arrays(df)
#         ax.plot(lam, y, linestyle="--", linewidth=lw, color="black", label=r"MMF $\kappa_{\mathrm{sca}}$")

#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.tick_params(axis="both", which="major", labelsize=ticksize)

#     if title:
#         ax.set_title(title, fontsize=ticksize + 2)

#     if show_legend:
#         ax.legend(fontsize=legend_size, frameon=False)


# def plot_ksca_six_panel(
#     particles,
#     titles=None,
#     save=None,
#     sharex=True,
#     sharey=True,
#     legend_mode="first",  # "first", "all", "none"
# ):
#     if len(particles) != 6:
#         raise ValueError(f"Expected exactly 6 particles, got {len(particles)}")

#     if titles is None:
#         titles = [f"Particle {i+1}" for i in range(6)]
#     if len(titles) != 6:
#         raise ValueError(f"Expected exactly 6 titles, got {len(titles)}")

#     fig, axes = plt.subplots(
#         nrows=3,
#         ncols=2,
#         figsize=(12, 14),
#         sharex=sharex,
#         sharey=sharey,
#         constrained_layout=True,
#     )

#     axes_flat = axes.ravel()

#     for i, (ax, p, t) in enumerate(zip(axes_flat, particles, titles)):
#         if legend_mode == "all":
#             show_legend = True
#         elif legend_mode == "none":
#             show_legend = False
#         else:
#             show_legend = (i == 0)

#         _plot_ksca_on_ax(ax, p, title=t, show_legend=show_legend)

#     fig.supxlabel("Wavelength (Micron)", fontsize=axis_size)
#     fig.supylabel(r"$\kappa_{\mathrm{sca}}$ (cm$^2$ g$^{-1}$)", fontsize=axis_size)

#     if save:
#         fig.savefig(save, dpi=300, bbox_inches="tight")

#     plt.show()
#     return fig, axes


# def _plot_g_on_ax(ax, particle, title=None, show_legend=True):
#     def _g_arrays(df):
#         lam = df["wavelength_um"].to_numpy(dtype=float)
#         g = df["g"].to_numpy(dtype=float)
#         return lam, g

#     plotted_any = False

#     if getattr(particle, "EMT_results", None) and particle.EMT_results[0] is not None:
#         df = particle.EMT_results[0]
#         if "g" in df.columns:
#             lam, y = _g_arrays(df)
#             ax.plot(lam, y, label="EMT g", linewidth=lw, color="black")
#             plotted_any = True

#     if getattr(particle, "Scatt_results", None) and particle.Scatt_results[0] is not None:
#         df = particle.Scatt_results[0]
#         if "g" in df.columns:
#             lam, y = _g_arrays(df)
#             ax.scatter(lam, y, color="black", s=35, zorder=3)
#             ax.scatter(lam, y, label="ML EMT g", s=25, zorder=4)
#             plotted_any = True

#     if getattr(particle, "MMF_results", None) and particle.MMF_results[0] is not None:
#         df = particle.MMF_results[0]
#         if "g" in df.columns:
#             lam, y = _g_arrays(df)
#             ax.plot(lam, y, linestyle="--", linewidth=lw, color="black", label="MMF g")
#             plotted_any = True

#     ax.set_xscale("log")
#     ax.tick_params(axis="both", which="major", labelsize=ticksize)

#     if title:
#         ax.set_title(title, fontsize=ticksize + 2)

#     if plotted_any and show_legend:
#         ax.legend(fontsize=legend_size, frameon=False)


# def plot_g_six_panel(
#     particles,
#     titles=None,
#     save=None,
#     sharex=True,
#     sharey=True,
#     legend_mode="first",  # "first", "all", "none"
# ):
#     if len(particles) != 6:
#         raise ValueError(f"Expected exactly 6 particles, got {len(particles)}")

#     if titles is None:
#         titles = [f"Particle {i+1}" for i in range(6)]
#     if len(titles) != 6:
#         raise ValueError(f"Expected exactly 6 titles, got {len(titles)}")

#     fig, axes = plt.subplots(
#         nrows=3,
#         ncols=2,
#         figsize=(12, 14),
#         sharex=sharex,
#         sharey=sharey,
#         constrained_layout=True,
#     )

#     axes_flat = axes.ravel()

#     for i, (ax, p, t) in enumerate(zip(axes_flat, particles, titles)):
#         if legend_mode == "all":
#             show_legend = True
#         elif legend_mode == "none":
#             show_legend = False
#         else:
#             show_legend = (i == 0)

#         _plot_g_on_ax(ax, p, title=t, show_legend=show_legend)

#     fig.supxlabel("Wavelength (Micron)", fontsize=axis_size)
#     fig.supylabel("g", fontsize=axis_size)

#     if save:
#         fig.savefig(save, dpi=300, bbox_inches="tight")

#     plt.show()
#     return fig, axes
