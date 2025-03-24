import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import bruges as bg
from scipy.interpolate import interp1d
import copy

class SeismicModeler:
    def __init__(self, excel_file, dt_time, wavelet_frequency, lithology='Sandstone', velocity_in_km_s=False):
        self.excel_file = excel_file
        self.velocity_in_km_s = velocity_in_km_s
        self.dt_time = dt_time / 1000  # Convert input dt (ms) to seconds
        self.wavelet_frequency = wavelet_frequency
        self.lithology = lithology

        # Data holders
        self.x = None
        self.y = None
        self.vp = None
        self.rhob = None

        # Results
        self.impedance = None
        self.reflection_depth = None
        self.seismic_depth = None
        self.R_time = None
        self.t_axis = None
        self.seismic_time = None
        self.impedance_time = None  # Added for impedance in time
        self.fluild_substution = None

        self.load_data()

    def load_data(self):
        """Load data from Excel sheets."""
        x_data = pd.read_excel(self.excel_file, sheet_name='X', header=None).values
        y_data = pd.read_excel(self.excel_file, sheet_name='Y', header=None).values

        self.x = x_data[0, :] if x_data.ndim == 2 else x_data.ravel()
        self.y = y_data[:, 0] if y_data.ndim == 2 else y_data.ravel()

        self.vp = pd.read_excel(self.excel_file, sheet_name='VP', header=None).values
        self.rhob = pd.read_excel(self.excel_file, sheet_name='RHOB', header=None).values

        if self.velocity_in_km_s:
            self.vp *= 1000.0  # Convert km/s to m/s
        
        if self.lithology == 'Sandstone':
            self.a, self.b, self.c = 0, 0.80416, -0.85588 
            self.rho_m = 2650
            self.K0 = 37                                                                # GPa   [Bulk Modulus, matrix]
        
        elif self.lithology == 'Limestone':
            self.a, self.b, self.c = -0.05508, 1.01677 , -1.03049 
            self.rho_m = 2710
            self.K0 = 63.7                                                              # GPa   [Bulk Modulus, matrix]
        
        elif self.lithology == 'Dolomite':
            self.a, self.b, self.c = 0, 0.58321, -0.07775 
            self.rho_m = 2840
            self.K0 = 76.8                                                              # GPa   [Bulk Modulus, matrix]
        
        else:
            raise NameError("The provided lithology is not supported!")
        

        print("Data shapes:")
        print(f"  x: {self.x.shape}, y: {self.y.shape}, vp: {self.vp.shape}, rhob: {self.rhob.shape}")
        print(f"Interval sampling time: {self.dt_time} seconds")
        print(f"Wavelet frequency: {self.wavelet_frequency} Hz")

    def compute_impedance(self):
        """Compute acoustic impedance."""
        self.impedance = self.vp * self.rhob

    def compute_reflectivity_in_depth(self):
        """Compute reflection coefficients in depth domain."""
        # Use 'same' mode to maintain layer count for plotting alignment
        self.reflection_depth = bg.reflection.acoustic_reflectivity(vp=self.vp, rho=self.rhob, mode='same')

        return self.reflection_depth

    def convert_time_to_depth(self):
        """Convert synthetic seismic from time domain back to depth domain.
        
        For each trace, the method:
        - Computes the TWT for each depth sample (using the velocity log).
        - Interpolates the time-domain synthetic seismic amplitude (defined on self.t_axis)
            at the computed TWT values.
        """
        nd = len(self.y)
        n_traces = self.vp.shape[1]
        seismic_depth = np.zeros((nd, n_traces))
        
        for i in range(n_traces):
            # (1) Compute TWT at each depth sample
            t_col = self.depth_to_time_single_trace(self.vp[:, i], process='Forward')  # skip i=0 indexing error
            # (2) Interpolate the time-domain amplitude
            f_amp = interp1d(self.t_axis, self.seismic_time[:, i],
                            kind='cubic', bounds_error=False, fill_value=0.0)
            seismic_depth[:, i] = f_amp(t_col)

        self.seismic_depth = seismic_depth
        return seismic_depth
    
    def generate_seismic_in_depth(self):
        """Generate depth-domain synthetic seismic via time-domain convolution and time-to-depth conversion.

        Workflow:
        1. Generate the time-domain synthetic seismic (using generate_seismic_in_time).
        2. Convert the time-domain seismic back to depth domain using a TWT-to-depth mapping.
        """
        # Compute time-domain synthetic seismic if not already available.
        if self.seismic_time is None:
            self.generate_seismic_in_time()
        # Convert the time-domain result back to the depth domain.
        self.convert_time_to_depth()

        return self.seismic_depth

    def depth_to_time_single_trace(self, vp_col, process='Forward'):
        """Convert depth to TWT (in seconds) for a single trace.
        
        This method computes the cumulative two-way travel time at each depth sample.
        """
        dz = np.abs(np.diff(self.y))  # depth intervals

        if process == 'Forward':
            self.t = np.zeros(len(vp_col))

            for i in range(1, len(vp_col)):
                v_avg = 0.5 * (vp_col[i] + vp_col[i-1])
                dt_i = 2 * dz[i-1] / v_avg
                self.t[i] = self.t[i-1] + dt_i
            return self.t

        elif process == 'Backward':
            self.depth = np.zeros(len(vp_col))
            for i in range(1, len(vp_col)):
                v_avg = 0.5 * (vp_col[i] + vp_col[i-1])
                dz_i = self.t[i-1] * v_avg / 2
                self.depth[i] = self.depth[i-1] + dz_i

            return self.depth

    def generate_seismic_in_time(self):
        """Generate time-domain synthetic seismic without interpolating the RC values.

        For each trace:
        1. Compute TWT (using midpoints) for each reflection coefficient.
        2. Build a new RC_time array on a uniform time axis defined by np.arange(start, end, dt_time).
        3. Insert each RC value into the RC_time array at the index corresponding to its TWT.
        4. Convolve the RC_time trace with the wavelet to produce the time-domain synthetic.
        """
        if self.reflection_depth is None:
            self.compute_reflectivity_in_depth()

        n_traces = self.reflection_depth.shape[1]
        TWT_all, RC_all = [], []

        # For each trace, compute the midpoints (TWT) corresponding to the RC values.
        for i in range(n_traces):
            vp_col = self.vp[:, i]
            t_col = self.depth_to_time_single_trace(vp_col, process='Forward')  # seconds
            TWT_all.append(t_col)
            RC_all.append(self.reflection_depth[:, i])

        # Determine overall min and max TWT across all traces
        overall_min_twt = min(t.min() for t in TWT_all)
        overall_max_twt = max(t.max() for t in TWT_all)
        self.before_gassman = overall_max_twt
        self.after_gassman = overall_max_twt

        self.wavelet_duration = overall_max_twt + self.dt_time                                      # seconds

        if self.fluild_substution is None:
            print(f"Minimum TWT: {round(overall_min_twt * 1000, 3)} ms")
            print(f"Maximum TWT: {round(overall_max_twt * 1000, 3)} ms")
        
        else:
            if self.fluild_substution:
                pass

        # Create a uniform time axis using np.arange with dt_time increment
        self.t_axis = np.arange(0, overall_max_twt + self.dt_time, self.dt_time)

        # Create RC_time array with zeros (dimensions: time samples x number of traces)
        self.R_time = np.zeros((len(self.t_axis), n_traces))

        # Insert each RC value into the RC_time array at the time index closest to its TWT.
        for i in range(n_traces):
            for rc_val, twt_val in zip(RC_all[i], TWT_all[i]):
                # Find the index in self.t_axis that is closest to the current TWT value
                idx = np.argmin(np.abs(self.t_axis - twt_val))
                self.R_time[idx, i] = rc_val

        # Generate the wavelet with the given dt (dt_time)
        w, _ = bg.filters.ricker(duration=self.wavelet_duration, dt=self.dt_time, f=self.wavelet_frequency)

        # Convolve the RC_time array with the wavelet to produce the time-domain synthetic seismic.
        self.seismic_time = np.zeros_like(self.R_time)
        for i in range(n_traces):
            self.seismic_time[:, i] = np.convolve(w, self.R_time[:, i], mode='same')

        return self.seismic_time
    
    def BiottGassman(self, avg_exp_Phi=0.192, avg_exp_Sw=0.15, avg_exp_So=0.85, avg_exp_oil_rho=700, avg_exp_gas_rho=150, avg_exp_W_rho=1000,
                     avg_exp_final_Sw=0.3, avg_exp_final_So=0.3, Khc=1.8, Kg=0.041, Kw=2.5, update=True, plot=True):
        
        def compute_K_dry(K0, K_sati, phi, K_fli):
            """
            Compute K_dry.

            :param K0:     K0 (bulk modulus of the matrix/mineral)
            :param K_sati:  K_sati (bulk modulus under saturated conditions)
            :param phi:     Porosity (phi)
            :param K_fli:   K_fli (bulk modulus of the fluid)
            :return:        K_dry (bulk modulus of the dry rock frame)
            """

            A = (K_sati / (K0 - K_sati)) - (1.0 / phi) * (K_fli / (K0 - K_fli))
            
            return (K0 * A) / (1.0 + A)
        
        def compute_Ksat_f(K0, Kdry, phi, K_flf):
            """
            Compute K_sat_f.

            :param K0:       K0 (bulk modulus of the solid matrix)
            :param Kdry:     Kdry (bulk modulus of the dry frame)
            :param phi:      Porosity
            :param K_flf:    K_flf (bulk modulus of the fluid)
            :return:         K_sat_f (bulk modulus under the new fluid saturation)
            """

            # Define R
            R = (Kdry / (K0 - Kdry)) + (1.0 / phi) * (K_flf / (K0 - K_flf))
            
            # Compute K_sat_f
            return (K0 * R) / (1.0 + R)
        
        Ko_pa = np.full(self.vp.shape, Khc * 10**9)                                                         # Pa   [Bulk Modulus, oil]
        Kw_pa = np.full(self.vp.shape, Kw * 10**9)                                                          # Pa   [Bulk Modulus, Water]
        Kg_pa = np.full(self.vp.shape, Kg * 10**9)                                                          # Pa   [Bulk Modulus, gas]

        # Porosity grid
        porosity = np.full(self.vp.shape, avg_exp_Phi)

        # Initial condition grid
        water_saturation_initial = np.full(self.vp.shape, avg_exp_Sw)                                       # -    [Water Saturation, initial]
        Oil_saturation_initial = np.full(self.vp.shape, avg_exp_So)                                         # -    [Oil Saturation, initial]
        Gas_saturation_initial = 1- (water_saturation_initial + Oil_saturation_initial)                     # -    [Gas Saturation, initial]

        Vp_i = self.vp                                                                                      # m/s  [Compressional Velocity, initial]
        Vp_kms = Vp_i / 1000                                                                                # km/s [Compressional Velocity, initial]
        Vs_kms = (self.a * (Vp_kms ** 2)) + (self.b * Vp_kms) + self.c                                      # km/s [Shear Velocity, initial]
        Vs_i = Vs_kms * 1000                                                                                # m/s  [Shear Velocity, initial]

        G_i = (Vs_i ** 2) * self.rhob                                                                       # Pa  [Shear Modulus, initial]    
        Ksat_i = (Vp_i ** 2) * self.rhob - (4/3) * G_i                                                      # Pa  [Bulk Modulus, Saturated initial]

        # Solve for Bulk Modulus of initial fluid mixture and matrix
        Kfl_i = (Ko_pa * Kw_pa) / (Oil_saturation_initial * Kw_pa + water_saturation_initial * Ko_pa)       # Pa   [Bulk Modulus Fluid Mixture, Initial]
        Kdry = compute_K_dry(K0=self.K0*10**9, K_sati=Ksat_i, phi=porosity, K_fli=Kfl_i)                    # Pa   [Bulk Modulus, dry]

        # Final condition grid
        water_saturation_final = np.full(self.vp.shape, avg_exp_final_Sw)
        Oil_saturation_final = np.full(self.vp.shape, avg_exp_final_So)                                      # -    [Oil Saturation, final]
        Gas_saturation_final = 1- (water_saturation_final + Oil_saturation_final)                            # -    [Gas Saturation, final]

        rhob_final = porosity * (water_saturation_final * avg_exp_W_rho + Oil_saturation_final * avg_exp_oil_rho + Gas_saturation_final * avg_exp_gas_rho) + (1 - porosity) * self.rho_m

        # Compute final properties
        Kfl_f = (Ko_pa * Kw_pa * Kg_pa) / (Oil_saturation_final * Kw_pa * Kg_pa + water_saturation_final * Ko_pa * Kg_pa + Gas_saturation_final * Ko_pa * Kw_pa)         # Pa   [Bulk Modulus Fluid Mixture, final]
        Ksat_f = compute_Ksat_f(K0=self.K0*10**9, Kdry=Kdry, phi=porosity, K_flf=Kfl_f)                     # Pa   [Bulk Modulus, Saturated Final]

        Vp_f = np.sqrt((Ksat_f + ((4/3) * G_i)) / rhob_final)                                               # m/s  [Compressional Velocity, final]
        Vs_f = np.sqrt(G_i / rhob_final)                                                                    # m/s  [Shear Velocity, final]

        if update:
            self.fluild_substution = True
            print("*********** Modelling Results ***********")
            print(f"Kdry: {round(Kdry.mean() * 1e-9, 2)} GPa")
            print(f"Initial Average Vp: {round(Vp_i.mean(), 2)} m/s")
            print(f"Initial Average Vs: {round(Vs_i.mean(), 2)} m/s")
            print(f"Initial Ksat: {round(Ksat_i.mean() * 1e-9, 2)} GPa")
            print(f"Initial Kfluid: {round(Kfl_i.mean() * 1.e-9, 2)} GPa")
            print(f"Initial Average Density: {round(self.rhob.mean(), 2)} kg/m3")
            print(f"Initial Maximum TWT: {round(self.before_gassman * 1000, 3)} ms")

            print("----------------------------------------------")

            print(f"Final Average Vp: {round(Vp_f.mean(), 2)} m/s")
            print(f"Final Average Vs: {round(Vs_f.mean(), 2)} m/s")
            print(f"Final Ksat: {round(Ksat_f.mean() * 1e-9, 2)} GPa")
            print(f"Final Kfluid: {round(Kfl_f.mean() * 1.e-9, 2)} GPa")
            print(f"Final Average Density: {round(rhob_final.mean(), 2)} kg/m3")

            self._update_model(Vp_f=Vp_f, rhob_final=rhob_final, plot=plot)
            print(f"Final Maximum TWT: {round(self.after_gassman * 1000, 3)} ms")

    def _update_model(self, Vp_f, rhob_final, plot):
        self.vp = Vp_f
        self.rhob = rhob_final

        self.compute_impedance()
        self.compute_reflectivity_in_depth()
        self.generate_seismic_in_time()  # Generate time synthetic
        self.convert_time_to_depth()     # Convert to depth
        
        if plot:
            self.plot_results()                   


    def plot_results(self):
        """Plot results with TWT in milliseconds."""
        if self.impedance is None:
            self.compute_impedance()

        # 1) Create a 4x1 figure instead of 2x2
        fig, axs = plt.subplots(4, 1, figsize=(10, 12))

        # 2) Acoustic Impedance
        vp_km_s = self.vp * 1e-3 if not self.velocity_in_km_s else self.vp
        rho_g_cc = self.rhob * 1e-3
        Z = vp_km_s * rho_g_cc

        divider1 = make_axes_locatable(axs[0])
        cax1 = divider1.append_axes("right", size="3%", pad=0.2)

        im1 = axs[0].imshow(
            Z, 
            aspect='equal', 
            cmap='jet',
            extent=[self.x[0], self.x[-1], self.y[-1], self.y[0]],
            vmin=Z.min(), 
            vmax=Z.max(),
            resample=False
        )
        axs[0].set_title("Acoustic Impedance (AI)")
        axs[0].set_xlabel("X Coordinate")
        axs[0].set_ylabel("Elevation (m)")

        cbar1 = fig.colorbar(im1, cax=cax1)
        cbar1.set_label("AI (km/s·g/cm³)", rotation=270, labelpad=10)

        # 3) Reflection Coefficients
        divider2 = make_axes_locatable(axs[1])
        cax2 = divider2.append_axes("right", size="3%", pad=0.2)

        im2 = axs[1].imshow(
            self.R_time, 
            aspect='auto', 
            cmap='seismic',
            extent=[self.x[0], self.x[-1], self.t_axis[-1]*1000*1.1, self.t_axis[0]*1000],
            vmin=self.R_time.min(), 
            vmax=self.R_time.max(),
            resample=False
        )
        axs[1].set_title("Reflection Coefficients (RC)")
        axs[1].set_xlabel("X Coordinate")
        axs[1].set_ylabel("TWT (ms)")

        cbar2 = fig.colorbar(im2, cax=cax2)
        cbar2.set_label("RC", rotation=270, labelpad=10)

        # 4) Depth-domain Seismic
        divider3 = make_axes_locatable(axs[2])
        cax3 = divider3.append_axes("right", size="3%", pad=0.2)

        im3 = axs[2].imshow(
            self.seismic_depth, 
            aspect='equal', 
            cmap='seismic',
            extent=[self.x[0], self.x[-1], self.y[-1], self.y[0]],
            interpolation='bicubic',
            vmin=self.seismic_depth.min(), 
            vmax=self.seismic_depth.max(),
            resample=False
        )
        axs[2].set_title("Depth-domain Synthetic")
        axs[2].set_xlabel("X Coordinate")
        axs[2].set_ylabel("Depth (m)")
        axs[2].invert_yaxis()

        cbar3 = fig.colorbar(im3, cax=cax3)
        cbar3.set_label("Amplitude", rotation=270, labelpad=10)

        # 5) Time-domain Seismic
        divider4 = make_axes_locatable(axs[3])
        cax4 = divider4.append_axes("right", size="3%", pad=0.2)

        im4 = axs[3].imshow(
            self.seismic_time, 
            aspect='auto', 
            cmap='seismic',
            extent=[self.x[0], self.x[-1], self.t_axis[-1]*1000*1.1, self.t_axis[0]*1000],
            interpolation='bicubic',
            vmin=self.seismic_time.min(), 
            vmax=self.seismic_time.max(),
            resample=False
        )
        axs[3].set_title("Time-domain Synthetic")
        axs[3].set_xlabel("X Coordinate")
        axs[3].set_ylabel("TWT (ms)")

        cbar4 = fig.colorbar(im4, cax=cax4)
        cbar4.set_label("Amplitude", rotation=270, labelpad=10)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_comparison(original, substituted):
        """Plots comparing original and substituted models."""
        original_imp = original.impedance * 1e-6
        substituted_imp = substituted.impedance * 1e-6

        fig, axs = plt.subplots(5, 3, figsize=(20, 10))
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        # Common parameters
        cmap_seismic = 'seismic'
        cmap_jet = 'jet'
        extent_depth = [original.x[0], original.x[-1], original.y[-1], original.y[0]]
        vmin_vp = min(original.vp.min(), substituted.vp.min())
        vmax_vp = max(original.vp.max(), substituted.vp.max())
        vmin_rhob = min(original.rhob.min(), substituted.rhob.min())
        vmax_rhob = max(original.rhob.max(), substituted.rhob.max())
        vmin_imp = min(original_imp.min(), substituted_imp.min())
        vmax_imp = max(original_imp.max(), substituted_imp.max())

        # Vp
        divider1 = make_axes_locatable(axs[0, 0])
        cax1 = divider1.append_axes("right", size="3%", pad=0.2)

        im1 = axs[0,0].imshow(original.vp, aspect='equal', cmap=cmap_jet, extent=extent_depth, vmin=vmin_vp, vmax=vmax_vp)
        axs[0,0].set_title("Original Vp")
        cbar1 = plt.colorbar(im1, cax=cax1)
        axs[0, 0].set_xlabel("X Coordinate")
        axs[0, 0].set_ylabel("Depth (m)")
        axs[0, 0].invert_yaxis()
        cbar1.set_label("Vp (m/s)", rotation=270, labelpad=10)

        divider2 = make_axes_locatable(axs[0, 1])
        cax2 = divider2.append_axes("right", size="3%", pad=0.2)

        im2 = axs[0, 1].imshow(substituted.vp, aspect='equal', cmap=cmap_jet, extent=extent_depth, vmin=vmin_vp, vmax=vmax_vp)
        axs[0, 1].set_title("Substituted Vp")
        cbar2 = plt.colorbar(im2, cax=cax2)
        axs[0, 1].set_xlabel("X Coordinate")
        axs[0, 1].set_ylabel("Depth (m)")
        axs[0, 1].invert_yaxis()
        cbar2.set_label("Vp (m/s)", rotation=270, labelpad=10)

        diff_vp = substituted.vp - original.vp
        vmax_diff_vp = np.max(np.abs(diff_vp))

        divider3 = make_axes_locatable(axs[0, 2])
        cax3 = divider3.append_axes("right", size="3%", pad=0.2)
        im3 = axs[0,2].imshow(diff_vp, aspect='equal', cmap=cmap_seismic, extent=extent_depth, vmin=-vmax_diff_vp, vmax=vmax_diff_vp)
        axs[0,2].set_title("ΔVp (Substituted - Original)")
        cbar3 = plt.colorbar(im3, cax=cax3)
        axs[0, 2].set_xlabel("X Coordinate")
        axs[0, 2].set_ylabel("Depth (m)")
        axs[0, 2].invert_yaxis()
        cbar3.set_label("ΔVp (m/s)", rotation=270, labelpad=10)

        # Density
        divider4 = make_axes_locatable(axs[1, 0])
        cax4 = divider4.append_axes("right", size="3%", pad=0.2)
        im4 = axs[1, 0].imshow(original.rhob, aspect='equal', cmap=cmap_jet, extent=extent_depth, vmin=vmin_rhob, vmax=vmax_rhob)
        axs[1, 0].set_title("Original Density")
        cbar4 = plt.colorbar(im4, cax=cax4)
        axs[1, 0].set_xlabel("X Coordinate")
        axs[1, 0].set_ylabel("Elevation (m)")
        cbar4.set_label("Density (kg/m³)", rotation=270, labelpad=10)

        divider5 = make_axes_locatable(axs[1, 1])
        cax5 = divider5.append_axes("right", size="3%", pad=0.2)
        im5 = axs[1,1].imshow(substituted.rhob, aspect='equal', cmap=cmap_jet, extent=extent_depth, vmin=vmin_rhob, vmax=vmax_rhob)
        axs[1,1].set_title("Substituted Density")
        cbar5 = plt.colorbar(im5, cax=cax5)
        axs[1, 1].set_xlabel("X Coordinate")
        axs[1, 1].set_ylabel("Elevation (m)")
        cbar5.set_label("Density (kg/m³)", rotation=270, labelpad=10)

        diff_rhob = substituted.rhob - original.rhob
        vmax_diff_rhob = np.max(np.abs(diff_rhob))
        divider6 = make_axes_locatable(axs[1, 2])
        cax6 = divider6.append_axes("right", size="3%", pad=0.2)
        im6 = axs[1, 2].imshow(diff_rhob, aspect='equal', cmap=cmap_seismic, extent=extent_depth, vmin=-vmax_diff_rhob, vmax=vmax_diff_rhob)
        axs[1, 2].set_title("ΔDensity (Substituted - Original)")
        cbar6 = plt.colorbar(im6, cax=cax6)
        axs[1, 2].set_xlabel("X Coordinate")
        axs[1, 2].set_ylabel("Elevation (m)")
        cbar6.set_label("ΔDensity (kg/m³)", rotation=270, labelpad=10)

        # Impedance
        divider7 = make_axes_locatable(axs[2, 0])
        cax7 = divider7.append_axes("right", size="3%", pad=0.2)
        im7 = axs[2, 0].imshow(original_imp, aspect='equal', cmap=cmap_jet, extent=extent_depth, vmin=vmin_imp, vmax=vmax_imp)
        axs[2, 0].set_title("Original Impedance")
        cbar7 = plt.colorbar(im7, cax=cax7)
        axs[2, 0].set_xlabel("X Coordinate")
        axs[2, 0].set_ylabel("Elevation (m)")
        cbar7.set_label("AI (km/s·g/cm³)", rotation=270, labelpad=10)

        divider8 = make_axes_locatable(axs[2, 1])
        cax8 = divider8.append_axes("right", size="3%", pad=0.2)
        im8 = axs[2, 1].imshow(substituted_imp, aspect='equal', cmap=cmap_jet, extent=extent_depth, vmin=vmin_imp, vmax=vmax_imp)
        axs[2, 1].set_title("Substituted Impedance")
        cbar8 = plt.colorbar(im8, cax=cax8)
        axs[2, 1].set_xlabel("X Coordinate")
        axs[2, 1].set_ylabel("Elevation (m)")
        cbar8.set_label("AI (km/s·g/cm³)", rotation=270, labelpad=10)

        diff_imp = (substituted.impedance - original.impedance) * 1e-6
        divider9 = make_axes_locatable(axs[2, 2])
        cax9 = divider9.append_axes("right", size="3%", pad=0.2)
        im9 = axs[2, 2].imshow(diff_imp, aspect='equal', cmap=cmap_seismic, extent=extent_depth, vmin=diff_imp.min(), vmax=diff_imp.max())
        axs[2, 2].set_title("ΔImpedance (Substituted - Original)")
        cbar9 = plt.colorbar(im9, cax=cax9)
        axs[2, 2].set_xlabel("X Coordinate")
        axs[2, 2].set_ylabel("Elevation (m)")
        cbar9.set_label("ΔAI (km/s·g/cm³)", rotation=270, labelpad=10)

        # Depth-domain Seismic
        divider10 = make_axes_locatable(axs[3, 0])
        cax10 = divider10.append_axes("right", size="3%", pad=0.2)
        vmax_seismic_depth = max(np.abs(original.seismic_depth).max(), np.abs(substituted.seismic_depth).max())
        im10 = axs[3,0].imshow(original.seismic_depth, aspect='equal', cmap=cmap_seismic, extent=extent_depth, vmin=-vmax_seismic_depth, vmax=vmax_seismic_depth, 
                               interpolation='bicubic')
        axs[3, 0].set_title("Original Seismic in Depth")
        axs[3, 0].invert_yaxis()
        cbar10 = plt.colorbar(im10, cax=cax10)
        axs[3, 0].set_xlabel("X Coordinate")
        axs[3, 0].set_ylabel("Depth (m)")
        cbar10.set_label("Amplitude", rotation=270, labelpad=10)

        divider11 = make_axes_locatable(axs[3, 1])
        cax11 = divider11.append_axes("right", size="3%", pad=0.2)
        im11 = axs[3,1].imshow(substituted.seismic_depth, aspect='equal', cmap=cmap_seismic, extent=extent_depth, vmin=-vmax_seismic_depth, vmax=vmax_seismic_depth,
                               interpolation='bicubic')
        axs[3, 1].set_title("Substituted Seismic in Depth")
        axs[3, 1].invert_yaxis()
        cbar11 = plt.colorbar(im11, cax=cax11)
        axs[3, 1].set_xlabel("X Coordinate")
        axs[3, 1].set_ylabel("Depth (m)")
        cbar11.set_label("Amplitude", rotation=270, labelpad=10)
        

        diff_seismic_depth = substituted.seismic_depth - original.seismic_depth
        vmax_diff_sd = np.max(np.abs(diff_seismic_depth))
        divider12 = make_axes_locatable(axs[3, 2])
        cax12 = divider12.append_axes("right", size="3%", pad=0.2)
        im12 = axs[3, 2].imshow(diff_seismic_depth, aspect='equal', cmap=cmap_seismic, extent=extent_depth, vmin=-vmax_diff_sd, vmax=vmax_diff_sd, interpolation='bicubic')
        axs[3, 2].set_title("ΔSeismic in Depth")
        axs[3, 2].invert_yaxis()
        cbar12 = plt.colorbar(im12, cax=cax12)
        axs[3, 2].set_xlabel("X Coordinate")
        axs[3, 2].set_ylabel("Depth (m)")
        cbar12.set_label("ΔAmplitude", rotation=270, labelpad=10)

        # Time-domain Seismic (assuming time axes are similar)
        try:
            extent_time_ori = [original.x[0], original.x[-1], original.t_axis[-1]*1000, original.t_axis[0]*1000]
            extent_time_sub = [substituted.x[0], substituted.x[-1], substituted.t_axis[-1]*1000, substituted.t_axis[0]*1000]
            vmax_seismic_time = max(np.abs(original.seismic_time).max(), np.abs(substituted.seismic_time).max())

            divider13 = make_axes_locatable(axs[4, 0])
            cax13 = divider13.append_axes("right", size="3%", pad=0.2)
            im13 = axs[4, 0].imshow(original.seismic_time, aspect='auto', cmap=cmap_seismic, extent=extent_time_ori, vmin=-vmax_seismic_time, vmax=vmax_seismic_time,
                                   interpolation='bicubic')
            axs[4, 0].set_title("Original Seismic in Time")
            cbar13 = plt.colorbar(im13, cax=cax13)
            axs[4, 0].set_xlabel("X Coordinate")
            axs[4, 0].set_ylabel("TWT (ms)")
            cbar13.set_label("Amplitude", rotation=270, labelpad=10)

            divider14 = make_axes_locatable(axs[4, 1])
            cax14 = divider14.append_axes("right", size="3%", pad=0.2)
            im14 = axs[4, 1].imshow(substituted.seismic_time, aspect='auto', cmap=cmap_seismic, extent=extent_time_sub, vmin=-vmax_seismic_time, vmax=vmax_seismic_time,
            interpolation='bicubic')
            axs[4, 1].set_title("Substituted Seismic in Time")
            cbar14 = plt.colorbar(im14, cax=cax14)
            axs[4, 1].set_xlabel("X Coordinate")
            axs[4, 1].set_ylabel("TWT (ms)")
            cbar14.set_label("Amplitude", rotation=270, labelpad=10)

            # If time axes are the same, compute difference
            if original.t_axis.shape == substituted.t_axis.shape and np.allclose(original.t_axis, substituted.t_axis):
                diff_seismic_time = substituted.seismic_time - original.seismic_time
                vmax_diff_st = np.max(np.abs(diff_seismic_time))
                divider15 = make_axes_locatable(axs[4, 2])
                cax15 = divider15.append_axes("right", size="3%", pad=0.2)
                im15 = axs[4, 2].imshow(diff_seismic_time, aspect='auto', cmap=cmap_seismic, extent=extent_time_ori, vmin=-vmax_diff_st, vmax=vmax_diff_st,
                                    interpolation='bicubic')
                axs[4, 2].set_title("ΔSeismic in Time")
                plt.colorbar(im15, cax=cax15)
                axs[4, 2].set_xlabel("X Coordinate")
                axs[4, 2].set_ylabel("TWT (ms)")
                cbar14.set_label("ΔAmplitude", rotation=270, labelpad=10)

            else:
                axs[4, 2].text(0.5, 0.5, 'Time axes differ, change has been observed!\nDifference map cannot be computed', ha='center', va='center')  

        except AttributeError:
            pass

        plt.show()
