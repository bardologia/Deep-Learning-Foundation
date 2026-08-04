from __future__ import annotations


class PhysicsLossLibrary:

    def _intro(self) -> dict:
        return {
            "kicker" : "training objective, signal-model supervision",
            "title"  : "The physics loss terms",
            "lead"   : "Five optional loss terms that do not compare curves bin by bin. Each one pushes the predicted Gaussian-mixture tomogram back through the TomoSAR forward model, extracts a physical quantity, and matches it against the same quantity computed from the ground-truth profile. They supervise what the radar actually measures, not the shape of a fitted spectrum.",
            "points" : [
                {
                    "label" : "Shared forward model",
                    "text"  : "Every term reuses one Fourier operator built from the acquisition geometry: the perpendicular baselines set the vertical wavenumbers, those set the steering matrix, and the steering matrix turns a profile into interferometric observables.",
                },
                {
                    "label" : "Supervised against the label",
                    "text"  : "Both the prediction and the reference are curve reconstructions of their Gaussian parameters, in denormalised physical units. The terms match the predicted physical quantity to the GT label, never to a re-estimated Capon profile.",
                },
                {
                    "label" : "Masked to real signal",
                    "text"  : "Each term reduces only over pixels whose ground-truth integrated power exceeds the physics floor; empty sky is never penalised. All five terms default off and must be scale-calibrated before they mix with the curve and parameter losses.",
                },
            ],
        }

    def _operator(self) -> dict:
        return {
            "title" : "The shared forward operator",
            "blurb" : "The vertical wavenumber of each pass, the steering response sampled on the elevation grid, and the steering outer product the covariance terms reuse. By default the wavenumber is per-pixel: the geometry field (previous section) gives every pass its own kz at every azimuth and range, and the steering is assembled per pixel from that kz map. When the geometry is constant across the scene this reduces to a single TomoGeometry matrix set built once per run. Either way the same operator feeds all five terms.",
            "items" : [
                {
                    "title" : "Vertical wavenumber",
                    "tex"   : r"k_z^{(i)} = \frac{4\pi\,b^{\perp}_i}{\lambda\,r\,\sin\theta}",
                    "note"  : "Each pass sees elevation through a phase ramp whose rate is set by its perpendicular baseline. The wider the baseline spread, the finer the elevation resolution the stack can resolve. This is the per-pixel height convention; the slant convention drops the sin theta. kz is recomputed for every azimuth and range pixel, and the reference pass has kz = 0.",
                    "vars"  : [
                        {"sym": r"k_z^{(i)}",   "desc": "vertical wavenumber of pass i, per azimuth and range (rad/m)"},
                        {"sym": r"b^{\perp}_i", "desc": "perpendicular baseline of pass i (m)"},
                        {"sym": r"\lambda",     "desc": "radar wavelength (m)"},
                        {"sym": r"r, \theta",   "desc": "per-range slant range (m) and look angle"},
                    ],
                },
                {
                    "title" : "Steering response",
                    "tex"   : r"A^{(p)}_{i,n} = \exp\!\left(j\,k_{z,p}^{(i)}\,\xi_n\right)",
                    "note"  : "The complex response of pass i to unit reflectivity at elevation bin n, for pixel p. Mapping a profile through it re-synthesises the multibaseline signal a real acquisition would have measured. In the per-pixel path the response is rebuilt from each pixel's own kz; in the constant-geometry path it is one steering matrix shared by every pixel.",
                    "vars"  : [
                        {"sym": r"A^{(p)}_{i,n}", "desc": "steering response, pass i at elevation bin n, pixel p"},
                        {"sym": r"k_{z,p}^{(i)}", "desc": "per-pixel vertical wavenumber from the kz map"},
                        {"sym": r"\xi_n",         "desc": "elevation grid sample (m)"},
                    ],
                },
                {
                    "title" : "Steering outer product",
                    "tex"   : r"O_{i,j,n} = A_{i,n}\,\overline{A_{j,n}}",
                    "note"  : "How the covariance and Capon-cycle terms synthesise a covariance matrix with a single contraction against the profile. With constant geometry it is precomputed once per elevation bin; with per-pixel geometry the covariance is synthesised per pixel from that pixel's kz, so this outer product is the shared-geometry optimisation of the same operation.",
                    "vars"  : [
                        {"sym": r"O_{i,j,n}",          "desc": "outer product of the steering columns at bin n"},
                        {"sym": r"\overline{A_{j,n}}", "desc": "complex conjugate of the steering entry"},
                    ],
                },
            ],
        }

    def _terms(self) -> list[dict]:
        return [
            {
                "key"        : "total_power",
                "index"      : 1,
                "name"       : "Total power",
                "code"       : "PhysicalLoss.total_power",
                "tagline"    : "Does the profile carry the right total backscatter?",
                "role"       : "radiometric",
                "role_label" : "kept, but biased",
                "quantity"   : "Integrated reflectivity (mass m0)",
                "invariant"  : "Absolute power",
                "cost"       : 1,
                "cost_label" : "cheapest",
                "tex"        : r"\ell_{\mathrm{pow}} = \left\langle \frac{\left|m_0^{P} - m_0^{T}\right|}{m_0^{T}} \right\rangle_{\text{valid}}, \qquad m_0 = \sum_n P(\xi_n)\,\Delta\xi",
                "story"      : "The simplest physical check: integrate the predicted spectrum and the reference spectrum over elevation and compare the total. It is the only one of the five that looks at absolute radiometric power rather than a normalised shape.",
                "caveat"     : "Because it is an absolute-power term it inherits the radiometric bias of the Capon estimator that produced the data, roughly 2 dB on-grid and worse off-grid. That bias is why its default weight is zero and why the scale-insensitive terms are preferred as the actual objective.",
                "vars"       : [
                    {"sym": r"\ell_{\mathrm{pow}}", "desc": "total-power relative-error term"},
                    {"sym": r"m_0^{P}, m_0^{T}",    "desc": "integrated power of predicted and GT profiles"},
                    {"sym": r"\Delta\xi",           "desc": "elevation bin spacing (m)"},
                ],
            },
            {
                "key"        : "moments",
                "index"      : 2,
                "name"       : "Profile moments",
                "code"       : "PhysicalLoss.moments",
                "tagline"    : "Is the scatterer at the right height and the right thickness?",
                "role"       : "metric",
                "role_label" : "validation-grade",
                "quantity"   : "Mass, centroid height, vertical spread",
                "invariant"  : "Absolute power (centroid and spread are ratios)",
                "cost"       : 1,
                "cost_label" : "cheap",
                "tex"        : r"\bar{z} = \frac{\sum_n P_n\,\xi_n}{\sum_n P_n}, \quad \sigma_z = \sqrt{\frac{\sum_n P_n\,\xi_n^2}{\sum_n P_n} - \bar{z}^2}, \qquad \ell_{\mathrm{mom}} = \left\langle \frac{w_0\frac{|\Delta m_0|}{m_0^{T}} + w_1\frac{|\Delta\bar{z}|}{\Delta\xi_R} + w_2\frac{|\Delta\sigma_z|}{\Delta\xi_R}}{w_0+w_1+w_2} \right\rangle",
                "story"      : "Summarises each profile by its first three descriptors: total mass, the centroid that locates the scattering phase centre in elevation, and the spread that measures the vertical extent of the scattering. These are the quantities a forest-height or layover analysis actually reads off a tomogram.",
                "caveat"     : "Centroid and spread are mass-normalised ratios, so they are robust to the Capon power bias even though the mass sub-term is not. Used mainly as a validation-grade physics metric rather than the primary loss; the sub-weights default to (1, 1, 1).",
                "vars"       : [
                    {"sym": r"\ell_{\mathrm{mom}}", "desc": "moments term value"},
                    {"sym": r"\bar{z}, \sigma_z",   "desc": "profile centroid and spread (m)"},
                    {"sym": r"w_0, w_1, w_2",       "desc": "mass / centroid / spread weights, default (1, 1, 1)"},
                    {"sym": r"\Delta\xi_R",         "desc": "elevation axis span x_max - x_min (m)"},
                ],
            },
            {
                "key"        : "coherence_resyn",
                "index"      : 3,
                "name"       : "Coherence re-synthesis",
                "code"       : "PhysicalLoss.coherence_resynthesis",
                "tagline"    : "Would the predicted profile produce the same interferograms?",
                "role"       : "front-runner",
                "role_label" : "scale-insensitive front-runner",
                "quantity"   : "Multibaseline interferometric coherences",
                "invariant"  : "Absolute power (unit-disc ratio)",
                "cost"       : 2,
                "cost_label" : "light",
                "tex"        : r"\gamma_P\!\left(k_z^{(i)}\right) = \frac{\sum_n P_n\,e^{j\,k_z^{(i)}\xi_n}}{\sum_n P_n}, \qquad \ell_{\mathrm{coh\text{-}r}} = \left\langle \frac{1}{N_s}\sum_i \left|\gamma_P^{(i)} - \gamma_T^{(i)}\right|^2 \right\rangle",
                "story"      : "Re-synthesises the complex coherence each baseline pair would observe: the normalised Fourier transform of the profile sampled at the track wavenumbers. This is the characteristic function of the normalised elevation distribution, exactly the interferometric observable the stack measures. For a Gaussian mixture it is a closed-form sum of complex Gaussians, so the term is exactly differentiable in the predicted amplitudes, means, and spreads.",
                "caveat"     : "Each coherence lives on the unit disc by construction, so the term is insensitive to absolute power and sidesteps the Capon radiometric bias entirely. One of the two recommended physics objectives.",
                "vars"       : [
                    {"sym": r"\ell_{\mathrm{coh\text{-}r}}", "desc": "coherence re-synthesis term value"},
                    {"sym": r"\gamma_P, \gamma_T",           "desc": "normalised coherences of predicted and GT profiles"},
                    {"sym": r"k_z^{(i)}",                    "desc": "vertical wavenumber of pass i"},
                    {"sym": r"N_s",                          "desc": "number of passes in the stack"},
                ],
            },
            {
                "key"        : "covariance_match",
                "index"      : 4,
                "name"       : "Covariance matching",
                "code"       : "PhysicalLoss.covariance_matching",
                "tagline"    : "Does the predicted profile rebuild the right sample covariance?",
                "role"       : "front-runner",
                "role_label" : "scale-insensitive front-runner",
                "quantity"   : "Synthesised multibaseline covariance matrix",
                "invariant"  : "Absolute scale (relative Frobenius)",
                "cost"       : 3,
                "cost_label" : "moderate",
                "tex"        : r"\mathbf{R}[P] = \mathbf{A}\,\mathrm{diag}(P)\,\mathbf{A}^{H}\,\Delta\xi, \qquad \ell_{\mathrm{cov}} = \left\langle \frac{\left\|\mathbf{R}[P] - \mathbf{R}[T]\right\|_F^2}{\left\|\mathbf{R}[T]\right\|_F^2} \right\rangle",
                "story"      : "Synthesises the full covariance matrix the profile would generate across the stack and matches it to the reference covariance in relative Frobenius norm. Where coherence re-synthesis checks the normalised diagonal of the interferometric structure, this term constrains the entire matrix at once. Because the covariance is linear in the profile, the implementation transforms only the prediction-minus-GT difference rather than two full matrices.",
                "caveat"     : "Follows the COMET covariance-matching lineage but uses an unweighted relative form with the reference synthesised from the GT profile, so it is insensitive to absolute scale. The second of the two recommended physics objectives.",
                "vars"       : [
                    {"sym": r"\ell_{\mathrm{cov}}",          "desc": "covariance matching term value"},
                    {"sym": r"\mathbf{R}[P], \mathbf{R}[T]", "desc": "covariances synthesised from predicted and GT profiles"},
                    {"sym": r"\mathbf{A}",                   "desc": "steering matrix exp(j kz xi)"},
                    {"sym": r"\|\cdot\|_F",                  "desc": "Frobenius norm over the pass-pair axes"},
                ],
            },
            {
                "key"        : "capon_cycle",
                "index"      : 5,
                "name"       : "Capon cycle-consistency",
                "code"       : "PhysicalLoss.capon_cycle",
                "tagline"    : "Run the prediction through the full estimator and back.",
                "role"       : "faithful",
                "role_label" : "most faithful, most expensive",
                "quantity"   : "Re-estimated Capon spectrum",
                "invariant"  : "Absolute power (mass-normalised spectra)",
                "cost"       : 5,
                "cost_label" : "most expensive",
                "tex"        : r"\hat{T}_P(\xi_n) = \frac{1}{\mathbf{a}^{H}(\xi_n)\big(\mathbf{R}[P] + \epsilon\,\bar{\sigma}\,\mathbf{I}\big)^{-1}\mathbf{a}(\xi_n)}, \qquad \ell_{\mathrm{cyc}} = \left\langle \frac{1}{N}\sum_n \left(\frac{\hat{T}_P(\xi_n)}{m_0^{\hat{T}}} - \frac{T(\xi_n)}{m_0^{T}}\right)^2 \right\rangle",
                "story"      : "The full cycle-consistency loss of inverse imaging: synthesise the covariance from the prediction, apply signal-adaptive diagonal loading, then run the same Capon beamformer that produced the data and compare the re-estimated spectrum with the reference, both mass-normalised. By construction it absorbs the Capon bias and point-spread function rather than fighting them.",
                "caveat"     : "Each pixel needs one N_s x N_s linear solve, making this the most expensive of the five. The loading term guarantees the covariance is invertible and the solve differentiable; the spectrum is formed by solving the system, not by an explicit inverse.",
                "vars"       : [
                    {"sym": r"\ell_{\mathrm{cyc}}",    "desc": "Capon cycle-consistency term value"},
                    {"sym": r"\hat{T}_P(\xi_n)",       "desc": "Capon spectrum re-estimated from the prediction"},
                    {"sym": r"\bar{\sigma}",           "desc": "mean covariance diagonal for adaptive loading"},
                    {"sym": r"\epsilon",               "desc": "capon_loading = 1e-2"},
                    {"sym": r"m_0^{\hat{T}}, m_0^{T}", "desc": "integrated-power normalisers"},
                ],
            },
        ]

    def _dataset(self) -> dict:
        return {
            "title"   : "What track data is used, and how the geometry is built",
            "blurb"   : "The forward operator above is not hand-set: its wavelength, slant range, look angle, per-pass baselines, and elevation grid are all extracted from the preprocessed dataset. This is the full schema of which dataset artifacts are read, what fields each one contributes, and the exact construction chain that turns them into the per-pixel vertical wavenumber and the elevation axis the loss steers with.",
            "sources" : [
                {
                    "file"      : "meta/track_parameters.json",
                    "container" : "TrackParameters",
                    "role"      : "Per-pass acquisition parameters, parsed from each track's INF/INF-RDP pp_*.xml (STEP processor) for the dataset polarisation. The reference track is labels[0]. Source of wavelength, slant-range vector, and the sensor/terrain heights behind the look angle.",
                    "fields"    : [
                        {"name": "lambda",            "shape": "scalar",         "desc": "radar wavelength (m); sets the 4 pi / lambda kz scale"},
                        {"name": "r",                 "shape": "[n_range_full]", "desc": "slant-range vector, one entry per range sample (m)"},
                        {"name": "h0",                "shape": "scalar",         "desc": "sensor altitude above the datum (m)"},
                        {"name": "terrain",           "shape": "scalar",         "desc": "reference terrain height (m); h0 - terrain is the height above ground"},
                        {"name": "antdir / da / rref","shape": "scalar",         "desc": "look side, depression angle, reference range (reported, diagnostic)"},
                    ],
                },
                {
                    "file"      : "data/track_profiles.npz",
                    "container" : "TrackProfiles",
                    "role"      : "Per-azimuth track positions for every pass over the full crop, in the resa frame. The horizontal and vertical profiles, referenced to the primary, become the per-azimuth baselines. azimuth_start keeps the profiles aligned to absolute SLC samples.",
                    "fields"    : [
                        {"name": "labels",            "shape": "[n_tracks]",        "desc": "ordered pass labels; must equal the parameter labels"},
                        {"name": "horizontal",        "shape": "[n_tracks, n_az]",  "desc": "horizontal track position per azimuth (m)"},
                        {"name": "vertical",          "shape": "[n_tracks, n_az]",  "desc": "vertical track position per azimuth (m)"},
                        {"name": "horizontal_std / vertical_std", "shape": "[n_tracks]", "desc": "windowed dispersion of each profile (validation)"},
                        {"name": "azimuth_start",     "shape": "scalar",            "desc": "absolute azimuth sample of profile column 0"},
                    ],
                },
                {
                    "file"      : "data/tomogram_full.npy",
                    "container" : "ndarray (complex64)",
                    "role"      : "The PyRAT Capon tomogram, axis order (elevation, azimuth, range). It is the ground-truth profile the loss target is fitted to, and its elevation dimension fixes the number of x-axis bins, so the steering grid and the GT curves share one axis.",
                    "fields"    : [
                        {"name": "shape[0]",          "shape": "n_elevation",       "desc": "elevation bins -> profile_length / x_axis length"},
                        {"name": "shape[1:]",         "shape": "[n_az, n_range]",   "desc": "azimuth x range extent of the crop"},
                    ],
                },
                {
                    "file"      : "meta/config_state.json",
                    "container" : "ProcessingConfig snapshot",
                    "role"      : "Frozen record of the preprocessing run. Supplies the elevation extent (the PyRAT beamforming range) and the global crop that bounds the range and azimuth windows.",
                    "fields"    : [
                        {"name": "tomogram_config.height_range", "shape": "[2]", "desc": "elevation axis [z_min, z_max] in metres (here -20, 80)"},
                        {"name": "crop",                         "shape": "[4]", "desc": "azimuth_start/end, range_start/end that slice r and the profiles"},
                        {"name": "tomogram_config.polarisation", "shape": "str", "desc": "polarisation selecting the pp_*.xml and SLC products"},
                    ],
                },
                {
                    "file"      : "meta/baselines.json + data/dataset.json",
                    "container" : "TrackBaselines / dataset manifest",
                    "role"      : "Windowed-mean baselines (and their stds) used for validation and the global-geometry fallback path, plus the dataset manifest holding the global crop, the ordered pass labels, and the artifact map. interferograms.npy (primary * conj of each DEM-deramped secondary) is the measured multibaseline signal that feeds the model input stack.",
                    "fields"    : [
                        {"name": "labels / reference",    "shape": "[n_tracks]", "desc": "pass order and the primary; reference is always index 0"},
                        {"name": "horizontal / vertical", "shape": "[n_tracks]", "desc": "windowed-mean baseline components (m), reference-subtracted"},
                        {"name": "global_crop / pass_labels", "shape": "manifest", "desc": "crop bounds and ordered passes echoed into the geometry build"},
                    ],
                },
            ],
            "pipeline" : [
                {
                    "step"   : "01",
                    "title"  : "Align passes",
                    "tex"    : "",
                    "detail" : "GeometryFieldBuilder checks that the track_profiles labels equal the track_parameters labels and fails loudly otherwise. The primary (index 0) is the interferometric reference; all baselines and phases are relative to it.",
                    "output" : "ordered track labels, reference = labels[0]",
                },
                {
                    "step"   : "02",
                    "title"  : "Range geometry from the reference track",
                    "tex"    : r"r_n = r^{\mathrm{ref}}[n_0{:}n_1], \qquad \theta_n = \arccos\!\frac{h_0 - h_t}{r_n}",
                    "detail" : "The reference track's slant-range vector is sliced to the crop range window, and the look angle per range bin comes from the sensor height above terrain over the slant range. Look angle grows from near to far range across the swath.",
                    "output" : "slant_range[n_range], look_angle[n_range]",
                },
                {
                    "step"   : "03",
                    "title"  : "Per-azimuth baselines",
                    "tex"    : r"b^{h}_{i,a} = H_{i,a} - H_{0,a}, \qquad b^{v}_{i,a} = V_{i,a} - V_{0,a}",
                    "detail" : "Horizontal and vertical track positions from track_profiles are sliced to the crop azimuth window and referenced to the primary, so the reference baseline is exactly zero and every other pass is an offset from it.",
                    "output" : "baseline_h[n_tracks, n_az], baseline_v[n_tracks, n_az]",
                },
                {
                    "step"   : "04",
                    "title"  : "Perpendicular baseline",
                    "tex"    : r"b^{\perp}_{i,a,n} = b^{h}_{i,a}\cos\theta_n + b^{v}_{i,a}\sin\theta_n",
                    "detail" : "The horizontal and vertical offsets are projected onto the direction perpendicular to the line of sight using the per-range look angle. Only this perpendicular component carries elevation sensitivity.",
                    "output" : "b_perp[n_tracks, n_az, n_range]",
                },
                {
                    "step"   : "05",
                    "title"  : "Per-pixel vertical wavenumber",
                    "tex"    : r"k_z = \frac{4\pi\,b^{\perp}}{\lambda\,r_n\,\sin\theta_n}\;\;(\text{height}) \qquad k_z = \frac{4\pi\,b^{\perp}}{\lambda\,r_n}\;\;(\text{slant})",
                    "detail" : "GeometryField.kz(convention) builds a wavenumber for every (pass, azimuth, range). The default height convention divides by sin theta to index a vertical axis; the slant convention indexes the line-of-sight-normal elevation. They differ only by 1/sin theta and give the same interferometric phase. The reference pass has kz = 0.",
                    "output" : "kz[n_tracks, n_az, n_range]",
                },
                {
                    "step"   : "06",
                    "title"  : "Elevation axis",
                    "tex"    : r"\xi = \mathrm{linspace}(z_{\min},\,z_{\max},\,N_\xi), \qquad N_\xi = \text{tomogram elevation bins}",
                    "detail" : "The axis span is the height_range recorded by the preprocessing run; the sample count is read directly from the tomogram's elevation dimension. Because both come from the same dataset, x_axis and the ground-truth curves are sampled identically and the integration step dx is exact.",
                    "output" : "x_axis[N_xi], dx",
                },
                {
                    "step"   : "07",
                    "title"  : "Slice into the loss",
                    "tex"    : "",
                    "detail" : "Per training region, GeometryField.slice crops the field to the patch and .kz(convention) yields the per-pixel kz_map. It is aligned bin-for-bin with the tomogram patch and fed to the steering operator, so each pixel is steered with its own geometry rather than a single scene-wide kz.",
                    "output" : "kz_map[B, n_tracks, H, W]",
                },
            ],
            "example"  : {
                "title" : "Resolved on the FL01 reference stack",
                "blurb" : "The values every step above produces on the transferred 17sartom-traun L-band stack, as verified against the dataset artifacts.",
                "rows"  : [
                    {"k": "Stack",                   "v": "17sartom-traun, L-band, hv"},
                    {"k": "Tracks",                  "v": "29 (1 primary + 28 secondaries)"},
                    {"k": "Reference pass",          "v": "FL01_PS02 (= pass_labels[0])"},
                    {"k": "Wavelength",              "v": "0.2262 m (L-band)"},
                    {"k": "Sensor altitude h0",      "v": "3719.2 m"},
                    {"k": "Terrain",                 "v": "683.9 m"},
                    {"k": "Height above terrain",    "v": "3035.3 m"},
                    {"k": "Slant range near to far", "v": "3598.9 to 3898.0 m"},
                    {"k": "Look angle near to far",  "v": "32.50 to 38.86 deg"},
                    {"k": "Crop",                    "v": "azimuth [1000, 2000), range [500, 1000)"},
                    {"k": "Elevation axis",          "v": "[-20, 80] m, 150 bins, dx = 0.671 m"},
                    {"k": "kz convention",           "v": "height (4 pi b_perp / lambda r sin theta)"},
                ],
            },
        }

    def _comparison(self) -> dict:
        return {
            "title"   : "How the five terms compare",
            "blurb"   : "All default off. The two scale-insensitive front-runners are the recommended objective; total power is kept but disqualified by radiometric bias; moments serve as a validation metric; the Capon cycle is the most faithful but the most expensive.",
            "columns" : ["Term", "Physical quantity", "Scale-invariant", "Compute cost", "Role"],
            "rows"    : [
                {"term": "Total power",            "quantity": "Integrated power",            "invariant": "No",  "cost": "Cheapest",  "role": "Kept, biased"},
                {"term": "Profile moments",        "quantity": "Mass, centroid, spread",      "invariant": "Mostly", "cost": "Cheap",  "role": "Validation metric"},
                {"term": "Coherence re-synthesis", "quantity": "Multibaseline coherences",    "invariant": "Yes", "cost": "Light",     "role": "Front-runner"},
                {"term": "Covariance matching",    "quantity": "Synthesised covariance",      "invariant": "Yes", "cost": "Moderate",  "role": "Front-runner"},
                {"term": "Capon cycle-consistency","quantity": "Re-estimated Capon spectrum", "invariant": "Yes", "cost": "Highest",   "role": "Most faithful"},
            ],
        }

    def _config(self) -> dict:
        return {
            "title"  : "Configuration reference",
            "blurb"  : "Per-term switches and weights live in LossConfig; the shared geometry lives in GeometryConfig. Every term is off by default; weights must be calibrated with the loss scale probe before physics terms are mixed with the curve and parameter losses.",
            "groups" : [
                {
                    "name"   : "LossConfig",
                    "fields" : [
                        {"field": "use_total_power / weight_total_power",       "default": "False / 0.0",    "meaning": "enable and weight the total-power term"},
                        {"field": "use_moments / weight_moments",               "default": "False / 0.0",    "meaning": "enable and weight the moments term"},
                        {"field": "moments_weights",                            "default": "(1, 1, 1)",      "meaning": "mass / centroid / spread sub-weights"},
                        {"field": "use_coherence_resyn / weight_coherence_resyn","default": "False / 0.0",   "meaning": "enable and weight coherence re-synthesis"},
                        {"field": "use_covariance_match / weight_covariance_match","default": "False / 0.0", "meaning": "enable and weight covariance matching"},
                        {"field": "use_capon_cycle / weight_capon_cycle",       "default": "False / 0.0",    "meaning": "enable and weight the Capon cycle"},
                        {"field": "capon_loading",                              "default": "1e-2",           "meaning": "diagonal loading fraction for the Capon cycle"},
                        {"field": "physics_floor",                              "default": "1e-3",           "meaning": "reference mass below which a pixel is masked out"},
                    ],
                },
                {
                    "name"   : "GeometryConfig",
                    "fields" : [
                        {"field": "wavelength",          "default": "0.23",          "meaning": "radar wavelength (m)"},
                        {"field": "slant_range",         "default": "5000.0",        "meaning": "reference slant range (m)"},
                        {"field": "look_angle_deg",      "default": "45.0",          "meaning": "projects baselines to the perpendicular component"},
                        {"field": "baselines",           "default": "9 values, 0-90", "meaning": "synthetic baselines (m); replaced per dataset"},
                        {"field": "baseline_component",  "default": "perpendicular", "meaning": "baseline component the dataset table resolves to"},
                        {"field": "baselines_source",    "default": "dataset",       "meaning": "dataset (require meta/baselines.json) / manual (use config values)"},
                        {"field": "kz_values",           "default": "()",            "meaning": "explicit kz override; bypasses the baselines"},
                    ],
                },
            ],
            "cli"    : [
                "python main/training/train_backbone.py --curriculum.complete.use_coherence_resyn true --curriculum.complete.weight_coherence_resyn 1.0",
                "python main/training/train_backbone.py --geometry.kz_values \"[0.0, 0.12, 0.25, 0.37, 0.49]\"",
            ],
        }

    def collect(self) -> dict:
        return {
            "intro"      : self._intro(),
            "operator"   : self._operator(),
            "dataset"    : self._dataset(),
            "terms"      : self._terms(),
            "comparison" : self._comparison(),
            "config"     : self._config(),
        }
