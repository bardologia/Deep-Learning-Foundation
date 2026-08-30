"""Training-time diagnostics of the Gaussian head: parameter, slot and example monitoring.

Collects capped samples of the physical Gaussian parameters for histograms, tracks per-slot
occupancy, amplitude and gradient magnitude to expose slot collapse, and logs reconstructed
z profiles for a fixed set of validation pixels chosen to span the component
configurations of interest.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


class ParamSampler:
    """Collects a capped sample of active Gaussian parameters for epoch histograms.

    Attributes:
        PARAM_NAMES          : Sampled parameter names in channel order: amplitude, mu and sigma.
        MAX_VALUES_PER_BATCH : Values taken from a single batch per parameter.
        MAX_VALUES_TOTAL     : Values retained per parameter across an epoch.
        ppg                  : Number of channels per Gaussian.
        thr                  : Amplitude above which a Gaussian counts as active, set when sampling begins.
        active               : Whether sampling is currently open.
    """

    PARAM_NAMES          = ("amp", "mu", "sigma")
    MAX_VALUES_PER_BATCH = 2048
    MAX_VALUES_TOTAL     = 200_000

    def __init__(self, params_per_gaussian: int):
        """Stores the channel count per Gaussian and clears the value store."""
        self.ppg    = params_per_gaussian
        self.thr    = None
        self.active = False

        self._reset_store()

    def begin(self, amp_zero_thr: float) -> None:
        """Opens a sampling window with the given active-amplitude threshold."""
        self.thr    = amp_zero_thr
        self.active = True
        self._reset_store()

    @torch.no_grad()
    def observe(self, pred_params_phys: torch.Tensor) -> None:
        """Samples the active Gaussian parameters of one batch into the store.

        Args:
            pred_params_phys : Denormalised predicted parameters of shape (batch, n_gaussians * ppg, height, width), with amplitude, mu in metres and sigma in metres per Gaussian.
        """
        B, C, H, W = pred_params_phys.shape
        p          = pred_params_phys.reshape(B, C // self.ppg, self.ppg, H, W)
        active     = p[:, :, 0] > self.thr

        for i, name in enumerate(self.PARAM_NAMES):
            budget = self.MAX_VALUES_TOTAL - self._counts[name]

            if budget <= 0:
                continue

            values = self._subsample(p[:, :, i][active], min(self.MAX_VALUES_PER_BATCH, budget))

            if values.numel() == 0:
                continue

            self._values[name].append(values.detach().float().cpu())
            self._counts[name] += values.numel()

    @staticmethod
    def _subsample(values: torch.Tensor, cap: int) -> torch.Tensor:
        """Returns at most cap values, evenly spaced over the flat input tensor."""
        if values.numel() <= cap:
            return values

        idx = torch.linspace(0, values.numel() - 1, cap, device=values.device).long()
        return values[idx]

    def histograms(self) -> dict:
        """Returns the collected values per parameter name as flat numpy arrays."""
        return {name: torch.cat(chunks).numpy() for name, chunks in self._values.items() if chunks}

    def end(self) -> None:
        """Closes the sampling window and clears the value store."""
        self.active = False
        self._reset_store()

    def _reset_store(self) -> None:
        """Empties the per-parameter value lists and their counters."""
        self._values = {name: [] for name in self.PARAM_NAMES}
        self._counts = {name: 0  for name in self.PARAM_NAMES}


class SlotVitals:
    """Tracks per-slot occupancy, mean amplitude and gradient norm to expose slot collapse.

    Attributes:
        ppg        : Number of channels per Gaussian.
        thr        : Amplitude above which a slot counts as active, set when observation begins.
        grad_scale : Factor undoing AMP loss scaling on the observed gradients.
        active     : Whether value observation is currently open.
    """

    def __init__(self, params_per_gaussian: int, grad_scale: float = 1.0):
        """Stores the channel count per Gaussian and the gradient rescaling factor."""
        self.ppg        = params_per_gaussian
        self.thr        = None
        self.grad_scale = float(grad_scale)
        self.active     = False

        self._reset_val()
        self._reset_grad()

    def begin(self, amp_zero_thr: float) -> None:
        """Opens an observation window with the given active-amplitude threshold."""
        self.thr    = amp_zero_thr
        self.active = True
        self._reset_val()

    @torch.no_grad()
    def observe(self, pred_params_phys: torch.Tensor) -> None:
        """Accumulates the active-pixel count and amplitude sum of every slot for one batch.

        Args:
            pred_params_phys : Denormalised predicted parameters of shape (batch, n_gaussians * ppg, height, width).
        """
        if not self.active:
            return

        B, C, H, W = pred_params_phys.shape
        p          = pred_params_phys.reshape(B, C // self.ppg, self.ppg, H, W)
        amp        = p[:, :, 0]
        active     = amp > self.thr

        counts = active.sum(dim=(0, 2, 3)).detach().float().cpu()
        amps   = (amp * active).sum(dim=(0, 2, 3)).detach().float().cpu()

        if self._active_count is None:
            self._active_count = counts
            self._amp_sum      = amps
        else:
            self._active_count += counts
            self._amp_sum      += amps

        self._n_pixels += B * H * W

    def observe_gradient(self, grad: torch.Tensor) -> None:
        """Accumulates the per-slot gradient norm of one backward pass.

        Args:
            grad : Gradient of the loss with respect to the head output, of shape (batch, n_gaussians * ppg, height, width), still carrying any AMP loss scaling.
        """
        B, C, H, W = grad.shape
        g          = grad.detach().reshape(B, C // self.ppg, self.ppg, H, W)
        norms      = g.pow(2).sum(dim=(0, 2, 3, 4)).sqrt().float() * self.grad_scale

        if self._grad_sum is None:
            self._grad_sum = norms
        else:
            self._grad_sum += norms

        self._grad_batches += 1

    def scalars(self) -> dict:
        """Summarises slot usage over the observation window.

        Returns:
            Per-slot active fraction and mean active amplitude, plus the normalised entropy
            of the slot-usage distribution; empty when nothing was observed.
        """
        if self._active_count is None or self._n_pixels == 0:
            return {}

        fractions = self._active_count / self._n_pixels
        out       = {f"slot_vitals/active_frac/slot_{k}": float(fractions[k]) for k in range(fractions.numel())}

        for k in range(fractions.numel()):
            n_active = float(self._active_count[k])
            out[f"slot_vitals/amp_mean/slot_{k}"] = float(self._amp_sum[k]) / n_active if n_active > 0 else 0.0

        usage = self._active_count / self._active_count.sum().clamp_min(1.0)
        ent   = -(usage.clamp_min(1e-12) * usage.clamp_min(1e-12).log()).sum()
        out["slot_vitals/usage_entropy"] = float(ent / torch.log(torch.tensor(float(max(usage.numel(), 2)))))

        return out

    def gradient_scalars(self) -> dict:
        """Returns the mean per-slot gradient norm and clears the gradient accumulator.

        Returns:
            Per-slot mean gradient norm since the last call; empty when no gradient was observed.
        """
        if self._grad_sum is None or self._grad_batches == 0:
            return {}

        means = (self._grad_sum / self._grad_batches).cpu()
        out   = {f"slot_vitals/grad_norm/slot_{k}": float(means[k]) for k in range(means.numel())}

        self._reset_grad()
        return out

    def end(self) -> None:
        """Closes the observation window and clears the value accumulators."""
        self.active = False
        self._reset_val()

    def _reset_val(self) -> None:
        """Clears the per-slot count and amplitude accumulators."""
        self._active_count = None
        self._amp_sum      = None
        self._n_pixels     = 0

    def _reset_grad(self) -> None:
        """Clears the per-slot gradient accumulator and its batch counter."""
        self._grad_sum     = None
        self._grad_batches = 0


class ExampleSelector:
    """Picks validation pixels spanning the component configurations worth watching.

    Candidates are classified from the ground-truth parameters into one-component pixels and
    three two-component regimes distinguished by the separation of the two strongest
    Gaussians relative to their widths, and by their amplitude ratio.

    Attributes:
        CANDIDATES_PER_BATCH : Candidates retained per category and batch.
        QUANTILES            : Quantiles of the ranking metric picked within each category.
        OVERLAP_RATIO_RANGE  : Separation-to-width ratio range counted as overlapping.
        SEPARATED_RATIO_MIN  : Minimum separation-to-width ratio counted as separated.
        DISTANT_RATIO_MIN    : Minimum separation-to-width ratio counted as distant.
        AMP_RATIO_MIN        : Amplitude ratio splitting balanced pairs from dominated ones.
        CATEGORIES           : Category names in report order.
        ppg                  : Number of channels per Gaussian.
        thr                  : Amplitude above which a Gaussian counts as active.
    """

    CANDIDATES_PER_BATCH = 256
    QUANTILES            = (0.1, 0.5, 0.9)
    OVERLAP_RATIO_RANGE  = (0.6, 1.5)
    SEPARATED_RATIO_MIN  = 1.5
    DISTANT_RATIO_MIN    = 3.0
    AMP_RATIO_MIN        = 2.0
    CATEGORIES           = ("single_gaussian", "two_overlapping", "two_separated", "two_distant")

    def __init__(self, params_per_gaussian: int, amp_zero_thr: float):
        """Stores the channel count per Gaussian and the active-amplitude threshold."""
        self.ppg = params_per_gaussian
        self.thr = amp_zero_thr

    @torch.no_grad()
    def _batch_candidates(self, phys: torch.Tensor, offset: int) -> dict:
        """Classifies the pixels of one batch into the example categories.

        Args:
            phys   : Denormalised ground-truth parameters of shape (batch, n_gaussians * ppg, height, width).
            offset : Number of samples seen before this batch, added to make sample indices global.

        Returns:
            Category name mapped to candidate rows of shape (n_candidates, 4), each row
            holding the global sample index, the height and width indices, and the ranking
            metric of that category.
        """
        B, C, H, W = phys.shape
        p          = phys.reshape(B, C // self.ppg, self.ppg, H, W)

        amp      = p[:, :, 0]
        mu       = p[:, :, 1]
        sigma    = p[:, :, 2]
        active   = amp > self.thr
        n_active = active.sum(dim=1)

        single     = n_active == 1
        single_amp = (amp * active).sum(dim=1)
        rows       = {"single_gaussian": self._rows(single, single_amp, offset)}

        if amp.shape[1] < 2:
            rows["two_overlapping"] = self._empty()
            rows["two_separated"]   = self._empty()
            rows["two_distant"]     = self._empty()
            return rows

        top2     = amp.topk(2, dim=1).indices
        mu_pair  = mu.gather(1, top2)
        sig_pair = sigma.gather(1, top2)
        amp_pair = amp.gather(1, top2)

        separation = (mu_pair[:, 0] - mu_pair[:, 1]).abs()
        sep_ratio  = separation / sig_pair.sum(dim=1).clamp_min(1e-6)
        amp_ratio  = amp_pair.amax(dim=1) / amp_pair.amin(dim=1).clamp_min(1e-6)

        pair      = n_active == 2
        overlap   = pair & (sep_ratio >= self.OVERLAP_RATIO_RANGE[0]) & (sep_ratio <= self.OVERLAP_RATIO_RANGE[1])
        separated = pair & (sep_ratio >= self.SEPARATED_RATIO_MIN) & (amp_ratio < self.AMP_RATIO_MIN)
        distant   = pair & (sep_ratio >= self.DISTANT_RATIO_MIN) & (amp_ratio >= self.AMP_RATIO_MIN)

        rows["two_overlapping"] = self._rows(overlap, sep_ratio, offset)
        rows["two_separated"]   = self._rows(separated, sep_ratio, offset)
        rows["two_distant"]     = self._rows(distant, amp_ratio, offset)

        return rows

    def _rows(self, mask: torch.Tensor, metric: torch.Tensor, offset: int) -> torch.Tensor:
        """Turns a pixel mask into capped candidate rows carrying their ranking metric.

        Args:
            mask   : Boolean pixel mask of shape (batch, height, width).
            metric : Ranking metric of the same shape as the mask.
            offset : Number of samples seen before this batch.

        Returns:
            Rows of shape (n_candidates, 4) with global sample index, height, width and metric.
        """
        coords = mask.nonzero(as_tuple=False)

        if coords.shape[0] == 0:
            return self._empty()

        rows = torch.stack([coords[:, 0].float() + offset, coords[:, 1].float(), coords[:, 2].float(), metric[mask].float()], dim=1)
        return self._subsample(rows, self.CANDIDATES_PER_BATCH)

    @staticmethod
    def _subsample(rows: torch.Tensor, cap: int) -> torch.Tensor:
        """Returns at most cap candidate rows, evenly spaced over the input rows."""
        if rows.shape[0] <= cap:
            return rows

        idx = torch.linspace(0, rows.shape[0] - 1, cap).long()
        return rows[idx]

    @staticmethod
    def _empty() -> torch.Tensor:
        """Returns an empty candidate-row tensor of shape (0, 4)."""
        return torch.zeros(0, 4)

    def _pick(self, pool: torch.Tensor) -> list:
        """Picks the pixels sitting at the configured quantiles of a category's ranking metric.

        Args:
            pool : Candidate rows of shape (n_candidates, 4) for one category.

        Returns:
            Deduplicated (sample, height, width) index triples, at most one per quantile.
        """
        if pool.shape[0] == 0:
            return []

        order  = pool[:, 3].argsort()
        picked = []
        seen   = set()

        for q in self.QUANTILES:
            row   = order[round(q * (order.shape[0] - 1))]
            pixel = tuple(int(v) for v in pool[row, :3])

            if pixel not in seen:
                seen.add(pixel)
                picked.append(pixel)

        return picked

    def select(self, loader, denormalize) -> dict:
        """Scans a loader once and picks the example pixels of every category.

        Args:
            loader      : Loader yielding (inputs, targets) batches over the validation split.
            denormalize : Callable mapping normalised targets to physical Gaussian parameters.

        Returns:
            Category name mapped to the selected (sample, height, width) index triples.
        """
        pools  = {name: [] for name in self.CATEGORIES}
        offset = 0

        for batch in loader:
            phys = denormalize(batch[1].float())

            for name, rows in self._batch_candidates(phys, offset).items():
                pools[name].append(rows)

            offset += batch[0].shape[0]

        return {name: self._pick(torch.cat(chunks)) for name, chunks in pools.items()}


class ReconstructionFigures:
    """Logs predicted-versus-label z profiles for a fixed set of validation pixels.

    The reference pixels and their y limits are captured once, so the figures logged each
    epoch are directly comparable over training.

    Attributes:
        HEADROOM : Factor applied to the label peak when fixing the y limit of a panel.
        tracker  : Experiment tracker receiving the figures.
        device   : Device the cached reference batch is held on.
    """

    HEADROOM = 1.3

    def __init__(self, tracker, device):
        """Stores the tracker and device and clears the cached reference batch."""
        self.tracker = tracker
        self.device  = device

        self._images   = None
        self._targets  = None
        self._pixels   = None
        self._ylims    = None
        self._disabled = False

    def _collect(self, loader, selected: dict):
        """Gathers the samples holding the selected pixels into one reference batch.

        Args:
            loader   : Loader yielding (inputs, targets) batches over the validation split.
            selected : Category name mapped to (sample, height, width) index triples.

        Returns:
            Tuple of the stacked inputs of shape (n_examples, in_channels, height, width),
            the stacked targets of shape (n_examples, target_channels, height, width), and
            the pixel records as (row, height, width, tag) tuples.
        """
        needed  = {g for rows in selected.values() for g, _, _ in rows}
        samples = {}
        offset  = 0

        for batch in loader:
            size = batch[0].shape[0]

            for g in needed:
                if offset <= g < offset + size:
                    samples[g] = (batch[0][g - offset], batch[1][g - offset])

            offset += size

        images  = []
        targets = []
        pixels  = []

        for name in ExampleSelector.CATEGORIES:
            for rank, (g, h, w) in enumerate(selected[name], start=1):
                image, target = samples[g]

                images.append(image)
                targets.append(target)
                pixels.append((len(pixels), h, w, f"{name}_{rank}"))

        return torch.stack(images), torch.stack(targets), pixels

    def _fixed_ylims(self, criterion) -> list:
        """Returns the fixed y limit of every panel, taken from its label curve peak.

        Args:
            criterion : Training criterion supplying the normalisation stats and curve reconstruction.

        Returns:
            One (0.0, upper) amplitude limit per selected pixel, in pixel order.
        """
        gt_phys   = criterion.norm_stats.denormalize_output(self._targets.float())
        gt_curves = criterion.reconstruct(gt_phys)

        return [(0.0, max(float(gt_curves[row, :, h, w].max()), 1e-6) * self.HEADROOM) for row, h, w, _ in self._pixels]

    def capture_reference(self, loader, criterion) -> None:
        """Selects and caches the reference pixels once, disabling the figures if none qualify.

        Args:
            loader    : Loader yielding (inputs, targets) batches over the validation split.
            criterion : Training criterion supplying the Gaussian layout, the active-amplitude threshold, the normalisation stats and the logger.
        """
        if self._images is not None or self._disabled:
            return

        selector = ExampleSelector(criterion.gaussian_cfg.params_per_gaussian, criterion.loss_cfg.amp_zero_thr)
        selected = selector.select(loader, criterion.norm_stats.denormalize_output)

        for name in ExampleSelector.CATEGORIES:
            if len(selected[name]) < len(ExampleSelector.QUANTILES):
                criterion.logger.warning(f"Reconstruction figures: found {len(selected[name])}/{len(ExampleSelector.QUANTILES)} validation pixels for category '{name}'.")

        if not any(len(rows) for rows in selected.values()):
            criterion.logger.warning("Reconstruction figures disabled: no validation pixel matches any example category.")
            self._disabled = True
            return

        images, targets, pixels = self._collect(loader, selected)

        self._images  = images.to(self.device)
        self._targets = targets.to(self.device)
        self._pixels  = pixels
        self._ylims   = self._fixed_ylims(criterion)

    def _figure(self, z_axis, pred_curve, gt_curve, ylim):
        """Draws one prediction-versus-label z profile.

        Args:
            z_axis     : Z axis in metres, of shape (n_z,).
            pred_curve : Predicted amplitudes of shape (n_z,).
            gt_curve   : Label amplitudes of shape (n_z,).
            ylim       : Fixed amplitude limits of the panel.

        Returns:
            The matplotlib figure, left open for the tracker to consume.
        """
        fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=120)

        ax.plot(z_axis, gt_curve,   color="black",   linestyle="--", linewidth=1.2, label="Ground truth")
        ax.plot(z_axis, pred_curve, color="#1f77b4", linestyle="-",  linewidth=1.4, label="Prediction")

        ax.set_xlabel("Z")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(*ylim)
        ax.legend(frameon=False)
        fig.tight_layout()

        return fig

    @torch.no_grad()
    def log(self, model, criterion, epoch: int) -> None:
        """Logs one reconstruction figure per reference pixel for the given epoch.

        Args:
            model     : Model evaluated on the cached reference inputs.
            criterion : Training criterion supplying the curve reconstruction and the z axis.
            epoch     : Epoch the figures are logged under.
        """
        if self._images is None:
            return

        pred                    = model(self._images)
        pred_curves, exp_curves = criterion.curves(pred, self._targets)
        x                       = criterion.z_axis.detach().cpu().numpy()

        for (row, h, w, tag), ylim in zip(self._pixels, self._ylims):
            pred_curve = pred_curves[row, :, h, w].float().cpu().numpy()
            gt_curve   = exp_curves[row, :, h, w].float().cpu().numpy()

            fig = self._figure(x, pred_curve, gt_curve, ylim)
            self.tracker.log_figure(f"reconstruction/{tag}/val", fig, epoch)
