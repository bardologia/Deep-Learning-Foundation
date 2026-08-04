from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Dict, List, Tuple

import numpy as np
import torch
from scipy.ndimage import uniform_filter

from configuration.inference            import InferenceConfig
from pipelines.backbone.inference.loader import Run
from tools.data.io                       import FileIO
from tools.loss.physical_loss            import PhysicalLoss
from tools.monitoring.logger             import Logger
from tools.sar                           import GeometryField


@dataclass
class DataConsistency:
    coherence_error_map  : np.ndarray
    covariance_error_map : np.ndarray
    valid_mask           : np.ndarray
    track_labels         : List[str]
    metrics              : Dict[str, float]
    track_rows           : List[dict]


class PhaseResidualStats:

    @staticmethod
    def offset(phasor_sum: complex) -> float:
        return float(np.angle(phasor_sum))

    @staticmethod
    def spread(agreement: float) -> float:
        r = float(np.clip(agreement, 1e-6, 1.0))
        return float(np.sqrt(-2.0 * np.log(r)))


class DataConsistencyEvaluator:
    def __init__(self, run: Run, cfg: InferenceConfig, logger: Logger) -> None:
        self._run   = run
        self.cfg    = cfg
        self.logger = logger
        self.device = torch.device(cfg.device)

    def _height_axis_convention(self) -> str:
        path = Path(self.cfg.run_directory) / "docs" / "trainer_config.json"
        if not path.is_file():
            raise FileNotFoundError(f"Data-consistency evaluation derives the height-axis convention from the training run, but {path} is missing; the run predates saved trainer configs, restore its docs/trainer_config.json or re-train.")

        convention = str(FileIO.load_json(path)["geometry"]["height_axis_convention"])
        self.logger.subsection(f"Height-axis convention from training run : {convention}")

        return convention

    def _load_kz(self) -> Tuple[np.ndarray, List[str]]:
        path = Path(self._run.dataset_config.preprocessing_run_directory) / "meta" / GeometryField.FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Data-consistency evaluation requires the per-pixel geometry field but {path} is missing; re-run preprocessing to generate it or disable compute_data_consistency.")

        field  = GeometryField.load(path).subset(self._run.dataset_config.secondary_labels)
        sliced = field.slice(*self._run.split_region.local_slices(self._run.global_crop))
        kz     = sliced.kz(self._height_axis_convention()).astype(np.float32)

        H = self._run.split_region.azimuth_size
        W = self._run.split_region.range_size
        if kz.shape[1:] != (H, W):
            raise ValueError(f"Geometry-field kz shape {kz.shape[1:]} does not match the split region {(H, W)}.")

        self.logger.kv_table(sliced.describe(), title="Geometry Field (data consistency)")

        return kz, list(sliced.labels)

    def _measured_unit_phasors(self, kz: np.ndarray) -> np.ndarray:
        inputs = self._run.complex_inputs
        if inputs is None or not np.iscomplexobj(inputs):
            raise ValueError("Data-consistency evaluation requires the complex input stack on the run; the loader did not provide complex inputs.")

        n_secondaries  = self._run.n_secondaries
        interferograms = np.asarray(inputs[1 + n_secondaries:])

        if interferograms.shape[0] != kz.shape[0] - 1:
            raise ValueError(f"Run provides {interferograms.shape[0]} interferograms but the geometry field lists {kz.shape[0] - 1} secondary tracks; the selections are inconsistent.")

        amplitude = np.abs(interferograms)
        phasor    = interferograms / (amplitude + 1e-30)

        size = int(self.cfg.phase_multilook)
        if size > 1:
            real   = uniform_filter(phasor.real, size=(1, size, size))
            imag   = uniform_filter(phasor.imag, size=(1, size, size))
            phasor = (real + 1j * imag).astype(np.complex64)

        magnitude = np.abs(phasor)
        phasor    = (phasor / (magnitude + 1e-30)).astype(np.complex64)
        phasor[amplitude <= 0.0] = 0.0

        return phasor

    def _chunk_rows(self, n_elev: int, width: int) -> int:
        return max(1, int(4_000_000 // max(1, n_elev * width)))

    def _evaluate_chunks(self, pred: np.ndarray, gt: np.ndarray, kz: np.ndarray, measured: np.ndarray, x_axis_np: np.ndarray) -> dict:
        n_elev, H, W = gt.shape
        n_tracks     = kz.shape[0]
        dx           = float(x_axis_np[1] - x_axis_np[0])
        floor        = float(self.cfg.physics_floor)
        x_axis       = torch.from_numpy(np.asarray(x_axis_np, dtype=np.float32)).to(self.device)

        coherence_map  = np.zeros((H, W), dtype=np.float32)
        covariance_map = np.zeros((H, W), dtype=np.float32)
        mask_map       = np.zeros((H, W), dtype=np.float32)

        track_error_sum = np.zeros(n_tracks, dtype=np.float64)
        track_mask_sum  = np.zeros(n_tracks, dtype=np.float64)

        n_secondaries = n_tracks - 1
        aligned_sum   = {"gt": np.zeros(n_secondaries, dtype=np.complex128), "pred": np.zeros(n_secondaries, dtype=np.complex128)}
        flipped_sum   = {"gt": np.zeros(n_secondaries, dtype=np.complex128), "pred": np.zeros(n_secondaries, dtype=np.complex128)}
        phase_valid   = np.zeros(n_secondaries, dtype=np.float64)

        rows = self._chunk_rows(n_elev, W)

        with torch.no_grad():
            for a0 in range(0, H, rows):
                a1 = min(a0 + rows, H)

                pred_t = torch.from_numpy(np.ascontiguousarray(pred[:, a0:a1], dtype=np.float32)).unsqueeze(0).to(self.device)
                gt_t   = torch.from_numpy(np.ascontiguousarray(gt[:, a0:a1],   dtype=np.float32)).unsqueeze(0).to(self.device)
                kz_t   = torch.from_numpy(np.ascontiguousarray(kz[:, a0:a1])).unsqueeze(0).to(self.device)

                cov_val, _ = PhysicalLoss.covariance_matching_pp_map(pred_t, gt_t, kz_t, x_axis, dx, floor)

                covariance_map[a0:a1] = cov_val[0].cpu().numpy()

                t0  = gt_t.sum(dim=1) * dx
                p0c = (pred_t.sum(dim=1) * dx).clamp(min=floor)
                t0c = t0.clamp(min=floor)

                mask_np       = (t0 > floor)[0].cpu().numpy()
                mask_count    = float(mask_np.sum())
                coherence_sum = np.zeros((a1 - a0, W), dtype=np.float64)

                mask_map[a0:a1] = mask_np.astype(np.float32)

                for track in range(n_tracks):
                    gp = PhysicalLoss.synthesise_track(pred_t, kz_t[:, track], x_axis, dx) / p0c
                    gs = PhysicalLoss.synthesise_track(gt_t,   kz_t[:, track], x_axis, dx) / t0c

                    err            = ((gp - gs).abs() ** 2)[0].cpu().numpy()
                    coherence_sum += err

                    track_error_sum[track] += float(err[mask_np].sum())
                    track_mask_sum[track]  += mask_count

                    if track == 0:
                        continue

                    syn_unit = {
                        "gt"   : (gs / (gs.abs() + 1e-30))[0].cpu().numpy(),
                        "pred" : (gp / (gp.abs() + 1e-30))[0].cpu().numpy(),
                    }

                    meas  = measured[track - 1, a0:a1]
                    valid = mask_np & (np.abs(meas) > 0.0)

                    phase_valid[track - 1] += float(valid.sum())
                    for source in ("gt", "pred"):
                        aligned_sum[source][track - 1] += complex((meas[valid] * np.conj(syn_unit[source][valid])).sum())
                        flipped_sum[source][track - 1] += complex((meas[valid] * syn_unit[source][valid]).sum())

                coherence_map[a0:a1] = (coherence_sum / n_tracks).astype(np.float32)

        return {
            "coherence_map"   : coherence_map,
            "covariance_map"  : covariance_map,
            "mask"            : mask_map > 0.0,
            "track_error_sum" : track_error_sum,
            "track_mask_sum"  : track_mask_sum,
            "aligned_sum"     : aligned_sum,
            "flipped_sum"     : flipped_sum,
            "phase_valid"     : phase_valid,
        }

    def _masked_stats(self, values: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
        selected = values[mask]
        if selected.size == 0:
            raise ValueError(f"Data-consistency mask selected zero pixels; physics_floor={self.cfg.physics_floor} excludes the whole split region.")

        return {
            f"{prefix}_mean"   : float(selected.mean(dtype=np.float64)),
            f"{prefix}_median" : float(np.median(selected)),
            f"{prefix}_p95"    : float(np.percentile(selected, 95)),
        }

    def _assemble_metrics(self, chunks: dict, labels: List[str]) -> Dict[str, float]:
        mask    = chunks["mask"]
        metrics = {}

        metrics.update(self._masked_stats(chunks["coherence_map"],  mask, "physics_coherence_error"))
        metrics.update(self._masked_stats(chunks["covariance_map"], mask, "physics_covariance_error"))
        metrics["physics_valid_fraction"] = float(mask.mean(dtype=np.float64))

        for track, label in enumerate(labels):
            denominator = max(chunks["track_mask_sum"][track], 1.0)
            metrics[f"physics_coherence_error_track_{label}"] = float(chunks["track_error_sum"][track] / denominator)

        for source in ("gt", "pred"):
            aligned = []
            flipped = []

            for index, label in enumerate(labels[1:]):
                count = max(chunks["phase_valid"][index], 1.0)
                r_al  = float(np.abs(chunks["aligned_sum"][source][index]) / count)
                r_fl  = float(np.abs(chunks["flipped_sum"][source][index]) / count)

                metrics[f"phase_agreement_{source}_track_{label}"]         = r_al
                metrics[f"phase_agreement_{source}_flipped_track_{label}"] = r_fl
                metrics[f"physics_phase_offset_{source}_track_{label}"]    = PhaseResidualStats.offset(chunks["aligned_sum"][source][index])
                metrics[f"physics_phase_spread_{source}_track_{label}"]    = PhaseResidualStats.spread(r_al)
                aligned.append(r_al)
                flipped.append(r_fl)

            metrics[f"phase_agreement_{source}_mean"]         = float(np.mean(aligned))
            metrics[f"phase_agreement_{source}_flipped_mean"] = float(np.mean(flipped))

        return metrics

    def _spectroscopy_rows(self, metrics: Dict[str, float], labels: List[str], kz: np.ndarray) -> List[dict]:
        rows = []
        for index, label in enumerate(labels[1:], start=1):
            rows.append({
                "label"           : label,
                "kz_abs_mean"     : float(np.abs(kz[index]).mean(dtype=np.float64)),
                "coherence_error" : metrics[f"physics_coherence_error_track_{label}"],
                "agreement_pred"  : metrics[f"phase_agreement_pred_track_{label}"],
                "agreement_gt"    : metrics[f"phase_agreement_gt_track_{label}"],
                "offset_pred"     : metrics[f"physics_phase_offset_pred_track_{label}"],
                "spread_pred"     : metrics[f"physics_phase_spread_pred_track_{label}"],
                "spread_gt"       : metrics[f"physics_phase_spread_gt_track_{label}"],
            })

        rows.sort(key=lambda row: row["kz_abs_mean"])

        for row in rows:
            metrics[f"physics_kz_abs_mean_track_{row['label']}"] = row["kz_abs_mean"]

        return rows

    def _report(self, metrics: Dict[str, float]) -> None:
        self.logger.kv_table({
            "Coherence error (mean)"      : f"{metrics['physics_coherence_error_mean']:.6f}",
            "Covariance error (mean)"     : f"{metrics['physics_covariance_error_mean']:.6f}",
            "Valid fraction"              : f"{metrics['physics_valid_fraction']:.4f}",
            "Phase agreement GT (mean)"   : f"{metrics['phase_agreement_gt_mean']:.4f}",
            "Phase agreement GT flipped"  : f"{metrics['phase_agreement_gt_flipped_mean']:.4f}",
            "Phase agreement pred (mean)" : f"{metrics['phase_agreement_pred_mean']:.4f}",
        }, title="Interferometric data consistency")

        if metrics["phase_agreement_gt_flipped_mean"] > metrics["phase_agreement_gt_mean"]:
            self.logger.subsection("WARNING: the measured interferogram phase agrees better with the SIGN-FLIPPED synthesized coherence; the kz sign convention (capon_phase_sign) is likely inverted for this stack.")

    def run(self, pred_curves: np.ndarray, gt_curves: np.ndarray, x_axis_np: np.ndarray) -> DataConsistency:
        self.logger.section("[Inference: Data Consistency]")

        kz, labels = self._load_kz()
        measured   = self._measured_unit_phasors(kz)
        chunks     = self._evaluate_chunks(pred_curves, gt_curves, kz, measured, x_axis_np)
        metrics    = self._assemble_metrics(chunks, labels)
        rows       = self._spectroscopy_rows(metrics, labels, kz)

        self._report(metrics)

        return DataConsistency(
            coherence_error_map  = chunks["coherence_map"],
            covariance_error_map = chunks["covariance_map"],
            valid_mask           = chunks["mask"],
            track_labels         = labels,
            metrics              = metrics,
            track_rows           = rows,
        )
