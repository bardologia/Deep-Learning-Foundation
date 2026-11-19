"use strict";

class FlowSketches {

  static pulse(svg, gsap) {
    if (!gsap) return null;

    const els = svg.querySelectorAll(".sk-live");
    if (!els.length) return null;

    return gsap.to(els, { opacity: 0.35, duration: 1.25, ease: "sine.inOut", repeat: -1, yoyo: true, stagger: { each: 0.16, from: "start" } });
  }

  static grid(cells, build) {
    let out = "";
    for (let r = 0; r < cells; r++) for (let c = 0; c < cells; c++) out += build(r, c);
    return out;
  }

  static CATALOG = {

  threshold: {
    tip: "Samples below t_f = 0.25 x peak are zeroed and bins past H_tr = 170 dropped before the fit sees the profile.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="120" x2="214" y2="120"/>
      <line class="skl-dash c-mid" x1="24" y1="96" x2="180" y2="96"/>
      <text x="26" y="92" style="fill:#f5b971;font-size:7px">floor</text>
      <line class="skl-dash c-faint" x1="180" y1="22" x2="180" y2="122"/>
      <text x="184" y="32" style="fill:#c3cbd8;font-size:7px">H_tr</text>
      <path class="skl-draw c-meas" d="M24 118 L52 70 L64 104 L98 44 L114 100 L150 78 L168 106 L186 90 L204 86" style="fill:none;opacity:.28"/>
      <path class="skl-draw c-cal" d="M24 120 L52 70 L64 104 L82 120 L98 44 L114 100 L132 120 L150 78 L168 120 L180 120" style="fill:none"/>`; },
    anim: null,
  },

  activity: {
    tip: "A profile is fitted only if its peak clears tau_a = 1e-3; otherwise skipped with zero parameters.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <line class="skl-dash c-mid" x1="24" y1="58" x2="214" y2="58"/>
      <text x="26" y="52" style="fill:#f5b971;font-size:7px">tau_a</text>
      <path class="skl-draw c-cal" d="M40 118 L52 110 L66 44 L80 112 L92 118" style="fill:none"/>
      <circle cx="66" cy="44" r="4" class="skl-pop f-cal"/>
      <path class="skl-draw c-faint" d="M150 118 L162 112 L176 88 L190 113 L202 118" style="fill:none;opacity:.5"/>
      <circle cx="176" cy="88" r="4" class="skl-pop f-faint"/>
      <text x="176" y="78" text-anchor="middle" style="fill:#aab6c3;font-size:8px">skip</text>`; },
    anim: null,
  },

  pnorm: {
    tip: "Dividing by the per-profile max sets the tallest peak to one, making the loss comparable across pixels.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="120" x2="214" y2="120"/>
      <line class="skl-axis" x1="24" y1="22" x2="24" y2="120"/>
      <line class="skl-dash c-cal" x1="24" y1="40" x2="214" y2="40"/>
      <text x="208" y="36" text-anchor="end" style="fill:#4fd6c4;font-size:7px">1.0</text>
      <path class="skl-draw c-faint" d="M24 116 L60 96 L96 70 L120 56 L150 92 L188 110 L214 114" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-cal" d="M24 112 L60 78 L96 40 L120 40 L150 84 L188 110 L214 116" style="fill:none"/>`; },
    anim: null,
  },

  peakfind: {
    tip: "find_peaks keeps a maximum only if its prominence reaches p_frac = 0.05 of the peak and it sits d_min bins from rivals.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="120" x2="214" y2="120"/>
      <path class="skl-draw c-meas" d="M24 118 L44 96 L60 40 L78 102 L96 110 L114 70 L132 104 L150 112 L168 58 L190 104 L214 116" style="fill:none"/>
      <circle class="sk-live skl-pop f-cal" cx="60" cy="40" r="4.5"/>
      <circle class="sk-live skl-pop f-cal" cx="168" cy="58" r="4.5"/>
      <circle cx="114" cy="70" r="4" class="skl-pop f-faint"/>`; },
    anim: FlowSketches.pulse,
  },

  geometry: {
    tip: "sigma0 = sigma_base / D_sigma (D_sigma = 4) seeds the width; Adam later clamps it between one bin and half the span.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="120" x2="40" y2="24"/>
      <line class="skl-dash c-cal" x1="34" y1="40" x2="200" y2="40"/>
      <text x="46" y="36" style="fill:#4fd6c4;font-size:7px">sigma_hi = span/2</text>
      <line class="skl-dash c-meas" x1="34" y1="108" x2="200" y2="108"/>
      <text x="46" y="120" style="fill:#6ea8ff;font-size:7px">sigma_lo = 1 bin</text>
      <path class="skl-draw c-mid" d="M126 116 q44 -76 88 0" style="fill:none"/>
      <line class="skl-draw c-mid" x1="148" y1="80" x2="192" y2="80" style="stroke-width:1.4"/>
      <text x="158" y="74" style="fill:#f5b971;font-size:8px">sigma0</text>`; },
    anim: null,
  },

  residfill: {
    tip: "If fewer than K peaks are found, each is masked over half-width d_min and the rest filled by repeated argmax.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <rect x="32" y="22" width="32" height="96" class="skl-pop f-faint" style="opacity:.4"/>
      <rect x="92" y="22" width="32" height="96" class="skl-pop f-faint" style="opacity:.4"/>
      <path class="skl-draw c-meas" d="M24 116 L48 70 L66 100 L86 84 L108 50 L130 96 L150 78 L172 104 L196 66 L214 110" style="fill:none;opacity:.45"/>
      <circle cx="48" cy="70" r="4" class="skl-pop f-cal"/>
      <circle cx="108" cy="50" r="4" class="skl-pop f-cal"/>
      <circle cx="196" cy="66" r="4.5" class="skl-pop f-mid"/>
      <text x="196" y="56" text-anchor="middle" style="fill:#f5b971;font-size:7px">argmax</text>`; },
    anim: null,
  },

  seed: {
    tip: "Amplitude and mean are read off the peaks as the seed; frozen in sigma mode, refined only when the mode frees them.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <path class="skl-draw c-meas" d="M24 116 L52 100 L78 44 L104 102 L130 110 L158 60 L190 104 L214 112" style="fill:none;opacity:.45"/>
      <line class="skl-dash c-cal" x1="78" y1="44" x2="78" y2="118"/>
      <line class="skl-draw c-cal" x1="62" y1="44" x2="94" y2="44" style="stroke-width:1.6"/>
      <circle cx="78" cy="44" r="4" class="skl-pop f-cal"/>
      <line class="skl-dash c-cal" x1="158" y1="60" x2="158" y2="118"/>
      <line class="skl-draw c-cal" x1="142" y1="60" x2="174" y2="60" style="stroke-width:1.6"/>
      <circle cx="158" cy="60" r="4" class="skl-pop f-cal"/>
      <text x="118" y="132" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">a, mu seed</text>`; },
    anim: null,
  },

  objective: {
    tip: "sigma is always fit (amp/mu per mode); the loss is the MSE between the K-Gaussian sum and the profile, sigma floored at 1e-6.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="116" x2="214" y2="116"/>
      <path class="skl-draw c-mid" d="M24 114 q40 -74 80 0 q40 -54 86 0 L214 114" style="fill:none"/>
      <path class="skl-draw c-cal" d="M24 114 q40 -54 80 0 q40 -40 86 0 L214 114" style="fill:none"/>
      <line class="skl-draw c-fin" x1="64" y1="74" x2="64" y2="92" style="stroke-width:1.6"/>
      <line class="skl-draw c-fin" x1="150" y1="82" x2="150" y2="98" style="stroke-width:1.6"/>
      <text x="120" y="22" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">L = mean(residual^2)</text>`; },
    anim: null,
  },

  adam: {
    tip: "Adam (eta = 0.2) runs as one lax.scan of T = 3000 steps, clamping each width to [sigma_lo, sigma_hi].",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="120" x2="214" y2="120"/>
      <line class="skl-axis" x1="28" y1="22" x2="28" y2="120"/>
      <line class="skl-dash c-faint" x1="28" y1="40" x2="214" y2="40"/>
      <line class="skl-dash c-faint" x1="28" y1="104" x2="214" y2="104"/>
      <path class="skl-draw c-cal" d="M28 100 C70 96 84 70 108 56 S170 44 214 44" style="fill:none"/>
      <circle class="sk-live skl-pop f-cal" cx="214" cy="44" r="4"/>`; },
    anim: FlowSketches.pulse,
  },

  scoreK: {
    tip: "Each order K scores as MSE_K + lambda_K x K, a flat per-slot charge, so a Gaussian is kept only when it earns it.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="32" y1="118" x2="214" y2="118"/>
      <rect x="44" y="58" width="20" height="60" class="skl-pop f-mid"/>
      <rect x="44" y="52" width="20" height="6" class="skl-pop f-fin"/>
      <rect x="92" y="92" width="20" height="26" class="skl-pop f-mid"/>
      <rect x="92" y="80" width="20" height="12" class="skl-pop f-fin"/>
      <rect x="140" y="86" width="20" height="32" class="skl-pop f-mid"/>
      <rect x="140" y="62" width="20" height="24" class="skl-pop f-fin"/>
      <rect x="188" y="80" width="20" height="38" class="skl-pop f-mid"/>
      <rect x="188" y="44" width="20" height="36" class="skl-pop f-fin"/>
      <text x="54" y="130" style="fill:#aab6c3;font-size:7px">1</text>
      <text x="146" y="130" style="fill:#aab6c3;font-size:7px">3</text>`; },
    anim: null,
  },

  selectK: {
    tip: "The penalised score is minimised over K, ties broken toward the smaller K.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="32" y1="116" x2="214" y2="116"/>
      <rect x="44" y="48" width="22" height="68" class="skl-pop f-mid"/>
      <rect class="sk-live skl-pop f-cal" x="92" y="84" width="22" height="32"/>
      <rect x="140" y="76" width="22" height="40" class="skl-pop f-mid"/>
      <rect x="188" y="62" width="22" height="54" class="skl-pop f-mid"/>
      <path class="skl-draw c-cal" d="M103 36 L103 62 M97 56 L103 62 L109 56" style="fill:none"/>
      <text x="103" y="30" text-anchor="middle" style="fill:#4fd6c4;font-size:9px;font-weight:700">K*=2</text>`; },
    anim: FlowSketches.pulse,
  },

  rescale: {
    tip: "The winner's amplitudes scale back by s to raw units; means and widths pass through unchanged.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <line class="skl-dash c-faint" x1="24" y1="56" x2="214" y2="56"/>
      <text x="28" y="52" style="fill:#c3cbd8;font-size:7px">1.0 (norm)</text>
      <path class="skl-draw c-faint" d="M40 116 q40 -56 80 0 q34 -40 70 0" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-fin" d="M40 116 q40 -84 80 0 q34 -60 70 0" style="fill:none"/>
      <text x="150" y="40" style="fill:#c4a3ff;font-size:8px">x s</text>`; },
    anim: null,
  },

  assemble: {
    tip: "Active components sort by ascending mean, inactive slots drop to the end, into the interleaved 3K target.",
    build(svg) { svg.innerHTML = `
      <rect x="40" y="40" width="26" height="20" class="skl-pop f-cal"/>
      <rect x="40" y="66" width="26" height="20" class="skl-pop f-cal"/>
      <rect x="40" y="92" width="26" height="20" class="skl-pop f-faint"/>
      <text x="86" y="80" style="fill:#c3cbd8;font-size:12px">&#8594;</text>
      <rect x="118" y="58" width="16" height="28" class="skl-pop f-fin"/>
      <rect x="136" y="58" width="16" height="28" class="skl-pop f-fin"/>
      <rect x="154" y="58" width="16" height="28" class="skl-pop f-fin"/>
      <rect x="172" y="58" width="16" height="28" class="skl-pop f-faint" style="opacity:.4"/>
      <rect x="190" y="58" width="16" height="28" class="skl-pop f-faint" style="opacity:.4"/>
      <text x="118" y="100" style="fill:#c4a3ff;font-size:7px">a mu sig a mu sig</text>`; },
    anim: null,
  },

  quality: {
    tip: "Per-pixel R-squared compares the reconstruction to the profile (1e-12 stabiliser), painting a fit-quality map.",
    build(svg) {
      const cells = [[0.95,"f-cal"],[0.9,"f-cal"],[0.6,"f-mid"],[0.7,"f-mid"],[0.92,"f-cal"],[0.88,"f-cal"],[0.85,"f-cal"],[0.4,"f-faint"],[0.9,"f-cal"]];
      let m = "";
      cells.forEach((cl, i) => { const x = 132 + (i % 3) * 26, y = 36 + Math.floor(i / 3) * 26; m += `<rect class="sk-live skl-pop ${cl[1]}" x="${x}" y="${y}" width="22" height="22" rx="2" style="opacity:${cl[0]}"/>`; });
      svg.innerHTML = `
        <line class="skl-axis" x1="24" y1="116" x2="108" y2="116"/>
        <path class="skl-draw c-meas" d="M24 114 L48 60 L72 96 L96 110" style="fill:none;opacity:.5"/>
        <path class="skl-draw c-cal" d="M24 114 q24 -58 44 -54 q22 4 22 40 q8 14 18 14" style="fill:none"/>
        <text x="40" y="34" style="fill:#4fd6c4;font-size:9px">R2</text>
        ${m}`;
    },
    anim: FlowSketches.pulse,
  },

  diagnostics: {
    tip: "Post-hoc: the runner-up margin over L_K* flags ambiguous pixels (< 0.05); contrast uses the lowest-amplitude quartile as floor.",
    build(svg) { svg.innerHTML = `
      <text x="34" y="30" style="fill:#c4a3ff;font-size:8px">m_rel</text>
      <rect x="34" y="36" width="92" height="12" rx="2" class="skl-pop f-faint" style="opacity:.3"/>
      <rect x="34" y="36" width="52" height="12" rx="2" class="skl-pop f-fin"/>
      <line class="skl-axis" x1="132" y1="120" x2="214" y2="120"/>
      <line class="skl-dash c-faint" x1="132" y1="96" x2="214" y2="96"/>
      <text x="214" y="92" text-anchor="end" style="fill:#c3cbd8;font-size:7px">floor (Q1)</text>
      <path class="skl-draw c-cal" d="M134 120 C152 50, 180 50, 198 120" style="fill:rgba(79,214,196,0.10)"/>
      <line class="skl-dash c-cal" x1="166" y1="96" x2="166" y2="62" style="stroke-width:1.4"/>
      <text x="171" y="82" style="fill:#4fd6c4;font-size:7px">C_dB</text>`; },
    anim: null,
  },

  splitgeom: {
    tip: "Row splits 70/15/15 into contiguous, non-overlapping train, val and test bands.",
    build(svg) { svg.innerHTML = `
      <rect x="40" y="22" width="160" height="74" class="skl-pop f-mid" style="opacity:.85"/>
      <rect x="40" y="96" width="160" height="16" class="skl-pop f-cal" style="opacity:.85"/>
      <rect x="40" y="112" width="160" height="16" class="skl-pop f-fin" style="opacity:.85"/>
      <text x="120" y="62" text-anchor="middle" style="fill:#0b1014;font-size:9px">train 70%</text>
      <text x="120" y="107" text-anchor="middle" style="fill:#0b1014;font-size:7px">val 15%</text>
      <text x="120" y="123" text-anchor="middle" style="fill:#0b1014;font-size:7px">test 15%</text>`; },
    anim: null,
  },

  localslice: {
    tip: "Subtracting the crop origin row0 turns absolute row bounds into zero-based slices.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="50" x2="214" y2="50"/>
      <text x="30" y="40" style="fill:#c3cbd8;font-size:7px">absolute</text>
      <rect x="92" y="42" width="78" height="16" class="skl-pop f-mid"/>
      <line class="skl-axis" x1="30" y1="104" x2="214" y2="104"/>
      <text x="30" y="124" style="fill:#c3cbd8;font-size:7px">local (0-based)</text>
      <rect x="30" y="96" width="78" height="16" class="skl-pop f-cal"/>
      <path class="skl-draw c-cal" d="M131 60 L60 94 M66 90 L60 94 L66 98" style="fill:none"/>
      <text x="138" y="80" style="fill:#4fd6c4;font-size:8px">- row0</text>`; },
    anim: null,
  },

  secselect: {
    tip: "Qualified labels in L_req map to positional indices, gathered alike from the secondaries and derived.",
    build(svg) {
      let row = "";
      for (let i = 0; i < 7; i++) { const sel = (i === 3 || i === 5 || i === 6); row += `<rect class="skl-pop ${sel ? "f-cal" : "f-faint"}" x="${28 + i * 24}" y="32" width="20" height="20" style="opacity:${sel ? 1 : 0.5}"/>`; if (sel) row += `<rect class="sk-live" x="${28 + i * 24}" y="32" width="20" height="20" style="fill:none;stroke:#4fd6c4;stroke-width:2.5"/>`; }
      svg.innerHTML = `
        ${row}
        <text x="120" y="80" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">pi = {3, 5, 6}</text>
        <rect x="100" y="96" width="20" height="20" class="skl-pop f-cal"/>
        <rect x="124" y="96" width="20" height="20" class="skl-pop f-cal"/>
        <rect x="148" y="96" width="20" height="20" class="skl-pop f-cal"/>`;
    },
    anim: FlowSketches.pulse,
  },

  stack: {
    tip: "Primary at slot 0, Ns secondaries in X[1:1+Ns], Ni derived after, into one complex buffer.",
    build(svg) { svg.innerHTML = `
      <rect x="38" y="30" width="44" height="18" class="skl-pop f-meas"/>
      <text x="60" y="43" text-anchor="middle" style="fill:#0b1014;font-size:8px">s0</text>
      <rect x="38" y="52" width="44" height="18" class="skl-pop f-mid"/>
      <rect x="38" y="74" width="44" height="18" class="skl-pop f-mid"/>
      <rect x="38" y="96" width="44" height="18" class="skl-pop f-cal"/>
      <rect x="38" y="118" width="44" height="18" class="skl-pop f-cal"/>
      <text x="100" y="86" style="fill:#c3cbd8;font-size:12px">&#8594;</text>
      <rect x="134" y="30" width="72" height="106" rx="3" class="skl-draw c-fin" style="fill:rgba(196,163,255,0.08)"/>
      <text x="170" y="92" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">X buffer</text>`; },
    anim: null,
  },

  patchgrid: {
    tip: "A strided PxP window tiles the region; the row/col counts let the last patch cover the far border.",
    build(svg) {
      let g = "";
      for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) g += `<rect x="${40 + c * 41}" y="${30 + r * 32}" width="40" height="30" class="skl-pop f-mid" style="opacity:.16;stroke:#f5b971;stroke-width:.8"/>`;
      svg.innerHTML = `${g}<rect class="sk-live" x="81" y="62" width="40" height="30" style="fill:none;stroke:#4fd6c4;stroke-width:2.2"/>`;
    },
    anim: FlowSketches.pulse,
  },

  padgeom: {
    tip: "The row deficit pv splits floor(pv/2) on top, the rest below; odd deficits go to the bottom.",
    build(svg) { svg.innerHTML = `
      <rect x="78" y="44" width="84" height="62" class="skl-pop f-mid" style="opacity:.8"/>
      <text x="120" y="80" text-anchor="middle" style="fill:#0b1014;font-size:8px">region</text>
      <rect x="78" y="28" width="84" height="14" class="skl-pop f-cal" style="opacity:.5"/>
      <text x="170" y="38" style="fill:#4fd6c4;font-size:7px">floor(pv/2)</text>
      <rect x="78" y="106" width="84" height="18" class="skl-pop f-cal" style="opacity:.5"/>
      <text x="170" y="120" style="fill:#4fd6c4;font-size:7px">pv-floor(pv/2)</text>`; },
    anim: null,
  },

  extract: {
    tip: "The read window is deep-copied (never aliases the mmap), then reflect-padded by the shared routine.",
    build(svg) { svg.innerHTML = `
      <rect x="28" y="32" width="74" height="74" class="skl-draw c-faint" style="fill:none"/>
      <rect x="58" y="62" width="44" height="44" class="skl-pop f-meas" style="opacity:.75"/>
      <text x="110" y="78" style="fill:#c3cbd8;font-size:12px">&#8594;</text>
      <rect x="138" y="42" width="64" height="64" class="skl-pop f-cal" style="opacity:.3"/>
      <rect x="148" y="52" width="44" height="44" class="skl-pop f-mid"/>
      <text x="170" y="124" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">pad(copy)</text>`; },
    anim: null,
  },

  represent: {
    tip: "An amplitude channel keeps |p|, a derived channel keeps its phase; normalised re/im divide by m=|p| (1 if |p|=0).",
    build(svg) { svg.innerHTML = `
      <circle cx="64" cy="70" r="32" class="skl-axis" style="fill:none;opacity:.5"/>
      <line class="skl-axis" x1="26" y1="70" x2="102" y2="70"/>
      <line class="skl-axis" x1="64" y1="34" x2="64" y2="106"/>
      <line class="skl-draw c-meas" x1="64" y1="70" x2="90" y2="52"/>
      <circle cx="90" cy="52" r="3.2" class="skl-pop f-meas"/>
      <path class="skl-dash c-mid" d="M82 70 A18 18 0 0 0 77 56" style="fill:none"/>
      <text x="64" y="118" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">complex p</text>
      <text x="120" y="74" style="fill:#c3cbd8;font-size:13px">&#8594;</text>
      <rect x="146" y="42" width="44" height="15" rx="2" class="skl-pop f-cal"/>
      <text x="196" y="53" style="fill:#4fd6c4;font-size:7px">|p|</text>
      <rect x="146" y="64" width="30" height="15" rx="2" class="skl-pop f-mid"/>
      <text x="182" y="75" style="fill:#f5b971;font-size:7px">ang</text>`; },
    anim: null,
  },

  assemble_in: {
    tip: "Real channels concatenate in fixed order: primary, secondaries, derived, optional Aux last.",
    build(svg) { svg.innerHTML = `
      <rect x="28" y="44" width="22" height="40" class="skl-pop f-meas"/>
      <rect x="54" y="44" width="22" height="40" class="skl-pop f-mid"/>
      <rect x="80" y="44" width="22" height="40" class="skl-pop f-mid"/>
      <rect x="106" y="44" width="22" height="40" class="skl-pop f-cal"/>
      <rect x="132" y="44" width="22" height="40" class="skl-pop f-faint"/>
      <text x="91" y="100" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">r0 | rS | rI | D</text>
      <rect x="40" y="110" width="130" height="22" rx="2" class="skl-draw c-cal" style="fill:rgba(79,214,196,0.12)"/>
      <text x="105" y="125" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">x : C_in</text>`; },
    anim: null,
  },

  target: {
    tip: "Channels are gathered from the interleaved (a, mu, sigma) GT; all three roles keep every n_g*3 channel.",
    build(svg) {
      const cls = ["f-meas", "f-mid", "f-cal"], roles = ["a", "mu", "s"];
      let row = "", out = "";
      for (let i = 0; i < 9; i++) { const x = 30 + i * 20, r = i % 3; row += `<rect class="skl-pop ${cls[r]}" x="${x}" y="40" width="16" height="22"/><text x="${x + 8}" y="55" text-anchor="middle" style="fill:#0b1014;font-size:7px">${roles[r]}</text>`; out += `<rect class="sk-live skl-pop f-cal" x="${x}" y="92" width="16" height="22"/>`; }
      svg.innerHTML = `<text x="120" y="30" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">interleaved theta</text>${row}<text x="120" y="80" text-anchor="middle" style="fill:#c3cbd8;font-size:12px">&#8595;</text>${out}`;
    },
    anim: FlowSketches.pulse,
  },

  augment_geo: {
    tip: "On train only, the sampled flip/rot90 is applied alike to x and y, keeping every pixel aligned.",
    build(svg) { svg.innerHTML = `
      <rect x="42" y="36" width="52" height="52" class="skl-draw c-meas" style="fill:none"/>
      <polygon points="50,80 50,44 74,44" style="fill:#6ea8ff;opacity:.7"/>
      <text x="68" y="100" text-anchor="middle" style="fill:#6ea8ff;font-size:8px">x</text>
      <text x="120" y="66" text-anchor="middle" style="fill:#f5b971;font-size:8px">flip</text>
      <rect x="146" y="36" width="52" height="52" class="skl-draw c-cal" style="fill:none"/>
      <polygon points="190,80 190,44 166,44" style="fill:#4fd6c4;opacity:.7"/>
      <text x="172" y="100" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">y</text>`; },
    anim: null,
  },

  slotkeys: {
    tip: "The strided layout labels each channel by family/slot, so chan/mag, derived/phase and aux/z get their own norm.",
    build(svg) {
      const L = [["chan/mag", "f-meas", "#6ea8ff"], ["chan/mag", "f-meas", "#6ea8ff"], ["derived/phase", "f-mid", "#f5b971"], ["derived/phase", "f-mid", "#f5b971"], ["aux/z", "f-faint", "#7e8aa0"]];
      let s   = "";
      L.forEach((l, i) => { const x = 34 + i * 36; s += `<rect class="skl-pop ${l[1]}" x="${x}" y="34" width="28" height="28"/><line x1="${x + 14}" y1="64" x2="${x + 14}" y2="92" style="stroke:${l[2]};stroke-width:1.3"/><text x="${x + 14}" y="106" text-anchor="middle" transform="rotate(38 ${x + 14} 106)" style="fill:${l[2]};font-size:6px">${l[0]}</text>`; });
      svg.innerHTML = s;
    },
    anim: null,
  },

  fitstats: {
    tip: "On train only (float64): magnitudes use robust-IQR-log1p, others z-score, derived phase fixed pi; scale floored at 1e-8.",
    build(svg) {
      const hs = [14, 26, 40, 52, 46, 32, 20, 10];
      let b = "";
      hs.forEach((h, i) => { const x = 60 + i * 14; b += `<rect class="skl-pop f-meas" x="${x}" y="${112 - h}" width="12" height="${h}" style="opacity:.8"/>`; });
      svg.innerHTML = `<line class="skl-axis" x1="40" y1="112" x2="200" y2="112"/>${b}<line class="skl-dash c-cal" x1="116" y1="40" x2="116" y2="112"/><text x="116" y="34" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">mu_c</text>`;
    },
    anim: null,
  },

  normalise: {
    tip: "Subtracting mu_c and dividing by s_c gives each slot zero mean and unit scale across every split.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="104" x2="214" y2="104"/>
      <line class="skl-dash c-faint" x1="120" y1="34" x2="120" y2="110"/>
      <text x="120" y="122" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">0</text>
      <path class="skl-draw c-faint" d="M156 104 C166 60 198 60 208 104" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-cal" d="M92 104 C106 48 134 48 148 104" style="fill:none"/>
      <path class="skl-draw c-mid" d="M168 104 L150 104 M156 100 L150 104 L156 108" style="fill:none"/>`; },
    anim: null,
  },

  noise: {
    tip: "On train only, with probability p_N, std-0.01 Gaussian noise jitters x-hat; the target stays untouched.",
    build(svg) { svg.innerHTML = `
      <path class="skl-draw c-faint" d="M30 80 Q70 42 110 80 T190 80" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-fin" d="M30 82 Q50 60 70 64 Q90 36 110 78 Q130 92 150 70 Q170 50 190 76" style="fill:none"/>
      <text x="120" y="116" text-anchor="middle" style="fill:#c4a3ff;font-size:7px">x-hat + N(0, 0.01^2)</text>`; },
    anim: null,
  },

  denorm: {
    tip: "The inverse scales by s_c, adds mu_c, and for log1p slots clips to [0, log1p(1000)] before expm1.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="104" x2="214" y2="104"/>
      <line class="skl-dash c-mid" x1="30" y1="48" x2="214" y2="48"/>
      <text x="208" y="44" text-anchor="end" style="fill:#f5b971;font-size:7px">log1p(1000)</text>
      <path class="skl-draw c-cal" d="M40 104 C72 50 150 48 200 48" style="fill:none"/>
      <text x="120" y="124" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">expm1(clip(x s + mu))</text>`; },
    anim: null,
  },

  forward: {
    tip: "One forward pass maps the input patch to 3K interleaved (a, mu, sigma) channels per pixel.",
    build(svg) { svg.innerHTML = `
      <rect x="24" y="56" width="20" height="40" rx="2" class="skl-pop f-meas"/>
      <path class="skl-draw c-faint" d="M56 48 L100 64 L100 96 L56 112 Z" style="fill:none"/>
      <path class="skl-draw c-faint" d="M108 64 L140 74 L140 94 L108 104 Z" style="fill:none"/>
      <path class="skl-draw c-faint" d="M148 74 L168 82 L168 100 L148 108 Z" style="fill:none"/>
      <rect x="178" y="46" width="34" height="9" rx="2" class="skl-pop f-cal"/>
      <rect x="178" y="58" width="34" height="9" rx="2" class="skl-pop f-cal" style="opacity:.7"/>
      <rect x="178" y="70" width="34" height="9" rx="2" class="skl-pop f-cal" style="opacity:.5"/>
      <text x="195" y="96" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">a mu s</text>`; },
    anim: null,
  },

  tdenorm: {
    tip: "expm1 inverts the log1p amplitude and sigma channels; the pre-exponent value is clamped so the physical output stays within [0, 1000].",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="214" y2="118"/>
      <line class="skl-axis" x1="30" y1="22" x2="30" y2="118"/>
      <line class="skl-dash c-mid" x1="30" y1="40" x2="214" y2="40"/>
      <text x="206" y="36" text-anchor="end" style="fill:#f5b971;font-size:7px">clamp</text>
      <path class="skl-draw c-cal" d="M30 116 Q120 114 166 40 L214 40" style="fill:none"/>`; },
    anim: null,
  },

  clamp: {
    tip: "Out-of-bounds a, mu and sigma clip to grid limits but keep a 0.1 leaky slope so gradients still flow.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="214" y2="118"/>
      <line class="skl-dash c-faint" x1="62" y1="20" x2="62" y2="120"/>
      <line class="skl-dash c-faint" x1="182" y1="20" x2="182" y2="120"/>
      <path class="skl-dash c-faint" d="M30 124 L214 18" style="opacity:.3"/>
      <path class="skl-draw c-cal" d="M30 108 L62 96 L182 34 L214 32" style="fill:none"/>
      <text x="48" y="132" style="fill:#c3cbd8;font-size:7px">min</text>
      <text x="170" y="132" style="fill:#c3cbd8;font-size:7px">max</text>`; },
    anim: null,
  },

  renorm: {
    tip: "(log1p - offset) / scale maps the clamped physical predictions back into label-normalised units.",
    build(svg) { svg.innerHTML = `
      <rect x="26" y="50" width="44" height="50" rx="3" class="skl-pop f-cal"/>
      <text x="48" y="79" text-anchor="middle" style="fill:#0b1014;font-size:9px">12.4</text>
      <text x="48" y="116" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">phys</text>
      <path class="skl-draw c-mid" d="M76 75 L162 75 M154 69 L162 75 L154 81" style="fill:none"/>
      <text x="119" y="68" text-anchor="middle" style="fill:#f5b971;font-size:7px">(log1p-l)/s</text>
      <rect x="170" y="50" width="44" height="50" rx="3" class="skl-pop f-meas"/>
      <text x="192" y="79" text-anchor="middle" style="fill:#0b1014;font-size:9px">0.74</text>
      <text x="192" y="116" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">norm</text>`; },
    anim: null,
  },

  reconstruct: {
    tip: "Predicted and GT parameters each sum K Gaussians into an z curve; the GT curve is built once under no_grad.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="118" x2="216" y2="118"/>
      <path class="skl-draw c-faint" d="M40 118 Q70 70 100 118" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-faint" d="M96 118 Q128 48 160 118" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-faint" d="M150 118 Q176 82 202 118" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-cal" d="M40 118 Q70 70 100 100 Q128 48 160 92 Q176 82 202 118" style="fill:none"/>
      <path class="skl-dash c-meas" d="M40 118 Q72 76 102 98 Q130 54 162 90 Q178 86 202 118" style="fill:none;opacity:.6"/>`; },
    anim: null,
  },

  residual: {
    tip: "The elementwise y-hat - y is the residual shared by the MSE, L1, Huber and Charbonnier terms.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="80" x2="216" y2="80"/>
      <path class="skl-draw c-cal" d="M30 70 Q60 30 90 56 Q120 18 150 50 Q180 44 214 70" style="fill:none"/>
      <path class="skl-dash c-meas" d="M30 74 Q60 40 90 60 Q120 30 150 56 Q180 52 214 74" style="fill:none"/>
      <line class="sk-live skl-pop" x1="60" y1="48" x2="60" y2="60" style="stroke:#f5b971;stroke-width:3;opacity:1"/>
      <line class="sk-live skl-pop" x1="120" y1="22" x2="120" y2="32" style="stroke:#f5b971;stroke-width:3;opacity:1"/>
      <line class="sk-live skl-pop" x1="150" y1="50" x2="150" y2="58" style="stroke:#f5b971;stroke-width:3;opacity:1"/>`; },
    anim: FlowSketches.pulse,
  },

  curvepoint: {
    tip: "Four pointwise residual reductions: MSE squares, L1 magnitude, Huber bends at delta, Charbonnier smooths with epsilon.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="120" y1="118" x2="120" y2="22"/>
      <line class="skl-axis" x1="28" y1="100" x2="214" y2="100"/>
      <path class="skl-draw c-cal" d="M40 30 Q120 134 200 30" style="fill:none"/>
      <path class="skl-dash c-mid" d="M40 38 L120 100 L200 38" style="fill:none"/>
      <path class="skl-draw c-faint" d="M40 40 Q86 54 100 86 L120 100 L140 86 Q154 54 200 40" style="fill:none"/>
      <text x="150" y="40" style="fill:#4fd6c4;font-size:7px">MSE</text>
      <text x="44" y="44" style="fill:#f5b971;font-size:7px">L1</text>`; },
    anim: null,
  },

  curveshape: {
    tip: "The default curve term: magnitude-free cosine angle between predicted and target profiles over valid pixels.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="102" y1="116" x2="102" y2="32"/>
      <line class="skl-axis" x1="80" y1="96" x2="176" y2="96"/>
      <line class="skl-draw c-cal" x1="102" y1="96" x2="164" y2="42"/>
      <line class="skl-draw c-meas" x1="102" y1="96" x2="148" y2="56"/>
      <path class="skl-dash c-faint" d="M132 70 A 34 34 0 0 1 138 78" style="fill:none"/>
      <text x="140" y="66" style="fill:#c3cbd8;font-size:7px">cos</text>`; },
    anim: null,
  },

  physmoments: {
    tip: "Ratio terms compare integrated power and the mass, centroid and spread moments over GT-strong pixels.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="216" y2="118"/>
      <path class="skl-draw c-cal" d="M40 118 Q90 40 124 70 Q150 92 200 118" style="fill:rgba(79,214,196,0.12)"/>
      <line class="skl-dash c-mid" x1="112" y1="118" x2="112" y2="50"/>
      <circle cx="112" cy="50" r="3.5" class="skl-pop f-mid"/>
      <text x="116" y="48" style="fill:#f5b971;font-size:7px">z-bar</text>
      <line class="skl-dash c-faint" x1="84" y1="100" x2="140" y2="100"/>
      <text x="92" y="114" style="fill:#c3cbd8;font-size:7px">sigma_z</text>`; },
    anim: null,
  },

  paramterms: {
    tip: "GT sorts by mu and predictions are permuted onto it by optimal assignment (hungarian, the default); empty slots mask to zero weight. Param-L1 is active-normalised in normalised space, TV penalises roughness.",
    build(svg) { svg.innerHTML = `
      <rect x="34" y="42" width="22" height="20" rx="2" class="skl-pop f-cal"/>
      <rect x="34" y="66" width="22" height="20" rx="2" class="skl-pop f-cal"/>
      <rect x="34" y="90" width="22" height="20" rx="2" class="skl-dash c-faint" style="fill:none"/>
      <text x="45" y="124" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">mask</text>
      <text x="86" y="78" style="fill:#c3cbd8;font-size:12px">&#8594;</text>
      <rect x="150" y="40" width="62" height="62" class="skl-axis" style="fill:none;opacity:.3"/>
      <path class="skl-draw c-cal" d="M158 92 L170 60 L182 80 L194 52 L206 72" style="fill:none"/>
      <text x="170" y="116" style="fill:#4fd6c4;font-size:7px">TV</text>`; },
    anim: null,
  },

  composite: {
    tip: "Each enabled term is scaled by its user weight; the weighted terms sum and divide by the total weight into one loss.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="140" y2="118"/>
      <rect x="36" y="92" width="14" height="26" class="skl-pop f-cal"/>
      <rect x="56" y="74" width="14" height="44" class="skl-pop f-cal"/>
      <rect x="76" y="100" width="14" height="18" class="skl-pop f-cal"/>
      <rect x="96" y="58" width="14" height="60" class="skl-pop f-cal"/>
      <rect x="116" y="86" width="14" height="32" class="skl-pop f-cal"/>
      <path class="skl-draw c-mid" d="M146 75 L176 75 M168 70 L176 75 L168 80" style="fill:none"/>
      <text x="160" y="68" text-anchor="middle" style="fill:#f5b971;font-size:7px">/ sum</text>
      <rect class="sk-live skl-pop f-fin" x="192" y="48" width="24" height="70"/>
      <text x="204" y="132" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">L</text>`; },
    anim: FlowSketches.pulse,
  },

  gradclip: {
    tip: "If the gradient norm exceeds tau, all gradients scale by tau/norm, landing exactly on the limit.",
    build(svg) { svg.innerHTML = `
      <circle cx="76" cy="80" r="44" class="skl-dash c-faint" style="fill:none;opacity:.5"/>
      <text x="40" y="36" style="fill:#c3cbd8;font-size:7px">tau</text>
      <line class="skl-dash c-faint" x1="76" y1="80" x2="124" y2="34" style="opacity:.5"/>
      <line class="skl-draw c-cal" x1="76" y1="80" x2="107" y2="50"/>
      <circle cx="76" cy="80" r="3" class="skl-pop f-faint"/>
      <circle cx="107" cy="50" r="3.5" class="skl-pop f-cal"/>
      <text x="130" y="80" style="fill:#4fd6c4;font-size:7px">g min(1, tau/|g|)</text>`; },
    anim: null,
  },

  adamw: {
    tip: "Adaptive moments with decoupled weight decay step the weights downhill, lowering the loss each epoch.",
    build(svg) { svg.innerHTML = `
      <path class="skl-draw c-faint" d="M30 36 Q120 156 210 36" style="fill:none;opacity:.5"/>
      <circle cx="42" cy="60" r="2.5" class="skl-pop f-mid" style="opacity:.4"/>
      <circle cx="70" cy="92" r="2.5" class="skl-pop f-mid" style="opacity:.4"/>
      <circle cx="98" cy="114" r="2.5" class="skl-pop f-mid" style="opacity:.4"/>
      <circle class="sk-live skl-pop f-fin" cx="120" cy="122" r="5"/>`; },
    anim: FlowSketches.pulse,
  },

  schedule: {
    tip: "Effective LR = base x cosine decay x linear warmup; the curriculum swaps objectives at swap epoch 15.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="118" x2="216" y2="118"/>
      <line class="skl-axis" x1="28" y1="22" x2="28" y2="118"/>
      <path class="skl-draw c-mid" d="M28 110 L58 40" style="fill:none"/>
      <path class="skl-draw c-cal" d="M58 40 Q120 44 150 78 Q180 108 210 114" style="fill:none"/>
      <line class="skl-dash c-fin" x1="138" y1="22" x2="138" y2="118"/>
      <text x="142" y="32" style="fill:#c4a3ff;font-size:7px">swap</text>`; },
    anim: null,
  },

  checkpoint: {
    tip: "Validation checkpoints the best epoch; early stopping reverts to it after 15 evals without a new minimum.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="118" x2="216" y2="118"/>
      <path class="skl-draw c-cal" d="M30 50 L62 72 L94 60 L120 90 L150 78 L180 96 L210 88" style="fill:none"/>
      <circle class="sk-live skl-pop f-fin" cx="120" cy="90" r="5"/>
      <text x="120" y="108" text-anchor="middle" style="fill:#c4a3ff;font-size:7px">best</text>`; },
    anim: FlowSketches.pulse,
  },




































  load: {
    tip: "The architecture is rebuilt from the saved config, the best-epoch weights load, x-axis and norm stats restore; the input must be one contiguous region.",
    build(svg) { svg.innerHTML = `
      <rect x="38" y="40" width="64" height="70" rx="4" class="skl-draw c-faint" style="fill:#1b242f"/>
      <rect x="48" y="52" width="44" height="6" rx="2" class="skl-pop f-meas"/>
      <rect x="48" y="64" width="44" height="6" rx="2" class="skl-pop f-meas"/>
      <rect x="48" y="76" width="44" height="6" rx="2" class="skl-pop f-meas"/>
      <rect x="48" y="88" width="44" height="6" rx="2" class="skl-pop f-meas"/>
      <text x="70" y="34" text-anchor="middle" style="fill:#6ea8ff;font-size:9px">theta*</text>
      <path class="skl-draw c-cal" d="M108 75 L150 75 M142 69 L150 75 L142 81" style="fill:none"/>
      <rect x="160" y="48" width="50" height="50" rx="3" class="skl-draw c-cal" style="fill:rgba(79,214,196,0.12)"/>
      <text x="185" y="118" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">1 region</text>`; },
    anim: null,
  },

  predict: {
    tip: "The frozen model emits raw normalised z-hat for every sliding-window patch in grid order, leaving no holes.",
    build(svg) {
      let g = "";
      for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) g += `<rect x="${44 + c * 34}" y="${40 + r * 28}" width="30" height="24" rx="2" style="fill:#1b242f;stroke:#303d4c"/>`;
      svg.innerHTML = `${g}<rect class="sk-live" x="44" y="40" width="30" height="24" rx="2" style="fill:rgba(110,168,255,0.25);stroke:#6ea8ff;stroke-width:2"/><text x="120" y="138" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">sliding window</text>`;
    },
    anim: FlowSketches.pulse,
  },

  idenorm: {
    tip: "Predictions are denormalised then hard-clamped (leaky_slope 0), pinning amplitude into [0, a_max].",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="110" x2="210" y2="110"/>
      <line class="skl-axis" x1="40" y1="110" x2="40" y2="28"/>
      <line class="skl-dash c-faint" x1="40" y1="44" x2="210" y2="44"/>
      <text x="34" y="48" text-anchor="end" style="fill:#c3cbd8;font-size:7px">a_max</text>
      <path class="skl-dash c-mid" d="M30 124 L210 18" style="opacity:.3"/>
      <path class="skl-draw c-cal" d="M40 110 L100 110 L155 44 L210 44" style="fill:none"/>`; },
    anim: null,
  },

  align: {
    tip: "GT slots sort by mu (inactive last); the prediction keeps its raw order, so matching later resolves the correspondence.",
    build(svg) { svg.innerHTML = `
      <text x="60" y="34" text-anchor="middle" style="fill:#6ea8ff;font-size:8px">GT</text>
      <text x="180" y="34" text-anchor="middle" style="fill:#f5b971;font-size:8px">mu-sorted</text>
      <rect x="44" y="44" width="32" height="16" rx="2" class="skl-pop f-meas"/>
      <rect x="44" y="64" width="32" height="16" rx="2" class="skl-pop f-faint"/>
      <rect x="44" y="84" width="32" height="16" rx="2" class="skl-pop f-meas"/>
      <rect x="44" y="104" width="32" height="16" rx="2" class="skl-pop f-meas"/>
      <line class="skl-dash c-faint" x1="78" y1="52" x2="162" y2="52"/>
      <line class="skl-dash c-faint" x1="78" y1="72" x2="162" y2="112"/>
      <line class="skl-dash c-faint" x1="78" y1="92" x2="162" y2="72"/>
      <line class="skl-dash c-faint" x1="78" y1="112" x2="162" y2="92"/>
      <rect x="164" y="44" width="32" height="16" rx="2" class="skl-pop f-mid"/>
      <rect x="164" y="64" width="32" height="16" rx="2" class="skl-pop f-mid"/>
      <rect x="164" y="84" width="32" height="16" rx="2" class="skl-pop f-mid"/>
      <rect x="164" y="104" width="32" height="16" rx="2" class="skl-pop f-faint" style="opacity:.4"/>`; },
    anim: null,
  },

  recon: {
    tip: "Each patch's clamped Gaussians sum on the z axis, amplitudes rectified at zero, denominator 2 sigma^2 + 1e-8.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="34" y1="116" x2="214" y2="116"/>
      <path class="skl-draw c-faint" d="M34 116 C70 116 78 64 96 64 C114 64 122 116 158 116" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-faint" d="M110 116 C140 116 148 80 162 80 C176 80 184 116 214 116" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-mid" d="M34 116 C70 116 78 64 96 64 C108 64 116 92 134 92 C150 92 156 80 162 80 C176 80 184 116 214 116" style="fill:none"/>`; },
    anim: null,
  },

  window: {
    tip: "A separable Hann taper (each axis floored at 1e-3) gives every covered position a positive overlap-add weight.",
    build(svg) {
      let cells = "";
      for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) { const d = Math.hypot(r - 2, c - 2) / 2.83; const op = (0.95 - d * 0.85).toFixed(2); cells += `<rect x="${150 + c * 12}" y="${40 + r * 12}" width="11" height="11" style="fill:#f5b971;opacity:${op}"/>`; }
      svg.innerHTML = `
        <line class="skl-axis" x1="40" y1="118" x2="120" y2="118"/>
        <path class="skl-draw c-mid" d="M40 118 C58 118 64 50 80 50 C96 50 102 118 120 118" style="fill:none"/>
        <text x="80" y="134" text-anchor="middle" style="fill:#f5b971;font-size:7px">Hann</text>
        ${cells}`;
    },
    anim: null,
  },

  ola: {
    tip: "Each windowed curve patch adds into accumulator A at its origin; the bare window adds into the weight buffer W.",
    build(svg) { svg.innerHTML = `
      <rect x="46" y="44" width="64" height="64" class="skl-axis" style="fill:#15302d"/>
      <text x="78" y="36" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">A += p w</text>
      <rect x="46" y="44" width="40" height="40" class="skl-pop f-cal" style="opacity:.5"/>
      <rect x="70" y="68" width="40" height="40" class="skl-pop f-cal" style="opacity:.5"/>
      <rect x="140" y="44" width="64" height="64" class="skl-axis" style="fill:#2a2418"/>
      <text x="172" y="36" text-anchor="middle" style="fill:#f5b971;font-size:8px">W += w</text>
      <rect x="140" y="44" width="40" height="40" class="skl-pop f-mid" style="opacity:.5"/>
      <rect x="164" y="68" width="40" height="40" class="skl-pop f-mid" style="opacity:.5"/>`; },
    anim: null,
  },

  finalise: {
    tip: "A is divided by W and the padding trimmed to the scene; a coverage guard aborts if any scene pixel got zero weight.",
    build(svg) { svg.innerHTML = `
      <rect x="38" y="52" width="44" height="44" class="skl-pop f-cal" style="opacity:.7"/>
      <text x="60" y="78" text-anchor="middle" style="fill:#0b1014;font-size:11px">A</text>
      <text x="98" y="80" text-anchor="middle" style="fill:#d5dee7;font-size:16px">&#247;</text>
      <rect x="114" y="52" width="44" height="44" class="skl-pop f-mid" style="opacity:.7"/>
      <text x="136" y="80" text-anchor="middle" style="fill:#0b1014;font-size:11px">W</text>
      <text x="174" y="80" text-anchor="middle" style="fill:#d5dee7;font-size:16px">=</text>
      <rect x="190" y="56" width="34" height="34" class="skl-draw c-fin" style="fill:rgba(196,163,255,0.15)"/>
      <text x="207" y="106" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">cube</text>`; },
    anim: null,
  },

  paramstitch: {
    tip: "Parameters are not blended: at each pixel the patch with the largest Hann centrality overwrites the value, then inactive GT mu/sigma go to NaN.",
    build(svg) { svg.innerHTML = `
      <rect x="40" y="40" width="74" height="74" class="skl-axis" style="fill:#1b242f"/>
      <rect x="40" y="40" width="46" height="46" class="skl-pop f-mid" style="opacity:.4"/>
      <rect x="68" y="68" width="46" height="46" class="skl-pop f-cal" style="opacity:.45"/>
      <circle class="sk-live skl-pop f-cal" cx="91" cy="77" r="5"/>
      <text x="77" y="130" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">centrality wins</text>
      <path class="skl-draw c-fin" d="M120 77 L152 77 M144 71 L152 77 L144 83" style="fill:none"/>
      <rect x="160" y="52" width="50" height="50" rx="3" class="skl-draw c-fin" style="fill:rgba(196,163,255,0.15)"/>
      <text x="185" y="118" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">params</text>`; },
    anim: FlowSketches.pulse,
  },

  pixelmaps: {
    tip: "Five per-pixel maps reduce over the z bins: MSE, MAE, R-squared, cosine similarity, and peak-bin error.",
    build(svg) {
      const vals = [0.9, 0.7, 0.95, 0.6, 0.8, 0.4, 0.85, 0.5, 0.92, 0.65, 0.75, 0.55];
      let cells = "";
      vals.forEach((v, i) => { const x = 120 + (i % 4) * 24, y = 44 + Math.floor(i / 4) * 24; cells += `<rect class="sk-live" x="${x}" y="${y}" width="22" height="22" style="fill:#4fd6c4;opacity:${0.2 + v * 0.6}"/>`; });
      svg.innerHTML = `
        <rect x="40" y="60" width="40" height="40" class="skl-draw c-faint" style="fill:#1b242f"/>
        <rect x="46" y="54" width="40" height="40" class="skl-draw c-faint" style="fill:#1b242f"/>
        <rect x="52" y="48" width="40" height="40" class="skl-draw c-meas" style="fill:#1d3350"/>
        <path class="skl-draw c-cal" d="M98 74 L112 74 M106 69 L112 74 L106 79" style="fill:none"/>
        ${cells}`;
    },
    anim: FlowSketches.pulse,
  },

  globalcurve: {
    tip: "Cube-wide scalars at physical scale: MSE, RMSE, R-squared, and a PSNR over the GT range C_max - C_min.",
    build(svg) { svg.innerHTML = `
      <rect x="40" y="40" width="160" height="70" rx="5" class="skl-axis" style="fill:#1b242f"/>
      <text x="58" y="64" style="fill:#d5dee7;font-size:11px">R2</text>
      <text x="184" y="64" text-anchor="end" style="fill:#c4a3ff;font-size:13px;font-weight:700">0.94</text>
      <line x1="56" y1="74" x2="184" y2="74" class="skl-axis"/>
      <text x="58" y="96" style="fill:#d5dee7;font-size:11px">PSNR</text>
      <text x="184" y="96" text-anchor="end" style="fill:#4fd6c4;font-size:13px;font-weight:700">28.4 dB</text>`; },
    anim: null,
  },

  elevssim: {
    tip: "Per z bin: MAE, RMSE, R-squared and a column-normalised cross-entropy, plus mean SSIM over the slices.",
    build(svg) {
      const h = [20, 34, 46, 40, 28, 18];
      let b = "";
      h.forEach((v, i) => { b += `<rect class="skl-pop f-cal" x="${44 + i * 16}" y="${108 - v}" width="11" height="${v}" rx="1"/>`; });
      svg.innerHTML = `<line class="skl-axis" x1="40" y1="108" x2="148" y2="108"/>${b}<text x="92" y="124" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">R2 per bin</text><rect x="162" y="44" width="52" height="52" class="skl-axis" style="fill:#1b242f"/><rect x="162" y="62" width="52" height="9" class="skl-pop f-fin"/><text x="188" y="110" text-anchor="middle" style="fill:#c4a3ff;font-size:7px">SSIM</text>`;
    },
    anim: null,
  },

  paramslot: {
    tip: "On active pixels, per-Gaussian mu/sigma errors and detection F1 pool after brute-force optimal mu-matching over all K! permutations.",
    build(svg) { svg.innerHTML = `
      <circle cx="60" cy="58" r="9" class="skl-dash c-meas" style="fill:none"/>
      <circle cx="60" cy="58" r="4" class="skl-pop f-cal"/>
      <text x="60" y="84" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">mu err</text>
      <line class="skl-axis" x1="104" y1="100" x2="196" y2="100"/>
      <rect x="110" y="86" width="14" height="14" class="skl-pop f-faint"/>
      <rect x="132" y="60" width="14" height="40" class="skl-pop f-fin"/>
      <rect x="154" y="90" width="14" height="10" class="skl-pop f-faint"/>
      <rect x="176" y="84" width="14" height="16" class="skl-pop f-faint"/>
      <text x="150" y="50" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">best perm</text>`; },
    anim: null,
  },




























  bench_reference: {
    tip: "The reference model (unet) at default width sets N*, the parameter budget every architecture must match.",
    build(svg) { svg.innerHTML = `
      <path class="skl-draw c-meas" d="M34 46 L74 58 L74 92 L34 104 Z" style="fill:none"/>
      <text x="54" y="122" text-anchor="middle" style="fill:#6ea8ff;font-size:8px">unet</text>
      <text x="96" y="78" style="fill:#c3cbd8;font-size:12px">&#8594;</text>
      <rect class="sk-live skl-pop f-cal" x="118" y="64" width="94" height="20" rx="2"/>
      <text x="165" y="58" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">N* (budget)</text>
      <text x="165" y="78" text-anchor="middle" style="fill:#0b1014;font-size:8px">params</text>`; },
    anim: FlowSketches.pulse,
  },

  bench_search: {
    tip: "Scale is bisected by geometric mean s = sqrt(l h); the candidate count converges onto the budget N*.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="100" x2="214" y2="100"/>
      <line class="skl-axis" x1="40" y1="95" x2="40" y2="105"/>
      <line class="skl-axis" x1="196" y1="95" x2="196" y2="105"/>
      <text x="40" y="118" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">l</text>
      <text x="196" y="118" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">h</text>
      <line class="skl-dash c-cal" x1="132" y1="42" x2="132" y2="104"/>
      <text x="132" y="36" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">N*</text>
      <circle class="sk-live skl-pop f-mid" cx="120" cy="100" r="4.5"/>
      <text x="120" y="128" text-anchor="middle" style="fill:#f5b971;font-size:7px">s = sqrt(l h)</text>`; },
    anim: FlowSketches.pulse,
  },

  bench_widths: {
    tip: "The scaled width is rounded to the divisor 8 and floored there; locked embedding dims are never scaled.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="46" y1="120" x2="204" y2="120"/>
      <line class="skl-dash c-faint" x1="46" y1="98" x2="204" y2="98" style="opacity:.35"/>
      <line class="skl-dash c-faint" x1="46" y1="76" x2="204" y2="76" style="opacity:.35"/>
      <line class="skl-dash c-faint" x1="46" y1="54" x2="204" y2="54" style="opacity:.35"/>
      <rect class="skl-pop f-cal" x="56" y="76" width="18" height="44"/>
      <rect class="skl-pop f-cal" x="86" y="54" width="18" height="66"/>
      <rect class="sk-live skl-pop f-cal" x="116" y="98" width="18" height="22"/>
      <rect class="skl-pop f-cal" x="146" y="76" width="18" height="44"/>
      <rect class="skl-pop f-faint" x="176" y="54" width="18" height="66" style="opacity:.5"/>
      <text x="185" y="50" text-anchor="middle" style="fill:#c3cbd8;font-size:6px">locked</text>
      <text x="204" y="134" text-anchor="end" style="fill:#4fd6c4;font-size:7px">round&#183;8</text>`; },
    anim: FlowSketches.pulse,
  },

  bench_context: {
    tip: "After a cache clear, resident memory (total - free) is the CUDA context added to every peak; >1.5 GB warns.",
    build(svg) { svg.innerHTML = `
      <rect class="skl-draw c-faint" x="92" y="30" width="56" height="100" rx="2" style="fill:none"/>
      <line class="skl-dash c-cal" x1="84" y1="44" x2="156" y2="44"/>
      <text x="156" y="40" text-anchor="end" style="fill:#4fd6c4;font-size:7px">budget 40 GB</text>
      <rect class="sk-live skl-pop f-meas" x="92" y="114" width="56" height="16"/>
      <text x="120" y="126" text-anchor="middle" style="fill:#0b1014;font-size:7px">C_ctx</text>
      <text x="120" y="24" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">device VRAM</text>`; },
    anim: FlowSketches.pulse,
  },

  bench_probe: {
    tip: "Each doubling batch runs a real 3-step train loop; peak reserved VRAM plus context is the measured footprint.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="122" x2="214" y2="122"/>
      <line class="skl-dash c-cal" x1="30" y1="44" x2="214" y2="44"/>
      <text x="212" y="40" text-anchor="end" style="fill:#4fd6c4;font-size:7px">40 GB</text>
      <rect class="skl-pop f-mid" x="42" y="108" width="20" height="14"/>
      <rect class="skl-pop f-mid" x="72" y="94" width="20" height="28"/>
      <rect class="skl-pop f-mid" x="102" y="76" width="20" height="46"/>
      <rect class="sk-live skl-pop f-mid" x="132" y="52" width="20" height="70"/>
      <rect class="skl-pop f-faint" x="162" y="30" width="20" height="92" style="opacity:.5"/>
      <text x="120" y="138" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">b = 1, 2, 4, 8, 16</text>`; },
    anim: FlowSketches.pulse,
  },

  bench_maxbatch: {
    tip: "The largest batch whose bar stays under the budget line is kept as B*; the scan stops at the first over/OOM.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="122" x2="214" y2="122"/>
      <line class="skl-dash c-cal" x1="30" y1="52" x2="214" y2="52"/>
      <text x="212" y="48" text-anchor="end" style="fill:#4fd6c4;font-size:7px">budget</text>
      <rect class="skl-pop f-cal" x="42" y="108" width="20" height="14" style="opacity:.6"/>
      <rect class="skl-pop f-cal" x="72" y="92" width="20" height="30" style="opacity:.75"/>
      <rect class="skl-pop f-cal" x="102" y="72" width="20" height="50" style="opacity:.85"/>
      <rect class="sk-live skl-pop f-cal" x="132" y="58" width="20" height="64"/>
      <rect x="130" y="56" width="24" height="68" class="sk-live" style="fill:none;stroke:#4fd6c4;stroke-width:2"/>
      <rect class="skl-pop f-faint" x="162" y="34" width="20" height="88" style="opacity:.35"/>
      <text x="142" y="50" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">B*</text>`; },
    anim: FlowSketches.pulse,
  },

  bench_units: {
    tip: "The grid is models x loss components x seeds; each cell is one trained/inferred benchmark unit.",
    build(svg) {
      let g = "";
      for (let r = 0; r < 4; r++) for (let c = 0; c < 3; c++) { const x = 74 + c * 42, y = 40 + r * 22; g += `<rect class="skl-pop f-cal" x="${x}" y="${y}" width="34" height="16" rx="2" style="opacity:${0.55 + 0.1 * c}"/>`; }
      svg.innerHTML = `
        <text x="132" y="30" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">models x components x seeds</text>
        ${g}
        <rect x="72" y="38" width="122" height="20" class="sk-live" style="fill:none;stroke:#4fd6c4;stroke-width:2"/>
        <text x="52" y="52" text-anchor="middle" style="fill:#6ea8ff;font-size:7px">m1</text>`;
    },
    anim: FlowSketches.pulse,
  },

  bench_train: {
    tip: "Each unit trains at matched width and measured batch; the best-epoch val-loss minimum is checkpointed.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="34" y1="120" x2="212" y2="120"/>
      <line class="skl-axis" x1="34" y1="30" x2="34" y2="120"/>
      <path class="skl-draw c-faint" d="M40 44 C80 96 130 108 206 112" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-faint" d="M40 52 C84 100 140 100 206 104" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-cal" d="M40 40 C82 92 120 96 150 90 C176 86 194 88 206 90" style="fill:none"/>
      <circle class="sk-live skl-pop f-cal" cx="150" cy="90" r="4.5"/>
      <text x="150" y="82" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">best</text>
      <text x="120" y="136" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">val loss / epoch</text>`; },
    anim: FlowSketches.pulse,
  },

  bench_infer: {
    tip: "Sliding-window inference on the held-out test split yields the scalar metric vector q for the unit.",
    build(svg) { svg.innerHTML = `
      <rect x="30" y="48" width="96" height="56" rx="2" class="skl-pop f-faint" style="opacity:.35"/>
      <text x="78" y="118" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">test split</text>
      <rect class="sk-live" x="42" y="58" width="30" height="36" style="fill:none;stroke:#4fd6c4;stroke-width:2.2"/>
      <text x="136" y="80" style="fill:#c3cbd8;font-size:12px">&#8594;</text>
      <rect class="skl-pop f-cal" x="160" y="86" width="12" height="18"/>
      <rect class="skl-pop f-cal" x="176" y="70" width="12" height="34"/>
      <rect class="skl-pop f-cal" x="192" y="58" width="12" height="46"/>
      <text x="182" y="118" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">metrics q</text>`; },
    anim: FlowSketches.pulse,
  },

  bench_aggregate: {
    tip: "Per-model seed runs collapse to a mean and an error bar, separating seed noise from architecture effect.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="116" x2="200" y2="116"/>
      <circle class="skl-pop f-cal" cx="94" cy="66" r="3" style="opacity:.7"/>
      <circle class="skl-pop f-cal" cx="122" cy="56" r="3" style="opacity:.7"/>
      <circle class="skl-pop f-cal" cx="112" cy="82" r="3" style="opacity:.7"/>
      <line class="skl-draw c-cal" x1="108" y1="46" x2="108" y2="92"/>
      <line class="skl-draw c-cal" x1="100" y1="46" x2="116" y2="46"/>
      <line class="skl-draw c-cal" x1="100" y1="92" x2="116" y2="92"/>
      <circle class="sk-live skl-pop f-cal" cx="108" cy="68" r="4.5"/>
      <text x="108" y="132" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">mean &#177; std</text>`; },
    anim: FlowSketches.pulse,
  },

  bench_leaderboard: {
    tip: "Headline metrics min-max to [0,1] and average into a composite Score; models rank, best on top.",
    build(svg) { svg.innerHTML = `
      <rect class="sk-live skl-pop f-fin" x="40" y="34" width="152" height="18" rx="2"/>
      <rect class="skl-pop f-cal" x="40" y="60" width="118" height="18" rx="2" style="opacity:.85"/>
      <rect class="skl-pop f-cal" x="40" y="86" width="90" height="18" rx="2" style="opacity:.7"/>
      <rect class="skl-pop f-cal" x="40" y="112" width="62" height="18" rx="2" style="opacity:.55"/>
      <text x="32" y="47" text-anchor="end" style="fill:#c3cbd8;font-size:7px">1</text>
      <text x="32" y="73" text-anchor="end" style="fill:#c3cbd8;font-size:7px">2</text>
      <text x="32" y="99" text-anchor="end" style="fill:#c3cbd8;font-size:7px">3</text>
      <text x="32" y="125" text-anchor="end" style="fill:#c3cbd8;font-size:7px">4</text>
      <text x="200" y="47" style="fill:#c4a3ff;font-size:8px">Score</text>`; },
    anim: FlowSketches.pulse,
  },

  cv_partition: {
    tip: "The fold row window is cut into K equal contiguous blocks; the last block absorbs the remainder.",
    build(svg) {
      let b = "";
      for (let i = 0; i < 6; i++) b += `<rect x="${26 + i * 31}" y="52" width="29" height="46" class="skl-pop f-mid" style="opacity:${0.4 + 0.12 * (i % 2)}"/>`;
      svg.innerHTML = `<text x="120" y="40" text-anchor="middle" style="fill:#6ea8ff;font-size:8px">row window [row0, row1)</text>${b}<line class="skl-axis" x1="26" y1="108" x2="212" y2="108"/><text x="26" y="122" style="fill:#c3cbd8;font-size:7px">row0 = 1000</text><text x="212" y="122" text-anchor="end" style="fill:#c3cbd8;font-size:7px">row1 = 16000</text>`;
    },
    anim: null,
  },

  cv_guard: {
    tip: "Interior boundaries erode by margin = g/2 on each side, opening a guard gap so no patch straddles two folds.",
    build(svg) { svg.innerHTML = `
      <rect x="24" y="44" width="80" height="60" class="skl-pop f-mid" style="opacity:.6"/>
      <rect x="136" y="44" width="80" height="60" class="skl-pop f-mid" style="opacity:.6"/>
      <rect x="104" y="44" width="32" height="60" class="skl-pop f-faint" style="opacity:.35"/>
      <line class="skl-dash c-faint" x1="120" y1="36" x2="120" y2="112"/>
      <text x="64" y="78" text-anchor="middle" style="fill:#0b1014;font-size:8px">B_k</text>
      <text x="176" y="78" text-anchor="middle" style="fill:#0b1014;font-size:8px">B_k+1</text>
      <text x="120" y="128" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">guard g = 64</text>`; },
    anim: null,
  },

  cv_assign: {
    tip: "For fold k the test band is block k, val is the next block, and the rest merge into the training regions.",
    build(svg) {
      const roles = ["train", "train", "test", "val", "train", "train"];
      const cls = { train: "f-meas", test: "f-fin", val: "f-cal" };
      let b = "";
      roles.forEach((r, i) => { const live = (r === "test" || r === "val") ? " sk-live" : ""; b += `<rect class="skl-pop ${cls[r]}${live}" x="${28 + i * 31}" y="50" width="28" height="42"/>`; });
      svg.innerHTML = `${b}<text x="104" y="42" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">test</text><text x="135" y="42" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">val</text><text x="120" y="110" text-anchor="middle" style="fill:#6ea8ff;font-size:8px">train = merge(rest)</text>`;
    },
    anim: FlowSketches.pulse,
  },

  cv_units: {
    tip: "Each fold is replicated once per seed into fold_k_seedS run units, queued across the configured GPUs.",
    build(svg) {
      let g = "";
      for (let r = 0; r < 3; r++) { g += `<text x="30" y="${58 + r * 26}" style="fill:#6ea8ff;font-size:7px">fold_${r}</text>`; for (let c = 0; c < 3; c++) g += `<rect class="sk-live skl-pop f-cal" x="${76 + c * 40}" y="${46 + r * 26}" width="34" height="18" style="opacity:${0.45 + 0.16 * c}"/>`; }
      svg.innerHTML = `<text x="146" y="36" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">seed 0    seed 1    seed 2</text>${g}`;
    },
    anim: FlowSketches.pulse,
  },

  cv_train: {
    tip: "Each fold trains an independent model on its train/val regions; the best-epoch checkpoint and its val loss are kept.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="34" y1="28" x2="34" y2="116"/>
      <line class="skl-axis" x1="34" y1="116" x2="212" y2="116"/>
      <path class="skl-draw c-faint" d="M34 46 C82 98 142 108 206 112" style="fill:none;opacity:.45"/>
      <path class="skl-draw c-faint" d="M34 56 C82 102 142 110 206 114" style="fill:none;opacity:.45"/>
      <path class="skl-draw c-mid sk-live" d="M34 40 C82 92 142 105 206 109" style="fill:none"/>
      <circle class="sk-live skl-pop f-cal" cx="150" cy="104" r="3.4"/>
      <text x="150" y="96" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">L*_k</text>
      <text x="120" y="132" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">epoch</text>`; },
    anim: FlowSketches.pulse,
  },

  cv_infer: {
    tip: "Each trained fold is inferred on its held-out val and test bands into per-split metrics (AE folds skip this).",
    build(svg) { svg.innerHTML = `
      <rect x="26" y="42" width="58" height="26" class="skl-pop f-cal" style="opacity:.8"/>
      <text x="55" y="59" text-anchor="middle" style="fill:#0b1014;font-size:8px">val</text>
      <rect x="26" y="86" width="58" height="26" class="skl-pop f-fin" style="opacity:.8"/>
      <text x="55" y="103" text-anchor="middle" style="fill:#0b1014;font-size:8px">test</text>
      <text x="100" y="82" style="fill:#c3cbd8;font-size:13px">&#8594;</text>
      <rect x="130" y="38" width="84" height="78" rx="3" class="skl-draw c-cal" style="fill:rgba(79,214,196,0.08)"/>
      <text x="172" y="58" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">MAE</text>
      <text x="172" y="74" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">RMSE</text>
      <text x="172" y="90" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">R2</text>
      <text x="172" y="106" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">SSIM</text>`; },
    anim: null,
  },

  cv_collect: {
    tip: "Seed replicas of a fold reduce to the across-seed mean and the within-fold seed standard deviation.",
    build(svg) { svg.innerHTML = `
      <circle class="skl-pop f-cal" cx="52" cy="50" r="3.4"/>
      <circle class="skl-pop f-cal" cx="52" cy="74" r="3.4"/>
      <circle class="skl-pop f-cal" cx="52" cy="98" r="3.4"/>
      <text x="52" y="122" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">seeds</text>
      <text x="102" y="80" style="fill:#c3cbd8;font-size:13px">&#8594;</text>
      <line class="skl-dash c-mid" x1="168" y1="50" x2="168" y2="98"/>
      <line class="skl-dash c-mid" x1="160" y1="50" x2="176" y2="50"/>
      <line class="skl-dash c-mid" x1="160" y1="98" x2="176" y2="98"/>
      <circle class="sk-live skl-pop f-mid" cx="168" cy="74" r="4.2"/>
      <text x="168" y="122" text-anchor="middle" style="fill:#f5b971;font-size:7px">mean, seed std</text>`; },
    anim: FlowSketches.pulse,
  },

  cv_aggregate: {
    tip: "Per-fold values aggregate to the cross-fold mean and sample std (ddof=1, only when >=2 folds contribute).",
    build(svg) {
      const ys = [64, 82, 70, 88, 72, 66, 84, 78];
      let d = "";
      ys.forEach((y, i) => { d += `<circle class="skl-pop f-cal" cx="${44 + i * 22}" cy="${y}" r="3"/>`; });
      svg.innerHTML = `<line class="skl-axis" x1="30" y1="116" x2="214" y2="116"/><rect class="sk-live skl-pop f-fin" x="34" y="62" width="176" height="28" style="opacity:.16"/><line class="skl-dash c-fin" x1="34" y1="76" x2="210" y2="76"/><text x="34" y="54" style="fill:#c4a3ff;font-size:7px">mean +/- std</text><text x="120" y="132" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">folds</text>${d}`;
    },
    anim: FlowSketches.pulse,
  },

  cv_report: {
    tip: "The fold plan, training summary and metric aggregates land in the report, the summary JSON and per-split comparisons.",
    build(svg) { svg.innerHTML = `
      <rect x="36" y="32" width="50" height="72" rx="3" class="skl-pop f-fin" style="opacity:.85"/>
      <line class="skl-axis" x1="44" y1="46" x2="78" y2="46"/>
      <line class="skl-axis" x1="44" y1="56" x2="78" y2="56"/>
      <line class="skl-axis" x1="44" y1="66" x2="78" y2="66"/>
      <text x="61" y="118" text-anchor="middle" style="fill:#c4a3ff;font-size:7px">report.md</text>
      <rect x="98" y="42" width="46" height="62" rx="3" class="skl-pop f-cal" style="opacity:.8"/>
      <text x="121" y="118" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">summary.json</text>
      <rect x="158" y="50" width="42" height="54" rx="3" class="skl-pop f-mid" style="opacity:.7"/>
      <text x="179" y="118" text-anchor="middle" style="fill:#f5b971;font-size:7px">val / test</text>`; },
    anim: null,
  },

  spacelr: {
    tip: "Encoder, bottleneck, decoder and head LRs draw log-uniform over 1e-5 to 1e-2; dropout draws linearly over 0 to 0.5.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <text x="30" y="132" style="fill:#c3cbd8;font-size:7px">1e-5</text>
      <text x="190" y="132" text-anchor="end" style="fill:#c3cbd8;font-size:7px">1e-2</text>
      <circle class="sk-live skl-pop f-mid" cx="58" cy="118" r="4"/>
      <circle class="sk-live skl-pop f-mid" cx="112" cy="118" r="4"/>
      <circle class="sk-live skl-pop f-mid" cx="150" cy="118" r="4"/>
      <circle class="sk-live skl-pop f-mid" cx="192" cy="118" r="4"/>
      <line class="skl-axis" x1="30" y1="44" x2="210" y2="44"/>
      <text x="30" y="38" style="fill:#c3cbd8;font-size:7px">p_drop</text>
      <circle class="sk-live skl-pop f-cal" cx="96" cy="44" r="4"/>`; },
    anim: FlowSketches.pulse,
  },

  spacearch: {
    tip: "Five categorical knobs are sampled; the features index k decodes through a lookup to a list like [64,128,256,512].",
    build(svg) { svg.innerHTML = `
      <rect x="28" y="26" width="26" height="16" rx="3" class="skl-pop f-faint"/>
      <rect x="58" y="26" width="26" height="16" rx="3" class="skl-pop f-faint"/>
      <rect x="88" y="26" width="26" height="16" rx="3" class="skl-pop f-mid"/>
      <rect x="118" y="26" width="26" height="16" rx="3" class="skl-pop f-faint"/>
      <path class="skl-draw c-mid" d="M101 44 q24 16 58 16" style="fill:none"/>
      <rect x="150" y="50" width="62" height="18" rx="3" class="skl-pop f-cal"/>
      <text x="181" y="63" text-anchor="middle" style="fill:#0b1014;font-size:7px">[64,128,256,512]</text>
      <rect x="28" y="88" width="40" height="14" rx="3" class="skl-pop f-faint"/>
      <rect x="74" y="88" width="30" height="14" rx="3" class="skl-pop f-faint"/>
      <rect x="110" y="88" width="34" height="14" rx="3" class="skl-pop f-faint"/>
      <rect x="150" y="88" width="32" height="14" rx="3" class="skl-pop f-faint"/>`; },
    anim: null,
  },

  merge: {
    tip: "The 9-dim learning and 5-dim architecture blocks stack into one 14-D space that multivariate TPE samples jointly.",
    build(svg) { svg.innerHTML = `
      <rect x="34" y="42" width="46" height="66" rx="4" class="skl-pop f-mid" style="opacity:.85"/>
      <text x="57" y="34" text-anchor="middle" style="fill:#f5b971;font-size:8px">lr (9)</text>
      <text x="100" y="80" text-anchor="middle" style="fill:#c3cbd8;font-size:14px">+</text>
      <rect x="118" y="42" width="46" height="66" rx="4" class="skl-pop f-cal" style="opacity:.85"/>
      <text x="141" y="34" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">arch (5)</text>
      <text x="178" y="80" text-anchor="middle" style="fill:#c3cbd8;font-size:14px">=</text>
      <rect x="190" y="48" width="28" height="54" rx="4" class="skl-pop f-meas"/>
      <text x="204" y="116" text-anchor="middle" style="fill:#6ea8ff;font-size:8px">d=14</text>`; },
    anim: null,
  },

  tpesplit: {
    tip: "Trials split at the gamma quantile of loss into good and bad sets, each fit with its own KDE l(theta) and g(theta).",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <line class="skl-dash c-faint" x1="120" y1="26" x2="120" y2="122"/>
      <text x="110" y="22" style="fill:#c3cbd8;font-size:7px">y_g</text>
      <path class="skl-draw c-cal" d="M34 118 C70 118 86 40 110 40 C116 40 118 118 120 118 Z" style="fill:rgba(79,214,196,.18)"/>
      <path class="skl-draw c-mid" d="M120 118 C122 118 132 70 158 70 C190 70 196 118 206 118 Z" style="fill:rgba(245,185,113,.16)"/>
      <text x="64" y="60" style="fill:#4fd6c4;font-size:7px">l</text>
      <text x="160" y="92" style="fill:#f5b971;font-size:7px">g</text>`; },
    anim: null,
  },

  tpeacq: {
    tip: "TPE proposes the theta maximising l(theta)/g(theta), except the first n0 = 8 trials which sample uniformly.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <path class="skl-draw c-cal" d="M34 110 C70 110 86 48 108 48 C120 48 124 110 130 110" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-mid" d="M40 110 C90 110 110 80 150 80 C188 80 196 110 206 110" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-fin" d="M34 116 C66 116 80 36 104 30 C124 26 132 116 150 116 C170 116 206 116 206 116" style="fill:none"/>
      <line class="skl-dash c-fin" x1="104" y1="30" x2="104" y2="118"/>
      <circle class="sk-live skl-pop f-fin" cx="104" cy="30" r="5"/>
      <text x="60" y="26" style="fill:#c4a3ff;font-size:7px">l/g</text>`; },
    anim: FlowSketches.pulse,
  },

  liar: {
    tip: "Each pending trial gets the worst objective max f as a phantom value, so parallel workers avoid the same point.",
    build(svg) { svg.innerHTML = `
      <rect x="96" y="14" width="48" height="18" rx="4" class="skl-pop f-faint"/>
      <text x="120" y="27" text-anchor="middle" style="fill:#cfd8e8;font-size:7px">study</text>
      <rect x="28" y="74" width="36" height="22" rx="4" class="skl-pop f-faint"/>
      <rect x="84" y="74" width="36" height="22" rx="4" class="skl-pop f-faint"/>
      <rect x="140" y="74" width="36" height="22" rx="4" class="skl-pop f-faint"/>
      <line class="skl-axis" x1="120" y1="32" x2="46" y2="74" style="opacity:.4"/>
      <line class="skl-axis" x1="120" y1="32" x2="102" y2="74" style="opacity:.4"/>
      <line class="skl-axis" x1="120" y1="32" x2="158" y2="74" style="opacity:.4"/>
      <circle cx="46" cy="116" r="6" class="skl-pop f-mid"/>
      <circle cx="102" cy="116" r="6" class="skl-pop f-mid"/>
      <circle cx="158" cy="116" r="6" class="skl-pop f-mid"/>
      <text x="120" y="138" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">phantom max f</text>`; },
    anim: null,
  },

  trialsetup: {
    tip: "The base config is deep-copied per trial, overridden with E = 30, patience = 8, seed = 42 + trial.number.",
    build(svg) { svg.innerHTML = `
      <rect x="30" y="50" width="56" height="50" rx="5" class="skl-pop f-faint"/>
      <text x="58" y="116" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">base cfg</text>
      <path class="skl-draw c-faint" d="M90 75 L128 75 M120 70 L128 75 L120 80" style="fill:none"/>
      <rect x="134" y="46" width="60" height="58" rx="5" class="skl-pop f-mid"/>
      <line class="skl-draw c-cal" x1="142" y1="60" x2="186" y2="60"/>
      <line class="skl-draw c-cal" x1="142" y1="72" x2="186" y2="72"/>
      <line class="skl-draw c-cal" x1="142" y1="84" x2="186" y2="84"/>
      <text x="164" y="118" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">E=30 pat=8</text>`; },
    anim: null,
  },

  trial: {
    tip: "A trial trains on the fixed split up to 30 epochs and returns f(theta), the minimum validation loss.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="120" x2="210" y2="120"/>
      <line class="skl-axis" x1="30" y1="24" x2="30" y2="120"/>
      <path class="skl-draw c-cal" d="M30 36 C60 70 78 92 104 100 C140 110 170 104 206 102" style="fill:none"/>
      <line class="skl-dash c-fin" x1="30" y1="102" x2="206" y2="102"/>
      <circle class="sk-live skl-pop f-cal" cx="158" cy="102" r="4"/>
      <text x="158" y="96" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">min L</text>`; },
    anim: FlowSketches.pulse,
  },

  prune: {
    tip: "A trial whose loss stays above the running median m at step t is pruned, once 8 trials complete and t >= 8.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="120" x2="210" y2="120"/>
      <line class="skl-dash c-cal" x1="30" y1="72" x2="210" y2="72"/>
      <text x="34" y="66" style="fill:#4fd6c4;font-size:7px">median m</text>
      <line class="skl-dash c-faint" x1="92" y1="24" x2="92" y2="120"/>
      <text x="68" y="20" style="fill:#c3cbd8;font-size:7px">t>=8</text>
      <path class="skl-draw c-mid" d="M30 40 C50 56 70 64 92 66 C104 67 110 66 116 64" style="fill:none"/>
      <line class="skl-draw c-mid" x1="118" y1="58" x2="130" y2="70" style="stroke-width:2"/>
      <line class="skl-draw c-mid" x1="130" y1="58" x2="118" y2="70" style="stroke-width:2"/>`; },
    anim: null,
  },

  best: {
    tip: "Remaining trials run in GPU chunks; after each, theta* is rewritten as argmin f, decoding the features index to channels.",
    build(svg) { svg.innerHTML = `
      <rect x="28" y="26" width="14" height="18" rx="2" class="skl-pop f-faint"/>
      <rect x="46" y="26" width="14" height="18" rx="2" class="skl-pop f-faint"/>
      <rect x="64" y="26" width="14" height="18" rx="2" class="skl-pop f-mid"/>
      <rect x="82" y="26" width="14" height="18" rx="2" class="skl-pop f-mid"/>
      <text x="120" y="40" style="fill:#c3cbd8;font-size:7px">n_rem of 100</text>
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <circle cx="52" cy="86" r="3" class="skl-pop f-cal" style="opacity:.5"/>
      <circle cx="86" cy="70" r="3" class="skl-pop f-cal" style="opacity:.5"/>
      <circle cx="118" cy="98" r="3" class="skl-pop f-cal" style="opacity:.5"/>
      <circle class="sk-live skl-pop f-fin" cx="150" cy="110" r="5"/>
      <circle cx="184" cy="82" r="3" class="skl-pop f-cal" style="opacity:.5"/>
      <text x="150" y="132" text-anchor="middle" style="fill:#c4a3ff;font-size:7px">argmin f</text>`; },
    anim: FlowSketches.pulse,
  },

  feed_target: {
    tip: "The mode's adapter wires the real dataset and model; the benchmark drives a genuine forward/backward/AdamW step.",
    build(svg) { svg.innerHTML = `
      <rect x="30" y="44" width="20" height="16" class="skl-pop f-meas"/>
      <rect x="30" y="64" width="20" height="16" class="sk-live skl-pop f-meas"/>
      <rect x="30" y="84" width="20" height="16" class="skl-pop f-meas"/>
      <text x="40" y="118" text-anchor="middle" style="fill:#6ea8ff;font-size:7px">dataset</text>
      <text x="64" y="78" style="fill:#c3cbd8;font-size:12px">&#8594;</text>
      <path class="skl-draw c-faint" d="M92 52 L140 66 L140 96 L92 110 Z" style="fill:none"/>
      <text x="116" y="126" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">f_theta step</text>
      <path class="skl-dash c-mid" d="M140 70 C180 58 180 104 140 92" style="fill:none"/>
      <text x="184" y="66" style="fill:#f5b971;font-size:7px">back</text>
      <circle cx="152" cy="120" r="4" class="skl-pop f-cal"/>
      <text x="180" y="124" style="fill:#4fd6c4;font-size:7px">L(x)</text>`; },
    anim: FlowSketches.pulse,
  },

  feed_grid: {
    tip: "Every (batch, workers) pair is a spec; the sweep visits each, dropping worker counts above the core budget C.",
    build(svg) {
      let g = "";
      for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) {
        const x = 52 + c * 30, y = 26 + r * 22, hot = (r === 1 && c === 2);
        g += `<rect class="skl-pop ${hot ? "sk-live f-cal" : "f-mid"}" x="${x}" y="${y}" width="26" height="18" style="opacity:${hot ? 1 : 0.22};stroke:#f5b971;stroke-width:.7"/>`;
      }
      svg.innerHTML = `${g}
        <text x="36" y="68" text-anchor="middle" transform="rotate(-90 36 68)" style="fill:#6ea8ff;font-size:7px">batch b</text>
        <text x="127" y="134" text-anchor="middle" style="fill:#6ea8ff;font-size:7px">workers w</text>`;
    },
    anim: FlowSketches.pulse,
  },

  feed_loader: {
    tip: "With the GPU idle, 60 timed batches after 8 warm-up give the pure data-pipeline throughput R_load.",
    build(svg) { svg.innerHTML = `
      <rect x="28" y="58" width="18" height="30" class="sk-live skl-pop f-meas"/>
      <rect x="50" y="58" width="18" height="30" class="sk-live skl-pop f-meas" style="opacity:.75"/>
      <rect x="72" y="58" width="18" height="30" class="sk-live skl-pop f-meas" style="opacity:.5"/>
      <text x="108" y="78" style="fill:#c3cbd8;font-size:12px">&#8594;</text>
      <rect x="134" y="52" width="74" height="42" rx="3" class="skl-draw c-cal" style="fill:rgba(79,214,196,0.1)"/>
      <text x="171" y="77" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">R_load</text>
      <text x="118" y="116" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">GPU idle (loader only)</text>`; },
    anim: FlowSketches.pulse,
  },

  feed_ceiling: {
    tip: "One batch is reused 60 times through the real train step; with zero data cost this is the GPU compute ceiling.",
    build(svg) { svg.innerHTML = `
      <rect x="34" y="60" width="22" height="30" class="skl-pop f-meas"/>
      <text x="45" y="106" text-anchor="middle" style="fill:#6ea8ff;font-size:7px">one x0</text>
      <path class="sk-live skl-dash c-mid" d="M58 66 C88 40 88 108 58 84" style="fill:none"/>
      <path class="skl-draw c-faint" d="M98 56 L138 68 L138 92 L98 104 Z" style="fill:none"/>
      <text x="118" y="120" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">f_theta step</text>
      <line class="skl-axis" x1="158" y1="104" x2="158" y2="34"/>
      <rect x="168" y="40" width="20" height="64" class="skl-pop f-cal"/>
      <text x="188" y="120" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">R_gpu</text>`; },
    anim: FlowSketches.pulse,
  },

  feed_e2e: {
    tip: "Each batch's fetch-wait t_d and compute t_c are timed apart; their split is the data-wait fraction w_d.",
    build(svg) { svg.innerHTML = `
      <text x="30" y="42" style="fill:#c3cbd8;font-size:7px">one step time</text>
      <rect x="40" y="56" width="46" height="30" class="sk-live skl-pop f-mid"/>
      <text x="63" y="102" text-anchor="middle" style="fill:#f5b971;font-size:7px">t_d wait</text>
      <rect x="86" y="56" width="118" height="30" class="skl-pop f-cal"/>
      <text x="145" y="102" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">t_c compute</text>
      <line class="skl-axis" x1="40" y1="90" x2="204" y2="90"/>
      <text x="122" y="128" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">w_d = t_d / (t_d + t_c)</text>`; },
    anim: FlowSketches.pulse,
  },

  feed_ratios: {
    tip: "Feed ratio phi = loader / ceiling (>= 1 means CPU can outpace GPU); efficiency eta = achieved / ceiling.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="120" x2="214" y2="120"/>
      <line class="skl-dash c-faint" x1="30" y1="52" x2="214" y2="52"/>
      <text x="210" y="48" text-anchor="end" style="fill:#c3cbd8;font-size:7px">1.0</text>
      <rect x="70" y="72" width="34" height="48" class="skl-pop f-cal"/>
      <text x="87" y="134" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">phi</text>
      <rect x="134" y="80" width="34" height="40" class="skl-pop f-cal" style="opacity:.7"/>
      <text x="151" y="134" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">eta</text>`; },
    anim: null,
  },

  feed_saturate: {
    tip: "Configs with data-wait <= 0.05 or feed-ratio >= 1 are GPU-saturated; none qualifying flags the run CPU-bound.",
    build(svg) {
      const pts = [[54, 46, 0], [84, 66, 0], [110, 98, 1], [140, 106, 1], [168, 112, 1], [196, 80, 0]];
      let d = "";
      pts.forEach(p => { const sat = p[2] === 1; d += `<circle cx="${p[0]}" cy="${p[1]}" r="4" class="${sat ? "sk-live skl-pop f-cal" : "skl-pop f-faint"}" style="opacity:${sat ? 1 : 0.5}"/>`; });
      svg.innerHTML = `
        <line class="skl-axis" x1="34" y1="120" x2="214" y2="120"/>
        <line class="skl-axis" x1="34" y1="28" x2="34" y2="120"/>
        <line class="skl-dash c-mid" x1="34" y1="92" x2="214" y2="92"/>
        <text x="210" y="88" text-anchor="end" style="fill:#f5b971;font-size:7px">tau = 0.05</text>
        ${d}
        <text x="124" y="134" text-anchor="middle" style="fill:#c3cbd8;font-size:7px">workers</text>
        <text x="24" y="40" style="fill:#c3cbd8;font-size:7px">w_d</text>`;
    },
    anim: FlowSketches.pulse,
  },

  feed_recommend: {
    tip: "Within the saturated pool the highest end-to-end throughput wins; ties break to fewer workers, then smaller batch.",
    build(svg) {
      const hs = [34, 52, 68, 60, 40];
      let b = "";
      hs.forEach((h, i) => { const x = 52 + i * 30, win = (i === 2); b += `<rect class="skl-pop ${win ? "sk-live f-fin" : "f-faint"}" x="${x}" y="${118 - h}" width="22" height="${h}" style="opacity:${win ? 1 : 0.45}"/>`; });
      svg.innerHTML = `<line class="skl-axis" x1="40" y1="118" x2="210" y2="118"/>${b}
        <circle cx="123" cy="42" r="4" class="sk-live skl-pop f-fin"/>
        <text x="120" y="134" text-anchor="middle" style="fill:#c4a3ff;font-size:7px">max R_e2e -> l*</text>`;
    },
    anim: FlowSketches.pulse,
  },

  feed_refine: {
    tip: "Batch and workers fixed, prefetch {2,4,8,16} x pin on/off is swept; the fastest cell tunes the transfer path.",
    build(svg) {
      let g = "";
      for (let r = 0; r < 2; r++) for (let c = 0; c < 4; c++) {
        const x = 66 + c * 30, y = 44 + r * 30, best = (r === 0 && c === 2);
        g += `<rect class="skl-pop ${best ? "sk-live f-cal" : "f-mid"}" x="${x}" y="${y}" width="26" height="26" style="opacity:${best ? 1 : 0.3}"/>`;
      }
      svg.innerHTML = `${g}
        <text x="128" y="30" text-anchor="middle" style="fill:#f5b971;font-size:7px">prefetch {2,4,8,16}</text>
        <text x="46" y="60" text-anchor="middle" transform="rotate(-90 46 60)" style="fill:#c3cbd8;font-size:6px">pin 1</text>
        <text x="46" y="90" text-anchor="middle" transform="rotate(-90 46 90)" style="fill:#c3cbd8;font-size:6px">pin 0</text>
        <text x="128" y="112" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">best (p*, pin*)</text>`;
    },
    anim: FlowSketches.pulse,
  },

  feed_final: {
    tip: "Batch and workers come from the recommendation; refine overrides prefetch and pin, else defaults 4 / on.",
    build(svg) { svg.innerHTML = `
      <rect x="60" y="30" width="120" height="92" rx="5" class="skl-draw c-fin" style="fill:rgba(196,163,255,0.08)"/>
      <text x="120" y="46" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">final loader</text>
      <line class="skl-axis" x1="70" y1="54" x2="170" y2="54" style="opacity:.4"/>
      <text x="76" y="70" style="fill:#6ea8ff;font-size:7px">batch b*</text>
      <text x="76" y="84" style="fill:#6ea8ff;font-size:7px">workers w*</text>
      <text x="76" y="98" style="fill:#4fd6c4;font-size:7px">prefetch p*</text>
      <text x="76" y="112" style="fill:#4fd6c4;font-size:7px">pin*, persist</text>`; },
    anim: null,
  },

  feed_report: {
    tip: "results.json plus four figures: throughput-vs-batch, wait-vs-workers, util-vs-throughput, feed-ratio-vs-workers.",
    build(svg) { svg.innerHTML = `
      <rect x="28" y="40" width="54" height="70" rx="3" class="skl-pop f-fin" style="opacity:.85"/>
      <text x="55" y="80" text-anchor="middle" style="fill:#0b1014;font-size:8px">.json</text>
      <text x="94" y="80" style="fill:#c3cbd8;font-size:12px">&#8594;</text>
      <rect x="120" y="40" width="42" height="30" rx="2" class="skl-pop f-cal" style="opacity:.5"/>
      <rect x="168" y="40" width="42" height="30" rx="2" class="skl-pop f-cal" style="opacity:.5"/>
      <rect x="120" y="78" width="42" height="30" rx="2" class="skl-pop f-cal" style="opacity:.5"/>
      <rect x="168" y="78" width="42" height="30" rx="2" class="skl-pop f-cal" style="opacity:.5"/>
      <text x="165" y="124" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">4 figures</text>`; },
    anim: null,
  },

};
}

window.FlowSketches = FlowSketches;
