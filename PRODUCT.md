# Product

## Register

product

## Users

A single ML researcher (and occasional colleagues) operating a shared multi-GPU Linux server for SAR tomography deep learning. The web UI is glanced at on a side monitor while long trainings run, and consulted closely when something goes wrong (GPU intrusion, RAM pressure, contention with other users).

## Product Purpose

Local control room for the DLR-TomoSAR codebase: launch and monitor training/inference runs, inspect configs, browse results and tomograms, and watch live server telemetry (GPU/CPU/RAM/disk/processes) with watchdog and neighbour-impact guards.

## Brand Personality

Instrument-grade, scientific, quietly futuristic. A measurement device, not a SaaS product. Dark surfaces are the app's existing vocabulary for live machine output (consoles, CLI panes, process stages); light surfaces are for configuration and reading.

## Anti-references

- Packed uniform card walls where every metric shouts equally.
- Consumer dashboard gloss: rounded pastel cards, gradient text, glassmorphism.
- Grafana-default look: undifferentiated panel soup.

## Design Principles

- Hierarchy by importance: GPUs are the reason the server exists; they read first. Guards and alarms surface only when they matter.
- Telemetry is an instrument panel: mono numerals, hairlines, phosphor-like traces, high legibility at a distance.
- Alarm states are unmissable and unambiguous; idle states are calm.
- Density is allowed (process tables, logs) but rhythm and grouping keep it scannable.

## Accessibility & Inclusion

Reduced-motion honoured everywhere (REDUCED_MOTION gates all canvas loops and CSS animation). Text contrast at or above 4.5:1 on both light and dark surfaces. State never conveyed by colour alone (lights are paired with text labels).
