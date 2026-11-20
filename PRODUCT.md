# Product

## Register

product

## Users

Researchers who start a new machine-learning project from ML-Template, alone or in a small
group, on a workstation or a shared multi-GPU Linux server. The console sits on a side
monitor while long training runs proceed and is read closely when something needs
attention: a failed job, GPU intrusion by another user, memory pressure or contention.

## Product Purpose

The local control room of a project built on the template. It launches and queues the
entry points with their full configuration, streams their output, chains experiments into
sequences, browses runs, reports and figures, inspects trained models, and watches the
machine (GPUs, CPU, memory, disk, processes) with watchdogs and neighbour-impact guards.

## Brand Personality

Instrument-grade, scientific and quiet. A measurement device, not a SaaS product. The
console uses one dark theme throughout, the vocabulary of live machine output: consoles,
terminals and telemetry traces.

## Anti-references

- Packed uniform card walls where every metric shouts equally.
- Consumer dashboard gloss: pastel cards, gradient text, decorative effects that carry no
  information.
- Grafana-default look: undifferentiated panel soup.

## Design Principles

- Hierarchy by importance: GPUs are the scarce resource and read first. Guards and alarms
  surface only when they matter.
- Telemetry is an instrument panel: monospaced numerals, hairlines, trace plots, high
  legibility at a distance.
- Alarm states are unmissable and unambiguous; idle states are calm.
- Density is allowed (process tables, logs, configuration forms), but rhythm and grouping
  keep it scannable.
- Every launch control maps to a configuration field of its entry point, and the command
  preview shows the exact command line a launch runs.

## Accessibility and Inclusion

Reduced motion is honoured everywhere: it gates the canvas loops, the line-figure
autopilot and the CSS animations. Text contrast is at least 4.5:1, and state is never
conveyed by colour alone; lights are paired with text labels.
