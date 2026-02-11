# Reaction Diffusion Synthesizer

Small Rust experiment: a reaction-diffusion simulation drives a real-time audio synth and a live GUI visualization.

## Theory (short)
The visual core is a Gray-Scott reaction-diffusion system: two virtual chemicals diffuse and react, creating evolving spatial patterns.
Those pattern statistics (motion, edges, centroid, variance) are mapped into synth parameters to produce changing timbre and pitch behavior.

## Run
Ensure Rust is installed

```bash
cd reaction_diffusion_synthesizer
~/.cargo/bin/cargo run
```

