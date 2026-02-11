# Reaction Diffusion Synthesizer

<img width="1196" height="792" alt="image" src="https://github.com/user-attachments/assets/aa7a6554-0028-41ca-bd08-742b076b58fe" />


Small Rust experiment: a reaction-diffusion simulation drives a real-time audio synth and a live GUI visualization.

## Mechanism
The visual core is a Gray-Scott reaction-diffusion system: two virtual chemicals diffuse and react, creating evolving spatial patterns.
Those pattern statistics (motion, edges, centroid, variance) are mapped into synth parameters to produce changing timbre and pitch behavior.

## Run
Ensure Rust is installed

```bash
cd reaction_diffusion_synthesizer
~/.cargo/bin/cargo run
```

