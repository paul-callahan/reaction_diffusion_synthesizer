use std::f32::consts::TAU;
use std::sync::{Arc, RwLock};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::types::{SimulationFeatures, SimulationParams, SynthParams};

#[derive(Clone)]
struct SharedAudioState {
    params: Arc<RwLock<SynthParams>>,
    sim_params: Arc<RwLock<SimulationParams>>,
    features: Arc<RwLock<SimulationFeatures>>,
}

pub struct AudioEngine {
    shared: SharedAudioState,
    _stream: cpal::Stream,
    pub device_name: String,
    pub sample_rate: u32,
}

impl AudioEngine {
    pub fn new() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "No default audio output device found".to_owned())?;
        let device_name = device
            .name()
            .unwrap_or_else(|_| "Unknown output device".to_owned());

        let supported_config = device
            .default_output_config()
            .map_err(|err| format!("Failed to read default output config: {err}"))?;
        let config = supported_config.config();
        let sample_rate = config.sample_rate.0;

        let shared = SharedAudioState {
            params: Arc::new(RwLock::new(SynthParams::default())),
            sim_params: Arc::new(RwLock::new(SimulationParams::default())),
            features: Arc::new(RwLock::new(SimulationFeatures::default())),
        };

        let stream = match supported_config.sample_format() {
            cpal::SampleFormat::F32 => build_stream::<f32>(&device, &config, shared.clone())?,
            cpal::SampleFormat::I16 => build_stream::<i16>(&device, &config, shared.clone())?,
            cpal::SampleFormat::U16 => build_stream::<u16>(&device, &config, shared.clone())?,
            other => {
                return Err(format!(
                    "Unsupported output sample format from audio device: {other:?}"
                ));
            }
        };

        stream
            .play()
            .map_err(|err| format!("Failed to start audio stream: {err}"))?;

        Ok(Self {
            shared,
            _stream: stream,
            device_name,
            sample_rate,
        })
    }

    pub fn set_params(&self, params: SynthParams) {
        write_copy(&self.shared.params, params);
    }

    pub fn set_simulation_params(&self, sim_params: SimulationParams) {
        write_copy(&self.shared.sim_params, sim_params);
    }

    pub fn update_features(&self, features: SimulationFeatures) {
        write_copy(&self.shared.features, features);
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: SharedAudioState,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;
    let mut voice = SpectralVoice::new(sample_rate);

    let params_handle = shared.params.clone();
    let sim_params_handle = shared.sim_params.clone();
    let features_handle = shared.features.clone();

    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [T], _| {
                let params = read_copy(&params_handle);
                let sim_params = read_copy(&sim_params_handle);
                let features = read_copy(&features_handle);
                write_audio_buffer(data, channels, &mut voice, params, sim_params, features);
            },
            move |err| {
                eprintln!("Audio stream error: {err}");
            },
            None,
        )
        .map_err(|err| format!("Failed to build output stream: {err}"))?;

    Ok(stream)
}

fn write_audio_buffer<T>(
    output: &mut [T],
    channels: usize,
    voice: &mut SpectralVoice,
    params: SynthParams,
    sim_params: SimulationParams,
    features: SimulationFeatures,
) where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    for frame in output.chunks_mut(channels) {
        let (left, right) = voice.next_sample(params, sim_params, features);
        frame[0] = T::from_sample(left);

        if channels > 1 {
            frame[1] = T::from_sample(right);
            if channels > 2 {
                for sample in &mut frame[2..] {
                    *sample = T::from_sample(0.5 * (left + right));
                }
            }
        }
    }
}

const PARTIAL_COUNT: usize = 10;
const TRITAVE_SCALE: [i32; 8] = [0, 2, 3, 5, 7, 8, 10, 12];

#[derive(Clone, Copy)]
struct Partial {
    harmonic_ratio: f32,
    exotic_ratio: f32,
    gain: f32,
    pan: f32,
    carrier_phase: f32,
    mod_phase: f32,
    shimmer_phase: f32,
}

impl Partial {
    const fn new(harmonic_ratio: f32, exotic_ratio: f32, gain: f32, pan: f32) -> Self {
        Self {
            harmonic_ratio,
            exotic_ratio,
            gain,
            pan,
            carrier_phase: 0.0,
            mod_phase: 0.0,
            shimmer_phase: 0.0,
        }
    }
}

struct SpectralVoice {
    sample_rate: f32,
    base_freq_smooth: f32,
    pulse_phase: f32,
    pulse_env: f32,
    pulse_target: f32,
    core_phase: f32,
    tritave_phase: f32,
    spectral_phase: f32,
    texture_phase: f32,
    chaos_state: f32,
    noise_state: u64,
    partials: [Partial; PARTIAL_COUNT],
    reson_l: f32,
    reson_r: f32,
    chorus: StereoChorus,
    high_pass_left: HighPass,
    high_pass_right: HighPass,
    limiter: Limiter,
}

impl SpectralVoice {
    fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            base_freq_smooth: 140.0,
            pulse_phase: 0.0,
            pulse_env: 0.0,
            pulse_target: 0.0,
            core_phase: 0.0,
            tritave_phase: 0.0,
            spectral_phase: 0.0,
            texture_phase: 0.0,
            chaos_state: 0.36,
            noise_state: 0xA7F4_19B3_14C0_77E1,
            partials: [
                Partial::new(1.00, 1.09, 0.30, -0.55),
                Partial::new(1.50, 1.74, 0.26, 0.50),
                Partial::new(2.00, 2.37, 0.22, -0.25),
                Partial::new(2.50, 3.06, 0.18, 0.70),
                Partial::new(3.00, 3.95, 0.15, -0.74),
                Partial::new(3.80, 4.88, 0.13, 0.18),
                Partial::new(4.60, 6.01, 0.11, -0.05),
                Partial::new(5.60, 7.38, 0.09, 0.82),
                Partial::new(6.80, 9.05, 0.07, -0.90),
                Partial::new(8.20, 11.10, 0.06, 0.35),
            ],
            reson_l: 0.0,
            reson_r: 0.0,
            chorus: StereoChorus::new(sample_rate),
            high_pass_left: HighPass::new(),
            high_pass_right: HighPass::new(),
            limiter: Limiter::new(),
        }
    }

    fn next_sample(
        &mut self,
        params: SynthParams,
        sim_params: SimulationParams,
        features: SimulationFeatures,
    ) -> (f32, f32) {
        let feed_norm = normalize(sim_params.feed, 0.005, 0.095);
        let kill_norm = normalize(sim_params.kill, 0.02, 0.09);
        let dt_norm = normalize(sim_params.dt, 0.2, 1.4);
        let step_norm = normalize(sim_params.steps_per_frame as f32, 1.0, 20.0);
        let diff_balance = ((sim_params.diff_a - sim_params.diff_b)
            / (sim_params.diff_a + sim_params.diff_b + 1.0e-6))
            .clamp(-1.0, 1.0);

        let flux = (features.temporal_flux * 18.0).clamp(0.0, 1.0);
        let reaction = (features.reaction_energy * 24.0).clamp(0.0, 1.0);
        let variance = (features.variance_b * 40.0).clamp(0.0, 1.0);
        let edge = features.edge_activity.clamp(0.0, 1.0);
        let movement = (0.52 * flux + 0.36 * reaction + 0.28 * edge).clamp(0.0, 1.0);

        let pitch_offset = (features.centroid_y - 0.5) * (2.0 + 4.0 * params.pitch_sensitivity)
            + (feed_norm - kill_norm) * 1.8
            + diff_balance * 1.1
            + (reaction - 0.5) * 1.4;
        let raw_freq = params.base_pitch_hz.max(45.0) * 2.0_f32.powf(pitch_offset / 12.0);

        let tritave_root = (params.base_pitch_hz * 0.92 + 22.0).clamp(55.0, 260.0);
        let quantized_freq =
            quantize_to_tritave_scale(raw_freq, tritave_root, &TRITAVE_SCALE).clamp(90.0, 1_800.0);

        let glide = (0.0004 + movement * 0.0014 + dt_norm * 0.0006).clamp(0.0004, 0.003);
        self.base_freq_smooth += (quantized_freq - self.base_freq_smooth) * glide;
        let base_freq = self.base_freq_smooth;

        self.core_phase = wrap_phase(self.core_phase + TAU * base_freq / self.sample_rate);
        self.tritave_phase =
            wrap_phase(self.tritave_phase + TAU * (base_freq * 3.0) / self.sample_rate);
        let spectral_rate = 0.07 + movement * 0.35 + variance * 0.25;
        self.spectral_phase =
            wrap_phase(self.spectral_phase + TAU * spectral_rate / self.sample_rate);
        let texture_rate = 0.19 + reaction * 0.7 + flux * 0.9;
        self.texture_phase = wrap_phase(self.texture_phase + TAU * texture_rate / self.sample_rate);

        let core_tone = self.core_phase.sin() * 0.7 + self.tritave_phase.sin() * 0.3;
        let spectral_warp = self.spectral_phase.sin();

        let pulse_rate_hz = 0.3 + movement * 3.4 + step_norm * 2.1;
        let prev_pulse_phase = self.pulse_phase;
        self.pulse_phase = wrap_phase(self.pulse_phase + TAU * pulse_rate_hz / self.sample_rate);
        if self.pulse_phase < prev_pulse_phase {
            self.pulse_target = (self.pulse_target + 0.28 + 0.48 * movement).min(1.2);
        }
        self.pulse_target *= (0.995 - dt_norm * 0.002).clamp(0.986, 0.998);
        let pulse_slew = if self.pulse_target > self.pulse_env {
            (0.012 + movement * 0.010).clamp(0.012, 0.030)
        } else {
            (0.003 + dt_norm * 0.002).clamp(0.003, 0.008)
        };
        self.pulse_env += (self.pulse_target - self.pulse_env) * pulse_slew;
        let pulse = self.pulse_env.clamp(0.0, 1.2);

        let chaos_r = 3.56 + 0.20 * feed_norm + 0.14 * movement;
        self.chaos_state =
            (chaos_r * self.chaos_state * (1.0 - self.chaos_state)).clamp(0.001, 0.999);
        let chaos = self.chaos_state * 2.0 - 1.0;
        let noise = self.next_noise();

        let alien = params.alien_blend.clamp(0.0, 1.0);
        let exoticity =
            (alien * 0.85 + movement * 0.35 + (1.0 - params.resonance) * 0.28).clamp(0.0, 1.0);

        let excitation = params.excitation
            * (0.22 + pulse * 0.42 + core_tone * 0.11 + chaos * 0.05 + noise * 0.02);

        let mut left = 0.0;
        let mut right = 0.0;

        for (idx, partial) in self.partials.iter_mut().enumerate() {
            let ratio = lerp(partial.harmonic_ratio, partial.exotic_ratio, exoticity);
            let harmonic_pos = idx as f32 / (PARTIAL_COUNT as f32 - 1.0);
            let spectral_tilt =
                (params.brightness - 0.5) * 1.6 + (variance - 0.5) * 1.2 + spectral_warp * 0.5;
            let tilt_gain = (1.15 + spectral_tilt * (0.5 - harmonic_pos)).clamp(0.2, 1.8);
            let detune = 1.0
                + (features.centroid_x - 0.5) * 0.03
                + (feed_norm - kill_norm) * 0.02 * partial.pan;
            let texture_warp =
                self.texture_phase.sin() * 0.6 + self.texture_phase.cos() * 0.25 + chaos * 0.4;
            let warp = 1.0
                + 0.035 * texture_warp * partial.pan
                + 0.018 * spectral_warp * (harmonic_pos * 2.0 - 1.0);
            let freq = (base_freq * ratio * detune * warp).clamp(70.0, 14_000.0);

            partial.carrier_phase =
                wrap_phase(partial.carrier_phase + TAU * freq / self.sample_rate);

            let mod_freq = freq * (1.24 + 0.18 * partial.pan + 0.24 * alien);
            partial.mod_phase = wrap_phase(partial.mod_phase + TAU * mod_freq / self.sample_rate);

            let shimmer_rate = 0.12 + movement * 1.4 + alien * 0.9 + partial.pan.abs() * 0.35;
            partial.shimmer_phase =
                wrap_phase(partial.shimmer_phase + TAU * shimmer_rate / self.sample_rate);

            let fm_index =
                0.10 + params.brightness * 0.55 + movement * 0.55 + alien * 0.60 + variance * 0.45;
            let modulator = partial.mod_phase.sin() * fm_index;

            let shimmer = 0.68 + 0.32 * partial.shimmer_phase.sin();
            let grain = 0.85 + 0.15 * (partial.shimmer_phase * (1.0 + harmonic_pos * 2.0)).sin();
            let sample = (partial.carrier_phase + modulator).sin()
                * partial.gain
                * tilt_gain
                * shimmer
                * grain
                * (0.40 + 0.55 * excitation.abs());

            let pan = (partial.pan * params.stereo_spread).clamp(-1.0, 1.0);
            let left_gain = ((1.0 - pan) * 0.5).sqrt();
            let right_gain = ((1.0 + pan) * 0.5).sqrt();

            left += sample * left_gain;
            right += sample * right_gain;
        }

        let ring = (left * right) * (0.11 + 0.35 * alien * movement);
        left += ring;
        right -= ring * 0.8;

        let air = noise * 0.01 * (0.15 + alien * 0.55);
        left += air;
        right -= air * 0.7;

        let reson_decay = (0.965 + params.resonance * 0.028 - movement * 0.010).clamp(0.940, 0.995);
        self.reson_l = (self.reson_l * reson_decay + left * 0.08).clamp(-1.5, 1.5);
        self.reson_r = (self.reson_r * reson_decay + right * 0.08).clamp(-1.5, 1.5);
        let reson_mix = 0.08 + params.resonance * 0.18 + alien * 0.10;
        left += self.reson_l * reson_mix;
        right += self.reson_r * reson_mix;

        let richness =
            (0.25 + alien * 0.40 + params.brightness * 0.25 + movement * 0.20).clamp(0.15, 0.95);
        let chorus_mix = (0.08 + 0.20 * richness).clamp(0.08, 0.30);
        let chorus_depth_sec = (0.0016 + 0.0036 * richness).clamp(0.0012, 0.006);
        let chorus_rate_hz = (0.06 + 0.25 * movement + 0.12 * variance).clamp(0.05, 0.8);
        let chorus_feedback = (0.04 + params.resonance * 0.07 + alien * 0.05).clamp(0.03, 0.18);
        (left, right) = self.chorus.process(
            left,
            right,
            chorus_mix,
            chorus_depth_sec,
            chorus_rate_hz,
            params.stereo_spread,
            chorus_feedback,
        );

        let drive = 1.0 + params.drive * (2.0 + alien * 1.8);
        left = (left * drive).tanh();
        right = (right * drive).tanh();

        let hp_cutoff = (120.0 + 180.0 * alien + 70.0 * movement + 50.0 * (1.0 - params.resonance))
            .clamp(80.0, 420.0);
        left = self
            .high_pass_left
            .process(left, hp_cutoff, self.sample_rate);
        right = self
            .high_pass_right
            .process(right, hp_cutoff, self.sample_rate);

        self.limiter.process(left, right, params.master_gain)
    }

    fn next_noise(&mut self) -> f32 {
        self.noise_state ^= self.noise_state << 13;
        self.noise_state ^= self.noise_state >> 7;
        self.noise_state ^= self.noise_state << 17;
        let normalized = (self.noise_state as f64 / u64::MAX as f64) as f32;
        normalized * 2.0 - 1.0
    }
}

struct HighPass {
    x1: f32,
    y1: f32,
}

impl HighPass {
    fn new() -> Self {
        Self { x1: 0.0, y1: 0.0 }
    }

    fn process(&mut self, input: f32, cutoff_hz: f32, sample_rate: f32) -> f32 {
        let rc = 1.0 / (TAU * cutoff_hz.max(20.0));
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);
        let output = alpha * (self.y1 + input - self.x1);
        self.x1 = input;
        self.y1 = output;
        output
    }
}

struct Limiter {
    envelope: f32,
}

impl Limiter {
    fn new() -> Self {
        Self { envelope: 0.0 }
    }

    fn process(&mut self, left: f32, right: f32, gain: f32) -> (f32, f32) {
        let peak = left.abs().max(right.abs());
        if peak > self.envelope {
            self.envelope = peak;
        } else {
            self.envelope *= 0.999;
        }

        let threshold = 0.92;
        let limiter_gain = if self.envelope > threshold {
            threshold / self.envelope
        } else {
            1.0
        };

        (
            (left * limiter_gain * gain).clamp(-1.0, 1.0),
            (right * limiter_gain * gain).clamp(-1.0, 1.0),
        )
    }
}

struct StereoChorus {
    sample_rate: f32,
    phase: f32,
    write_idx: usize,
    buf_l: Vec<f32>,
    buf_r: Vec<f32>,
    fb_l: f32,
    fb_r: f32,
}

impl StereoChorus {
    fn new(sample_rate: f32) -> Self {
        let len = (sample_rate * 0.08).ceil() as usize + 4;
        Self {
            sample_rate,
            phase: 0.0,
            write_idx: 0,
            buf_l: vec![0.0; len],
            buf_r: vec![0.0; len],
            fb_l: 0.0,
            fb_r: 0.0,
        }
    }

    fn process(
        &mut self,
        left: f32,
        right: f32,
        mix: f32,
        depth_sec: f32,
        rate_hz: f32,
        stereo_spread: f32,
        feedback: f32,
    ) -> (f32, f32) {
        let spread = stereo_spread.clamp(0.0, 1.0);
        let rate = rate_hz.clamp(0.05, 1.2);
        self.phase = wrap_phase(self.phase + TAU * rate / self.sample_rate);

        let len_f = self.buf_l.len() as f32;
        let max_delay = (len_f - 3.0).max(4.0);
        let depth = (depth_sec * self.sample_rate).clamp(1.0, 0.018 * self.sample_rate);

        let base_l = (0.0125 * self.sample_rate).clamp(4.0, max_delay);
        let base_r = (0.0187 * self.sample_rate).clamp(4.0, max_delay);
        let lfo_l = 0.5 + 0.5 * self.phase.sin();
        let lfo_r = 0.5 + 0.5 * (self.phase + TAU * (0.23 + 0.20 * spread)).sin();

        let delay_l = (base_l + depth * lfo_l).clamp(2.0, max_delay);
        let delay_r = (base_r + depth * lfo_r).clamp(2.0, max_delay);

        let tap_l = Self::read_tap(&self.buf_l, self.write_idx, delay_l);
        let tap_r = Self::read_tap(&self.buf_r, self.write_idx, delay_r);

        let wet_l = tap_l * 0.82 + tap_r * 0.18;
        let wet_r = tap_r * 0.82 + tap_l * 0.18;

        let fb = feedback.clamp(0.0, 0.28);
        let cross = 0.30 + 0.50 * spread;
        let write_l =
            (left + (self.fb_l * (1.0 - cross) + self.fb_r * cross) * fb).clamp(-1.6, 1.6);
        let write_r =
            (right + (self.fb_r * (1.0 - cross) + self.fb_l * cross) * fb).clamp(-1.6, 1.6);

        self.buf_l[self.write_idx] = write_l;
        self.buf_r[self.write_idx] = write_r;
        self.write_idx = (self.write_idx + 1) % self.buf_l.len();
        self.fb_l = wet_l;
        self.fb_r = wet_r;

        let blend = mix.clamp(0.0, 0.45);
        (
            left + (wet_l - left) * blend,
            right + (wet_r - right) * blend,
        )
    }

    fn read_tap(buffer: &[f32], write_idx: usize, delay_samples: f32) -> f32 {
        let len = buffer.len();
        let len_f = len as f32;
        let pos = (write_idx as f32 - delay_samples).rem_euclid(len_f);
        let idx0 = pos.floor() as usize;
        let idx1 = (idx0 + 1) % len;
        let frac = pos - idx0 as f32;
        buffer[idx0] + (buffer[idx1] - buffer[idx0]) * frac
    }
}

fn read_copy<T: Copy>(lock: &RwLock<T>) -> T {
    match lock.read() {
        Ok(guard) => *guard,
        Err(poisoned) => *poisoned.into_inner(),
    }
}

fn write_copy<T: Copy>(lock: &RwLock<T>, value: T) {
    match lock.write() {
        Ok(mut guard) => *guard = value,
        Err(poisoned) => *poisoned.into_inner() = value,
    }
}

fn wrap_phase(mut phase: f32) -> f32 {
    while phase >= TAU {
        phase -= TAU;
    }
    phase
}

fn normalize(value: f32, min: f32, max: f32) -> f32 {
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn quantize_to_tritave_scale(freq: f32, root: f32, scale: &[i32]) -> f32 {
    let root = root.max(20.0);
    let freq = freq.max(20.0);

    let mut best = freq;
    let mut best_distance = f32::INFINITY;

    for tritave in -5..=5 {
        for degree in scale {
            let steps = *degree as f32 + 13.0 * tritave as f32;
            let candidate = root * 3.0_f32.powf(steps / 13.0);
            let distance = (candidate / freq).ln().abs();
            if distance < best_distance {
                best_distance = distance;
                best = candidate;
            }
        }
    }

    best
}
