#[derive(Clone, Copy, Debug)]
pub struct SimulationParams {
    pub feed: f32,
    pub kill: f32,
    pub diff_a: f32,
    pub diff_b: f32,
    pub dt: f32,
    pub steps_per_frame: usize,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            feed: 0.0367,
            kill: 0.0649,
            diff_a: 1.0,
            diff_b: 0.5,
            dt: 1.0,
            steps_per_frame: 8,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SynthParams {
    pub master_gain: f32,
    pub base_pitch_hz: f32,
    pub pitch_sensitivity: f32,
    pub excitation: f32,
    pub resonance: f32,
    pub brightness: f32,
    pub drive: f32,
    pub stereo_spread: f32,
    pub alien_blend: f32,
}

impl Default for SynthParams {
    fn default() -> Self {
        Self {
            master_gain: 0.2,
            base_pitch_hz: 110.0,
            pitch_sensitivity: 0.6,
            excitation: 0.8,
            resonance: 0.65,
            brightness: 0.55,
            drive: 0.2,
            stereo_spread: 0.85,
            alien_blend: 0.35,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SimulationFeatures {
    pub mean_b: f32,
    pub variance_b: f32,
    pub edge_activity: f32,
    pub temporal_flux: f32,
    pub reaction_energy: f32,
    pub centroid_x: f32,
    pub centroid_y: f32,
}

impl Default for SimulationFeatures {
    fn default() -> Self {
        Self {
            mean_b: 0.0,
            variance_b: 0.0,
            edge_activity: 0.0,
            temporal_flux: 0.0,
            reaction_energy: 0.0,
            centroid_x: 0.5,
            centroid_y: 0.5,
        }
    }
}
