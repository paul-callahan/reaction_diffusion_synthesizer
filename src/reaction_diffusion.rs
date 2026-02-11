use crate::types::{SimulationFeatures, SimulationParams};

pub struct ReactionDiffusionField {
    width: usize,
    height: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    next_a: Vec<f32>,
    next_b: Vec<f32>,
    time: f32,
    seed_state: u64,
    low_flux_counter: usize,
}

impl ReactionDiffusionField {
    pub fn new(width: usize, height: usize) -> Self {
        let len = width * height;
        let mut field = Self {
            width,
            height,
            a: vec![1.0; len],
            b: vec![0.0; len],
            next_a: vec![0.0; len],
            next_b: vec![0.0; len],
            time: 0.0,
            seed_state: 0xD1B5_4A32_9C8E_2711 ^ ((width as u64) << 16) ^ height as u64,
            low_flux_counter: 0,
        };
        field.seed();
        field
    }

    pub fn dimensions(&self) -> [usize; 2] {
        [self.width, self.height]
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    pub fn reset_seed(&mut self) {
        self.a.fill(1.0);
        self.b.fill(0.0);
        self.time = 0.0;
        self.low_flux_counter = 0;
        self.seed_state ^= 0x9E37_79B9_7F4A_7C15;
        self.seed();
    }

    pub fn step(&mut self, params: &SimulationParams) -> SimulationFeatures {
        let mut temporal_flux_sum = 0.0;
        let mut reaction_energy_sum = 0.0;

        for step_idx in 0..params.steps_per_frame {
            for y in 0..self.height {
                for x in 0..self.width {
                    let idx = self.idx(x, y);
                    let a = self.a[idx];
                    let b = self.b[idx];
                    let lap_a = Self::laplacian(&self.a, self.width, self.height, x, y);
                    let lap_b = Self::laplacian(&self.b, self.width, self.height, x, y);
                    let reaction = a * b * b;

                    let next_a = a
                        + (params.diff_a * lap_a - reaction + params.feed * (1.0 - a)) * params.dt;
                    let next_b = b
                        + (params.diff_b * lap_b + reaction - (params.kill + params.feed) * b)
                            * params.dt;

                    let next_a = next_a.clamp(0.0, 1.0);
                    let next_b = next_b.clamp(0.0, 1.0);

                    temporal_flux_sum += (next_b - b).abs();
                    reaction_energy_sum += reaction;

                    self.next_a[idx] = next_a;
                    self.next_b[idx] = next_b;
                }
            }

            std::mem::swap(&mut self.a, &mut self.next_a);
            std::mem::swap(&mut self.b, &mut self.next_b);

            if step_idx % 3 == 0 {
                self.inject_micro_perturbation(0.0015);
            }

            self.time += params.dt;
        }

        let norm = (self.width * self.height * params.steps_per_frame) as f32;
        let temporal_flux = (temporal_flux_sum / norm).clamp(0.0, 1.0);
        let reaction_energy = (reaction_energy_sum / norm).clamp(0.0, 1.0);

        if temporal_flux < 0.00045 {
            self.low_flux_counter += 1;
        } else {
            self.low_flux_counter = 0;
        }

        if self.low_flux_counter > 14 {
            self.inject_sparks(14, 0.85);
            self.low_flux_counter = 0;
        }

        self.measure(temporal_flux, reaction_energy)
    }

    pub fn to_rgba8(&self) -> Vec<u8> {
        let mut min_b = f32::INFINITY;
        let mut max_b = f32::NEG_INFINITY;

        for &value in &self.b {
            min_b = min_b.min(value);
            max_b = max_b.max(value);
        }

        let span = (max_b - min_b).max(1.0e-5);
        let mut rgba = Vec::with_capacity(self.width * self.height * 4);

        for idx in 0..self.a.len() {
            let a = self.a[idx];
            let b = self.b[idx];

            let b_norm = ((b - min_b) / span).clamp(0.0, 1.0);
            let edge = (b - a).abs().powf(0.55).clamp(0.0, 1.0);
            let t = (0.65 * b_norm + 0.35 * edge).clamp(0.0, 1.0);
            let [mut r, mut g, mut bl] = spectral_color(t);

            let glow = edge * 0.45;
            r = (r + glow * 0.30).clamp(0.0, 1.0);
            g = (g + glow * 0.45).clamp(0.0, 1.0);
            bl = (bl + (1.0 - b_norm) * 0.08).clamp(0.0, 1.0);

            rgba.extend_from_slice(&[
                (r * 255.0) as u8,
                (g * 255.0) as u8,
                (bl * 255.0) as u8,
                255,
            ]);
        }

        rgba
    }

    fn seed(&mut self) {
        let blob_count = 9 + self.rand_range_usize(16);

        for _ in 0..blob_count {
            let cx = self.rand_range_usize(self.width.max(1));
            let cy = self.rand_range_usize(self.height.max(1));
            let radius = 3 + self.rand_range_usize((self.width.min(self.height) / 10).max(4));
            let radius_f = radius as f32;

            for y in cy.saturating_sub(radius)..(cy + radius + 1).min(self.height) {
                for x in cx.saturating_sub(radius)..(cx + radius + 1).min(self.width) {
                    let dx = x as f32 - cx as f32;
                    let dy = y as f32 - cy as f32;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist <= radius_f {
                        let falloff = 1.0 - (dist / radius_f);
                        let idx = self.idx(x, y);
                        self.a[idx] = (self.a[idx] - 0.85 * falloff).clamp(0.0, 1.0);
                        self.b[idx] = (self.b[idx] + 0.95 * falloff).clamp(0.0, 1.0);
                    }
                }
            }
        }

        self.inject_sparks(10, 0.75);
    }

    fn inject_micro_perturbation(&mut self, chance: f32) {
        if self.rand01() > chance {
            return;
        }

        let cx = self.rand_range_usize(self.width);
        let cy = self.rand_range_usize(self.height);
        for y in cy.saturating_sub(1)..(cy + 2).min(self.height) {
            for x in cx.saturating_sub(1)..(cx + 2).min(self.width) {
                let idx = self.idx(x, y);
                self.b[idx] = (self.b[idx] + 0.22).clamp(0.0, 1.0);
                self.a[idx] = (self.a[idx] - 0.12).clamp(0.0, 1.0);
            }
        }
    }

    fn inject_sparks(&mut self, count: usize, intensity: f32) {
        for _ in 0..count {
            let cx = self.rand_range_usize(self.width);
            let cy = self.rand_range_usize(self.height);
            let radius = 1 + self.rand_range_usize(4);
            let radius_f = radius as f32;

            for y in cy.saturating_sub(radius)..(cy + radius + 1).min(self.height) {
                for x in cx.saturating_sub(radius)..(cx + radius + 1).min(self.width) {
                    let dx = x as f32 - cx as f32;
                    let dy = y as f32 - cy as f32;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist <= radius_f {
                        let falloff = 1.0 - (dist / radius_f);
                        let idx = self.idx(x, y);
                        self.b[idx] = (self.b[idx] + intensity * falloff).clamp(0.0, 1.0);
                        self.a[idx] = (self.a[idx] - 0.55 * intensity * falloff).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn laplacian(field: &[f32], width: usize, height: usize, x: usize, y: usize) -> f32 {
        let xm = if x == 0 { width - 1 } else { x - 1 };
        let xp = if x + 1 == width { 0 } else { x + 1 };
        let ym = if y == 0 { height - 1 } else { y - 1 };
        let yp = if y + 1 == height { 0 } else { y + 1 };

        let idx = |xx: usize, yy: usize| -> usize { yy * width + xx };

        -field[idx(x, y)]
            + 0.2 * (field[idx(xm, y)] + field[idx(xp, y)] + field[idx(x, ym)] + field[idx(x, yp)])
            + 0.05
                * (field[idx(xm, ym)]
                    + field[idx(xp, ym)]
                    + field[idx(xm, yp)]
                    + field[idx(xp, yp)])
    }

    fn measure(&self, temporal_flux: f32, reaction_energy: f32) -> SimulationFeatures {
        let mut mean = 0.0;
        for value in &self.b {
            mean += *value;
        }
        mean /= self.b.len() as f32;

        let mut variance = 0.0;
        let mut edge = 0.0;
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;
        let mut weight_sum = 0.0;

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.idx(x, y);
                let b = self.b[idx];
                let d = b - mean;
                variance += d * d;

                let xm = if x == 0 { self.width - 1 } else { x - 1 };
                let xp = if x + 1 == self.width { 0 } else { x + 1 };
                let ym = if y == 0 { self.height - 1 } else { y - 1 };
                let yp = if y + 1 == self.height { 0 } else { y + 1 };

                let gx = (self.b[self.idx(xp, y)] - self.b[self.idx(xm, y)]).abs();
                let gy = (self.b[self.idx(x, yp)] - self.b[self.idx(x, ym)]).abs();
                edge += 0.5 * (gx + gy);

                weight_sum += b;
                weighted_x += b * x as f32;
                weighted_y += b * y as f32;
            }
        }

        variance /= self.b.len() as f32;
        edge = (edge / self.b.len() as f32).clamp(0.0, 1.0);

        let centroid_x = if weight_sum > 1.0e-6 {
            (weighted_x / weight_sum) / self.width as f32
        } else {
            0.5
        };
        let centroid_y = if weight_sum > 1.0e-6 {
            (weighted_y / weight_sum) / self.height as f32
        } else {
            0.5
        };

        SimulationFeatures {
            mean_b: mean,
            variance_b: variance,
            edge_activity: edge,
            temporal_flux,
            reaction_energy,
            centroid_x,
            centroid_y,
        }
    }

    fn next_rand_u64(&mut self) -> u64 {
        let mut x = self.seed_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.seed_state = x;
        x.wrapping_mul(2685821657736338717)
    }

    fn rand01(&mut self) -> f32 {
        let v = (self.next_rand_u64() >> 40) as u32;
        v as f32 / u32::MAX as f32
    }

    fn rand_range_usize(&mut self, max_exclusive: usize) -> usize {
        if max_exclusive <= 1 {
            return 0;
        }
        (self.next_rand_u64() as usize) % max_exclusive
    }
}

fn spectral_color(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let anchors = [
        (0.00, [0.02, 0.03, 0.13]),
        (0.25, [0.00, 0.46, 0.95]),
        (0.50, [0.05, 0.92, 0.35]),
        (0.75, [0.98, 0.86, 0.10]),
        (1.00, [0.95, 0.12, 0.18]),
    ];

    for pair in anchors.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if t >= t0 && t <= t1 {
            let alpha = (t - t0) / (t1 - t0);
            return [
                c0[0] + (c1[0] - c0[0]) * alpha,
                c0[1] + (c1[1] - c0[1]) * alpha,
                c0[2] + (c1[2] - c0[2]) * alpha,
            ];
        }
    }

    anchors[anchors.len() - 1].1
}
