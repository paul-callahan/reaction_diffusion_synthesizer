use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};

use crate::audio_engine::AudioEngine;
use crate::reaction_diffusion::ReactionDiffusionField;
use crate::types::{SimulationFeatures, SimulationParams, SynthParams};

pub struct SynthApp {
    field: ReactionDiffusionField,
    sim_params: SimulationParams,
    synth_params: SynthParams,
    features: SimulationFeatures,
    texture: Option<TextureHandle>,
    audio: Option<AudioEngine>,
    audio_error: Option<String>,
    paused: bool,
}

impl SynthApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut field = ReactionDiffusionField::new(220, 220);
        let sim_params = SimulationParams::default();
        let synth_params = SynthParams::default();
        let features = field.step(&sim_params);

        let (audio, audio_error) = match AudioEngine::new() {
            Ok(engine) => (Some(engine), None),
            Err(err) => (None, Some(err)),
        };

        let app = Self {
            field,
            sim_params,
            synth_params,
            features,
            texture: None,
            audio,
            audio_error,
            paused: false,
        };

        app.sync_audio();
        app
    }

    fn sync_audio(&self) {
        if let Some(audio) = &self.audio {
            audio.set_params(self.synth_params);
            audio.set_simulation_params(self.sim_params);
            audio.update_features(self.features);
        }
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        let image =
            ColorImage::from_rgba_unmultiplied(self.field.dimensions(), &self.field.to_rgba8());

        if let Some(texture) = &mut self.texture {
            texture.set(image, TextureOptions::LINEAR);
        } else {
            self.texture =
                Some(ctx.load_texture("reaction-diffusion", image, TextureOptions::LINEAR));
        }
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Reaction-Diffusion");
        ui.add(egui::Slider::new(&mut self.sim_params.feed, 0.005..=0.095).text("feed"));
        ui.add(egui::Slider::new(&mut self.sim_params.kill, 0.02..=0.09).text("kill"));
        ui.add(egui::Slider::new(&mut self.sim_params.diff_a, 0.2..=1.6).text("diff A"));
        ui.add(egui::Slider::new(&mut self.sim_params.diff_b, 0.05..=1.2).text("diff B"));
        ui.add(egui::Slider::new(&mut self.sim_params.dt, 0.2..=1.4).text("dt"));
        ui.add(egui::Slider::new(&mut self.sim_params.steps_per_frame, 1..=20).text("steps/frame"));

        ui.horizontal(|ui| {
            if ui
                .button(if self.paused {
                    "Resume simulation"
                } else {
                    "Pause simulation"
                })
                .clicked()
            {
                self.paused = !self.paused;
            }

            if ui.button("Reseed field").clicked() {
                self.field.reset_seed();
            }
        });

        ui.separator();
        ui.heading("Resonator Synth");
        ui.add(
            egui::Slider::new(&mut self.synth_params.master_gain, 0.0..=0.9).text("master gain"),
        );
        ui.add(
            egui::Slider::new(&mut self.synth_params.base_pitch_hz, 20.0..=440.0)
                .logarithmic(true)
                .text("base pitch (Hz)"),
        );
        ui.add(
            egui::Slider::new(&mut self.synth_params.pitch_sensitivity, 0.0..=1.5)
                .text("pitch sensitivity"),
        );
        ui.add(egui::Slider::new(&mut self.synth_params.excitation, 0.0..=2.0).text("excitation"));
        ui.add(egui::Slider::new(&mut self.synth_params.resonance, 0.05..=1.0).text("resonance"));
        ui.add(egui::Slider::new(&mut self.synth_params.brightness, 0.0..=1.0).text("brightness"));
        ui.add(
            egui::Slider::new(&mut self.synth_params.alien_blend, 0.0..=1.0)
                .text("alien blend / exotica"),
        );
        ui.add(egui::Slider::new(&mut self.synth_params.drive, 0.0..=1.0).text("drive"));
        ui.add(
            egui::Slider::new(&mut self.synth_params.stereo_spread, 0.0..=1.0)
                .text("stereo spread"),
        );

        ui.separator();
        if let Some(audio) = &self.audio {
            ui.label(format!("Audio device: {}", audio.device_name));
            ui.label(format!("Sample rate: {} Hz", audio.sample_rate));
        } else if let Some(err) = &self.audio_error {
            ui.colored_label(
                egui::Color32::from_rgb(230, 100, 100),
                format!("Audio offline: {err}"),
            );
        }
    }

    fn draw_visuals(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(format!("mean B: {:.4}", self.features.mean_b));
            ui.separator();
            ui.label(format!("variance: {:.5}", self.features.variance_b));
            ui.separator();
            ui.label(format!("edge: {:.4}", self.features.edge_activity));
            ui.separator();
            ui.label(format!("flux: {:.5}", self.features.temporal_flux));
            ui.separator();
            ui.label(format!("reaction: {:.5}", self.features.reaction_energy));
            ui.separator();
            ui.label(format!(
                "centroid: ({:.3}, {:.3})",
                self.features.centroid_x, self.features.centroid_y
            ));
            ui.separator();
            ui.label(format!("sim time: {:.1}", self.field.time()));
        });

        ui.separator();

        if let Some(texture) = &self.texture {
            let image_size = texture.size_vec2();
            let available = ui.available_size();
            let scale = (available.x / image_size.x)
                .min(available.y / image_size.y)
                .clamp(0.8, 3.0);
            ui.image((texture.id(), image_size * scale));
        }
    }
}

impl eframe::App for SynthApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.paused {
            self.features = self.field.step(&self.sim_params);
        }

        self.sync_audio();
        self.update_texture(ctx);

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(290.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.draw_controls(ui);
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_visuals(ui);
        });

        ctx.request_repaint();
    }
}
