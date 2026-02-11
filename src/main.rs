mod app;
mod audio_engine;
mod reaction_diffusion;
mod types;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 760.0])
            .with_min_inner_size([900.0, 620.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Reaction Diffusion Synthesizer",
        options,
        Box::new(|cc| Ok(Box::new(app::SynthApp::new(cc)))),
    )
}
