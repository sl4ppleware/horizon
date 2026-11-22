//! Transport for the vanilla quad sorting / viewpoint parity gate.
use horizon_client::gfx::translucency::{SortState, Sorter, quad_center};
use serde::Deserialize;
use serde_json::json;
use std::{
    fs,
    io::{BufWriter, Write},
};

#[derive(Deserialize)]
struct Case {
    origin: [i32; 3],
    quads: Vec<[[f32; 3]; 4]>,
    cameras: Vec<[f64; 3]>,
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = crate::args().skip(1).collect();
    let [input, output] = args.as_slice() else {
        return Err("usage: translucency_fixture INPUT OUTPUT".into());
    };
    let mut out = BufWriter::new(fs::File::create(output)?);
    for line in fs::read_to_string(input)?.lines() {
        let case: Case = serde_json::from_str(line)?;
        let centers: Vec<_> = case.quads.iter().map(|q| quad_center(q[0], q[2])).collect();
        let mut state = SortState::from_quads(&case.quads, |&p| p);
        let mut sorter = Sorter::default();
        let mut last_indices = Vec::new();
        let frames: Vec<_> = case
            .cameras
            .iter()
            .map(|&camera| {
                let update = if let Some(indices) = state.update(camera, case.origin, &mut sorter) {
                    last_indices.clear();
                    last_indices.extend_from_slice(indices);
                    true
                } else {
                    false
                };
                json!({"indices": &last_indices, "update": update})
            })
            .collect();
        let bits: Vec<_> = centers
            .iter()
            .map(|c| c.map(|f| f.to_bits() as i32))
            .collect();
        writeln!(out, "{}", json!({"centers": bits, "frames": frames}))?;
    }
    Ok(())
}
