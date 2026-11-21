//! Apply ordered carving-mask intervals to pre-lighting terrain.
use super::{
    material_condition::{
        CompiledSurfaceRule, MaterialConditions, MaterialDensitySources, MaterialRule,
        MaterialSurfaceQueries,
    },
    material_stage::{SurfaceError, TopMaterialInputs, top_material_cell},
    terrain::{TerrainAquifer, TerrainBlockStates, TerrainChunk, TerrainError},
};
use crate::{
    heightmap::HeightmapState,
    registry::{BlockRegistry, BlockStateId},
};
use horizon_core::{
    carving::CarvingMask, density::SamplerContext, noise::DensityVolume, position::BlockPos,
    random::RandomSource,
};
use std::collections::BTreeSet;

/// Per-state predicates resolved once from the world's block registry and
/// `minecraft:uncarvable` tag. Keep these bound to the same registry as the chunk.
pub struct CarvingBlocks {
    flags: Vec<u8>,
}
impl CarvingBlocks {
    pub fn from_registry(
        registry: &BlockRegistry,
        uncarvable: &BTreeSet<String>,
    ) -> Result<Self, TerrainError> {
        for name in uncarvable.iter().map(String::as_str).chain([
            "minecraft:grass_block",
            "minecraft:mycelium",
            "minecraft:dirt",
        ]) {
            if registry.default_state(name).is_none() {
                return Err(TerrainError::InvalidState);
            }
        }
        let flags = (0..registry.len())
            .map(|index| {
                let name = &registry.state(BlockStateId(index as u32)).unwrap().name;
                u8::from(uncarvable.contains(name))
                    | (u8::from(matches!(
                        name.as_str(),
                        "minecraft:grass_block" | "minecraft:mycelium"
                    )) << 1)
                    | (u8::from(name == "minecraft:dirt") << 2)
            })
            .collect();
        Ok(Self { flags })
    }
    fn flags(&self, state: BlockStateId) -> Result<u8, TerrainError> {
        self.flags
            .get(state.0 as usize)
            .copied()
            .ok_or(TerrainError::InvalidState)
    }
}

/// Visit mask runs in packed order and each interval top to bottom. The filter
/// receives local X/Z and absolute Y before any block or aquifer lookup. A run
/// remembers grass/mycelium even if its aquifer returns no replacement. Exposed
/// dirt below subsequent carved cells is restored through `top_material`.
/// Earlier writes, aquifer state and fluid postprocessing survive later errors.
pub fn apply_carving_mask<R: RandomSource, S: MaterialSurfaceQueries, B: TerrainBlockStates>(
    chunk: &mut TerrainChunk<'_, B>,
    mask: &CarvingMask,
    rule: MaterialRule,
    inputs: TopMaterialInputs<'_, R, S>,
    aquifer: &mut (impl TerrainAquifer + ?Sized),
    blocks: &CarvingBlocks,
    filter: impl FnMut([i32; 3]) -> bool,
) -> Result<(), SurfaceError>
where
    R::Positional: Clone,
{
    apply_carving_mask_shared(
        chunk, mask, &rule, &mut None, inputs, aquifer, blocks, filter,
    )
}
/// `apply_carving_mask` with the rule compiled once for the chunks `shared`
/// serves (`MaterialConditions::add_shared_rule`).
pub fn apply_carving_mask_shared<
    R: RandomSource,
    S: MaterialSurfaceQueries,
    B: TerrainBlockStates,
>(
    chunk: &mut TerrainChunk<'_, B>,
    mask: &CarvingMask,
    rule: &MaterialRule,
    shared: &mut Option<CompiledSurfaceRule<R>>,
    inputs: TopMaterialInputs<'_, R, S>,
    aquifer: &mut (impl TerrainAquifer + ?Sized),
    blocks: &CarvingBlocks,
    mut filter: impl FnMut([i32; 3]) -> bool,
) -> Result<(), SurfaceError>
where
    R::Positional: Clone,
{
    if mask.is_empty() {
        return Ok(());
    }
    // Most carved columns never expose dirt below grass/mycelium. When `shared`
    // is already compiled for this generation (steady-state after surface
    // building), defer constructing the per-cell top-material worker and
    // sampling ore-vein densities until an exposed dirt block actually needs
    // repair. Otherwise validate and bind the surface rule before the first
    // terrain write so invalid rules (such as unknown `NoiseThreshold` noises)
    // still fail even when no dirt is exposed.
    let TopMaterialInputs {
        generation,
        root,
        surface,
        preliminary_surface,
        density_context,
    } = inputs;
    let mut top_worker = None;
    let mut top_inputs = if shared
        .as_ref()
        .is_some_and(|cached| cached.matches_generation(generation))
    {
        Some((generation, root, surface, preliminary_surface))
    } else {
        let volume =
            DensityVolume::new([1; 3], [0; 3], [1; 3]).map_err(|_| SurfaceError::InvalidVolume)?;
        let sources = MaterialDensitySources::new(
            surface,
            volume,
            preliminary_surface,
            SamplerContext::default(),
        )?;
        let mut worker =
            MaterialConditions::<R, _>::with_possible_biomes(generation, root, sources, None);
        let rule_id = worker.add_shared_rule(rule, shared)?;
        top_worker = Some((worker, rule_id));
        None
    };
    let cx = chunk.position().x.wrapping_shl(4);
    let cz = chunk.position().z.wrapping_shl(4);
    for column in mask.columns() {
        let mut found_surface = false;
        let x = cx.wrapping_add(column.x);
        let z = cz.wrapping_add(column.z);
        for y in (column.min_y..=column.max_y).rev() {
            if !filter([column.x, y, column.z]) {
                continue;
            }
            let flags = blocks.flags(chunk.block(x, y, z))?;
            if flags & 1 != 0 {
                continue;
            }
            found_surface |= flags & 2 != 0;
            let Some(state) = aquifer
                .compute_substance(density_context, [x, y, z], 0.)
                .map_err(TerrainError::from)?
            else {
                continue;
            };
            chunk.set_block_worldgen(x, y, z, state)?;
            if aquifer.should_schedule_fluid_update() && chunk.block_properties(state).has_fluid() {
                chunk.mark_postprocessing(x, y, z);
            }
            let below = y.wrapping_sub(1);
            if found_surface && blocks.flags(chunk.block(x, below, z))? & 4 != 0 {
                let submerged = chunk.block_properties(state).has_fluid();
                let (worker, rule_id) = match &mut top_worker {
                    Some((worker, rule_id)) => {
                        worker.retarget_surface_densities([x, below, z])?;
                        (worker, *rule_id)
                    }
                    None => {
                        let (generation, root, surface, preliminary_surface) =
                            top_inputs.take().expect("top material inputs");
                        let volume = DensityVolume::new([1; 3], [x, below, z], [1; 3])
                            .map_err(|_| SurfaceError::InvalidVolume)?;
                        let sources = MaterialDensitySources::new(
                            surface,
                            volume,
                            preliminary_surface,
                            SamplerContext::default(),
                        )?;
                        let mut worker = MaterialConditions::<R, _>::with_possible_biomes(
                            generation, root, sources, None,
                        );
                        let rule_id = worker.add_shared_rule(rule, shared)?;
                        let (worker, rule_id) = top_worker.insert((worker, rule_id));
                        (worker, *rule_id)
                    }
                };
                let replacement = top_material_cell(
                    worker,
                    chunk,
                    rule_id,
                    BlockPos::new(x, below, z),
                    submerged,
                )?;
                if let Some(replacement) = replacement {
                    chunk.set_block_worldgen(x, below, z, replacement)?;
                    if chunk.block_properties(replacement).has_fluid() {
                        chunk.mark_postprocessing(x, below, z);
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DATA_VERSION,
        worldgen::{
            anchor::WorldGenerationContext,
            aquifer::{DisabledAquifer, GlobalFluidPicker},
            material_condition::MaterialCondition,
            terrain::TerrainChunk,
        },
    };
    use horizon_core::{
        density::{DensitySampler, SampleError},
        nbt::Tag,
        position::ChunkPos,
        random::LegacyRandom,
    };
    use std::sync::Arc;

    #[derive(Clone, Copy)]
    struct SimpleState {
        air: bool,
        fluid: bool,
    }
    impl HeightmapState for SimpleState {
        fn is_air(self) -> bool {
            self.air
        }
        fn is_air_block(self) -> bool {
            self.air
        }
        fn has_fluid(self) -> bool {
            self.fluid
        }
        fn blocks_motion_in_heightmap(self) -> bool {
            !self.air && !self.fluid
        }
        fn blocks_motion_in_heightmap_no_leaves(self) -> bool {
            !self.air && !self.fluid
        }
    }
    struct SimpleStates {
        air: BlockStateId,
        water: BlockStateId,
        lava: BlockStateId,
    }
    impl TerrainBlockStates for SimpleStates {
        type State = SimpleState;
        fn heightmap_state(&self, id: BlockStateId) -> SimpleState {
            SimpleState {
                air: id == self.air,
                fluid: id == self.water || id == self.lava,
            }
        }
    }
    #[derive(Debug)]
    struct ZeroSampler;
    impl DensitySampler for ZeroSampler {
        fn sample_value(&self, _: &mut SamplerContext, _: [i32; 3]) -> Result<f32, SampleError> {
            Ok(0.0)
        }
    }
    struct EmptySurface;
    impl MaterialSurfaceQueries for EmptySurface {
        fn surface_depth(&mut self, _: i32, _: i32) -> i32 {
            0
        }
        fn surface_secondary(&mut self, _: i32, _: i32) -> f64 {
            0.0
        }
    }

    #[test]
    fn invalid_noise_threshold_fails_before_carving_writes_any_block() {
        let names = [
            "minecraft:air",
            "minecraft:void_air",
            "minecraft:cave_air",
            "minecraft:water",
            "minecraft:lava",
            "minecraft:stone",
            "minecraft:grass_block",
            "minecraft:mycelium",
            "minecraft:dirt",
        ];
        let defaults: Vec<_> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (*n, Tag::Int(i as i32)))
            .collect();
        let states: Vec<_> = names
            .iter()
            .map(|n| Tag::compound([("id", Tag::String((*n).into()))]))
            .collect();
        let registry = BlockRegistry::from_nbt(&Tag::compound([
            ("DataVersion", Tag::Int(DATA_VERSION)),
            ("states", Tag::List(states)),
            ("defaults", Tag::compound(defaults)),
        ]))
        .unwrap();
        let air = registry.default_state("minecraft:air").unwrap();
        let water = registry.default_state("minecraft:water").unwrap();
        let lava = registry.default_state("minecraft:lava").unwrap();
        let stone = registry.default_state("minecraft:stone").unwrap();
        let blocks = CarvingBlocks::from_registry(&registry, &BTreeSet::new()).unwrap();
        let mut chunk = TerrainChunk::new(
            ChunkPos::new(0, 0),
            0,
            32,
            &registry,
            SimpleStates { air, water, lava },
        )
        .unwrap();
        chunk.set_block_worldgen(2, 10, 3, stone).unwrap();
        let mut mask = CarvingMask::new(0, 31).unwrap();
        mask.carve(2, 10, 3).unwrap();
        let mut aquifer = DisabledAquifer::new(GlobalFluidPicker::new(0, water, lava, air), air);
        let invalid_rule = MaterialRule::Condition {
            if_true: MaterialCondition::NoiseThreshold {
                noise: "minecraft:no_such_noise".to_owned(),
                min: 0.0,
                max: 1.0,
                is_3d: false,
            },
            then_run: Box::new(MaterialRule::Block(stone)),
        };
        let mut density_context = SamplerContext::default();
        let root = LegacyRandom::new(1).fork_positional();
        let err = apply_carving_mask::<LegacyRandom, _, _>(
            &mut chunk,
            &mask,
            invalid_rule,
            TopMaterialInputs {
                generation: WorldGenerationContext::new(0, 32, 0),
                root,
                surface: EmptySurface,
                preliminary_surface: Arc::new(ZeroSampler),
                density_context: &mut density_context,
            },
            &mut aquifer,
            &blocks,
            |_| true,
        );
        assert!(matches!(err, Err(SurfaceError::Decode(_))));
        assert_eq!(chunk.block(2, 10, 3), stone);
    }
}
