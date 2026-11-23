//! Removing entities a server dropped (`RemoveEntities`), out of remote.rs,
//! which may not grow.
use super::Game;
use super::remote::EntityBucket;

impl Game<'_> {
    pub(super) fn remove_entity(&mut self, id: i32) {
        self.remove_entities(&[id]);
    }

    /// `ClientboundRemoveEntitiesPacket`: a packet's ids go together, the big
    /// entity lists filtered once (#13007), each id's own cleanup in order.
    pub(super) fn remove_entities(&mut self, ids: &[i32]) {
        if ids.is_empty() {
            return;
        }
        let removed_set: Option<horizon_client::fast_hash::FxHashSet<u32>> =
            (ids.len() > 1).then(|| ids.iter().map(|&id| id as u32).collect());
        let single = ids[0] as u32;
        let is_removed = |id: u32| match &removed_set {
            Some(set) => set.contains(&id),
            None => id == single,
        };
        let mut bucket_mask = 0u16;
        if let Some(r) = &self.remote {
            for &id in ids {
                if let Some(t) = r.entities.get(&id) {
                    bucket_mask |= t.bucket.bit();
                } else {
                    bucket_mask = EntityBucket::ALL_MASK;
                    break;
                }
            }
        } else {
            bucket_mask = EntityBucket::ALL_MASK;
        }
        let mut vehicles = horizon_client::fast_hash::FxHashSet::default();
        let mut include = |id| {
            if is_removed(id) {
                vehicles.insert(id);
            }
        };
        if let Some(r) = &self.remote {
            for &id in r.passenger_vehicles.values() {
                include(id);
            }
        }
        if let Some(id) = self.riding {
            include(id);
        }
        // Shared/server riding paths can also set these links without a
        // SetPassengers entry. Preserve their cleanup when any removed entity
        // could be a vehicle or rider, scanning once per packet.
        if bucket_mask & EntityBucket::VEHICLE_CANDIDATE_MASK != 0 {
            for m in &self.mobs {
                if let Some(id) = m.extra.mount {
                    include(id);
                }
                if m.extra.rider.is_some() {
                    include(m.id);
                }
            }
            for id in self
                .text_displays
                .values()
                .filter_map(|d| d.vehicle)
                .chain(self.armor_stands.iter().filter_map(|a| a.vehicle))
                .chain(self.interactions.values().filter_map(|i| i.vehicle))
                .chain(
                    self.boats
                        .iter()
                        .filter(|b| b.paddle_controller)
                        .map(|b| b.id),
                )
            {
                include(id);
            }
        }
        // Snapshot before removing passenger keys: a removed passenger can
        // precede its vehicle, whose other seat state still needs clearing.
        for &id in ids {
            let bucket = self
                .remote
                .as_ref()
                .and_then(|r| r.entities.get(&id).map(|t| t.bucket));
            self.detach_remote_tracking_entity(id);
            self.audio.remove_sound_entity(id as u32);
            // Entity.setRemoved detaches every passenger, even when no empty
            // SetPassengers packet preceded the vehicle's removal. Otherwise
            // mobs and other players keep their passenger pose after it is gone.
            if vehicles.contains(&(id as u32)) {
                self.remote_set_passengers(id, &[]);
            }
            // Before the crystals go: the dragon's beam index follows them.
            match bucket {
                Some(EntityBucket::Dragon) => self.remove_remote_dragon(id),
                Some(EntityBucket::Object) => self.remove_remote_object(id as u32),
                Some(EntityBucket::Hook) => self.remove_remote_hook(id as u32),
                Some(EntityBucket::Other) => {
                    self.crystals.retain(|c| c.id != id as u32);
                    self.remove_remote_small_projectiles(id as u32);
                }
                Some(
                    EntityBucket::Mob
                    | EntityBucket::Dropped
                    | EntityBucket::ArmorStand
                    | EntityBucket::Boat
                    | EntityBucket::Cart
                    | EntityBucket::Orb
                    | EntityBucket::Projectile
                    | EntityBucket::Falling
                    | EntityBucket::Tnt
                    | EntityBucket::Trident
                    | EntityBucket::Rocket,
                ) => {}
                None => {
                    self.remove_remote_dragon(id);
                    self.crystals.retain(|c| c.id != id as u32);
                    self.remove_remote_object(id as u32);
                    self.remove_remote_small_projectiles(id as u32);
                }
            }
            self.persist.uuids.remove(&(id as u32));
            if let Some(r) = self.remote.as_mut() {
                r.glowing.remove(&id);
                r.entities.remove(&id);
                r.passenger_vehicles.remove(&(id as u32));
                r.lerps.remove(&id);
                r.cart_lerps.remove(&id);
                r.stepped.remove(&id);
                r.head_lerps.remove(&id);
                r.pending_items.remove(&id);
            }
        }
        let keep = |id: u32| !is_removed(id);
        if bucket_mask & (EntityBucket::Mob.bit() | EntityBucket::Dragon.bit()) != 0 {
            self.mobs.retain(|m| keep(m.id));
        }
        if bucket_mask & EntityBucket::Orb.bit() != 0 {
            self.orbs.retain(|o| keep(o.id));
        }
        if bucket_mask & EntityBucket::Dropped.bit() != 0 {
            self.dropped.retain(|d| keep(d.id));
        }
        if bucket_mask & EntityBucket::Falling.bit() != 0 {
            self.falling.retain(|f| keep(f.id));
        }
        if bucket_mask & EntityBucket::Tnt.bit() != 0 {
            self.tnts.retain(|t| keep(t.id));
        }
        if bucket_mask & EntityBucket::ArmorStand.bit() != 0 {
            self.armor_stands.retain(|a| keep(a.id));
        }
        if bucket_mask & EntityBucket::Boat.bit() != 0 {
            self.boats.retain(|b| keep(b.id));
        }
        if bucket_mask & EntityBucket::Cart.bit() != 0 {
            self.carts.retain(|c| keep(c.id));
        }
        if bucket_mask & EntityBucket::Projectile.bit() != 0 {
            self.creatures.projectiles.retain(|p| keep(p.id));
        }
        if bucket_mask & EntityBucket::Trident.bit() != 0 {
            self.tridents.retain(|t| keep(t.id));
        }
        if bucket_mask & EntityBucket::Rocket.bit() != 0 {
            self.rockets.retain(|r| keep(r.id));
        }
        let mut camera_removed = false;
        if let Some(r) = self.remote.as_mut() {
            r.passenger_vehicles.retain(|_, v| keep(*v));
            if r.camera.is_some_and(|id| is_removed(id as u32)) {
                r.camera = None;
                camera_removed = true;
            }
        }
        if camera_removed {
            // Minecraft.setCameraEntity returns to the local player when the
            // spectated entity leaves, including clearing its post chain.
            self.check_entity_post_effect(true);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "daily"]
    fn remove_entities_detaches_only_removed_vehicles() {
        use crate::game::{
            Audio, Game, Options, Progress, Settings, execution_role::ExecutionRole, load_resources,
        };
        use horizon_protocol::packets::play::{self, Clientbound as C};
        let root = std::path::PathBuf::from(std::env::var_os("HORIZON_TEST_RESOURCES").unwrap());
        let settings = Settings::default();
        let loaded = load_resources(
            horizon_assets::resource_source::ResourceRoot::directory(root.clone()),
            Vec::new(),
            None,
            &root,
            &settings,
            std::time::Instant::now(),
            false,
            crate::game::execution_role::ExecutionRole::IntegratedServer,
            &Progress::new(),
            &mut |_, _| unreachable!(),
        )
        .unwrap();
        let mut options = Options::parse(std::iter::empty()).unwrap().unwrap();
        options.resources = root.clone();
        options.execution_role = ExecutionRole::Client;
        let mut game = Game::new(
            &loaded.resources,
            loaded.pack,
            loaded.gui.unwrap(),
            options,
            settings,
            Audio::silent(&horizon_assets::resource_source::ResourceSource::directory(
                root.clone(),
            )),
            None,
        )
        .unwrap();
        let (client, _server) = horizon_protocol::connection::Pipe::pair();
        game.connect_local(client, "PassengerRemovalTest").unwrap();
        for (id, name) in [
            (40, "minecart"),
            (41, "minecart"),
            (42, "armor_stand"),
            (43, "zombie"),
            (44, "falling_block"),
            (45, "falling_block"),
            (46, "falling_block"),
        ] {
            let kind = game
                .remote
                .as_ref()
                .unwrap()
                .builtin
                .get("minecraft:entity_type")
                .unwrap()
                .id(&format!("minecraft:{name}"))
                .unwrap();
            game.handle_packet(C::AddEntity(play::AddEntity {
                id,
                uuid: id as u128,
                kind,
                position: [0.0, 64.0, 0.0],
                velocity: [0.0; 3],
                pitch: 0,
                yaw: 0,
                head_yaw: 0,
                data: 0,
            }))
            .unwrap();
        }
        game.remote_set_passengers(40, &[42]);
        game.remote_set_passengers(41, &[43]);
        assert_eq!(game.armor_stands[0].vehicle, Some(40));
        assert_eq!(game.mobs[0].extra.mount, Some(41));
        game.handle_packet(C::RemoveEntities(play::RemoveEntities {
            ids: vec![44, 40, 46],
        }))
        .unwrap();
        let r = game.remote.as_ref().unwrap();
        assert_eq!(r.entities[&42].vehicle, None);
        assert_eq!(game.armor_stands[0].vehicle, None);
        assert_eq!(r.entities[&43].vehicle, Some(41));
        assert_eq!(game.mobs[0].extra.mount, Some(41));
        assert!(game.mobs[0].extra.passenger);
        assert_eq!(r.passenger_vehicles.get(&42), None);
        assert_eq!(r.passenger_vehicles.get(&43), Some(&41));
        assert_eq!(game.falling.iter().map(|e| e.id).collect::<Vec<_>>(), [45]);
    }

    #[test]
    #[ignore = "daily"]
    fn add_entity_replaces_falling_block_without_removing_other_ids() {
        use crate::game::{
            Audio, Game, Options, Progress, Settings, execution_role::ExecutionRole, load_resources,
        };
        use horizon_protocol::packets::play::{self, Clientbound as C};

        let root = std::path::PathBuf::from(std::env::var_os("HORIZON_TEST_RESOURCES").unwrap());
        let settings = Settings::default();
        let loaded = load_resources(
            horizon_assets::resource_source::ResourceRoot::directory(root.clone()),
            Vec::new(),
            None,
            &root,
            &settings,
            std::time::Instant::now(),
            false,
            ExecutionRole::Client,
            &Progress::new(),
            &mut |_, _| unreachable!(),
        )
        .unwrap();
        let mut options = Options::parse(std::iter::empty()).unwrap().unwrap();
        options.resources = root.clone();
        options.execution_role = ExecutionRole::Client;
        let mut game = Game::new(
            &loaded.resources,
            loaded.pack,
            loaded.gui.unwrap(),
            options,
            settings,
            Audio::silent(&horizon_assets::resource_source::ResourceSource::directory(
                root.clone(),
            )),
            None,
        )
        .unwrap();
        let (client, _server) = horizon_protocol::connection::Pipe::pair();
        game.connect_local(client, "SpawnTest").unwrap();
        let kind = game
            .remote
            .as_ref()
            .unwrap()
            .builtin
            .get("minecraft:entity_type")
            .unwrap()
            .id("minecraft:falling_block")
            .unwrap();
        for (id, position) in [
            (40, [1.0, 64.0, 2.0]),
            (41, [3.0, 65.0, 4.0]),
            (40, [5.0, 66.0, 6.0]),
            (42, [7.0, 67.0, 8.0]),
        ] {
            game.handle_packet(C::AddEntity(play::AddEntity {
                id,
                uuid: id as u128,
                kind,
                position,
                velocity: [0.0, -0.1, 0.0],
                pitch: 0,
                yaw: 0,
                head_yaw: 0,
                data: 0,
            }))
            .unwrap();
            assert_eq!(game.falling.iter().filter(|f| f.id == id as u32).count(), 1);
            let block = game.falling.iter().find(|f| f.id == id as u32).unwrap();
            assert_eq!(
                [block.position.x, block.position.y, block.position.z],
                position
            );
            assert_eq!(block.velocity.y, -0.1);
        }
        let blocks: Vec<_> = game
            .falling
            .iter()
            .map(|f| (f.id, [f.position.x, f.position.y, f.position.z]))
            .collect();
        assert_eq!(
            blocks,
            [
                (41, [3.0, 65.0, 4.0]),
                (40, [5.0, 66.0, 6.0]),
                (42, [7.0, 67.0, 8.0]),
            ]
        );
    }

    #[test]
    #[ignore = "daily"]
    fn remove_entities_filters_mixed_entity_lists() {
        use crate::game::{
            Audio, Game, Options, Progress, Settings, execution_role::ExecutionRole, load_resources,
        };
        use horizon_protocol::packets::play::{self, Clientbound as C};

        let root = std::path::PathBuf::from(std::env::var_os("HORIZON_TEST_RESOURCES").unwrap());
        let settings = Settings::default();
        let loaded = load_resources(
            horizon_assets::resource_source::ResourceRoot::directory(root.clone()),
            Vec::new(),
            None,
            &root,
            &settings,
            std::time::Instant::now(),
            false,
            crate::game::execution_role::ExecutionRole::IntegratedServer,
            &Progress::new(),
            &mut |_, _| unreachable!(),
        )
        .unwrap();
        let mut options = Options::parse(std::iter::empty()).unwrap().unwrap();
        options.resources = root.clone();
        options.execution_role = ExecutionRole::Client;
        let mut game = Game::new(
            &loaded.resources,
            loaded.pack,
            loaded.gui.unwrap(),
            options,
            settings,
            Audio::silent(&horizon_assets::resource_source::ResourceSource::directory(
                root.clone(),
            )),
            None,
        )
        .unwrap();
        let (client, _server) = horizon_protocol::connection::Pipe::pair();
        game.connect_local(client, "RemovalTest").unwrap();
        for (id, kind) in [
            (40, "minecraft:falling_block"),
            (41, "minecraft:falling_block"),
            (42, "minecraft:falling_block"),
            (43, "minecraft:tnt"),
            (44, "minecraft:tnt"),
            (45, "minecraft:experience_orb"),
            (46, "minecraft:experience_orb"),
            (47, "minecraft:zombie"),
            (48, "minecraft:zombie"),
        ] {
            if matches!(kind, "minecraft:falling_block" | "minecraft:tnt") {
                assert!(game.spawn_remote_projectile(
                    kind,
                    &horizon_protocol::packets::play::AddEntity {
                        id,
                        uuid: id as u128,
                        kind: 0,
                        position: [0.0, 64.0, 0.0],
                        velocity: [0.0; 3],
                        pitch: 0,
                        yaw: 0,
                        head_yaw: 0,
                        data: 0,
                    },
                    (0.0, 0.0),
                ));
            } else {
                game.spawn_entity(id, kind, [0.0, 64.0, 0.0], [0.0; 3]);
            }
        }
        for id in [49, 50] {
            game.dropped.push(horizon_client::entities::ItemEntity::new(
                id,
                horizon_client::items::ItemStack::new("minecraft:sand", 1),
                horizon_core::geometry::Vec3::new(0.0, 64.0, 0.0),
                horizon_core::geometry::Vec3::ZERO,
                0.0,
            ));
        }
        game.handle_packet(C::RemoveEntities(play::RemoveEntities {
            ids: vec![40, 42, 43, 45, 47, 49],
        }))
        .unwrap();
        assert_eq!(game.falling.iter().map(|e| e.id).collect::<Vec<_>>(), [41]);
        assert_eq!(game.tnts.iter().map(|e| e.id).collect::<Vec<_>>(), [44]);
        assert_eq!(game.orbs.iter().map(|e| e.id).collect::<Vec<_>>(), [46]);
        assert_eq!(game.mobs.iter().map(|e| e.id).collect::<Vec<_>>(), [48]);
        assert_eq!(game.dropped.iter().map(|e| e.id).collect::<Vec<_>>(), [50]);
    }

    #[test]
    #[ignore = "daily"]
    fn bucket_dispatch_handles_mixed_updates_duplicate_spawns_and_selective_removal() {
        use crate::game::{
            Audio, Game, Options, Progress, Settings, execution_role::ExecutionRole, load_resources,
        };
        use horizon_protocol::entity_data::{Value, serializer};
        use horizon_protocol::packets::play::{self, Clientbound as C};

        let root = std::path::PathBuf::from(std::env::var_os("HORIZON_TEST_RESOURCES").unwrap());
        let settings = Settings::default();
        let loaded = load_resources(
            horizon_assets::resource_source::ResourceRoot::directory(root.clone()),
            Vec::new(),
            None,
            &root,
            &settings,
            std::time::Instant::now(),
            false,
            ExecutionRole::Client,
            &Progress::new(),
            &mut |_, _| unreachable!(),
        )
        .unwrap();
        let mut options = Options::parse(std::iter::empty()).unwrap().unwrap();
        options.resources = root.clone();
        options.execution_role = ExecutionRole::Client;
        let mut game = Game::new(
            &loaded.resources,
            loaded.pack,
            loaded.gui.unwrap(),
            options,
            settings,
            Audio::silent(&horizon_assets::resource_source::ResourceSource::directory(
                root.clone(),
            )),
            None,
        )
        .unwrap();
        let (client, _server) = horizon_protocol::connection::Pipe::pair();
        game.connect_local(client, "BucketDispatchTest").unwrap();

        let spawn = |game: &mut Game<'_>, id: i32, name: &str, position: [f64; 3]| {
            let kind = game
                .remote
                .as_ref()
                .unwrap()
                .builtin
                .get("minecraft:entity_type")
                .unwrap()
                .id(&format!("minecraft:{name}"))
                .unwrap();
            game.handle_packet(C::AddEntity(play::AddEntity {
                id,
                uuid: id as u128,
                kind,
                position,
                velocity: [0.0; 3],
                pitch: 0,
                yaw: 0,
                head_yaw: 0,
                data: 0,
            }))
            .unwrap();
        };

        spawn(&mut game, 100, "zombie", [1.0, 64.0, 1.0]);
        spawn(&mut game, 101, "zombie", [2.0, 64.0, 2.0]);
        // Duplicate mob id replaces the previous entry without leaving two entries.
        spawn(&mut game, 100, "zombie", [3.0, 64.0, 3.0]);
        assert_eq!(
            game.mobs.iter().map(|m| m.id).collect::<Vec<_>>(),
            [101, 100]
        );
        assert_eq!(game.mobs[1].position.x, 3.0);

        spawn(&mut game, 200, "armor_stand", [4.0, 64.0, 4.0]);
        spawn(&mut game, 201, "experience_orb", [5.0, 64.0, 5.0]);
        spawn(&mut game, 202, "tnt", [6.0, 64.0, 6.0]);
        spawn(&mut game, 203, "arrow", [7.0, 64.0, 7.0]);
        spawn(&mut game, 204, "oak_boat", [8.0, 64.0, 8.0]);
        spawn(&mut game, 205, "minecart", [9.0, 64.0, 9.0]);

        // Bucket-routed SetEntityData & SetEntityMotion across distinct collections.
        game.handle_packet(C::SetEntityData(play::SetEntityData {
            id: 200,
            values: vec![(15, serializer::BYTE, Value::Byte(1 | 4))],
        }))
        .unwrap();
        assert!(game.armor_stands[0].small);
        assert!(game.armor_stands[0].show_arms);

        game.handle_packet(C::SetEntityData(play::SetEntityData {
            id: 201,
            values: vec![(8, serializer::INT, Value::Int(17))],
        }))
        .unwrap();
        assert_eq!(game.orbs[0].value, 17);

        game.handle_packet(C::SetEntityData(play::SetEntityData {
            id: 202,
            values: vec![(8, serializer::INT, Value::Int(35))],
        }))
        .unwrap();
        assert_eq!(game.tnts[0].fuse, 35);

        game.handle_packet(C::SetEntityMotion(play::SetEntityMotion {
            id: 203,
            velocity: [0.25, 0.5, -0.25],
        }))
        .unwrap();
        assert_eq!(game.creatures.projectiles[0].velocity.y, 0.5);

        game.handle_packet(C::SetEntityData(play::SetEntityData {
            id: 204,
            values: vec![(13, serializer::INT, Value::Int(12))],
        }))
        .unwrap();
        assert_eq!(game.boats[0].bubble_time, 12);

        game.handle_packet(C::SetEntityData(play::SetEntityData {
            id: 205,
            values: vec![(12, serializer::INT, Value::Int(9))],
        }))
        .unwrap();
        assert_eq!(game.carts[0].custom_offset, Some(9));

        // Removing only the orb and arrow must not disturb surviving buckets' structure stamps.
        let mobs_stamp = game.mobs.structure_stamp();
        let stands_stamp = game.armor_stands.structure_stamp();
        let boats_stamp = game.boats.structure_stamp();
        let carts_stamp = game.carts.structure_stamp();
        let tnts_stamp = game.tnts.structure_stamp();
        game.handle_packet(C::RemoveEntities(play::RemoveEntities {
            ids: vec![201, 203],
        }))
        .unwrap();
        assert!(game.orbs.is_empty());
        assert!(game.creatures.projectiles.is_empty());
        assert_eq!(game.mobs.structure_stamp(), mobs_stamp);
        assert_eq!(game.armor_stands.structure_stamp(), stands_stamp);
        assert_eq!(game.boats.structure_stamp(), boats_stamp);
        assert_eq!(game.carts.structure_stamp(), carts_stamp);
        assert_eq!(game.tnts.structure_stamp(), tnts_stamp);

        // Removing the boat and armor stand removes only those collections.
        game.handle_packet(C::RemoveEntities(play::RemoveEntities {
            ids: vec![200, 204],
        }))
        .unwrap();
        assert!(game.armor_stands.is_empty());
        assert!(game.boats.is_empty());
        assert_eq!(game.mobs.structure_stamp(), mobs_stamp);
        assert_eq!(game.carts.structure_stamp(), carts_stamp);
        assert_eq!(game.tnts.structure_stamp(), tnts_stamp);
    }
}
