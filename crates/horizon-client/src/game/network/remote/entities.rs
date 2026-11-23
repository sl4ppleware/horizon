//! Spawning and moving the server's entities.

use super::*;

struct MoveUpdate {
    position: [f64; 3],
    path: Option<Path>,
    rotation: Option<(f32, f32)>,
    on_ground: bool,
    packet: MovePacket,
}

fn step_remote_mob(
    stepped: &mut HashMap<i32, SteppedInterpolation>,
    id: i32,
    m: &mut Mob,
    update: MoveUpdate,
) {
    // Shulker.createInterpolationHandler is NO_OP, so a move packet
    // is setPos at once. The 6-tick slide starts from the block
    // the shulker occupied before this packet.
    if m.species() == horizon_client::mobs::Species::Shulker {
        stepped.remove(&id);
        if update.path.is_some() {
            horizon_client::mobs::shulker::client_set_pos(
                m,
                update.position[0],
                update.position[1],
                update.position[2],
            );
        }
        if let Some((yaw, pitch)) = update.rotation {
            m.yaw += wrap_degrees(yaw - m.yaw);
            m.head_pitch = (pitch % 360.0).clamp(-90.0, 90.0);
        }
        m.on_ground = update.on_ground;
        return;
    }
    // A living entity: LivingEntity.createInterpolationHandler's
    // SteppedInterpolationHandler, over its type's update interval.
    let entity = PosRot {
        position: [m.position.x, m.position.y, m.position.z],
        yaw: m.yaw,
        pitch: m.head_pitch,
    };
    let handler = stepped.entry(id).or_insert_with(|| {
        let interval = horizon_client::lan::tracking::settings(&m.kind.id).1;
        SteppedInterpolation::new(interval as i32)
    });
    if let Some(at) = stepped_move(
        handler,
        update.packet,
        entity,
        update.position,
        update.path,
        update.rotation,
    ) {
        // snapTo: setOldPosAndRot too.
        place_interpolated(m, at);
        m.previous = m.position;
        m.previous_yaw = m.yaw;
        m.previous_head_pitch = m.head_pitch;
    }
    m.on_ground = update.on_ground;
}

fn step_remote_armor_stand(
    stepped: &mut HashMap<i32, SteppedInterpolation>,
    id: i32,
    a: &mut horizon_client::entities::ArmorStand,
    update: MoveUpdate,
) {
    let entity = PosRot {
        position: [a.position.x, a.position.y, a.position.z],
        yaw: a.yaw,
        pitch: 0.0,
    };
    let handler = stepped.entry(id).or_insert_with(|| {
        let interval = horizon_client::lan::tracking::settings("minecraft:armor_stand").1;
        SteppedInterpolation::new(interval as i32)
    });
    if let Some(at) = stepped_move(
        handler,
        update.packet,
        entity,
        update.position,
        update.path,
        update.rotation,
    ) {
        a.snap(
            Vec3::new(at.position[0], at.position[1], at.position[2]),
            at.yaw,
        );
    }
    a.on_ground = update.on_ground;
}

impl Game<'_> {
    /// `Entity.isVehicle` on a client: the entities a tracked entity rides.
    pub(super) fn remote_vehicle_ids_body(&self) -> impl Iterator<Item = u32> + '_ {
        self.remote
            .iter()
            .flat_map(|r| r.entities.values().filter_map(|t| t.vehicle))
    }
    /// The `SetPassengers` bookkeeping of the tracked entities.
    pub(super) fn remote_seat_passengers_body(&mut self, vehicle: u32, passengers: &[i32]) {
        if let Some(r) = self.remote.as_mut() {
            seat_passengers(&mut r.entities, vehicle, passengers);
        }
    }
    /// `recreateFromPacket`: yRot and xRot from the packet, the body and head
    /// both at its head yaw (LivingEntity).
    pub(super) fn client_attachment_entity_exists_body(&self, holder: Holder) -> bool {
        self.remote.as_ref().is_some_and(|remote| match holder {
            Holder::Entity(id) => remote.entities.contains_key(&(id as i32)),
            Holder::Player(uuid) => {
                remote.profile == uuid || remote.level_players.values().any(|p| *p == uuid)
            }
            _ => false,
        })
    }

    pub(super) fn spawn_entity_body(
        &mut self,
        id: i32,
        kind: &str,
        position: [f64; 3],
        rotation: [f32; 3],
    ) {
        self.detach_remote_tracking_entity(id);
        let [yaw, pitch, head] = rotation;
        let at = Vec3::new(position[0], position[1], position[2]);
        let mob_type = if kind == "minecraft:player" {
            self.remote.as_ref().and_then(|r| r.player_type.clone())
        } else {
            self.res.mob_types().get(kind).cloned()
        };
        let is_mob = mob_type.is_some();
        if kind == "minecraft:ender_dragon" {
            self.spawn_remote_dragon(id, position, yaw);
        }
        if kind == "minecraft:experience_orb" {
            // Its value (ExperienceOrb.DATA_VALUE) follows in SetEntityData.
            let orb = ExperienceOrb::new(id as u32, at, Vec3::ZERO, 1);
            if let Some(r) = self.remote.as_mut() {
                r.entity_indices
                    .orbs
                    .replace_or_push(&mut self.orbs, |o| o.id, orb);
            } else {
                self.orbs.retain(|o| o.id != id as u32);
                self.orbs.push(orb);
            }
        }
        if let Some(t) = mob_type {
            // The window's copy of the server's mob: no mod part goals —
            // their `register_goals` ran on the authoritative side (arismoko/horizon#1077).
            let mut mob = Mob::new_remote(id as u32, t, at, head);
            super::super::remote_mob_data::client_defaults(&mut mob);
            (mob.yaw, mob.previous_yaw) = (yaw, yaw);
            (mob.head_pitch, mob.previous_head_pitch) = (pitch, pitch);
            mob.persistent = true;
            self.mob_index.replace_or_push(&mut self.mobs, mob);
        }
        if let Some(r) = self.remote.as_mut() {
            // A new entity has a new interpolation handler (a mob's or an
            // armor stand's), even under an id the server used before.
            r.stepped.remove(&id);
            if kind == "minecraft:item" {
                r.pending_items.insert(id, (position, Vec3::ZERO));
            }
            r.entities.insert(
                id,
                Tracked {
                    kind: kind.to_owned(),
                    position,
                    codec_base: position,
                    silent: false,
                    bucket: EntityBucket::classify(kind, is_mob),
                    vehicle: None,
                },
            );
        }
    }
    /// A move packet for entity `id`: to the end of `path` (None: a
    /// rotation alone), turned to `rotation` if it has one. `packet` says
    /// whether a living entity more than 64 blocks away is put there at once
    /// (its `snaps`).
    pub(super) fn move_entity_body(
        &mut self,
        id: i32,
        path: Option<Path>,
        rotation: Option<(f32, f32)>,
        on_ground: bool,
        packet: MovePacket,
    ) {
        // The dragon is drawn from its own state (game/mobs/dragon.rs), which
        // tick_remote_dragon eases to the server's moves through `lerps`. Its
        // copy in the mob list draws nothing: given the move, it kept the
        // dragon where it spawned (arismoko/horizon#2083).
        let dragon = self.dragon.as_ref().is_some_and(|d| d.id as i32 == id);
        let driven = self.driven_mount().is_some_and(|m| m.id as i32 == id);
        let tracked = self
            .remote
            .as_ref()
            .and_then(|r| r.entities.get(&id))
            .map(|t| (t.position, t.bucket));
        let bucket = tracked.map(|(_, b)| b);
        let position = path
            .as_ref()
            .map(Path::end)
            .or_else(|| tracked.map(|(p, _)| p))
            .or_else(|| {
                (!dragon)
                    .then(|| {
                        self.mob_index
                            .get_stamped(&self.mobs, id as u32)
                            .map(|m| [m.position.x, m.position.y, m.position.z])
                    })
                    .flatten()
            })
            .unwrap_or([0.0; 3]);
        let at = Vec3::new(position[0], position[1], position[2]);
        let mut boat_turn: Option<usize> = None;
        let mut object_display = false;
        let mut untracked_tail = false;
        if let Some(r) = self.remote.as_mut() {
            if let Some(t) = r.entities.get_mut(&id) {
                t.position = position;
            }
            if driven {
                // handleMoveEntity, handleEntityPositionSync: the mount this
                // client moves (isLocalInstanceAuthoritative) keeps its own
                // position, and no steps pile up for after the ride.
                r.stepped.remove(&id);
                return;
            }
            let insert_lerp = |lerps: &mut HashMap<i32, Lerp>| {
                // AbstractInterpolationHandler.interpolateTo: without a
                // rotation, the one it was heading for.
                let rotation = rotation.or_else(|| lerps.get(&id).and_then(|l| l.rotation));
                lerps.insert(
                    id,
                    Lerp {
                        target: position,
                        rotation,
                        steps: 3,
                    },
                );
            };
            match bucket {
                Some(EntityBucket::Mob) => {
                    if !dragon && let Some(m) = self.mob_index.get_mut(&mut self.mobs, id as u32) {
                        step_remote_mob(
                            &mut r.stepped,
                            id,
                            m,
                            MoveUpdate {
                                position,
                                path,
                                rotation,
                                on_ground,
                                packet,
                            },
                        );
                        return;
                    }
                    insert_lerp(&mut r.lerps);
                    return;
                }
                Some(EntityBucket::Dragon) => {
                    insert_lerp(&mut r.lerps);
                    return;
                }
                Some(EntityBucket::ArmorStand) => {
                    // An armor stand is a living entity too: the same handler, over
                    // its type's update interval, stepped in `tick_remote_entities`.
                    if let Some(a) = r.entity_indices.armor_stands.get_mut_stamped(
                        &mut self.armor_stands,
                        |a| a.id,
                        id as u32,
                    ) {
                        step_remote_armor_stand(
                            &mut r.stepped,
                            id,
                            a,
                            MoveUpdate {
                                position,
                                path,
                                rotation,
                                on_ground,
                                packet,
                            },
                        );
                        return;
                    }
                    insert_lerp(&mut r.lerps);
                    return;
                }
                Some(EntityBucket::Projectile) => {
                    // Projectiles keep Entity's NO_OP InterpolationHandler too: the
                    // server's position replaces theirs at once and their own tick
                    // flies them on (`tick_remote_projectiles`).
                    if let Some(i) = r.entity_indices.projectiles.find_stamped(
                        &self.creatures.projectiles,
                        |p| p.id,
                        id as u32,
                    ) {
                        self.creatures.projectiles.update(i, |p| {
                            if path.is_some() {
                                p.position = at;
                            }
                            if let Some((yaw, pitch)) = rotation {
                                p.yaw = yaw;
                                p.pitch = pitch;
                            }
                        });
                        return;
                    }
                    insert_lerp(&mut r.lerps);
                    return;
                }
                Some(EntityBucket::Hook) => {
                    // A fishing hook keeps a LinearInterpolationHandler: eased towards the
                    // update over three ticks, on top of its own movement (`tick_remote_hooks`).
                    if path.is_some() {
                        if let Some(h) = self.hook.as_mut().filter(|h| h.id() == id as u32) {
                            h.move_to(position, on_ground);
                            return;
                        }
                        if let Some(h) = self.remote_hooks.iter_mut().find(|h| h.id() == id as u32)
                        {
                            h.move_to(position, on_ground);
                            return;
                        }
                    }
                    insert_lerp(&mut r.lerps);
                    return;
                }
                Some(EntityBucket::Trident) => {
                    // A thrown trident is an arrow: placed at once, flown on by its tick.
                    if let Some(t) = r.entity_indices.tridents.get_mut_stamped(
                        &mut self.tridents,
                        |t| t.id,
                        id as u32,
                    ) {
                        if path.is_some() {
                            t.position = at;
                        }
                        if let Some((yaw, pitch)) = rotation {
                            t.yaw = yaw;
                            t.pitch = pitch;
                        }
                        return;
                    }
                    insert_lerp(&mut r.lerps);
                    return;
                }
                Some(EntityBucket::Rocket) => {
                    // So does a firework rocket: its own tick flies it on.
                    if path.is_some()
                        && let Some(f) = r.entity_indices.rockets.get_mut_stamped(
                            &mut self.rockets,
                            |f| f.id,
                            id as u32,
                        )
                    {
                        f.position = at;
                        return;
                    }
                    insert_lerp(&mut r.lerps);
                    return;
                }
                Some(EntityBucket::Dropped) => {
                    if let Some(p) = r.pending_items.get_mut(&id) {
                        p.0 = position;
                    }
                    // ItemEntity keeps Entity's NO_OP InterpolationHandler.
                    if let Some(d) = r.entity_indices.dropped.get_mut_stamped(
                        &mut self.dropped,
                        |d| d.id,
                        id as u32,
                    ) {
                        d.position = at;
                        d.on_ground = on_ground;
                        return;
                    }
                    insert_lerp(&mut r.lerps);
                    return;
                }
                Some(EntityBucket::Falling) => {
                    if let Some(i) =
                        r.entity_indices
                            .falling
                            .find_stamped(&self.falling, |f| f.id, id as u32)
                    {
                        self.falling.update(i, |f| f.position = at);
                        return;
                    }
                    insert_lerp(&mut r.lerps);
                    return;
                }
                Some(EntityBucket::Tnt) => {
                    if let Some(t) =
                        r.entity_indices
                            .tnts
                            .get_mut_stamped(&mut self.tnts, |t| t.id, id as u32)
                    {
                        t.position = at;
                        t.on_ground = on_ground;
                        return;
                    }
                    insert_lerp(&mut r.lerps);
                    return;
                }
                Some(EntityBucket::Orb) => {
                    // ExperienceOrb.createInterpolationHandler is a
                    // LinearInterpolationHandler: the correction is spread over 3
                    // ticks (`tick_remote_entities`), while `ExperienceOrb.tick` keeps
                    // moving it.
                    if let Some(o) =
                        r.entity_indices
                            .orbs
                            .get_mut_stamped(&mut self.orbs, |o| o.id, id as u32)
                    {
                        o.on_ground = on_ground;
                        r.lerps.insert(
                            id,
                            Lerp {
                                target: position,
                                rotation: None,
                                steps: 3,
                            },
                        );
                        return;
                    }
                    insert_lerp(&mut r.lerps);
                    return;
                }
                Some(EntityBucket::Cart) => {
                    let cart_idx =
                        r.entity_indices
                            .carts
                            .find_stamped(&self.carts, |c| c.id, id as u32);
                    if r.cart_lerps.contains_key(&id) {
                        // A minecart on NewMinecartBehavior keeps Entity's NO_OP
                        // InterpolationHandler: a teleport puts it there at once.
                        if let Some(i) = cart_idx {
                            self.carts.update(i, |c| {
                                if path.is_some() {
                                    c.position = at;
                                }
                                if let Some((yaw, pitch)) = rotation {
                                    c.yaw = yaw;
                                    c.pitch = pitch;
                                }
                            });
                        }
                    } else {
                        insert_lerp(&mut r.lerps);
                        if let Some(i) = cart_idx
                            && let Some((yaw, _)) = rotation
                        {
                            self.carts.update(i, |c| c.yaw = yaw);
                        }
                    }
                    return;
                }
                Some(EntityBucket::Boat) => {
                    insert_lerp(&mut r.lerps);
                    boat_turn =
                        r.entity_indices
                            .boats
                            .find_stamped(&self.boats, |b| b.id, id as u32);
                }
                Some(EntityBucket::Object) => {
                    // Interaction extends Entity and keeps its NO_OP interpolation
                    // handler: the server's position replaces its own at once.
                    if let Some(i) = self.interactions.get_mut(&id) {
                        if path.is_some() {
                            i.position = at;
                        }
                        return;
                    }
                    insert_lerp(&mut r.lerps);
                    object_display = true;
                }
                Some(EntityBucket::Other) => {
                    if path.is_some()
                        && (self.potions.move_remote_thrown(id as u32, at)
                            || self.end.move_remote_eye(id as u32, position))
                    {
                        return;
                    }
                    insert_lerp(&mut r.lerps);
                    return;
                }
                None => {
                    if !dragon && let Some(m) = self.mob_index.get_mut(&mut self.mobs, id as u32) {
                        step_remote_mob(
                            &mut r.stepped,
                            id,
                            m,
                            MoveUpdate {
                                position,
                                path,
                                rotation,
                                on_ground,
                                packet,
                            },
                        );
                        return;
                    }
                    if let Some(a) = r.entity_indices.armor_stands.get_mut_stamped(
                        &mut self.armor_stands,
                        |a| a.id,
                        id as u32,
                    ) {
                        step_remote_armor_stand(
                            &mut r.stepped,
                            id,
                            a,
                            MoveUpdate {
                                position,
                                path,
                                rotation,
                                on_ground,
                                packet,
                            },
                        );
                        return;
                    }
                    if let Some(i) = self.interactions.get_mut(&id) {
                        if path.is_some() {
                            i.position = at;
                        }
                        return;
                    }
                    if let Some(i) = r.entity_indices.projectiles.find_stamped(
                        &self.creatures.projectiles,
                        |p| p.id,
                        id as u32,
                    ) {
                        self.creatures.projectiles.update(i, |p| {
                            if path.is_some() {
                                p.position = at;
                            }
                            if let Some((yaw, pitch)) = rotation {
                                p.yaw = yaw;
                                p.pitch = pitch;
                            }
                        });
                        return;
                    }
                    if path.is_some()
                        && (self.potions.move_remote_thrown(id as u32, at)
                            || self.end.move_remote_eye(id as u32, position))
                    {
                        return;
                    }
                    if path.is_some() {
                        if let Some(h) = self.hook.as_mut().filter(|h| h.id() == id as u32) {
                            h.move_to(position, on_ground);
                            return;
                        }
                        if let Some(h) = self.remote_hooks.iter_mut().find(|h| h.id() == id as u32)
                        {
                            h.move_to(position, on_ground);
                            return;
                        }
                    }
                    if let Some(t) = r.entity_indices.tridents.get_mut_stamped(
                        &mut self.tridents,
                        |t| t.id,
                        id as u32,
                    ) {
                        if path.is_some() {
                            t.position = at;
                        }
                        if let Some((yaw, pitch)) = rotation {
                            t.yaw = yaw;
                            t.pitch = pitch;
                        }
                        return;
                    }
                    if path.is_some()
                        && let Some(f) = r.entity_indices.rockets.get_mut_stamped(
                            &mut self.rockets,
                            |f| f.id,
                            id as u32,
                        )
                    {
                        f.position = at;
                        return;
                    }
                    if let Some(p) = r.pending_items.get_mut(&id) {
                        p.0 = position;
                    }
                    if let Some(d) = r.entity_indices.dropped.get_mut_stamped(
                        &mut self.dropped,
                        |d| d.id,
                        id as u32,
                    ) {
                        d.position = at;
                        d.on_ground = on_ground;
                        return;
                    }
                    if let Some(i) =
                        r.entity_indices
                            .falling
                            .find_stamped(&self.falling, |f| f.id, id as u32)
                    {
                        self.falling.update(i, |f| f.position = at);
                        return;
                    }
                    if let Some(t) =
                        r.entity_indices
                            .tnts
                            .get_mut_stamped(&mut self.tnts, |t| t.id, id as u32)
                    {
                        t.position = at;
                        t.on_ground = on_ground;
                        return;
                    }
                    if let Some(o) =
                        r.entity_indices
                            .orbs
                            .get_mut_stamped(&mut self.orbs, |o| o.id, id as u32)
                    {
                        o.on_ground = on_ground;
                        r.lerps.insert(
                            id,
                            Lerp {
                                target: position,
                                rotation: None,
                                steps: 3,
                            },
                        );
                        return;
                    }
                    if r.cart_lerps.contains_key(&id) {
                        if let Some(i) =
                            r.entity_indices
                                .carts
                                .find_stamped(&self.carts, |c| c.id, id as u32)
                        {
                            self.carts.update(i, |c| {
                                if path.is_some() {
                                    c.position = at;
                                }
                                if let Some((yaw, pitch)) = rotation {
                                    c.yaw = yaw;
                                    c.pitch = pitch;
                                }
                            });
                        }
                    } else {
                        insert_lerp(&mut r.lerps);
                    }
                    untracked_tail = true;
                }
            }
        } else {
            untracked_tail = true;
        }
        if let Some(i) = boat_turn {
            // The boat the player drives turns here (isLocalInstanceAuthoritative).
            let driven_boat = self.riding.filter(|&bid| self.controls_boat(bid));
            self.boats.update(i, |b| {
                if Some(id as u32) == driven_boat {
                    return;
                }
                b.on_ground = on_ground;
            });
        } else if object_display {
            self.move_remote_text_display(id, position, rotation);
        } else if untracked_tail {
            self.turn_remote_object(id, rotation.map(|(yaw, _)| yaw), on_ground);
            self.move_remote_text_display(id, position, rotation);
            self.turn_remote_projectile(id, rotation);
        }
    }
    pub(super) fn detach_remote_tracking_entity_body(&mut self, id: i32) {
        let mobs = &self.mobs;
        let index = &mut self.mob_index;
        self.particles
            .detach_tracking_target(TrackingTarget::Entity(id), || {
                index.get_stamped(mobs, id as u32).map(mob_tracking_body)
            });
    }
}
