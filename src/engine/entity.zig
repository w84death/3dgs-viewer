const std = @import("std");
const rl = @import("raylib");
const Vec2 = @import("math.zig").Vec2;
const IVec2 = @import("math.zig").IVec2;
const GRAVITY = 98.1;
const WINDOW_WIDTH = 800;
const WINDOW_HEIGHT = 600;

pub const EntityType = enum { Player, Enemy, Pickup, Powerup };
pub const Entity = struct {
    kind: EntityType,
    pos: Vec2,
    size: IVec2,
    vel: Vec2 = Vec2.init(0, 0),
    max_vel: f32 = 250.0,
    speed: f32 = 0.0,
    color: rl.Color = rl.Color.white,
    health: i16 = 100,

    frames: []const rl.Texture,
    frame_count: usize = 0,
    current_frame: usize = 0,
    frame_timer: f32 = 0.0,
    frame_duration: f32 = 0.1,

    flying: bool = false,
    sfx: rl.Sound,

    pub fn init(kind: EntityType, x: f32, y: f32, speed: f32, frames: []const rl.Texture, frame_duration: f32, sfx: rl.Sound, flying: bool) Entity {
        return .{
            .kind = kind,
            .pos = Vec2.init(x, y),
            .size = IVec2{ .x = frames[0].width, .y = frames[0].height },
            .speed = speed,
            .frames = frames,
            .frame_count = frames.len,
            .frame_duration = frame_duration,
            .sfx = sfx,
            .flying = flying,
        };
    }

    pub fn getRect(self: Entity) struct { x: i32, y: i32, w: i32, h: i32 } {
        return .{
            .x = @intFromFloat(self.pos.x),
            .y = @intFromFloat(self.pos.y),
            .w = self.size.x,
            .h = self.size.y,
        };
    }

    pub fn update(self: *Entity, dt: f32) void {
        if (!self.flying) self.vel.y += GRAVITY * dt;
        const new_pos = self.pos.add(.{ .x = self.vel.x * dt, .y = self.vel.y * dt });
        const window_w: f32 = @floatFromInt(WINDOW_WIDTH);
        const window_h: f32 = @floatFromInt(WINDOW_HEIGHT);
        const size_w: f32 = @floatFromInt(self.size.x);
        const size_h: f32 = @floatFromInt(self.size.y);
        var play_sfx = false;

        if (new_pos.x <= 0 or new_pos.x + size_w >= window_w) {
            self.vel.x = -self.vel.x;
            play_sfx = true;
        }

        if (new_pos.y <= 0 or new_pos.y + size_h >= window_h) {
            self.vel.y = -self.vel.y;
            play_sfx = true;
        }

        self.pos = self.pos.add(.{ .x = self.vel.x * dt, .y = self.vel.y * dt });
        self.pos.x = std.math.clamp(self.pos.x, 0.0, window_w - size_w);
        self.pos.y = std.math.clamp(self.pos.y, 0.0, window_h - size_h);

        if (self.flying) {
            self.vel.x *= 0.98;
            self.vel.y *= 0.98;
        }

        self.updateAnimation(dt);
        if (play_sfx) rl.playSound(self.sfx);
    }

    pub fn updateAnimation(self: *Entity, dt: f32) void {
        if (self.frame_count <= 1) return;
        self.frame_timer += dt;
        if (self.frame_timer >= self.frame_duration) {
            self.frame_timer -= self.frame_duration;
            self.current_frame = (self.current_frame + 1) % self.frame_count;
        }
    }

    pub fn draw(self: Entity) void {
        const tex = self.frames[self.current_frame];
        var w: f32 = @floatFromInt(tex.width);
        const h: f32 = @floatFromInt(tex.height);
        if (self.vel.x < 0) w *= -1.0;
        rl.drawTextureRec(tex, .{ .x = 0, .y = 0, .width = w, .height = h }, rl.Vector2.init(self.pos.x, self.pos.y), self.color);
    }

    pub fn setVelocity(self: *Entity, vx: f32, vy: f32) void {
        self.vel = Vec2.init(vx, vy);
    }

    pub fn addVel(self: *Entity, vel: Vec2) void {
        self.vel = self.vel.add(Vec2.init(vel.x * self.speed, vel.y * self.speed));
        self.vel = self.vel.clamp(-self.max_vel, self.max_vel);
    }

    pub fn collidesWidth(self: Entity, other: Entity) bool {
        const a = self.getRect();
        const b = other.getRect();
        return a.x < b.x + b.w and
            a.x + a.w > b.x and
            a.y < b.y + b.h and
            a.y + a.h > b.y;
    }
};
