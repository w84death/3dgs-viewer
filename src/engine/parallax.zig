const rl = @import("raylib");
const Vec2 = @import("math.zig").Vec2;
const std = @import("std");
const SEED = 666;
const MAX_SPRITES = 64;
const MAX_SPRITES_PER_LAYER = 64;
const WORLD_WIDTH = 1200.0;

const ParallaxSprite = struct {
    x: f32,
    y: f32,
    tex_index: u8,
    tint: rl.Color,
};

pub const ParallaxLayer = struct {
    sprites: [MAX_SPRITES]ParallaxSprite,
    count: usize,
    parallax: f32,
    textures: []const rl.Texture,

    pub fn draw(self: *const ParallaxLayer, camera_x: f32) void {
        const offset_x = camera_x * self.parallax;
        for (self.sprites[0..self.count]) |spr| {
            const screen_x = spr.x - offset_x;
            if (screen_x > -400 and screen_x < 800 + 400) {
                const tex = self.textures[spr.tex_index];
                rl.drawTexture(tex, @intFromFloat(screen_x), @intFromFloat(spr.y), spr.tint);
            }
        }
    }
};

pub fn make(
    textures: []const rl.Texture,
    parallax: f32,
    ground_y: f32,
    density: f32,
    tint: rl.Color,
    seed_offset: u64,
) ParallaxLayer {
    var layer = ParallaxLayer{
        .sprites = undefined,
        .count = 0,
        .parallax = parallax,
        .textures = textures,
    };
    fillLayer(&layer, ground_y, density, tint, seed_offset, WORLD_WIDTH);
    return layer;
}

fn fillLayer(
    layer: *ParallaxLayer,
    ground_y: f32,
    base_density: f32,
    tint: rl.Color,
    seed_offset: u64,
    world_width: f32,
) void {
    var rng = std.Random.DefaultPrng.init(SEED + seed_offset);
    const rand = rng.random();

    var x: f32 = -500;
    var idx: usize = 0;

    while (x < world_width + 500 and idx < MAX_SPRITES_PER_LAYER) : ({
        x += base_density + rand.float(f32) * base_density * 0.7;
        idx += 1;
    }) {
        const tex_idx = rand.uintLessThan(u8, @as(u8, @intCast(layer.textures.len)));
        const tex = layer.textures[tex_idx];
        const h_offset = rand.float(f32) * @as(f32, @floatFromInt(tex.height)) * 0.6;
        const y = ground_y - @as(f32, @floatFromInt(tex.height)) + h_offset;

        layer.sprites[idx] = ParallaxSprite{
            .x = x,
            .y = y,
            .tex_index = tex_idx,
            .tint = tint,
        };
    }
    layer.count = idx;
}
